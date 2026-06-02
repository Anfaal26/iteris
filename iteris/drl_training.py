"""
DRL training loop.

Top-level entry point: `run_drl_training(cfg, train_samples, val_samples)`.
Each sample dict has keys {image, gt_mask, init_mask, patient, ...} produced
by `warm_start.precompute_init_masks()`.

Handles:
  - per-sample state caches (avoid recomputing image + init_mask channels)
  - buffer pre-fill with random rollouts
  - epsilon-greedy decay for DQN-family
  - periodic validation evaluation with greedy policy
  - best-checkpoint saving + training history return
"""

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from .env     import (SegmentationEnv, signed_dt, dice_score, hd95_px)
from .buffer  import ReplayBuffer
from .agents  import DQNAgent, DuelingDQNAgent, DDPGAgent


# ─── Environment registry ────────────────────────────────────────────────────
# All discrete agents use SegmentationEnv (14-action local mask refinement).
# 'default' is the only active env_class; it applies to both CAMUS and BRISC
# (BRISC uses fail_thresh / fail_n config keys for fail-fast termination).
# Archived: ContourTracingEnv (Paradigm 1, boundary tracing) — see
#   iteris/archive/paradigm1_boundary_tracing/env_contour.py
ENV_REGISTRY = {
    'default': SegmentationEnv,
}

CONTINUOUS_ACTION_DIM = SegmentationEnv.CONTINUOUS_ACTION_DIM   # 3


AGENT_REGISTRY = {
    'DQN':     (DQNAgent,        'discrete'),   # 14-action local mask refinement
    'DUELING': (DuelingDQNAgent, 'discrete'),   # same, with V+A dueling heads
    'DDPG':    (DDPGAgent,       'continuous'), # 3-D continuous morph+shift baseline
}


def _build_state_caches(samples: List[dict], image_size: int) -> dict:
    """Stack image + init_mask arrays for vectorised lookup at sample time."""
    n = len(samples)
    return dict(
        image     = np.stack([s['image']     for s in samples]).astype(np.float32),
        gt_mask   = np.stack([s['gt_mask']   for s in samples]).astype(np.uint8),
        init_mask = np.stack([s['init_mask'] for s in samples]).astype(np.uint8),
    )


def _make_state_builder(caches: dict, sdt_clip: float):
    """
    Return a builder (idx, current_mask, cached_sdt=None) → torch (4, H, W).

    If `cached_sdt` is provided (e.g. from ReplayBuffer with cache_sdt=True),
    skip the expensive scipy.ndimage SDT recomputation. This is the single
    biggest training-speed optimisation — without it, agent.update() spends
    ~80% of its time recomputing SDTs that were already computed during
    env.step().
    """
    def build(idx, current_mask, cached_sdt=None):
        image = caches['image'][idx]
        init  = caches['init_mask'][idx].astype(np.float32)
        cur   = current_mask.astype(np.float32)
        if cached_sdt is not None:
            sdt = cached_sdt.astype(np.float32)
        else:
            sdt = signed_dt(current_mask, sdt_clip)
        return torch.from_numpy(np.stack([image, cur, sdt, init], axis=0))
    return build


def _make_env(caches: dict, idx: int, env_kwargs: dict,
              env_cls=SegmentationEnv) -> SegmentationEnv:
    return env_cls(
        image     = caches['image'][idx],
        gt_mask   = caches['gt_mask'][idx],
        init_mask = caches['init_mask'][idx],
        **env_kwargs,
    )


def _save_agent(agent, history, best_dice, path):
    torch.save({
        'agent':     agent.state_dict(),
        'best_dice': best_dice,
        'history':   history,
    }, path)


@torch.no_grad()
def evaluate_agent(agent, samples_or_caches, env_kwargs, env_cls=SegmentationEnv) -> dict:
    """Greedy policy evaluation over a sample list. Returns aggregate metrics.

    ``env_cls`` must match the env class used at training time — otherwise the
    agent's num_actions head will mismatch the env's discrete action space.
    """
    if isinstance(samples_or_caches, dict):
        caches = samples_or_caches
        n = len(caches['image'])
        samples = None
    else:
        samples = samples_or_caches
        n = len(samples)
        caches = _build_state_caches(samples, samples[0]['image'].shape[0])

    init_d, final_d, final_h = [], [], []
    for i in range(n):
        env = _make_env(caches, i, env_kwargs, env_cls=env_cls)
        state = env.reset()
        init_d.append(env.dice_history[0])
        while True:
            if agent.action_type == 'discrete':
                a = agent.select_action(state, epsilon=0.0)
            else:
                a = agent.select_action(state, explore=False)
            state, _, done, info = env.step(a)
            if done:
                break
        final_d.append(info['dice'])
        final_h.append(info['hd95'])

    return dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),
        final_hd95_mean = float(np.nanmean(final_h)),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
    )


def run_drl_training(
    cfg: dict,
    train_samples: List[dict],
    val_samples:   List[dict],
) -> dict:
    """Train one DRL agent for one (dataset, structure) combination."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.get('env_class') == 'contour_tracing':

    seed = cfg.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent_cls, action_type = AGENT_REGISTRY[cfg['agent_type'].upper()]
    image_size = cfg['image_size']
    H = image_size

    # ── Environment class selection ──────────────────────────────────────────
    # All agents use SegmentationEnv (14-action local mask refinement).
    # env_class is always 'default' — kept as a config key for forward compat.
    env_class_name = cfg.get('env_class', 'default')
    if action_type == 'continuous':
        env_cls = SegmentationEnv
    else:
        if env_class_name not in ENV_REGISTRY:
            raise ValueError(
                f"Unknown env_class '{env_class_name}'. "
                f"Available: {list(ENV_REGISTRY)}"
            )
        env_cls = ENV_REGISTRY[env_class_name]
    num_actions_for_cls = env_cls.NUM_DISCRETE_ACTIONS
    print(f"[drl] env_class={env_class_name} ({env_cls.__name__}) "
          f"| discrete actions={num_actions_for_cls} | action_type={action_type}")

    # ── Caches ───────────────────────────────────────────────────────────────
    train_caches = _build_state_caches(train_samples, image_size)
    val_caches   = _build_state_caches(val_samples,   image_size)

    # ── Hard sample mining ────────────────────────────────────────────────────
    # Problem: most training samples already have high init Dice (e.g. 0.93 for
    # CAMUS LV_endo). On those, every action except no-op degrades performance.
    # The Q-network is overwhelmed by "everything is bad" signal from the easy
    # majority and never properly learns to refine the rare hard samples where
    # improvement is actually achievable.
    #
    # Fix: exponential weighting that preferentially samples lower-init-Dice
    # cases. Weight ∝ exp((1-init_dice) * hard_mining_scale).
    #   scale=0  → uniform (off)
    #   scale=5  → mild: dice=0.75 gets ~2.5× more training than dice=0.93
    #   scale=10 → strong: dice=0.75 gets ~7× more training than dice=0.93
    #
    # Keeps easy samples in the mix (they teach "no-op when already good"),
    # but amplifies learning signal from samples where improvement is possible.
    _hard_mining_scale = cfg.get('hard_mining_scale', 0.0)
    if _hard_mining_scale > 0:
        _init_dices = np.array([
            dice_score(train_caches['init_mask'][i], train_caches['gt_mask'][i])
            for i in range(len(train_samples))
        ])
        _raw = (1.0 - _init_dices) * _hard_mining_scale
        _raw -= _raw.max()   # numerical stability before exp
        _sample_weights = np.exp(_raw)
        _sample_weights /= _sample_weights.sum()
        hard_pct = (_init_dices < 0.90).mean() * 100
        print(f'[drl] Hard sample mining ON: scale={_hard_mining_scale:.1f} | '
              f'init_dice mean={_init_dices.mean():.4f} | '
              f'hard (<0.90): {hard_pct:.1f}% of train')
        def _sample_idx():
            return int(np.random.choice(len(train_samples), p=_sample_weights))
    else:
        def _sample_idx():
            return np.random.randint(len(train_samples))

    # cont_action_scale is the legacy single-scale key (old per-agent configs).
    # New per-class configs supply cont_morph_scale / cont_trans_scale separately.
    _legacy_scale = cfg.get('cont_action_scale', 0.02)
    # action_type is always passed; everything else only goes through if the
    # cfg explicitly sets it.  This lets the chosen env class (e.g. BRISC)
    # apply its own defaults for max_steps / stop_n / stop_eps_dice rather
    # than being clobbered by hardcoded base-env defaults from this dict.
    env_kwargs = {'action_type': action_type}
    _env_optional_keys = (
        'max_steps', 'shift_px', 'sdt_clip', 'reward_clip',
        'cont_morph_scale', 'cont_trans_scale',
        'reward_mode', 'reward_alpha', 'reward_beta', 'hd_norm',
        'stop_eps_dice', 'stop_eps_hd', 'stop_n',
        'fail_thresh', 'fail_n',                # BRISC env extras
    )
    for _k in _env_optional_keys:
        if _k in cfg:
            env_kwargs[_k] = cfg[_k]
    # cont_trans_scale legacy fallback (old configs use cont_action_scale)
    if 'cont_trans_scale' not in env_kwargs and 'cont_action_scale' in cfg:
        env_kwargs['cont_trans_scale'] = cfg['cont_action_scale']
    # state_builder needs sdt_clip even if not in cfg (env class default applies).
    state_builder = _make_state_builder(train_caches, cfg.get('sdt_clip', 20.0))

    # ── Agent ─────────────────────────────────────────────────────────────────
    # branch below is retained for completeness but no shipped config routes a
    common = dict(in_channels=4, gamma=cfg.get('gamma', 0.99),
                  tau=cfg.get('tau', 0.005),
                  embed_dim=cfg.get('embed_dim', 256), device=device)
    if action_type == 'discrete':
        # num_actions is env-class-dependent — agent's output head must match
        # exactly or argmax will produce invalid indices.
        agent = agent_cls(num_actions=num_actions_for_cls,
                          lr=cfg.get('lr', 1e-4), **common)
    else:
        common.pop('lr', None)
        _legacy_scale = cfg.get('cont_action_scale', 0.02)
        morph_scale   = cfg.get('cont_morph_scale', 0.25)
        trans_scale   = cfg.get('cont_trans_scale', _legacy_scale)
        # 3-component: [morph, dy, dx] — see env.SegmentationEnv docstring.
        action_scale  = [morph_scale, trans_scale, trans_scale]
        # ou_sigma from config may be scalar (legacy) or list (per-component)
        ou_sigma_raw  = cfg.get('ou_sigma', None)   # None → DDPGAgent default (10% of scale)
        agent = agent_cls(
            action_dim         = CONTINUOUS_ACTION_DIM,
            action_scale       = action_scale,
            actor_lr           = cfg.get('actor_lr',  1e-4),
            critic_lr          = cfg.get('critic_lr', 1e-3),
            ou_theta           = cfg.get('ou_theta', 0.15),
            ou_sigma           = ou_sigma_raw,
            actor_freeze_steps = cfg.get('actor_freeze_steps', 2000),
            **common,
        )

    # ── Buffer ────────────────────────────────────────────────────────────────
    # cache_sdt=True is the key speed optimisation: skips ~80% of SDT recomputation
    # in agent.update() by reusing the SDT computed during env.step().
    buffer = ReplayBuffer(
        capacity   = cfg.get('buffer_size', 10000),
        mask_shape = (H, H),
        # 3-component continuous action: (morph, dy_norm, dx_norm).
        action_dim = CONTINUOUS_ACTION_DIM if action_type == 'continuous' else None,
        discrete   = (action_type == 'discrete'),
        # cache_sdt FORCED ON: reusing the SDT computed during env.step() skips
        # ~80% of SDT recomputation in agent.update() — the single biggest
        # speed optimisation in the training loop.
        cache_sdt  = True,
    )

    # ── Pre-fill buffer with random rollouts ──────────────────────────────────
    prefill_steps = cfg.get('prefill_steps', 2000)
    print(f'[drl] Pre-filling buffer with {prefill_steps} random transitions...')
    # Build per-component ranges for random continuous actions
    _legacy_scale = cfg.get('cont_action_scale', 0.02)
    _morph_scale  = cfg.get('cont_morph_scale', 0.25)
    _trans_scale  = cfg.get('cont_trans_scale', _legacy_scale)

    while len(buffer) < prefill_steps:
        idx = _sample_idx()
        env = _make_env(train_caches, idx, env_kwargs, env_cls=env_cls)
        prev_state = env.reset()           # state = stack([img, mask, SDT, init])
        while True:
            if action_type == 'discrete':
                a = np.random.randint(num_actions_for_cls)
            else:
                # 3-component: (morph, dy_norm, dx_norm)
                a = np.array([
                    np.random.uniform(-_morph_scale,  _morph_scale),
                    np.random.uniform(-_trans_scale,  _trans_scale),
                    np.random.uniform(-_trans_scale,  _trans_scale),
                ], dtype=np.float32)
            prev_mask = env.mask.copy()
            prev_sdt  = prev_state[2]      # SDT slot of state tensor
            next_state, r, done, _ = env.step(a)
            buffer.push(idx, prev_mask, a, r, env.mask.copy(), done,
                        current_sdt=prev_sdt, next_sdt=next_state[2])
            prev_state = next_state
            if done or len(buffer) >= prefill_steps:
                break
    print(f'[drl] Buffer size: {len(buffer)}  (SDT cache: {buffer.cache_sdt})')

    # ── Main training loop ────────────────────────────────────────────────────
    train_steps         = cfg.get('train_steps', 50000)
    eval_every          = cfg.get('eval_every',   2000)
    batch_size          = cfg.get('batch_size',   64)
    eps_start           = cfg.get('epsilon_start',       1.0)
    eps_end             = cfg.get('epsilon_end',         0.05)
    eps_decay_steps     = cfg.get('epsilon_decay_steps', 40000)
    target_class        = cfg.get('target_class', 1)

    history    = []
    best_dice  = 0.0
    ckpt_dir   = Path(cfg.get('checkpoint_dir', '/kaggle/working'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = ckpt_dir / f"{cfg.get('dataset','camus').lower()}_{cfg['agent_type'].lower()}_c{target_class}_best.pt"

    print(f'[drl] Training: {train_steps} steps  →  {ckpt_path}')
    pbar = trange(train_steps, desc=f"{cfg['agent_type']} c{target_class}")

    step = 0
    while step < train_steps:
        idx = _sample_idx()
        env = _make_env(train_caches, idx, env_kwargs, env_cls=env_cls)
        state = env.reset()

        if action_type == 'continuous':
            agent.noise.reset()
            epsilon = None
        else:
            epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)

        episode_steps = 0
        while True:
            if action_type == 'discrete':
                a = agent.select_action(state, epsilon)
            else:
                a = agent.select_action(state, explore=True)

            prev_mask = env.mask.copy()
            prev_sdt  = state[2]           # cached SDT from previous state
            next_state, r, done, _ = env.step(a)
            buffer.push(idx, prev_mask, a, r, env.mask.copy(), done,
                        current_sdt=prev_sdt, next_sdt=next_state[2])

            if len(buffer) >= batch_size:
                agent.update(buffer.sample(batch_size), state_builder)

            state = next_state
            step += 1
            episode_steps += 1
            pbar.update(1)

            # Periodic eval
            if step % eval_every == 0:
                metrics = evaluate_agent(agent, val_caches, env_kwargs, env_cls=env_cls)
                metrics['step']    = step
                metrics['epsilon'] = epsilon if epsilon is not None else None
                history.append(metrics)
                improved = metrics['final_dice_mean'] > best_dice
                if improved:
                    best_dice = metrics['final_dice_mean']
                    _save_agent(agent, history, best_dice, ckpt_path)
                pbar.write(
                    f"step {step:6d} | init {metrics['init_dice_mean']:.4f} "
                    f"→ final {metrics['final_dice_mean']:.4f} "
                    f"(Δ {metrics['delta_dice_mean']:+.4f}, HD95 {metrics['final_hd95_mean']:.2f}px)"
                    f"{' ✓' if improved else ''}"
                )

            if done or step >= train_steps:
                break

    pbar.close()
    print(f'[drl] Done. Best val final-Dice: {best_dice:.4f}')

    return dict(
        agent      = agent,
        history    = pd.DataFrame(history),
        best_dice  = best_dice,
        checkpoint = str(ckpt_path),
    )
