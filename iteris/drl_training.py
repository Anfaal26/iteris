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
from .agents  import DQNAgent, DuelingDQNAgent, DDPGAgent, TD3Agent
from .env_contour_refine import ContourRefineEnv
from . import bc_demo


# ─── Environment registry ────────────────────────────────────────────────────
# 'default' : SegmentationEnv — 24-action GLOBAL morphological mask refinement.
# 'contour' : ContourRefineEnv — control-point + spline LOCAL contour refinement
#             (DeepSnake/MARL-style sector push-out/in/smooth/stop). Same 4-channel
#             state and discrete-agent interface as 'default', so it is a drop-in
#             env swap selected via `env_class: contour` in the per-agent config.
# Archived : ContourTracingEnv (Paradigm 1, 8-dir boundary tracing) — retired for
#            staircase artefacts; see iteris/archive/paradigm1_boundary_tracing/.
ENV_REGISTRY = {
    'default': SegmentationEnv,
    'contour': ContourRefineEnv,
}

CONTINUOUS_ACTION_DIM = SegmentationEnv.CONTINUOUS_ACTION_DIM   # 3


AGENT_REGISTRY = {
    'DQN':     (DQNAgent,        'discrete'),   # 14-action local mask refinement
    'DUELING': (DuelingDQNAgent, 'discrete'),   # same, with V+A dueling heads
    'DDPG':    (DDPGAgent,       'continuous'), # 3-D global morph+shift (baseline-capped)
    'TD3':     (TD3Agent,        'continuous'), # robust DDPG on angular-sector contour action
}


def _build_state_caches(samples: List[dict], image_size: int) -> dict:
    """Stack image + init_mask (+ optional prob_map) arrays for vectorised lookup."""
    caches = dict(
        image     = np.stack([s['image']     for s in samples]).astype(np.float32),
        gt_mask   = np.stack([s['gt_mask']   for s in samples]).astype(np.uint8),
        init_mask = np.stack([s['init_mask'] for s in samples]).astype(np.uint8),
    )
    # prob_map is present when warm_start was run with the updated code.
    # Stored as float16 to save RAM; env converts to float32 on use.
    if all('prob_map' in s for s in samples):
        caches['prob_map'] = np.stack([s['prob_map'] for s in samples]).astype(np.float16)
    return caches


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
        # 5th channel: U-Net probability map (static per sample). Falls back to
        # the binary init mask when warm_start produced no prob_map. MUST match
        # the channel order in SegmentationEnv._state().
        if 'prob_map' in caches:
            prob = caches['prob_map'][idx].astype(np.float32)
        else:
            prob = init
        return torch.from_numpy(np.stack([image, cur, sdt, init, prob], axis=0))
    return build


def _make_env(caches: dict, idx: int, env_kwargs: dict,
              env_cls=SegmentationEnv) -> SegmentationEnv:
    # Pass prob_map if warm_start produced one (enables ADD_UNCERTAIN/REMOVE_UNCERTAIN)
    prob = caches['prob_map'][idx].astype(np.float32) \
           if 'prob_map' in caches else None
    return env_cls(
        image     = caches['image'][idx],
        gt_mask   = caches['gt_mask'][idx],
        init_mask = caches['init_mask'][idx],
        prob_map  = prob,
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

    init_d, final_d, final_h, best_d = [], [], [], []
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
        # best_dice = highest Dice seen during the episode (the achievable ceiling
        # if STOP timing were perfect). final_dice = the deployment number.
        best_d.append(info.get('best_dice', info['dice']))

    final_h_arr = np.asarray(final_h, dtype=float)
    valid_h = final_h_arr[~np.isnan(final_h_arr)]
    return dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),
        best_dice_mean  = float(np.mean(best_d)),    # diagnostic ceiling
        final_hd95_mean = float(valid_h.mean()) if valid_h.size else float('nan'),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
    )


def run_drl_training(
    cfg: dict,
    train_samples: List[dict],
    val_samples:   List[dict],
) -> dict:
    """Train one DRL agent for one (dataset, structure) combination."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # Both discrete and continuous agents may select the env via env_class.
    # TD3 in particular runs on env_class: contour (angular-sector continuous
    # action). DDPG defaults to 'default' (SegmentationEnv 3-D global action).
    env_class_name = cfg.get('env_class', 'default')
    if env_class_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env_class '{env_class_name}'. Available: {list(ENV_REGISTRY)}")
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
        'fail_thresh', 'fail_n',                       # small-target extras
        'reward_step_penalty', 'disable_auto_stop',    # STOP-incentive shaping
        'terminal_bonus_scale',                        # path-independent terminal reward
        'reward_potential_scale',                      # PBRS potential scale (baseline-centred Φ)
        'n_points', 'disp_px', 'spline_smooth', 'smooth_lambda', 'cont_sectors',  # contour env
        'uncertainty_gate', 'gate_lo', 'gate_hi', 'gate_margin',  # U-Net confidence gate
    )
    for _k in _env_optional_keys:
        if _k in cfg:
            env_kwargs[_k] = cfg[_k]
    # Potential-based reward shaping (dice_pbrs / dice_hd_pbrs) needs the SAME
    # discount the agent uses, so the shaped return telescopes exactly to
    # γ^T·Φ_T − Φ_0. Always pass it; the env ignores it for non-pbrs modes.
    env_kwargs['pbrs_gamma'] = cfg.get('gamma', 0.99)
    # cont_trans_scale legacy fallback (old configs use cont_action_scale)
    if 'cont_trans_scale' not in env_kwargs and 'cont_action_scale' in cfg:
        env_kwargs['cont_trans_scale'] = cfg['cont_action_scale']
    # state_builder needs sdt_clip even if not in cfg (env class default applies).
    state_builder = _make_state_builder(train_caches, cfg.get('sdt_clip', 20.0))

    # ── Continuous action dim (env-dependent) ─────────────────────────────────
    #   contour env → cont_sectors angular wedges (TD3, action in [-1,1]^K)
    #   default env → 3-D global (morph, dy, dx)  (DDPG)
    is_contour = (env_cls is ContourRefineEnv)
    cont_action_dim = None
    if action_type == 'continuous':
        cont_action_dim = (int(cfg.get('cont_sectors', 16)) if is_contour
                           else CONTINUOUS_ACTION_DIM)

    # ── Agent ─────────────────────────────────────────────────────────────────
    common = dict(in_channels=5, gamma=cfg.get('gamma', 0.99),
                  tau=cfg.get('tau', 0.005),
                  embed_dim=cfg.get('embed_dim', 256), device=device)
    if action_type == 'discrete':
        # num_actions is env-class-dependent — agent's output head must match
        # exactly or argmax will produce invalid indices.
        agent = agent_cls(num_actions=num_actions_for_cls,
                          lr=cfg.get('lr', 1e-4),
                          spatial=cfg.get('spatial_head', False), **common)
    else:
        common.pop('lr', None)
        if is_contour:
            action_scale = [1.0] * cont_action_dim   # env multiplies by disp_px
        else:
            _legacy_scale = cfg.get('cont_action_scale', 0.02)
            morph_scale   = cfg.get('cont_morph_scale', 0.25)
            trans_scale   = cfg.get('cont_trans_scale', _legacy_scale)
            action_scale  = [morph_scale, trans_scale, trans_scale]

        if cfg['agent_type'].upper() == 'TD3':
            agent = agent_cls(
                action_dim   = cont_action_dim,
                action_scale = action_scale,
                actor_lr     = cfg.get('actor_lr',  1e-4),
                critic_lr    = cfg.get('critic_lr', 1e-3),
                policy_delay = cfg.get('policy_delay', 2),
                expl_noise   = cfg.get('expl_noise', 0.2),
                target_noise = cfg.get('target_noise', 0.2),
                noise_clip   = cfg.get('noise_clip', 0.5),
                spatial      = cfg.get('spatial_head', False),
                **common,
            )
        else:   # DDPG
            ou_sigma_raw = cfg.get('ou_sigma', None)
            agent = agent_cls(
                action_dim         = cont_action_dim,
                action_scale       = action_scale,
                actor_lr           = cfg.get('actor_lr',  1e-4),
                critic_lr          = cfg.get('critic_lr', 1e-3),
                ou_theta           = cfg.get('ou_theta', 0.15),
                ou_sigma           = ou_sigma_raw,
                actor_freeze_steps = cfg.get('actor_freeze_steps', 2000),
                **common,
            )

    # ── TD3+BC: oracle-demonstration actor warm-start (opt-in) ─────────────────
    # Fujimoto & Gu 2021. Vanilla TD3's deterministic actor starts near-identity
    # (small-uniform final-layer init) and the reward landscape around zero is
    # the only "safe" region, so the actor never escapes it. Pretraining on
    # GT-privileged oracle demonstrations (train-time only — never used at
    # deployment) gives it a competent starting policy instead; `bc_lambda`
    # regularisation during RL fine-tuning (see below) keeps it from drifting
    # back to identity. Only applies to TD3 on the continuous contour env.
    demo_buffer = None
    bc_warm_start = (cfg['agent_type'].upper() == 'TD3' and is_contour
                     and cfg.get('bc_warm_start', False))
    if bc_warm_start:
        bc_demo_episodes = cfg.get('bc_demo_episodes', 150)
        bc_demo_max_steps = cfg.get('bc_demo_max_steps', 8)
        bc_pretrain_epochs = cfg.get('bc_pretrain_epochs', 30)
        print(f'[drl] BC warm-start: collecting {bc_demo_episodes} oracle demo '
              f'episodes (max_steps={bc_demo_max_steps})...')
        demos = bc_demo.collect_continuous_oracle_demos(
            train_samples, env_kwargs, cont_action_dim,
            bc_demo_episodes, bc_demo_max_steps, seed=cfg.get('seed', 42))
        demo_buffer = bc_demo.DemoBuffer(demos, mask_shape=(H, H))
        bc_loss = agent.pretrain_actor_bc(
            demo_buffer, state_builder,
            epochs=bc_pretrain_epochs, batch_size=cfg.get('batch_size', 64))
        print(f'[drl] BC warm-start: {len(demos)} demo transitions, '
              f'pretrain loss={bc_loss:.4f}')

    bc_lambda_start = cfg.get('bc_lambda_start', 0.5)
    bc_lambda_end = cfg.get('bc_lambda_end', 0.05)
    bc_lambda_decay_steps = cfg.get('bc_lambda_decay_steps', 20000)

    # ── Buffer ────────────────────────────────────────────────────────────────
    # cache_sdt=True is the key speed optimisation: skips ~80% of SDT recomputation
    # in agent.update() by reusing the SDT computed during env.step().
    buffer = ReplayBuffer(
        capacity   = cfg.get('buffer_size', 10000),
        mask_shape = (H, H),   # BRISC/CAMUS are square (256×256); extend to (H,W) if needed
        # continuous action width: 3 (global DDPG) or cont_sectors (contour TD3).
        action_dim = cont_action_dim if action_type == 'continuous' else None,
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
            elif is_contour:
                # angular-sector action in [-1, 1]^K
                a = np.random.uniform(-1.0, 1.0, size=cont_action_dim).astype(np.float32)
            else:
                # 3-component global: (morph, dy_norm, dx_norm)
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
            # DDPG keeps temporally-correlated OU noise that must reset per
            # episode; TD3 uses i.i.d. Gaussian exploration and has no such state.
            if hasattr(agent, 'noise'):
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
                if demo_buffer is not None and len(demo_buffer) > 0:
                    bc_batch = demo_buffer.sample(batch_size)
                    bc_lambda = max(bc_lambda_end, bc_lambda_start
                                    - (bc_lambda_start - bc_lambda_end) * step / bc_lambda_decay_steps)
                    agent.update(buffer.sample(batch_size), state_builder,
                                bc_batch=bc_batch, bc_lambda=bc_lambda)
                else:
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
                    f"| best-seen {metrics.get('best_dice_mean', float('nan')):.4f} "
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
