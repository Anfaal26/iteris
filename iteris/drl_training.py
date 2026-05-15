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

from .env     import SegmentationEnv, signed_dt
from .buffer  import ReplayBuffer
from .agents  import DQNAgent, DDQNAgent, DuelingDQNAgent, DDPGAgent


AGENT_REGISTRY = {
    'DQN':     (DQNAgent,        'discrete'),
    'DDQN':    (DDQNAgent,       'discrete'),
    'DUELING': (DuelingDQNAgent, 'discrete'),
    'DDPG':    (DDPGAgent,       'continuous'),
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
    """Return a function (idx, current_mask) → torch tensor (4, H, W)."""
    def build(idx, current_mask):
        image = caches['image'][idx]
        init  = caches['init_mask'][idx].astype(np.float32)
        cur   = current_mask.astype(np.float32)
        sdt   = signed_dt(current_mask, sdt_clip)
        return torch.from_numpy(np.stack([image, cur, sdt, init], axis=0))
    return build


def _make_env(caches: dict, idx: int, env_kwargs: dict) -> SegmentationEnv:
    return SegmentationEnv(
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
def evaluate_agent(agent, samples_or_caches, env_kwargs) -> dict:
    """Greedy policy evaluation over a sample list. Returns aggregate metrics."""
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
        env = _make_env(caches, i, env_kwargs)
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

    seed = cfg.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent_cls, action_type = AGENT_REGISTRY[cfg['agent_type'].upper()]
    image_size = cfg['image_size']
    H = image_size

    # ── Caches ───────────────────────────────────────────────────────────────
    train_caches = _build_state_caches(train_samples, image_size)
    val_caches   = _build_state_caches(val_samples,   image_size)

    env_kwargs = dict(
        action_type       = action_type,
        max_steps         = cfg.get('max_steps', 20),
        shift_px          = cfg.get('shift_px', 2),
        sdt_clip          = cfg.get('sdt_clip', 20.0),
        reward_clip       = cfg.get('reward_clip', 1.0),
        cont_action_scale = cfg.get('cont_action_scale', 0.04),
    )
    state_builder = _make_state_builder(train_caches, env_kwargs['sdt_clip'])

    # ── Agent ─────────────────────────────────────────────────────────────────
    common = dict(in_channels=4, gamma=cfg.get('gamma', 0.99),
                  tau=cfg.get('tau', 0.005),
                  embed_dim=cfg.get('embed_dim', 256), device=device)
    if action_type == 'discrete':
        agent = agent_cls(num_actions=7, lr=cfg.get('lr', 1e-4), **common)
    else:
        common.pop('lr', None)
        agent = agent_cls(
            action_dim=2,
            action_scale=cfg.get('cont_action_scale', 0.04),
            actor_lr=cfg.get('actor_lr',  1e-4),
            critic_lr=cfg.get('critic_lr', 1e-3),
            ou_theta=cfg.get('ou_theta', 0.15),
            ou_sigma=cfg.get('ou_sigma', 0.02),
            actor_freeze_steps=cfg.get('actor_freeze_steps', 2000),
            **common,
        )

    # ── Buffer ────────────────────────────────────────────────────────────────
    buffer = ReplayBuffer(
        capacity   = cfg.get('buffer_size', 10000),
        mask_shape = (H, H),
        action_dim = 2 if action_type == 'continuous' else None,
        discrete   = (action_type == 'discrete'),
    )

    # ── Pre-fill buffer with random rollouts ──────────────────────────────────
    prefill_steps = cfg.get('prefill_steps', 2000)
    print(f'[drl] Pre-filling buffer with {prefill_steps} random transitions...')
    while len(buffer) < prefill_steps:
        idx = np.random.randint(len(train_samples))
        env = _make_env(train_caches, idx, env_kwargs)
        env.reset()
        while True:
            if action_type == 'discrete':
                a = np.random.randint(7)
            else:
                scale = cfg.get('cont_action_scale', 0.04)
                a = np.random.uniform(-scale, scale, size=2).astype(np.float32)
            prev = env.mask.copy()
            _, r, done, _ = env.step(a)
            buffer.push(idx, prev, a, r, env.mask.copy(), done)
            if done or len(buffer) >= prefill_steps:
                break
    print(f'[drl] Buffer size: {len(buffer)}')

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
        idx = np.random.randint(len(train_samples))
        env = _make_env(train_caches, idx, env_kwargs)
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

            prev = env.mask.copy()
            next_state, r, done, _ = env.step(a)
            buffer.push(idx, prev, a, r, env.mask.copy(), done)

            if len(buffer) >= batch_size:
                agent.update(buffer.sample(batch_size), state_builder)

            state = next_state
            step += 1
            episode_steps += 1
            pbar.update(1)

            # Periodic eval
            if step % eval_every == 0:
                metrics = evaluate_agent(agent, val_caches, env_kwargs)
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
