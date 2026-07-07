"""
TD3+BC oracle-demonstration collection (Fujimoto & Gu 2021).

TD3's deterministic actor starts near-identity (small-uniform final-layer init)
and the reward landscape around zero is the only "safe" region — without help
the actor never escapes that basin (see `iteris/diagnostics.py::oracle_greedy`
for the discrete analogue of the same problem). The fix here is the continuous
counterpart: a GT-privileged greedy oracle builds continuous action vectors
that improve Dice, and those (state, action) pairs are used to warm-start the
TD3 actor via behaviour cloning (see `TD3Agent.pretrain_actor_bc` in agents.py)
BEFORE RL fine-tuning begins. The oracle sees ground truth — that is valid only
at TRAIN time; it is never available, and never used, at deployment.

This module provides:
  - collect_continuous_oracle_demos(...): greedy coordinate-ascent oracle that
    builds (state, action) demonstrations over the angular-sector continuous
    contour action space.
  - DemoBuffer: array-based storage for those demonstrations, following the
    exact dtype/shape conventions of `iteris/buffer.py::ReplayBuffer` so it
    can be consumed by `TD3Agent._build_states` unmodified.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np

from .env_contour_refine import ContourRefineEnv
from .geometry import signed_dt


# Candidate values swept for each action component during the greedy
# coordinate-ascent search (mirrors the discrete oracle's exhaustive-action
# sweep in diagnostics.py::oracle_greedy, adapted to a continuous component).
_CANDIDATES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def collect_continuous_oracle_demos(
    samples: List[dict],
    env_kwargs: dict,
    cont_action_dim: int,
    n_episodes: int,
    max_steps: int,
    seed: int = 42,
) -> List[dict]:
    """Greedy coordinate-ascent oracle over the continuous contour action space.

    For each of `n_episodes` randomly-sampled (seeded) entries from `samples`,
    roll out up to `max_steps` continuous-action steps. At each step, build an
    action vector ``a`` of length `cont_action_dim` one component at a time:
    for component `g`, try each candidate in `_CANDIDATES` for `a[g]` (holding
    already-decided components fixed), deep-copy the env, step the FULL trial
    vector, and keep whichever candidate value maximises `info['dice']`
    (mirrors `diagnostics.py::oracle_greedy`'s deepcopy-and-discard-trial
    pattern exactly — the real env is never mutated during the search).

    Once all components have been swept once, the assembled vector ``a`` is
    applied to the REAL env via `env.step(a)`, advancing the episode. A demo
    entry is recorded for the state BEFORE that step is applied, paired with
    the action that produced the improvement.

    Returns a plain list of demo dicts:
        {'sample_idx': int, 'mask': uint8 (H, W), 'sdt': float16 (H, W),
         'action': float32 (cont_action_dim,)}
    """
    rng = np.random.RandomState(seed)
    n_samples = len(samples)
    episode_idx = rng.randint(0, n_samples, size=n_episodes)

    sdt_clip = float(env_kwargs.get('sdt_clip', 20.0))
    # Oracle search is always continuous-action; discrete/contour irrelevant here.
    base_kwargs = dict(env_kwargs)
    base_kwargs['action_type'] = 'continuous'

    demos: List[dict] = []

    for idx in episode_idx:
        sample = samples[int(idx)]
        env = ContourRefineEnv(
            image=sample['image'], gt_mask=sample['gt_mask'],
            init_mask=sample['init_mask'], prob_map=sample.get('prob_map'),
            **base_kwargs,
        )

        for _ in range(max_steps):
            best_a = np.zeros(cont_action_dim, dtype=np.float32)
            best_dice = -1.0
            for g in range(cont_action_dim):
                best_val_for_g = best_a[g]
                for cand in _CANDIDATES:
                    trial_a = best_a.copy()
                    trial_a[g] = cand
                    trial = deepcopy(env)
                    _, _, _, info = trial.step(trial_a)
                    if info['dice'] > best_dice:
                        best_dice = info['dice']
                        best_val_for_g = cand
                best_a[g] = best_val_for_g

            # Record the demo BEFORE applying this step's action to the real env.
            pre_mask = env.mask.copy().astype(np.uint8)
            pre_sdt = signed_dt(pre_mask, sdt_clip).astype(np.float16)
            demos.append({
                'sample_idx': int(idx),
                'mask': pre_mask,
                'sdt': pre_sdt,
                'action': best_a.astype(np.float32),
            })

            _, _, done, _ = env.step(best_a)
            if done:
                break

    return demos


def collect_discrete_oracle_demos(
    samples: List[dict],
    env_kwargs: dict,
    n_episodes: int,
    max_steps: int,
    seed: int = 42,
    stop_eps: float = 0.005,
) -> List[dict]:
    """Greedy discrete oracle over the 18-action contour action space, for
    warm-starting the DuelingDDQN/DQN Q-network (the discrete counterpart of
    `collect_continuous_oracle_demos`).

    For each of `n_episodes` seeded-random samples, roll out up to `max_steps`
    steps. At each step, try every non-STOP discrete action on a deep-copied env
    and keep whichever most raises Dice vs GT (same deepcopy-and-discard pattern
    as `diagnostics.oracle_greedy`). If no push raises Dice by more than
    `stop_eps`, the oracle records STOP — so the demos teach BOTH "push the
    right sector" and "stop when converged". A demo is recorded for the state
    BEFORE the chosen action is applied. The oracle sees GT — valid at TRAIN
    time only.

    `stop_eps` (default matches the env's own `stop_eps_dice`, ~0.005) guards
    against floating-point/rasterization jitter: on an already-converged mask,
    re-rasterizing a barely-nudged spline can shift Dice by a sub-pixel amount
    in EITHER direction purely from numerical noise, so a strict `>` bar is
    almost always cleared by at least one of 17 candidates — the oracle then
    (almost) never emits STOP, teaching BC a noisy, near-arbitrary "best
    action" instead of a clean stop-when-converged signal.

    Returns demo dicts with an INTEGER action:
        {'sample_idx': int, 'mask': uint8 (H,W), 'sdt': float16 (H,W), 'action': int}
    """
    rng = np.random.RandomState(seed)
    episode_idx = rng.randint(0, len(samples), size=n_episodes)
    sdt_clip = float(env_kwargs.get('sdt_clip', 20.0))
    base_kwargs = dict(env_kwargs)
    base_kwargs['action_type'] = 'discrete'
    n_act = ContourRefineEnv.NUM_DISCRETE_ACTIONS
    stop = ContourRefineEnv.STOP

    demos: List[dict] = []
    for idx in episode_idx:
        sample = samples[int(idx)]
        env = ContourRefineEnv(
            image=sample['image'], gt_mask=sample['gt_mask'],
            init_mask=sample['init_mask'], prob_map=sample.get('prob_map'),
            **base_kwargs)
        for _ in range(max_steps):
            current = env.dice_history[-1]
            best_a, best_after = None, current
            for a in range(n_act):
                if a == stop:
                    continue
                trial = deepcopy(env)
                _, _, _, info = trial.step(a)
                if info['dice'] > best_after:
                    best_after, best_a = info['dice'], a
            # STOP unless the best candidate beats the current Dice by more
            # than stop_eps (not just any strictly-positive jitter).
            action = stop if (best_a is None or (best_after - current) <= stop_eps) else best_a
            pre_mask = env.mask.copy().astype(np.uint8)
            pre_sdt = signed_dt(pre_mask, sdt_clip).astype(np.float16)
            demos.append({
                'sample_idx': int(idx),
                'mask': pre_mask,
                'sdt': pre_sdt,
                'action': int(action),
            })
            _, _, done, _ = env.step(action)
            if done:
                break
    return demos


class DemoBuffer:
    """Array-based storage for oracle demonstrations, mirroring
    `iteris/buffer.py::ReplayBuffer`'s dtype/shape conventions so it can be
    consumed by `TD3Agent._build_states` / `DQNAgent` unmodified.

    Handles BOTH continuous (per-sector float vector) and discrete (single int)
    oracle actions, detected from the first demo's action.
    """

    def __init__(self, demos: List[dict], mask_shape: tuple):
        n = len(demos)
        self.mask_shape = mask_shape
        self.sample_idx = np.zeros(n, dtype=np.int32)
        self.mask = np.zeros((n, *mask_shape), dtype=np.uint8)
        self.sdt = np.zeros((n, *mask_shape), dtype=np.float16)
        a0 = demos[0]['action'] if n > 0 else 0.0
        self.discrete = np.isscalar(a0) or (hasattr(a0, 'ndim') and np.ndim(a0) == 0)
        if self.discrete:
            self.action = np.zeros(n, dtype=np.int64)
        else:
            self.action = np.zeros((n, np.asarray(a0).shape[0]), dtype=np.float32)

        for i, d in enumerate(demos):
            self.sample_idx[i] = d['sample_idx']
            self.mask[i] = d['mask']
            self.sdt[i] = d['sdt']
            self.action[i] = d['action']

        self.size = n

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            sample_idx   = self.sample_idx[idx],
            current_mask = self.mask[idx],
            current_sdt  = self.sdt[idx],
            action       = self.action[idx],
        )

    def __len__(self):
        return self.size
