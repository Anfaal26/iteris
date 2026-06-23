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


class DemoBuffer:
    """Array-based storage for oracle demonstrations, mirroring
    `iteris/buffer.py::ReplayBuffer`'s dtype/shape conventions so it can be
    consumed by `TD3Agent._build_states` unmodified.
    """

    def __init__(self, demos: List[dict], mask_shape: tuple):
        n = len(demos)
        self.mask_shape = mask_shape
        self.sample_idx = np.zeros(n, dtype=np.int32)
        self.mask = np.zeros((n, *mask_shape), dtype=np.uint8)
        self.sdt = np.zeros((n, *mask_shape), dtype=np.float16)
        action_dim = demos[0]['action'].shape[0] if n > 0 else 0
        self.action = np.zeros((n, action_dim), dtype=np.float32)

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
