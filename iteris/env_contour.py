"""
Sequential boundary-tracing environment (Paradigm 1).

The agent walks the object boundary one pixel at a time. At each step it picks
one of 8 directions; the episode ends when the trace closes back onto its seed,
walks off the image, or exceeds ``max_trace_length``. The final segmentation is
the polygon enclosed by the trajectory (rasterised at episode end).

Contrast with ``env.SegmentationEnv`` (refinement): there the agent edits a
whole U-Net mask via morphology; here the agent *draws* the mask itself, and the
U-Net only supplies the seed point + a reference init boundary in the state.

────────────────────────────────────────────────────────────────────────────────
State (4, patch_size, patch_size) float32 — a moving window anchored on the
current point (zero-padded at image edges):
    ch 0 : image intensity patch
    ch 1 : position layer — 1 at the patch centre (the current point)
    ch 2 : visited layer   — 1 along the trajectory so far (within the patch)
    ch 3 : U-Net init boundary — 1 on init-mask edge pixels (within the patch)

Action space — 8 directions (see contour_utils.DIRECTIONS):
    0=N 1=NE 2=E 3=SE 4=S 5=SW 6=W 7=NW   (single-pixel step)

Reward:
    r_step     = -clip(dist_to_GT_boundary, 0, max_distance_penalty)
    r_boundary = +reward_boundary_bonus    when dist < boundary_bonus_distance
    r_closure  = +reward_closure           (terminal: trace returns to seed)
    r_offimage = +reward_offimage          (terminal: trace exits image; negative)
    r_timeout  = +reward_length_penalty    (terminal: max_trace_length w/o closure)

Speed: the GT-boundary distance field is precomputed ONCE in reset() via
contour_utils.gt_boundary_edt, so the per-step distance reward is an O(1)
array lookup rather than a nearest-pixel scan (speedup option 2).
"""

from typing import Dict, List, Tuple

import numpy as np

from . import contour_utils as cu


class ContourTracingEnv:
    """Per-class boundary-tracing environment for the DQN family (8 actions)."""

    NUM_DISCRETE_ACTIONS = 8
    DISCRETE_NAMES = cu.DIRECTION_NAMES

    # Init Dice below this threshold → bypass U-Net mask, seed from GT instead
    # (Option A safeguard). 0.30 catches catastrophic U-Net failures while
    # accepting moderately-noisy predictions whose best-overlap CC is still
    # usable. Tunable per-dataset via subclassing if ever needed.
    _SEED_INIT_DICE_MIN: float = 0.30

    def __init__(
        self,
        image: np.ndarray,        # (H, W) float32 in [0, 1]
        gt_mask: np.ndarray,      # (H, W) uint8 in {0, 1}
        init_mask: np.ndarray,    # (H, W) uint8 in {0, 1} — U-Net prediction
        patch_size: int = 64,
        max_trace_length: int = 400,
        closure_tolerance: float = 3.0,
        min_perimeter_steps: int = 50,
        boundary_bonus_distance: float = 1.5,
        reward_boundary_bonus: float = 0.5,
        reward_offimage: float = -10.0,
        reward_closure: float = 5.0,
        reward_length_penalty: float = -5.0,
        max_distance_penalty: float = 10.0,
        seed_method: str = 'topmost',
        action_type: str = 'discrete',   # accepted for interface parity; only 'discrete'
    ):
        assert action_type == 'discrete', 'ContourTracingEnv is discrete-only'
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape

        self.patch_size              = int(patch_size)
        self.max_trace_length        = int(max_trace_length)
        self.closure_tolerance       = float(closure_tolerance)
        self.min_perimeter_steps     = int(min_perimeter_steps)
        self.boundary_bonus_distance = float(boundary_bonus_distance)
        self.reward_boundary_bonus   = float(reward_boundary_bonus)
        self.reward_offimage         = float(reward_offimage)
        self.reward_closure          = float(reward_closure)
        self.reward_length_penalty   = float(reward_length_penalty)
        self.max_distance_penalty    = float(max_distance_penalty)
        self.seed_method             = seed_method

        # Static within an episode: init-mask boundary as a full (H, W) edge mask.
        edge_px = cu.boundary_edge_pixels(cu.largest_cc(self.init_mask))
        self._init_edge_full = np.zeros((self.H, self.W), dtype=np.uint8)
        if edge_px.shape[0] > 0:
            self._init_edge_full[edge_px[:, 0], edge_px[:, 1]] = 1

        self.action_type = 'discrete'
        self.state_shape = (4, self.patch_size, self.patch_size)
        self.reset()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        # Seed from the U-Net init mask, GT-aware. Two safeguards (Option A):
        #   (1) Among the U-Net's connected components, pick the one with the
        #       highest IoU vs GT (not the largest CC). This prevents a stray
        #       FP blob from anchoring the trace far from the structure.
        #   (2) If the U-Net's overall init Dice is too poor (< _SEED_INIT_DICE
        #       MIN, default 0.30), bypass the U-Net entirely and seed from GT.
        #       Such samples are rare; without this they trace from nowhere
        #       useful and drag the val Dice down for reasons unrelated to the
        #       agent's policy quality.
        # Final fallback: if the init has no foreground at all, seed from GT
        # so the run does not crash.
        from .env import dice_score as _dice
        init_dice = _dice(self.init_mask, self.gt) if self.init_mask.any() else 0.0
        seed_init = self.init_mask if init_dice >= self._SEED_INIT_DICE_MIN else self.gt
        try:
            self.seed_point = cu.seed_point_from_init_mask(
                seed_init, self.seed_method, gt_mask=self.gt)
        except ValueError:
            self.seed_point = cu.seed_point_from_init_mask(
                self.gt, self.seed_method)
        self.current_point = self.seed_point
        self.trajectory: List[Tuple[int, int]] = [self.seed_point]
        self._visited = np.zeros((self.H, self.W), dtype=np.uint8)
        self._visited[self.seed_point[0], self.seed_point[1]] = 1
        # Precompute GT-boundary EDT once — O(1) per-step reward lookups after this.
        self._gt_edt = cu.gt_boundary_edt(self.gt)
        self.done = False
        return self._extract_patch()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        a = int(action)
        dy, dx = cu.DIRECTIONS[a]
        new_point = (self.current_point[0] + int(dy), self.current_point[1] + int(dx))

        info: Dict = {'closed': False, 'offimage': False, 'timeout': False}

        # Terminal: walked off the image. Do not advance the point.
        if cu.is_off_image(new_point, self.H, self.W):
            info['offimage'] = True
            self.done = True
            info['trace_length'] = len(self.trajectory)
            return self._extract_patch(), self.reward_offimage, True, info

        # Advance.
        self.current_point = new_point
        self.trajectory.append(new_point)
        self._visited[new_point[0], new_point[1]] = 1

        dist = float(self._gt_edt[new_point[0], new_point[1]])
        reward = -min(dist, self.max_distance_penalty)
        if dist < self.boundary_bonus_distance:
            reward += self.reward_boundary_bonus
        info['dist'] = dist

        done = False
        if cu.is_closed(self.trajectory, self.seed_point,
                        self.closure_tolerance, self.min_perimeter_steps):
            reward += self.reward_closure
            info['closed'] = True
            done = True
        elif len(self.trajectory) >= self.max_trace_length:
            reward += self.reward_length_penalty
            info['timeout'] = True
            done = True

        self.done = done
        info['trace_length'] = len(self.trajectory)
        return self._extract_patch(), float(reward), done, info

    # ── state construction ──────────────────────────────────────────────────────

    def _crop_centered(self, arr: np.ndarray) -> np.ndarray:
        """Zero-padded (patch_size, patch_size) crop of ``arr`` centred on the
        current point."""
        ps   = self.patch_size
        half = ps // 2
        y, x = self.current_point
        out = np.zeros((ps, ps), dtype=arr.dtype)
        y0, x0 = y - half, x - half                       # patch top-left in image coords
        iy0, ix0 = max(0, y0), max(0, x0)
        iy1, ix1 = min(self.H, y0 + ps), min(self.W, x0 + ps)
        if iy1 > iy0 and ix1 > ix0:
            oy0, ox0 = iy0 - y0, ix0 - x0
            out[oy0:oy0 + (iy1 - iy0), ox0:ox0 + (ix1 - ix0)] = arr[iy0:iy1, ix0:ix1]
        return out

    def _extract_patch(self) -> np.ndarray:
        ps   = self.patch_size
        half = ps // 2
        img = self._crop_centered(self.image).astype(np.float32)
        pos = np.zeros((ps, ps), dtype=np.float32)
        pos[half, half] = 1.0                              # current point at patch centre
        vis  = self._crop_centered(self._visited).astype(np.float32)
        edge = self._crop_centered(self._init_edge_full).astype(np.float32)
        return np.stack([img, pos, vis, edge], axis=0)

    # ── episode-end outputs ──────────────────────────────────────────────────────

    def get_final_mask(self) -> np.ndarray:
        """Rasterise the trajectory polygon to a binary (H, W) uint8 mask."""
        return cu.rasterise_trajectory(self.trajectory, self.H, self.W)


class VectorisedContourEnv:
    """Runs N ``ContourTracingEnv`` instances in lockstep for batched inference.

    The dominant training-speed lever (option 1): the training loop does ONE
    batched network forward over ``(N, 4, patch, patch)`` per step instead of N
    separate single-env forwards. This wrapper owns only the batching — each
    inner env keeps its own trajectory, seed, and cached GT-EDT.

    Auto-reset is explicit, not automatic: ``step_all`` never re-seeds, so the
    training loop can push the terminal transition for a finished slot BEFORE
    replacing it. After pushing, the loop calls ``replace(k, new_env)`` to drop
    a freshly-seeded env into slot ``k`` and gets back that env's initial patch
    to write into its running state array. This keeps all N slots busy without
    a Python-level per-episode loop.
    """

    def __init__(self, envs):
        assert len(envs) > 0
        self.envs = list(envs)
        self.n = len(self.envs)

    def reset_all(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step_all(self, actions):
        states, rewards, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            s, r, d, i = e.step(int(a))
            states.append(s); rewards.append(r); dones.append(d); infos.append(i)
        return (np.stack(states, axis=0),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool),
                infos)

    def replace(self, k: int, new_env: 'ContourTracingEnv') -> np.ndarray:
        """Drop ``new_env`` into slot ``k`` and return its initial patch state."""
        self.envs[k] = new_env
        return new_env.reset()
