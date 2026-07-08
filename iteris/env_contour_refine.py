"""
Control-point contour-refinement environment (Paradigm 3).

Motivation
──────────
The default ``SegmentationEnv`` (v4) refines the U-Net mask with *global*
morphological operations — "dilate north", "raise the SDT threshold everywhere",
"shift the whole mask". These are spatially uniform: they cannot correct a mask
that is over-segmented in one region and under-segmented in another without
trading one error for another. On already-strong U-Net baselines (CAMUS
~0.93, BRISC ~0.84) the best reachable mask under global ops barely exceeds the
baseline — a structural ceiling, independent of the reward design.

This environment removes that ceiling by representing the boundary as a set of
ordered **control points** on a closed contour and letting the agent deform it
*locally*. It is the supervisor-recommended paradigm ("control points with
interpolation … more robust than tracing every boundary pixel") and follows the
contour-deformation lineage in the literature — DeepSnake (Peng et al., CVPR'20,
vertex-wise offset prediction with circular convolutions), Curve-GCN (Ling et
al., CVPR'19), and MARL-MambaContour (2025, each contour point an RL agent
emitting a bounded 2-D displacement, optimised on a ΔIoU + boundary + smoothness
reward). Unlike the retired Paradigm-1 boundary *tracing* env (8-direction
single-pixel walk → staircase artefacts, 200–400-step credit-assignment), here
the contour is always a smooth closed spline and an episode is ≤ max_steps edits.

Contract with the existing framework
────────────────────────────────────
The state is the SAME 4-channel stack the default env exposes
``[image, current_mask, SDT(current_mask), init_mask]`` and ``self.mask`` is the
rasterised current contour. This means the existing CNN backbone (in_channels=4),
the DQN / Dueling agents, the replay buffer (which stores ``env.mask`` and
rebuilds the state), and the training/eval loops all work **unchanged** — only
the env class differs (selected via ``env_class: contour`` in the config).

Discrete action space (DQN family) — 2·SECTORS + 2 = 18 actions:
    0 .. S-1      push contour SECTOR g OUTWARD  (expand that boundary arc)
    S .. 2S-1     push contour SECTOR g INWARD   (shrink that boundary arc)
    2S            smooth   (Laplacian smoothing of the whole contour)
    2S+1          stop     (agent signals satisfied — explicit terminal)
where a "sector" is a contiguous arc of the ordered control points. Pushing an
arc moves only its points along their local outward normals — a genuinely local
expand/shrink/move-boundary-points operation. Heterogeneous errors are corrected
by editing different sectors on different steps.

Continuous action space (DDPG family) — per-point normal offset, dim = n_points:
    a[i] ∈ [-1, +1]  →  point i moves a[i]·disp_px along its outward normal.
    (Implemented for completeness — the DeepSnake/MARL offset-prediction form.
     NOT yet wired through drl_training's continuous path, which hardcodes
     action_dim = 3 for the global-morph DDPG baseline; see module note below.)

Reward / termination semantics are identical to ``SegmentationEnv`` (dice_delta /
dice_hd_composite / iou_delta, episode-start baseline, per-step penalty,
path-independent terminal bonus, best-mask tracking) so results are directly
comparable across paradigms.

Uncertainty gate (opt-in, ``uncertainty_gate=True``)
─────────────────────────────────────────────────────
A sector push can move a control point the U-Net was already confident and
correct about, causing large single-step Dice drops. When enabled, every edit's
per-point displacement is scaled by ``_gate_weights`` — a function of the
U-Net's own ``prob_map`` that is 1.0 in the uncertain band ``[gate_lo, gate_hi]``
and ramps to 0.0 (over ``gate_margin``) as the U-Net's confidence approaches 0
or 1. This bounds worst-case per-step degradation and concentrates edits where
the U-Net is actually unsure, without ever consulting ground truth. It is
disabled by default and only activates when a real ``prob_map`` is supplied.
"""

from typing import Tuple, Dict, List
import numpy as np
import scipy.ndimage as ndi

from .geometry import dice_score, signed_dt, sdt_direction_field, _largest_cc, STRUCT


# ── geometry helpers ─────────────────────────────────────────────────────────

def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an ordered closed contour to ``n`` equally arc-length-spaced
    points. ``points`` is (M, 2) in (y, x); the contour is implicitly closed
    (last → first). Returns (n, 2) float64."""
    p = np.vstack([points, points[:1]]).astype(np.float64)   # close the loop
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-6:
        return np.repeat(points[:1].astype(np.float64), n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    y = np.interp(targets, cum, p[:, 0])
    x = np.interp(targets, cum, p[:, 1])
    return np.stack([y, x], axis=1)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    """Longest ordered boundary polyline of ``mask`` as (M, 2) in (y, x).

    Uses skimage.measure.find_contours (sub-pixel marching squares) on the
    largest connected component. Returns an empty (0, 2) array if none.
    """
    from skimage.measure import find_contours
    cc = _largest_cc(mask)
    if not cc.any():
        return np.zeros((0, 2), dtype=np.float64)
    # pad by 1 so a structure touching the border still yields a closed contour
    padded = np.pad(cc.astype(np.float32), 1, mode='constant')
    contours = find_contours(padded, 0.5)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    longest = max(contours, key=len)
    return (longest - 1.0)   # undo pad offset; (row, col) == (y, x)


class ContourRefineEnv:
    """Per-class binary contour-refinement env (control points + spline)."""

    SECTORS = 8
    NUM_DISCRETE_ACTIONS = 2 * SECTORS + 2          # 18
    NUM_STATE_CHANNELS   = 5     # image, mask, sdt, init_mask, prob_map
    SMOOTH = 2 * SECTORS                            # 16
    STOP   = 2 * SECTORS + 1                        # 17
    # CONTINUOUS_ACTION_DIM is per-instance (= n_points); the class value is a
    # placeholder so introspection works before instantiation. The continuous
    # path is not wired through drl_training yet (see module docstring).
    CONTINUOUS_ACTION_DIM = 0

    DISCRETE_NAMES = (
        [f'out{g}' for g in range(SECTORS)]
        + [f'in{g}' for g in range(SECTORS)]
        + ['smooth', 'stop']
    )

    def __init__(
        self,
        image: np.ndarray,            # (H, W) float32 in [0, 1]
        gt_mask: np.ndarray,          # (H, W) uint8 in {0, 1}
        init_mask: np.ndarray,        # (H, W) uint8 in {0, 1} — U-Net prediction
        prob_map: np.ndarray = None,  # U-Net confidence — exposed as state channel
        pbrs_gamma: float = 0.99,     # discount used inside potential-based shaping
        action_type: str = 'discrete',
        max_steps: int = 20,
        sdt_clip: float = 20.0,
        reward_clip: float = 1.0,
        # ── reward shaping (shared with SegmentationEnv) ──────────────────────
        reward_mode: str = 'dice_delta',
        reward_alpha: float = 0.5,
        reward_beta: float = 0.5,
        hd_norm: float = 50.0,
        # ── termination heuristics ────────────────────────────────────────────
        stop_eps_dice: float = 0.001,
        stop_eps_hd: float = 0.5,
        stop_n: int = 3,
        fail_thresh: float = 0.0,
        fail_n: int = 2,
        reward_step_penalty: float = 0.0,
        disable_auto_stop: bool = False,
        terminal_bonus_scale: float = 0.0,
        reward_potential_scale: float = 1.0,   # PBRS potential scale (baseline-centred Φ)
        # ── contour-specific ──────────────────────────────────────────────────
        n_points: int = 32,           # number of control points on the contour
        disp_px: float = 1.5,         # per-edit displacement along the normal (px)
        spline_smooth: float = 2.0,   # periodic-spline smoothing factor (s in splprep)
        smooth_lambda: float = 0.5,   # Laplacian-smoothing strength for SMOOTH action
        cont_sectors: int = 16,       # continuous (DDPG/TD3) action dim: number of
                                      # angular wedges; action[g] ∈ [-1,1] pushes
                                      # every control point in wedge g along its
                                      # outward normal by action[g]·disp_px. Angular
                                      # (not index) binning makes the action↔location
                                      # mapping STABLE across samples — find_contours
                                      # starts at an arbitrary point, so a per-index
                                      # offset would be unlearnable (same bug fixed for
                                      # the discrete sectors). Lower-dim + spatially
                                      # grounded → the value surface is smooth and
                                      # TD3 can actually learn regional corrections.
        uncertainty_gate: bool = False,   # gate edit magnitude by U-Net confidence (prob_map)
        gate_lo: float = 0.35,            # below this prob, U-Net is confident background
        gate_hi: float = 0.65,            # above this prob, U-Net is confident foreground
        gate_margin: float = 0.10,        # linear ramp width outside [gate_lo, gate_hi]
        directional_state: bool = False,  # append 2 DeepSnake-style SDT-gradient direction channels (5->7)
        auto_smooth_lambda: float = 0.0,  # gentle Laplacian smoothing applied to the point cloud
                                          # after EVERY continuous action (0 = off, back-compat).
                                          # Continuous TD3/DDPG have no explicit SMOOTH action, so
                                          # with small disp_px + many steps the independent per-sector
                                          # pushes accumulate into a wavy contour. A small value
                                          # (~0.1-0.2) removes that high-frequency waviness each step
                                          # without erasing legitimate curvature. Analogous to the
                                          # discrete SMOOTH action (smooth_lambda) but automatic.
        **_ignored,                   # forward-compat: ignore unknown cfg keys
    ):
        assert action_type in ('discrete', 'continuous')
        assert reward_mode in ('dice_delta', 'dice_hd_composite', 'iou_delta',
                               'dice_pbrs', 'dice_hd_pbrs', 'contour_boundary')
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape
        self.prob_map  = (prob_map.astype(np.float32) if prob_map is not None
                          else init_mask.astype(np.float32))
        self.uncertainty_gate = bool(uncertainty_gate)
        self.gate_lo    = float(gate_lo)
        self.gate_hi    = float(gate_hi)
        self.gate_margin = float(gate_margin)
        self.directional_state = bool(directional_state)
        self.auto_smooth_lambda = float(auto_smooth_lambda)
        # Only gate when a REAL prob_map was supplied: the fallback prob_map
        # (= init_mask, a binary {0,1} array) would never fall inside the
        # uncertain band, so gating against it would freeze every point.
        self._gate_active = bool(uncertainty_gate) and (prob_map is not None)
        self.pbrs_gamma = float(pbrs_gamma)
        self._is_pbrs  = reward_mode in ('dice_pbrs', 'dice_hd_pbrs')
        self._needs_hd = reward_mode in ('dice_hd_composite', 'dice_hd_pbrs')
        # Dense per-control-point boundary reward: reward = reduction in the mean
        # distance from the N control points to the GT boundary (uses the already
        # precomputed self._gt_edt). High-SNR (one signal per point, not a single
        # global Dice scalar), self-regularising (moving a point already ON the
        # boundary *increases* its distance -> negative reward, so correct points
        # are left alone and the agent parks itself at the peak instead of
        # drifting). GT is used at TRAINING time only, exactly like every other
        # reward mode. See the module docstring / SKILLS.md.
        self._is_contour_boundary = (reward_mode == 'contour_boundary')

        self.action_type   = action_type
        self.max_steps     = int(max_steps)
        self.sdt_clip      = float(sdt_clip)
        self.reward_clip   = float(reward_clip)
        self.reward_mode   = reward_mode
        self.reward_alpha  = float(reward_alpha)
        self.reward_beta   = float(reward_beta)
        self.hd_norm       = float(hd_norm)
        self.stop_eps_dice = float(stop_eps_dice)
        self.stop_eps_hd   = float(stop_eps_hd)
        self.stop_n        = int(stop_n)
        self.fail_thresh   = float(fail_thresh)
        self.fail_n        = int(fail_n)
        self.reward_step_penalty  = float(reward_step_penalty)
        self.disable_auto_stop    = bool(disable_auto_stop)
        self.terminal_bonus_scale = float(terminal_bonus_scale)
        self.reward_potential_scale = float(reward_potential_scale)

        self.n_points      = int(n_points)
        self.disp_px       = float(disp_px)
        self.spline_smooth = float(spline_smooth)
        self.smooth_lambda = float(smooth_lambda)
        self.cont_sectors  = int(cont_sectors)
        # Continuous action dim = angular wedges (NOT n_points): spatially stable
        # and lower-dimensional than per-point offsets (see cont_sectors docstring).
        self.CONTINUOUS_ACTION_DIM = self.cont_sectors

        # State channels 4-5 (init_mask, prob_map) show ONLY the largest
        # connected component of the U-Net init — the region the single closed
        # contour can actually represent. Debris CCs (extra false blobs the
        # U-Net predicted) are dropped so the agent is never shown foreground it
        # has no action to reach or merge into the one contour. Left raw, they
        # leaked into the state and biased the policy (the dent carved into the
        # WORST-case replay sat exactly where the debris blobs were). GT-FREE —
        # derived only from the init mask. MUST stay identical to the buffer-
        # replay state builder in drl_training._build_state_caches.
        cc = _largest_cc(self.init_mask).astype(np.float32)
        self._init_repr = cc
        self._prob_repr = (self.prob_map * cc).astype(np.float32)

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.t = 0
        self.points = self._init_points()           # (N, 2) float (y, x)
        self.mask   = self._rasterize(self.points)
        if not self.mask.any():                      # spline failed on init
            self.mask = _largest_cc(self.init_mask).astype(np.uint8)
        self._prev_mask = self.mask.copy()

        self._precompute_gt_quantities()
        # Dense-boundary-reward tracker: mean distance of the current control
        # points to the GT boundary at episode start (needs _gt_edt, just built).
        self._prev_point_dist = self._point_boundary_dist(self.points)
        d0 = dice_score(self.mask, self.gt)
        self.dice_history = [d0]
        self.best_dice = d0
        self.best_mask = self.mask.copy()
        if self._needs_hd:
            self.hd95_history = [self._hd95_vs_gt(self.mask)]
        else:
            self.hd95_history = [float('nan')]
        return self._state()

    def _potential(self, dice: float, hd: float) -> float:
        """Baseline-centred, scaled state potential Φ(s) (see env.py for rationale)."""
        K = self.reward_potential_scale
        dice_0 = self.dice_history[0]
        if self.reward_mode == 'dice_hd_pbrs':
            hd_term = (0.0 if (np.isnan(hd) or np.isinf(hd))
                       else 1.0 - float(np.clip(hd / self.hd_norm, 0.0, 1.0)))
            hd0 = self.hd95_history[0]
            hd_term0 = (0.0 if (np.isnan(hd0) or np.isinf(hd0))
                        else 1.0 - float(np.clip(hd0 / self.hd_norm, 0.0, 1.0)))
            return K * (self.reward_alpha * (dice - dice_0)
                        + self.reward_beta * (hd_term - hd_term0))
        return K * (dice - dice_0)   # dice_pbrs

    def _init_points(self) -> np.ndarray:
        """Control points from the U-Net init-mask boundary (warm start)."""
        contour = _largest_contour(self.init_mask)
        if contour.shape[0] >= 3:
            return _resample_closed(contour, self.n_points)
        # Fallback: a small circle at the init-mask centroid (or image centre).
        cc = _largest_cc(self.init_mask)
        if cc.any():
            ys, xs = np.nonzero(cc)
            cy, cx = ys.mean(), xs.mean()
            r = max(4.0, np.sqrt(cc.sum() / np.pi))
        else:
            cy, cx, r = self.H / 2.0, self.W / 2.0, min(self.H, self.W) / 8.0
        ang = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        return np.stack([cy + r * np.sin(ang), cx + r * np.cos(ang)], axis=1)

    def get_best_mask(self) -> np.ndarray:
        return self.best_mask

    def get_final_mask(self) -> np.ndarray:
        return self.mask.copy()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        prev_points = self.points.copy()
        if self.action_type == 'discrete':
            a = int(action)
            if a == self.STOP:
                return self._terminal_step(stop_action=True)
            if a == self.SMOOTH:
                self.points = self._laplacian_smooth(self.points)
            else:
                outward = a < self.SECTORS
                sector  = a if outward else a - self.SECTORS
                self.points = self._push_sector(self.points, sector, outward)
        else:
            self.points = self._apply_continuous(np.asarray(action, dtype=np.float32))

        # Rasterise the new contour; revert points + mask on degenerate result so
        # a collapsed edit cannot poison subsequent steps.
        new_mask = self._rasterize(self.points)
        if not new_mask.any():
            self.points = prev_points
            new_mask = self._prev_mask.copy()
        self.mask = new_mask
        self._prev_mask = self.mask.copy()

        self.t += 1
        new_dice = dice_score(self.mask, self.gt)
        new_hd95 = self._hd95_vs_gt(self.mask) if self._needs_hd else float('nan')
        if self._is_pbrs:
            phi_prev = self._potential(self.dice_history[-1], self.hd95_history[-1])
        self.dice_history.append(new_dice)
        self.hd95_history.append(new_hd95)

        if new_dice > self.best_dice:
            self.best_dice = new_dice
            self.best_mask = self.mask.copy()

        done = self._check_termination()

        if self._is_contour_boundary:
            # Dense reward: how much closer (px) did the control points get to the
            # GT boundary this step. Positive when the contour moves toward truth;
            # NEGATIVE when a boundary-correct point is pushed off — this is the
            # built-in trust region that makes "leave correct points alone" and
            # "stop when converged" the reward-optimal behaviour.
            new_point_dist = self._point_boundary_dist(self.points)
            raw_reward = (self._prev_point_dist - new_point_dist) - self.reward_step_penalty
            self._prev_point_dist = new_point_dist
            clip = self.reward_clip
            if done and self.terminal_bonus_scale > 0.0:
                raw_reward += self.terminal_bonus_scale * (new_dice - self.dice_history[0])
                clip = self.reward_clip + self.terminal_bonus_scale
        elif self._is_pbrs:
            phi_new = self._potential(new_dice, new_hd95)
            raw_reward = (self.pbrs_gamma * phi_new - phi_prev) - self.reward_step_penalty
            clip = self.reward_clip
        else:
            raw_reward = self._compute_reward(new_dice, new_hd95) - self.reward_step_penalty
            clip = self.reward_clip
            if done and self.terminal_bonus_scale > 0.0:
                raw_reward += self.terminal_bonus_scale * (new_dice - self.dice_history[0])
                clip = self.reward_clip + self.terminal_bonus_scale
        reward = float(np.clip(raw_reward, -clip, clip))

        info = {'dice': new_dice, 'hd95': new_hd95, 'delta_dice': raw_reward,
                'reward_clipped': reward != raw_reward, 'step': self.t,
                'best_dice': self.best_dice}
        return self._state(), reward, done, info

    def _terminal_step(self, stop_action: bool) -> Tuple[np.ndarray, float, bool, Dict]:
        """Explicit STOP — terminate now, scoring the current contour."""
        self.t += 1
        new_dice = dice_score(self.mask, self.gt)
        new_hd95 = self._hd95_vs_gt(self.mask) if self._needs_hd else float('nan')
        if self._is_pbrs or self._is_contour_boundary:
            # STOP commits the current contour; per-step value already credited
            # (dense boundary reward has no separate terminal term to add here).
            raw_reward = 0.0
            clip = self.reward_clip
        else:
            raw_reward = self._compute_reward(new_dice, new_hd95)
            if self.terminal_bonus_scale > 0.0:
                raw_reward += self.terminal_bonus_scale * (new_dice - self.dice_history[0])
            clip = self.reward_clip + self.terminal_bonus_scale
        reward = float(np.clip(raw_reward, -clip, clip))
        info = {'dice': new_dice, 'hd95': new_hd95, 'delta_dice': raw_reward,
                'reward_clipped': reward != raw_reward, 'step': self.t,
                'stop_action': stop_action, 'best_dice': self.best_dice}
        return self._state(), reward, True, info

    def _check_termination(self) -> bool:
        done = self.t >= self.max_steps
        if not done and not self.disable_auto_stop and len(self.dice_history) > self.stop_n:
            dice_0 = self.dice_history[0]
            dd = [abs(self.dice_history[-k - 1] - self.dice_history[-k - 2])
                  for k in range(self.stop_n)]
            dice_converged = all(d < self.stop_eps_dice for d in dd)
            dice_improved = all(
                self.dice_history[-k - 1] > dice_0 + self.stop_eps_dice
                for k in range(self.stop_n))
            done = dice_converged or dice_improved
        if (not done and self.fail_thresh > 0.0
                and len(self.dice_history) > self.fail_n):
            dice_0 = self.dice_history[0]
            recent_bad = all(
                self.dice_history[-k - 1] < dice_0 - self.fail_thresh
                for k in range(self.fail_n))
            if recent_bad:
                done = True
        return done

    # ── contour operations ────────────────────────────────────────────────────

    def _gate_weights(self, points: np.ndarray) -> np.ndarray:
        """Per-control-point uncertainty gate in [0,1] from self.prob_map.
        1.0 inside the uncertain band [gate_lo, gate_hi]; ramps linearly to 0.0
        over `gate_margin` outside that band. Bounds how much a single edit can
        move a point the U-Net is already confident about, so the worst-case
        per-step Dice drop is bounded and learning concentrates on the U-Net's
        genuinely correctable errors. Uses prob_map (the U-Net's own confidence),
        never ground truth — fully valid at deployment time."""
        if not self._gate_active:
            return np.ones(points.shape[0], dtype=np.float32)
        ys = np.clip(points[:, 0].round().astype(int), 0, self.H - 1)
        xs = np.clip(points[:, 1].round().astype(int), 0, self.W - 1)
        p = self.prob_map[ys, xs]
        lo, hi, m = self.gate_lo, self.gate_hi, max(self.gate_margin, 1e-6)
        below = np.clip((p - (lo - m)) / m, 0.0, 1.0)
        above = np.clip(((hi + m) - p) / m, 0.0, 1.0)
        return np.minimum(below, above).astype(np.float32)

    def _outward_normals(self, points: np.ndarray) -> np.ndarray:
        """Unit outward normal at each control point (N, 2) in (y, x).

        Tangent t_i = p_{i+1} − p_{i-1} (wrap-around); normal ⟂ t_i, oriented
        away from the contour centroid. Degenerate (zero-length) tangents fall
        back to the radial direction.
        """
        c = points.mean(axis=0)
        nxt = np.roll(points, -1, axis=0)
        prv = np.roll(points, +1, axis=0)
        tang = nxt - prv                                   # (N, 2) (dy, dx)
        # perpendicular to (dy, dx) is (dx, -dy)
        norm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
        mag = np.sqrt((norm ** 2).sum(axis=1, keepdims=True))
        radial = points - c
        rmag = np.sqrt((radial ** 2).sum(axis=1, keepdims=True))
        unit = np.where(mag > 1e-6, norm / np.maximum(mag, 1e-6),
                        radial / np.maximum(rmag, 1e-6))
        # Orient outward: flip any normal pointing toward the centroid.
        sign = np.sign((unit * radial).sum(axis=1, keepdims=True))
        sign[sign == 0] = 1.0
        return unit * sign

    def _sector_indices(self, points: np.ndarray, sector: int) -> np.ndarray:
        """Indices of control points whose angle about the centroid falls in
        ``sector``'s 45° wedge.

        Sectors are defined SPATIALLY (compass wedges around the centroid), not
        by point order: find_contours starts tracing at an arbitrary point, so
        an index-based sector would map to a different region on every image and
        the action semantics would be unlearnable. Angle-based wedges give a
        stable mapping — "sector g" is always the same direction — so the CNN
        can learn "error in this direction → push this sector".
        """
        c = points.mean(axis=0)
        ang = np.arctan2(points[:, 0] - c[0], points[:, 1] - c[1])   # (y,x)→angle
        ang = np.mod(ang, 2.0 * np.pi)
        bins = np.floor(ang / (2.0 * np.pi / self.SECTORS)).astype(int)
        bins = np.clip(bins, 0, self.SECTORS - 1)
        return np.nonzero(bins == sector)[0]

    def _push_sector(self, points: np.ndarray, sector: int, outward: bool) -> np.ndarray:
        """Move one angular wedge of points along their outward normals.

        A distance-weighted taper (1 at the wedge centre angle → ~0 at its
        edges) keeps the deformation smooth — no kinks at wedge boundaries —
        the local analogue of a directional dilation/erosion.
        """
        normals = self._outward_normals(points)
        gate = self._gate_weights(points)
        idx = self._sector_indices(points, sector)
        out = points.copy()
        m = len(idx)
        if m == 0:
            return out                       # empty wedge (non-convex) → no-op
        # angular taper: weight by closeness to the wedge centre angle
        c = points.mean(axis=0)
        wedge = 2.0 * np.pi / self.SECTORS
        centre_ang = (sector + 0.5) * wedge
        ang = np.mod(np.arctan2(points[idx, 0] - c[0], points[idx, 1] - c[1]), 2.0 * np.pi)
        dtheta = np.abs(np.mod(ang - centre_ang + np.pi, 2.0 * np.pi) - np.pi)
        taper = np.clip(1.0 - dtheta / (wedge), 0.0, 1.0)
        sign = 1.0 if outward else -1.0
        out[idx] = points[idx] + sign * self.disp_px * taper[:, None] * gate[idx, None] * normals[idx]
        return self._clip_points(out)

    def _laplacian_smooth(self, points: np.ndarray) -> np.ndarray:
        """One pass of periodic Laplacian smoothing (the SMOOTH action)."""
        nxt = np.roll(points, -1, axis=0)
        prv = np.roll(points, +1, axis=0)
        target = 0.5 * (nxt + prv)
        out = points + self.smooth_lambda * (target - points)
        return self._clip_points(out)

    def _apply_continuous(self, a: np.ndarray) -> np.ndarray:
        """Angular-sector normal push (TD3/DDPG action).

        action[g] ∈ [-1, 1] is the signed displacement (× disp_px) applied along
        the outward normal to each control point, CIRCULARLY LINEARLY INTERPOLATED
        between the two nearest sector centres by angular position (not a hard
        per-sector bin assignment). At each sector's own centre angle, disp==a[g]
        exactly — same semantics as before — but it now blends smoothly toward the
        neighbouring sector's value near a wedge boundary instead of jumping
        discretely. With ~2 control points per wedge (n_points=32, cont_sectors=16
        in the CAMUS configs), a hard bin edge landing between two adjacent points
        with very different action values produces a visible geometric kink in the
        deformed contour — this is what that looked like in practice. Angular
        binning (now interpolation) is still angle-based, not point-index-based,
        for the same reason as before: find_contours seeds at an arbitrary point,
        so an index-keyed offset would map to a different region every episode.
        """
        a = np.asarray(a, dtype=np.float32).ravel()
        nb = a.shape[0]                                  # = self.cont_sectors
        c = self.points.mean(axis=0)
        ang = np.mod(np.arctan2(self.points[:, 0] - c[0],
                                self.points[:, 1] - c[1]), 2.0 * np.pi)
        wedge = 2.0 * np.pi / nb
        pos = ang / wedge - 0.5                          # continuous sector coordinate
        g0 = np.floor(pos).astype(int)
        frac = (pos - g0).astype(np.float32)
        g0m = np.mod(g0, nb)
        g1m = np.mod(g0 + 1, nb)
        disp = (1.0 - frac) * a[g0m] + frac * a[g1m]      # (N,) circularly interpolated
        gate = self._gate_weights(self.points)
        disp = disp * gate
        normals = self._outward_normals(self.points)
        out = self.points + (disp[:, None] * self.disp_px) * normals
        # Gentle per-step Laplacian smoothing removes the high-frequency waviness
        # that independent per-sector pushes accumulate over many small steps
        # (continuous agents have no explicit SMOOTH action). Off when lambda=0.
        if self.auto_smooth_lambda > 0.0:
            nxt = np.roll(out, -1, axis=0)
            prv = np.roll(out, +1, axis=0)
            out = out + self.auto_smooth_lambda * (0.5 * (nxt + prv) - out)
        return self._clip_points(out)

    def _clip_points(self, points: np.ndarray) -> np.ndarray:
        points[:, 0] = np.clip(points[:, 0], 0.0, self.H - 1.0)
        points[:, 1] = np.clip(points[:, 1], 0.0, self.W - 1.0)
        return points

    def _rasterize(self, points: np.ndarray) -> np.ndarray:
        """Control points → smooth closed spline → filled binary mask.

        Periodic cubic B-spline interpolation (scipy.interpolate.splprep,
        per=1) produces the smooth anatomical boundary the supervisor asked for
        (no staircase). Falls back to the straight-segment polygon if the spline
        fit fails (e.g. collapsed/duplicate points).
        """
        from skimage.draw import polygon as sk_polygon
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        if points.shape[0] < 3:
            return mask
        y = points[:, 0]
        x = points[:, 1]
        dense_y, dense_x = y, x
        try:
            from scipy.interpolate import splprep, splev
            # de-duplicate consecutive points (splprep requires distinct knots)
            keep = np.concatenate([[True], (np.abs(np.diff(y)) + np.abs(np.diff(x))) > 1e-3])
            ys, xs = y[keep], x[keep]
            if ys.shape[0] >= 4:
                tck, _ = splprep([xs, ys], s=self.spline_smooth, per=1, k=3)
                u = np.linspace(0.0, 1.0, max(200, 4 * self.n_points))
                dense_x, dense_y = splev(u, tck)
        except Exception:
            dense_y, dense_x = y, x
        rr, cc = sk_polygon(np.clip(dense_y, 0, self.H - 1),
                            np.clip(dense_x, 0, self.W - 1), shape=(self.H, self.W))
        mask[rr, cc] = 1
        return mask

    # ── reward (mirrors SegmentationEnv) ──────────────────────────────────────

    def _compute_reward(self, new_dice: float, new_hd: float) -> float:
        dice_0 = self.dice_history[0]
        if self.reward_mode == 'dice_delta':
            return new_dice - dice_0
        if self.reward_mode == 'dice_hd_composite':
            hd95_0 = self.hd95_history[0]
            r_dice = new_dice - dice_0
            if (np.isnan(hd95_0) or np.isnan(new_hd)
                    or np.isinf(hd95_0) or np.isinf(new_hd)):
                r_hd = 0.0
            else:
                r_hd = float(np.clip((hd95_0 - new_hd) / self.hd_norm, -1.0, 1.0))
            return self.reward_alpha * r_dice + self.reward_beta * r_hd
        # iou_delta
        iou_0 = dice_0   / max(2.0 - dice_0,   1e-6)
        iou_t = new_dice / max(2.0 - new_dice, 1e-6)
        return iou_t - iou_0

    # ── GT precompute + HD95 (mirrors SegmentationEnv) ────────────────────────

    def _precompute_gt_quantities(self) -> None:
        gt_bool = self.gt.astype(bool)
        gt_edge = gt_bool ^ ndi.binary_erosion(gt_bool, STRUCT)
        self._gt_edge     = gt_edge
        self._gt_edge_any = bool(gt_edge.any())
        self._gt_edt      = ndi.distance_transform_edt(~gt_edge)

    def _point_boundary_dist(self, points: np.ndarray) -> float:
        """Mean distance (px) from the control points to the GT boundary.

        Reads the precomputed `self._gt_edt` (Euclidean distance-to-GT-edge at
        every pixel) at each control point's rounded (y, x). This is the signal
        the dense `contour_boundary` reward differences step-to-step. GT-derived,
        so TRAINING-time only — same contract as the Dice/HD reward terms."""
        ys = np.clip(points[:, 0].round().astype(int), 0, self.H - 1)
        xs = np.clip(points[:, 1].round().astype(int), 0, self.W - 1)
        return float(self._gt_edt[ys, xs].mean())

    def _hd95_vs_gt(self, m1: np.ndarray) -> float:
        m1b = _largest_cc(m1)
        if not m1b.any() and not self._gt_edge_any:
            return 0.0
        if not m1b.any() or not self._gt_edge_any:
            return float('nan')
        edge_1 = m1b ^ ndi.binary_erosion(m1b, STRUCT)
        if not edge_1.any():
            return 0.0
        dt_1 = ndi.distance_transform_edt(~edge_1)
        d_12 = self._gt_edt[edge_1]
        d_21 = dt_1[self._gt_edge]
        return float(np.percentile(np.concatenate([d_12, d_21]), 95))

    # ── state ─────────────────────────────────────────────────────────────────

    def _state(self) -> np.ndarray:
        sdt = signed_dt(self.mask, self.sdt_clip)
        chans = [
            self.image,
            self.mask.astype(np.float32),
            sdt,
            self._init_repr,                    # largest-CC init (debris dropped)
            self._prob_repr,                    # U-Net confidence, masked to that CC
        ]
        # Optional directional (DeepSnake-style) channels, appended AT THE END
        # so the sdt stays at index 2 (the ReplayBuffer caches next_state[2] as
        # the SDT). Off by default → 5 channels, unchanged. MUST use the shared
        # geometry.sdt_direction_field so training/eval state builders match.
        if self.directional_state:
            field = sdt_direction_field(sdt)     # (2, H, W): dy, dx
            chans.extend([field[0], field[1]])
        return np.stack(chans, axis=0)
