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
"""

from typing import Tuple, Dict, List
import numpy as np
import scipy.ndimage as ndi

from .env import dice_score, signed_dt, _largest_cc, STRUCT


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
        prob_map: np.ndarray = None,  # accepted for interface parity (unused here)
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
        # ── contour-specific ──────────────────────────────────────────────────
        n_points: int = 32,           # number of control points on the contour
        disp_px: float = 1.5,         # per-edit displacement along the normal (px)
        spline_smooth: float = 2.0,   # periodic-spline smoothing factor (s in splprep)
        smooth_lambda: float = 0.5,   # Laplacian-smoothing strength for SMOOTH action
        **_ignored,                   # forward-compat: ignore unknown cfg keys
    ):
        assert action_type in ('discrete', 'continuous')
        assert reward_mode in ('dice_delta', 'dice_hd_composite', 'iou_delta')
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape

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

        self.n_points      = int(n_points)
        self.disp_px       = float(disp_px)
        self.spline_smooth = float(spline_smooth)
        self.smooth_lambda = float(smooth_lambda)
        self.CONTINUOUS_ACTION_DIM = self.n_points

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
        d0 = dice_score(self.mask, self.gt)
        self.dice_history = [d0]
        self.best_dice = d0
        self.best_mask = self.mask.copy()
        if self.reward_mode == 'dice_hd_composite':
            self.hd95_history = [self._hd95_vs_gt(self.mask)]
        else:
            self.hd95_history = [float('nan')]
        return self._state()

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
        new_hd95 = self._hd95_vs_gt(self.mask) \
                   if self.reward_mode == 'dice_hd_composite' else float('nan')
        self.dice_history.append(new_dice)
        self.hd95_history.append(new_hd95)

        if new_dice > self.best_dice:
            self.best_dice = new_dice
            self.best_mask = self.mask.copy()

        done = self._check_termination()

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
        new_hd95 = self._hd95_vs_gt(self.mask) \
                   if self.reward_mode == 'dice_hd_composite' else float('nan')
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
        out[idx] = points[idx] + sign * self.disp_px * taper[:, None] * normals[idx]
        return self._clip_points(out)

    def _laplacian_smooth(self, points: np.ndarray) -> np.ndarray:
        """One pass of periodic Laplacian smoothing (the SMOOTH action)."""
        nxt = np.roll(points, -1, axis=0)
        prv = np.roll(points, +1, axis=0)
        target = 0.5 * (nxt + prv)
        out = points + self.smooth_lambda * (target - points)
        return self._clip_points(out)

    def _apply_continuous(self, a: np.ndarray) -> np.ndarray:
        """Per-point normal offset (DeepSnake/MARL offset-prediction form)."""
        if a.shape[0] != self.n_points:
            raise ValueError(f'continuous action dim {a.shape[0]} != n_points {self.n_points}')
        normals = self._outward_normals(self.points)
        out = self.points + (a[:, None] * self.disp_px) * normals
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
        return np.stack([
            self.image,
            self.mask.astype(np.float32),
            sdt,
            self.init_mask.astype(np.float32),
        ], axis=0)
