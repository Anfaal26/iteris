"""
Boundary-refinement segmentation environment for DRL agents (v3-discrete / v2-continuous).

One agent operates on one target class at a time (per-class binary formulation).
The episode starts from the U-Net baseline mask (warm-start) and refines toward
GT over up to 20 steps.

────────────────────────────────────────────────────────────────────────────────
Sprint 1 (Nov 2026): directional discrete + reverted continuous
────────────────────────────────────────────────────────────────────────────────

Background — why discrete needed the refactor:
  v2 used a single global isotropic dilate / erode.  When the U-Net init is
  near-optimal (CAMUS LV_endo Dice ~ 0.94) the boundary is locally wrong in
  different directions: left edge too small, right edge too big, apex correct.
  A global dilate helps the left edge but hurts the right edge and the apex —
  almost always net-negative ΔDice.  No reward shaping or action penalty can
  rescue this: the action space genuinely had no good move on near-optimal
  samples.

  DISCRETE (DQN family) — 13 actions, directional structuring elements:
    0–3   directional dilate  (push boundary out 1 px in one cardinal direction)
    4–7   directional erode   (pull boundary in 1 px from one cardinal direction)
    8–11  whole-mask shift    (translate ±shift_px in one cardinal direction)
    12    no-op
    3-element SEs (e.g. [[1],[1],[0]] for NORTH) make each op affect only the
    boundary segment facing that direction.  1-px ops self-restrict to the
    immediate boundary band — no extra band_px knob needed.

  CONTINUOUS (DDPG family) — 3-component action (deliberately kept simple):
    a[0]  morph    ∈ [-cont_morph_scale, +cont_morph_scale]   SDT threshold shift
    a[1]  dy_norm  ∈ [-cont_trans_scale, +cont_trans_scale]   global y-translation
    a[2]  dx_norm  ∈ [-cont_trans_scale, +cont_trans_scale]   global x-translation
    Global isotropic morph has the same fundamental limitation as v2's discrete:
    the right algorithmic answer for DDPG is a contour-control-point action
    space (Sprint 2), not a window-selector bolted on top of continuous values.
    Until Sprint 2 lands, DDPG uses the same 3-component action as v2 — this
    keeps the algorithm coherent at the cost of staying on the limited action
    space.  DDPG results on the current action space serve as the "global
    continuous" baseline; Sprint-2 contour DDPG is the algorithmic upgrade.

────────────────────────────────────────────────────────────────────────────────

State (4, H, W) float32:
    ch 0 : preprocessed image                       (static within an episode)
    ch 1 : current binary mask for the target class (dynamic)
    ch 2 : signed distance transform of ch 1        (dynamic, derived from ch 1)
    ch 3 : U-Net init mask                          (static within an episode)

Reward modes (set per-class via YAML):
    dice_delta          r_t = Dice_t - Dice_0  (episode-start baseline)
    dice_hd_composite   α·ΔDice + β·ΔHD95_norm                — boundary precision
    iou_delta           r_t = IoU_t  - IoU_0                  — small targets

Episode: max_steps steps, OR composite stop
    (|ΔDice| < stop_eps_dice AND |ΔHD95| < stop_eps_hd) for stop_n consecutive
    steps, OR Dice has been > Dice_0 + stop_eps_dice for stop_n consecutive
    steps (terminates the dilate/erode oscillation trap if it ever occurs).
"""

from typing import Tuple, Dict
import numpy as np
import scipy.ndimage as ndi


# 4-connectivity cross — kept for HD95 boundary extraction
STRUCT = ndi.generate_binary_structure(2, 1)

# ── Directional structuring elements (3-element, 1-px move in one direction) ─
# binary_dilation with SE_N: each mask pixel marks the pixel ABOVE it → boundary
# extends NORTH.  binary_erosion with SE_N: pixel kept only if both itself AND
# pixel above are in mask → strips the NORTH (top) boundary.
SE_N = np.array([[1], [1], [0]], dtype=np.uint8)   # north  (offsets (-1, 0) and (0, 0))
SE_S = np.array([[0], [1], [1]], dtype=np.uint8)   # south  (offsets (+1, 0) and (0, 0))
SE_W = np.array([[1, 1, 0]],    dtype=np.uint8)    # west   (offsets (0, -1) and (0, 0))
SE_E = np.array([[0, 1, 1]],    dtype=np.uint8)    # east   (offsets (0, +1) and (0, 0))


def dice_score(m1: np.ndarray, m2: np.ndarray, eps: float = 1e-6) -> float:
    m1 = m1.astype(bool); m2 = m2.astype(bool)
    inter = (m1 & m2).sum()
    return (2.0 * inter + eps) / (m1.sum() + m2.sum() + eps)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return a binary mask keeping only the largest connected component.

    Stray isolated U-Net / GT pixels far from the main structure inflate HD95
    catastrophically (a single pixel in the image corner → ~200 px distance to
    the real boundary).  Keeping only the largest CC removes these artefacts
    before edge / EDT computation.
    """
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0             # ignore background
    return (labeled == sizes.argmax()).astype(bool)


def hd95_px(m1: np.ndarray, m2: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in pixels.

    Applies largest-connected-component filtering before edge extraction to
    prevent stray isolated pixels (U-Net FP fragments far from the structure)
    from inflating HD95 to hundreds of pixels.
    """
    m1b = _largest_cc(m1)
    m2b = _largest_cc(m2)
    if not m1b.any() and not m2b.any():
        return 0.0
    if not m1b.any() or not m2b.any():
        return float('nan')
    edges_1 = m1b ^ ndi.binary_erosion(m1b, STRUCT)
    edges_2 = m2b ^ ndi.binary_erosion(m2b, STRUCT)
    if not edges_1.any() or not edges_2.any():
        return 0.0
    dt_2 = ndi.distance_transform_edt(~edges_2)
    dt_1 = ndi.distance_transform_edt(~edges_1)
    d_12 = dt_2[edges_1]
    d_21 = dt_1[edges_2]
    return float(np.percentile(np.concatenate([d_12, d_21]), 95))


def signed_dt(mask: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Signed distance transform, clipped and normalised to [-1, +1]."""
    pos = ndi.distance_transform_edt(mask.astype(bool))
    neg = ndi.distance_transform_edt(~mask.astype(bool))
    sdt = pos - neg
    return (np.clip(sdt, -clip, clip) / clip).astype(np.float32)


def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate by (dy, dx) px with zero-fill, no wraparound."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    y_src = slice(max(0, -dy), H - max(0, dy))
    y_dst = slice(max(0, dy),  H - max(0, -dy))
    x_src = slice(max(0, -dx), W - max(0, dx))
    x_dst = slice(max(0, dx),  W - max(0, -dx))
    out[y_dst, x_dst] = mask[y_src, x_src]
    return out


class SegmentationEnv:
    """Per-class binary boundary-refinement environment (v3 action space)."""

    # ── Discrete action layout (see module docstring) ────────────────────────
    NUM_DISCRETE_ACTIONS  = 13
    DILATE_N, DILATE_E, DILATE_S, DILATE_W = 0, 1, 2, 3
    ERODE_N,  ERODE_E,  ERODE_S,  ERODE_W  = 4, 5, 6, 7
    SHIFT_U,  SHIFT_D,  SHIFT_L,  SHIFT_R  = 8, 9, 10, 11
    NOOP                                    = 12

    # ── Continuous action layout (see module docstring) ──────────────────────
    CONTINUOUS_ACTION_DIM = 3     # (morph, dy_norm, dx_norm)

    def __init__(
        self,
        image: np.ndarray,         # (H, W) float32 in [0, 1]
        gt_mask: np.ndarray,       # (H, W) uint8 in {0, 1}
        init_mask: np.ndarray,     # (H, W) uint8 in {0, 1} — U-Net prediction
        action_type: str = 'discrete',
        max_steps: int = 20,
        shift_px: int = 2,
        sdt_clip: float = 20.0,
        reward_clip: float = 1.0,
        stop_eps_dice: float = 0.001,
        stop_eps_hd: float = 0.5,
        stop_n: int = 3,
        # ── continuous action scales ─────────────────────────────────────────
        cont_morph_scale: float = 0.25,   # max SDT threshold shift (±5 px at sdt_clip=20)
        cont_trans_scale: float = 0.02,   # max translation as fraction of image dim (±5 px at H=256)
        # ── reward shaping (per-class tuned via YAML) ────────────────────────
        reward_mode: str = 'dice_delta',
        reward_alpha: float = 0.5,        # weight on Dice component (composite)
        reward_beta: float  = 0.5,        # weight on HD95 component (composite)
        hd_norm: float = 50.0,            # HD95 normalisation in px — tune to 2× expected HD95
    ):
        assert action_type in ('discrete', 'continuous')
        assert reward_mode in ('dice_delta', 'dice_hd_composite', 'iou_delta')
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape

        self.action_type      = action_type
        self.max_steps        = max_steps
        self.shift_px         = shift_px
        self.sdt_clip         = sdt_clip
        self.reward_clip      = reward_clip
        self.stop_eps_dice    = stop_eps_dice
        self.stop_eps_hd      = stop_eps_hd
        self.stop_n           = stop_n
        self.cont_morph_scale = cont_morph_scale
        self.cont_trans_scale = cont_trans_scale
        self.reward_mode      = reward_mode
        self.reward_alpha     = reward_alpha
        self.reward_beta      = reward_beta
        self.hd_norm          = hd_norm

        self.reset()

    def _compute_reward(self, new_dice: float, new_hd: float) -> float:
        """Dataset-tuned reward shaping. Returns RAW reward (will be clipped).

        IMPORTANT — baseline is the EPISODE START, not the previous step.

        Using r_t = Dice_t − Dice_0 (rather than the step-wise Δ Dice_t − Dice_{t-1})
        eliminates the oscillation trap that plagues step-wise delta rewards:

          • Staying at init:  r = 0 (neutral) every step
          • Improving:        r > 0 every step spent in the improved state
          • Degrading:        r < 0 every step spent below init
          • Best strategy:    reach an improved state ASAP and no-op there
        """
        dice_0 = self.dice_history[0]   # Dice of init_mask vs GT, fixed at reset()

        if self.reward_mode == 'dice_delta':
            return new_dice - dice_0

        if self.reward_mode == 'dice_hd_composite':
            hd95_0 = self.hd95_history[0]
            r_dice = new_dice - dice_0
            if (np.isnan(hd95_0) or np.isnan(new_hd) or
                np.isinf(hd95_0) or np.isinf(new_hd)):
                r_hd = 0.0
            else:
                r_hd = float(np.clip((hd95_0 - new_hd) / self.hd_norm, -1.0, 1.0))
            return self.reward_alpha * r_dice + self.reward_beta * r_hd

        if self.reward_mode == 'iou_delta':
            iou_0  = dice_0   / max(2.0 - dice_0,   1e-6)
            iou_t  = new_dice / max(2.0 - new_dice,  1e-6)
            return iou_t - iou_0

        raise ValueError(f'Unknown reward_mode: {self.reward_mode}')

    # ── GT precomputation (called once per episode in reset()) ───────────────

    def _precompute_gt_quantities(self) -> None:
        """Cache GT edge mask and GT EDT — GT is constant for the whole episode."""
        gt_bool = self.gt.astype(bool)
        gt_edge = gt_bool ^ ndi.binary_erosion(gt_bool, STRUCT)
        self._gt_edge     = gt_edge
        self._gt_edge_any = bool(gt_edge.any())
        self._gt_edt      = ndi.distance_transform_edt(~gt_edge)

    def _hd95_vs_gt(self, m1: np.ndarray) -> float:
        """HD95 between m1 and GT, reusing precomputed GT quantities."""
        m1b = _largest_cc(m1)
        if not m1b.any() and not self._gt_edge_any:
            return 0.0
        if not m1b.any() or not self._gt_edge_any:
            return float('nan')
        edge_1 = m1b ^ ndi.binary_erosion(m1b, STRUCT)
        if not edge_1.any():
            return 0.0
        dt_1  = ndi.distance_transform_edt(~edge_1)
        d_12  = self._gt_edt[edge_1]
        d_21  = dt_1[self._gt_edge]
        return float(np.percentile(np.concatenate([d_12, d_21]), 95))

    def reset(self) -> np.ndarray:
        self.mask = self.init_mask.copy()
        self.t    = 0
        self._precompute_gt_quantities()
        self.dice_history = [dice_score(self.mask, self.gt)]
        if self.reward_mode != 'dice_delta':
            self.hd95_history = [self._hd95_vs_gt(self.mask)]
        else:
            self.hd95_history = [float('nan')]
        return self._state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.action_type == 'discrete':
            self.mask = self._apply_discrete(int(action))
        else:
            self.mask = self._apply_continuous(action)

        # Mask-empty guard: catastrophic erosion (especially on small BRISC
        # tumours) can wipe the mask entirely.  Revert to init_mask so the
        # episode can recover instead of being stuck at Dice 0.
        if not self.mask.any():
            self.mask = self.init_mask.copy()

        self.t += 1
        new_dice = dice_score(self.mask, self.gt)

        if self.reward_mode == 'dice_hd_composite':
            new_hd95 = self._hd95_vs_gt(self.mask)
        else:
            new_hd95 = float('nan')

        raw_reward = self._compute_reward(new_dice, new_hd95)
        reward     = float(np.clip(raw_reward, -self.reward_clip, self.reward_clip))

        self.dice_history.append(new_dice)
        self.hd95_history.append(new_hd95)

        done = self.t >= self.max_steps
        if not done and len(self.dice_history) > self.stop_n:
            dice_0 = self.dice_history[0]

            # Convergence: Dice/HD95 has plateaued
            dd = [abs(self.dice_history[-k-1] - self.dice_history[-k-2])
                  for k in range(self.stop_n)]
            dice_converged = all(d < self.stop_eps_dice for d in dd)

            # Improvement maintained: last stop_n steps all above init
            dice_improved = all(
                self.dice_history[-k-1] > dice_0 + self.stop_eps_dice
                for k in range(self.stop_n)
            )

            if self.reward_mode == 'dice_hd_composite':
                dh = [abs(self.hd95_history[-k-1] - self.hd95_history[-k-2])
                      for k in range(self.stop_n)]
                hd_converged = all(
                    (np.isnan(h) or h < self.stop_eps_hd) for h in dh
                )
                done = (dice_converged and hd_converged) or dice_improved
            else:
                done = dice_converged or dice_improved

        info = {
            'dice': new_dice,
            'hd95': new_hd95,
            'delta_dice': raw_reward,
            'reward_clipped': reward != raw_reward,
            'step': self.t,
        }
        return self._state(), reward, done, info

    # ── private ──────────────────────────────────────────────────────────────

    def _apply_discrete(self, a: int) -> np.ndarray:
        """13-action discrete dispatcher — see module docstring for layout.

        Directional ops use 3-element SEs so each op only affects the boundary
        segment facing that cardinal direction.  Band restriction (#2) is
        automatic for 1-px ops: dilations only add pixels at raw_sdt = -1,
        erosions only remove pixels at raw_sdt = +1 — both inside any
        reasonable band_px (>= 1).
        """
        sp = self.shift_px
        # Directional dilate — push boundary out 1 px in one direction
        if a == 0:  return ndi.binary_dilation(self.mask, SE_N).astype(np.uint8)
        if a == 1:  return ndi.binary_dilation(self.mask, SE_E).astype(np.uint8)
        if a == 2:  return ndi.binary_dilation(self.mask, SE_S).astype(np.uint8)
        if a == 3:  return ndi.binary_dilation(self.mask, SE_W).astype(np.uint8)
        # Directional erode — pull boundary in 1 px from one direction
        if a == 4:  return ndi.binary_erosion(self.mask, SE_N).astype(np.uint8)
        if a == 5:  return ndi.binary_erosion(self.mask, SE_E).astype(np.uint8)
        if a == 6:  return ndi.binary_erosion(self.mask, SE_S).astype(np.uint8)
        if a == 7:  return ndi.binary_erosion(self.mask, SE_W).astype(np.uint8)
        # Whole-mask shifts
        if a == 8:  return shifted(self.mask, -sp,   0)   # ↑
        if a == 9:  return shifted(self.mask, +sp,   0)   # ↓
        if a == 10: return shifted(self.mask,   0, -sp)   # ←
        if a == 11: return shifted(self.mask,   0, +sp)   # →
        if a == 12: return self.mask                       # no-op
        raise ValueError(f'Bad discrete action: {a}')

    def _apply_continuous(self, a) -> np.ndarray:
        """
        3-component continuous action — global SDT-threshold morph + translation.

        a[0]  morph   ∈ [-cont_morph_scale, +cont_morph_scale]
              SDT threshold shift.  +morph → lower threshold → include exterior
              pixels → dilate.  -morph → raise threshold → erode.
              At sdt_clip=20, morph=0.25 corresponds to ±5 px boundary shift.
        a[1]  dy_norm ∈ [-cont_trans_scale, +cont_trans_scale]
              Fractional y-translation: dy_px = round(a[1] · H).
        a[2]  dx_norm ∈ [-cont_trans_scale, +cont_trans_scale]
              Fractional x-translation: dx_px = round(a[2] · W).

        Sprint-2 note: this is global isotropic morph (same fundamental
        limitation as the v2 discrete dilate/erode).  Contour-control-point
        DDPG in Sprint 2 will replace this with a (K, 2) continuous action
        over K boundary points — the algorithmically-correct continuous action
        for boundary refinement.
        """
        morph = float(a[0])
        dy    = int(round(float(a[1]) * self.H))
        dx    = int(round(float(a[2]) * self.W))

        # SDT-based morph (global, no window / band restriction)
        sdt    = signed_dt(self.mask, self.sdt_clip)        # (H, W) ∈ [-1, +1]
        thresh = -morph                                      # +morph → dilate
        morphed = (sdt >= thresh).astype(np.uint8)

        # Guard: catastrophic erosion can empty the mask; keep original instead.
        if not morphed.any():
            morphed = self.mask.copy()

        return shifted(morphed, dy, dx)

    def _state(self) -> np.ndarray:
        sdt = signed_dt(self.mask, self.sdt_clip)
        return np.stack([
            self.image,
            self.mask.astype(np.float32),
            sdt,
            self.init_mask.astype(np.float32),
        ], axis=0)
