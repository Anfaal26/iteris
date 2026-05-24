"""
Boundary-refinement segmentation environment for DRL agents (locked v2 contract).

One agent operates on one target class at a time (per-class binary formulation).
The episode starts from the U-Net baseline mask (warm-start) and refines toward
GT over up to 20 steps.

State (4, H, W) float32:
    ch 0 : preprocessed image                       (static within an episode)
    ch 1 : current binary mask for the target class (dynamic)
    ch 2 : signed distance transform of ch 1        (dynamic, derived from ch 1)
    ch 3 : U-Net init mask                          (static within an episode)

Reward modes (set per-class via YAML):
    dice_delta          r_t = Dice(t) - Dice(t-1)            — general purpose
    dice_hd_composite   α·ΔDice + β·ΔHD95_norm               — boundary precision
    iou_delta           r_t = IoU(t) - IoU(t-1)              — small targets

Continuous action (DDPG family) — 3-component vector:
    a[0]  morph   ∈ [-cont_morph_scale, +cont_morph_scale]
          Threshold shift on the normalised SDT (range [-1, +1]).
          +morph → lower threshold → include exterior pixels → dilate
          -morph → raise  threshold → exclude near-boundary pixels → erode
          Gives the DDPG agent local morphological power, unlike the old
          translation-only 2-component design.
    a[1]  dy_norm ∈ [-cont_trans_scale, +cont_trans_scale]
          Fractional y-translation (scaled by H at apply time).
    a[2]  dx_norm ∈ [-cont_trans_scale, +cont_trans_scale]
          Fractional x-translation (scaled by W at apply time).

Discrete action (DQN family) — 7 actions:
    0 dilate · 1 erode · 2 shift↑ · 3 shift↓ · 4 shift← · 5 shift→ · 6 no-op

Episode: max_steps steps, OR composite stop
    (|ΔDice| < stop_eps_dice AND |ΔHD95| < stop_eps_hd) for stop_n consecutive steps.
"""

from typing import Tuple, Dict
import numpy as np
import scipy.ndimage as ndi


STRUCT = ndi.generate_binary_structure(2, 1)   # 4-connectivity


def dice_score(m1: np.ndarray, m2: np.ndarray, eps: float = 1e-6) -> float:
    m1 = m1.astype(bool); m2 = m2.astype(bool)
    inter = (m1 & m2).sum()
    return (2.0 * inter + eps) / (m1.sum() + m2.sum() + eps)


def hd95_px(m1: np.ndarray, m2: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in pixels."""
    if not m1.any() and not m2.any():
        return 0.0
    if not m1.any() or not m2.any():
        return float('nan')
    edges_1 = m1 ^ ndi.binary_erosion(m1, STRUCT)
    edges_2 = m2 ^ ndi.binary_erosion(m2, STRUCT)
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
    """Per-class binary boundary-refinement environment."""

    NUM_DISCRETE_ACTIONS  = 7
    CONTINUOUS_ACTION_DIM = 3   # (morph, dy_norm, dx_norm) — see module docstring

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

    def _compute_reward(self, prev_dice, new_dice, prev_hd, new_hd):
        """Dataset-tuned reward shaping. Returns RAW reward (will be clipped)."""
        if self.reward_mode == 'dice_delta':
            # General-purpose: pure Dice improvement. CAMUS, HAM10000, Kvasir.
            return new_dice - prev_dice

        if self.reward_mode == 'dice_hd_composite':
            # Boundary-precision: weights both Dice gain AND HD95 improvement.
            # Suited for thin / topologically-complex structures (DRIVE vessels,
            # CAMUS LV_epi at the apex). HD improvement is +ve when HD95 drops.
            r_dice = new_dice - prev_dice
            if (np.isnan(prev_hd) or np.isnan(new_hd) or
                np.isinf(prev_hd) or np.isinf(new_hd)):
                r_hd = 0.0
            else:
                r_hd = float(np.clip((prev_hd - new_hd) / self.hd_norm, -1.0, 1.0))
            return self.reward_alpha * r_dice + self.reward_beta * r_hd

        if self.reward_mode == 'iou_delta':
            # IoU is less sensitive to area than Dice — better signal for small
            # targets where Dice fluctuates wildly (BRISC tumours <2% area).
            # IoU = Dice / (2 − Dice)
            prev_iou = prev_dice / max(2.0 - prev_dice, 1e-6)
            new_iou  = new_dice  / max(2.0 - new_dice,  1e-6)
            return new_iou - prev_iou

        raise ValueError(f'Unknown reward_mode: {self.reward_mode}')

    # ── GT precomputation (called once per episode in reset()) ───────────────

    def _precompute_gt_quantities(self) -> None:
        """Cache GT edge mask and GT EDT — GT is constant for the whole episode.

        Saves 2 scipy calls (binary_erosion + distance_transform_edt) on every
        single env.step() invocation.  For dice_hd_composite mode that trims
        each step from 4 scipy calls down to 2; for dice_delta it eliminates
        HD95 entirely (0 scipy calls for the GT side).
        """
        gt_bool = self.gt.astype(bool)
        gt_edge = gt_bool ^ ndi.binary_erosion(gt_bool, STRUCT)
        self._gt_edge     = gt_edge
        self._gt_edge_any = bool(gt_edge.any())
        # EDT on the complement of the edge — distance from any non-edge pixel
        # to the nearest GT boundary pixel.  Used in d_12 (pred→GT direction).
        self._gt_edt      = ndi.distance_transform_edt(~gt_edge)

    def _hd95_vs_gt(self, m1: np.ndarray) -> float:
        """HD95 between mask m1 and GT, reusing precomputed GT quantities.

        Only 2 scipy calls (pred erosion + pred EDT) instead of the 4 that the
        standalone hd95_px() function uses.  Must call _precompute_gt_quantities()
        first (done automatically in reset()).
        """
        m1b = m1.astype(bool)
        if not m1b.any() and not self._gt_edge_any:
            return 0.0
        if not m1b.any() or not self._gt_edge_any:
            return float('nan')
        edge_1 = m1b ^ ndi.binary_erosion(m1b, STRUCT)
        if not edge_1.any():
            return 0.0
        dt_1  = ndi.distance_transform_edt(~edge_1)
        d_12  = self._gt_edt[edge_1]        # pred boundary → nearest GT boundary
        d_21  = dt_1[self._gt_edge]         # GT boundary → nearest pred boundary
        return float(np.percentile(np.concatenate([d_12, d_21]), 95))

    def reset(self) -> np.ndarray:
        self.mask = self.init_mask.copy()
        self.t    = 0
        self._precompute_gt_quantities()    # 2 scipy calls, once per episode
        self.dice_history = [dice_score(self.mask, self.gt)]
        # HD95 only tracked when the reward mode actually needs it
        if self.reward_mode != 'dice_delta':
            self.hd95_history = [self._hd95_vs_gt(self.mask)]
        else:
            self.hd95_history = [float('nan')]
        return self._state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        prev_dice = self.dice_history[-1]
        prev_hd95 = self.hd95_history[-1]

        if self.action_type == 'discrete':
            self.mask = self._apply_discrete(int(action))
        else:
            self.mask = self._apply_continuous(action)

        self.t += 1
        new_dice = dice_score(self.mask, self.gt)

        # HD95 is expensive — skip entirely when reward_mode doesn't use it.
        # dice_delta and iou_delta rewards don't need HD95, so we save 2 scipy
        # calls per step (pred erosion + pred EDT).
        if self.reward_mode == 'dice_hd_composite':
            new_hd95 = self._hd95_vs_gt(self.mask)   # 2 scipy calls (GT side cached)
        else:
            new_hd95 = float('nan')                   # skip — not needed

        # Dataset-tuned reward (mode set in cfg), clipped BEFORE buffer push
        raw_reward = self._compute_reward(prev_dice, new_dice, prev_hd95, new_hd95)
        reward = float(np.clip(raw_reward, -self.reward_clip, self.reward_clip))

        self.dice_history.append(new_dice)
        self.hd95_history.append(new_hd95)

        done = self.t >= self.max_steps
        if not done and len(self.dice_history) > self.stop_n:
            dd = [abs(self.dice_history[-i-1] - self.dice_history[-i-2])
                  for i in range(self.stop_n)]
            dice_stopped = all(d < self.stop_eps_dice for d in dd)

            if self.reward_mode == 'dice_hd_composite':
                # Both Dice and HD95 must have converged
                dh = [abs(self.hd95_history[-i-1] - self.hd95_history[-i-2])
                      for i in range(self.stop_n)]
                hd_stopped = all(
                    (np.isnan(h) or h < self.stop_eps_hd) for h in dh
                )
                done = dice_stopped and hd_stopped
            else:
                # dice_delta / iou_delta: Dice-only convergence check
                done = dice_stopped

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
        sp = self.shift_px
        if a == 0: return ndi.binary_dilation(self.mask, STRUCT).astype(np.uint8)
        if a == 1: return ndi.binary_erosion (self.mask, STRUCT).astype(np.uint8)
        if a == 2: return shifted(self.mask, -sp,   0)
        if a == 3: return shifted(self.mask, +sp,   0)
        if a == 4: return shifted(self.mask,   0, -sp)
        if a == 5: return shifted(self.mask,   0, +sp)
        if a == 6: return self.mask
        raise ValueError(f'Bad discrete action: {a}')

    def _apply_continuous(self, a) -> np.ndarray:
        """
        3-component continuous action — morphological op then translation.

        a[0]  morph   ∈ [-cont_morph_scale, +cont_morph_scale]
              Threshold on the normalised SDT (values in [-1, +1]):
                thresh = -morph
                new_mask = (SDT >= thresh)
              +morph (positive) → thresh < 0 → includes exterior pixels → DILATE
              -morph (negative) → thresh > 0 → excludes boundary pixels → ERODE
               morph = 0        → thresh = 0 → SDT >= 0 ≡ original mask (identity)
              At sdt_clip=20, morph=0.25 corresponds to ±5 px boundary shift.

        a[1]  dy_norm  ∈ [-cont_trans_scale, +cont_trans_scale]
              Fractional y-translation: dy_px = round(a[1] × H).
        a[2]  dx_norm  ∈ [-cont_trans_scale, +cont_trans_scale]
              Fractional x-translation: dx_px = round(a[2] × W).

        This gives DDPG agents meaningful morphological power — dilate/erode the
        boundary — which the old translation-only 2-component design lacked.
        The discrete DQN family still uses explicit dilate/erode/shift primitives.
        """
        morph = float(a[0])
        dy    = int(round(float(a[1]) * self.H))
        dx    = int(round(float(a[2]) * self.W))

        # ── Morphological step via SDT threshold ─────────────────────────────
        # SDT in env.py is normalised to [-1, +1] (by sdt_clip).
        # Interior pixels: sdt > 0.  Exterior pixels: sdt < 0.
        # At morph = 0: thresh = 0 → (sdt >= 0) ≡ the original mask exactly.
        sdt    = signed_dt(self.mask, self.sdt_clip)   # (H, W) ∈ [-1, +1]
        thresh = -morph                                 # invert so +morph → dilate
        morphed = (sdt >= thresh).astype(np.uint8)

        # Guard: catastrophic erosion can empty the mask; keep original in that case.
        if not morphed.any():
            morphed = self.mask.copy()

        # ── Translation step ─────────────────────────────────────────────────
        return shifted(morphed, dy, dx)

    def _state(self) -> np.ndarray:
        sdt = signed_dt(self.mask, self.sdt_clip)
        return np.stack([
            self.image,
            self.mask.astype(np.float32),
            sdt,
            self.init_mask.astype(np.float32),
        ], axis=0)
