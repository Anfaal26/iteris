"""
Boundary-refinement segmentation environment for DRL agents.

One agent operates on one target class at a time (per-class binary formulation).
The episode starts from the U-Net baseline mask (warm-start) and refines toward
GT over up to 20 steps.

────────────────────────────────────────────────────────────────────────────────
Action spaces
────────────────────────────────────────────────────────────────────────────────

DISCRETE (DQN family) — 24 actions:
    0–3    Cardinal dilate / 4–7 Cardinal erode (directional SEs)
    8–11   Diagonal dilate / 12–15 Diagonal erode (corner SEs)
    16–19  Whole-mask shift
    20     Smooth (morphological closing)
    21     Add uncertain pixels / 22 Remove uncertain pixels (U-Net prob 0.35–0.65)
    23     Stop (explicit terminal)

CONTINUOUS (DDPG family) — 3-component action:
    a[0]  morph    ∈ [-cont_morph_scale, +cont_morph_scale]   SDT threshold shift
    a[1]  dy_norm  ∈ [-cont_trans_scale, +cont_trans_scale]   global y-translation
    a[2]  dx_norm  ∈ [-cont_trans_scale, +cont_trans_scale]   global x-translation
    Global isotropic SDT-threshold morph plus a small global translation.

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

# ── Directional structuring elements ─────────────────────────────────────────
# Cardinal (3-element): 1-px move in one direction, affects only boundary facing that way.
SE_N = np.array([[1], [1], [0]], dtype=np.uint8)
SE_S = np.array([[0], [1], [1]], dtype=np.uint8)
SE_W = np.array([[1, 1, 0]],    dtype=np.uint8)
SE_E = np.array([[0, 1, 1]],    dtype=np.uint8)

# Diagonal (2×2 corner): 1-px move along a diagonal, expands/contracts the corner.
# Crucial for irregular BRISC tumors — cardinal ops alone can't correct diagonal errors.
SE_NE = np.array([[0, 1], [1, 1]], dtype=np.uint8)  # northeast corner
SE_NW = np.array([[1, 0], [1, 1]], dtype=np.uint8)  # northwest corner
SE_SE = np.array([[1, 1], [0, 1]], dtype=np.uint8)  # southeast corner
SE_SW = np.array([[1, 1], [1, 0]], dtype=np.uint8)  # southwest corner


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
    """Per-class binary boundary-refinement environment (v4 action space).

    Action space (24 discrete actions):
    ────────────────────────────────────────────────────────────────
    0–3    Cardinal dilate  (N, E, S, W)   — expand boundary in one direction
    4–7    Cardinal erode   (N, E, S, W)   — shrink boundary from one direction
    8–11   Diagonal dilate  (NE, NW, SE, SW) — expand corner region
    12–15  Diagonal erode   (NE, NW, SE, SW) — shrink corner region
    16–19  Whole-mask shift (↑, ↓, ←, →)  — correct positional offset
    20     Smooth            — morphological closing, fills holes, rounds jagged edges
    21     Add uncertain     — include pixels where U-Net was uncertain (prob 0.35–0.65)
                               that are not yet in the current mask
    22     Remove uncertain  — remove pixels where U-Net was uncertain (prob 0.35–0.65)
                               that are still in the current mask
    23     Stop              — agent signals "satisfied"; episode terminates immediately
    ────────────────────────────────────────────────────────────────
    """

    # ── Discrete action layout (22 actions) ──────────────────────────────────
    # 0–3:   Cardinal dilate       (N, E, S, W)
    # 4–7:   Cardinal erode        (N, E, S, W)
    # 8–11:  Diagonal dilate       (NE, NW, SE, SW) — corner correction
    # 12–15: Diagonal erode        (NE, NW, SE, SW) — corner correction
    # 16–19: Whole-mask shift      (↑, ↓, ←, →)
    # 20:    Smooth                (morphological closing)
    # 21:    Add uncertain pixels  — include U-Net uncertain-borderline pixels
    # 22:    Remove uncertain pixels — exclude U-Net uncertain-borderline pixels
    # 23:    Stop
    NUM_DISCRETE_ACTIONS  = 24
    DILATE_N, DILATE_E, DILATE_S, DILATE_W            = 0, 1, 2, 3
    ERODE_N,  ERODE_E,  ERODE_S,  ERODE_W             = 4, 5, 6, 7
    DILATE_NE, DILATE_NW, DILATE_SE, DILATE_SW        = 8, 9, 10, 11
    ERODE_NE,  ERODE_NW,  ERODE_SE,  ERODE_SW         = 12, 13, 14, 15
    SHIFT_U,  SHIFT_D,  SHIFT_L,  SHIFT_R             = 16, 17, 18, 19
    SMOOTH                                             = 20
    ADD_UNCERTAIN                                      = 21
    REMOVE_UNCERTAIN                                   = 22
    STOP                                               = 23

    DISCRETE_NAMES = [
        'dil-N',  'dil-E',  'dil-S',  'dil-W',
        'ero-N',  'ero-E',  'ero-S',  'ero-W',
        'dil-NE', 'dil-NW', 'dil-SE', 'dil-SW',
        'ero-NE', 'ero-NW', 'ero-SE', 'ero-SW',
        'sh-↑', 'sh-↓', 'sh-←', 'sh-→',
        'smooth', 'add-unc', 'rem-unc', 'stop',
    ]

    # ── Continuous action layout (see module docstring) ──────────────────────
    CONTINUOUS_ACTION_DIM = 3     # (morph, dy_norm, dx_norm)

    def __init__(
        self,
        image: np.ndarray,         # (H, W) float32 in [0, 1]
        gt_mask: np.ndarray,       # (H, W) uint8 in {0, 1}
        init_mask: np.ndarray,     # (H, W) uint8 in {0, 1} — U-Net prediction
        prob_map: np.ndarray = None, # (H, W) float — U-Net probability for target class;
                                     # enables ADD_UNCERTAIN / REMOVE_UNCERTAIN actions.
                                     # None → those actions fall back to no-op.
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
        # ── fail-fast (small-target datasets) ───────────────────────────────
        fail_thresh: float = 0.0,         # >0 enables fail-fast: terminate if Dice
                                          # drops ≥ fail_thresh below episode-start
                                          # Dice for fail_n consecutive steps.
                                          # 0.0 = disabled (default).
        fail_n: int = 2,                  # consecutive bad steps before termination
        # ── STOP-incentive shaping (critical for near-optimal baselines) ─────
        reward_step_penalty: float = 0.0, # subtracted from every NON-stop step's
                                          # reward. Breaks the baseline-indifference
                                          # trap: without it, wandering at baseline
                                          # (r=0/step) ties STOP (r=0), so the agent
                                          # never learns to stop and noisily degrades.
                                          # With it, STOP-at-baseline strictly
                                          # dominates idle wandering.
        disable_auto_stop: bool = False,  # True → episode ends ONLY on STOP action,
                                          # max_steps, or fail-fast. Removes the
                                          # convergence/improvement heuristic that
                                          # otherwise terminates at degraded states
                                          # before the agent can choose STOP.
        terminal_bonus_scale: float = 0.0,
                                          # >0 adds `scale * (Dice_final - Dice_0)`
                                          # to the reward of the LAST transition only
                                          # (whatever ends the episode: STOP, max_steps,
                                          # or fail-fast). Default 0.0 = fully backward
                                          # compatible / off.
                                          #
                                          # WHY: with r_t = Dice_t − Dice_0 (or the
                                          # composite/IoU equivalents) summed under a
                                          # discount factor γ<1, an agent that spikes to
                                          # a good Dice early and degrades later still
                                          # nets a *positive* return — γ-weighting makes
                                          # early high-Dice steps worth more than the
                                          # later low-Dice steps that erase the gain.
                                          # That "spike-then-fade" path is rewarded even
                                          # though the FINAL mask (what we actually keep
                                          # at inference time) is worse than baseline —
                                          # which is exactly the failure mode visible in
                                          # training logs (best-seen ≫ final). A reward
                                          # that depends only on the terminal state is
                                          # path-independent and removes that exploit;
                                          # this term makes "reach the best state and
                                          # STOP there" dominate "wander and hope".
    ):
        assert action_type in ('discrete', 'continuous')
        assert reward_mode in ('dice_delta', 'dice_hd_composite', 'iou_delta')
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape
        # Precompute uncertainty zone masks from the U-Net probability map.
        # uncertain_hi: pixels the U-Net considered "maybe foreground" (0.35–0.65)
        # that aren't in the binary init_mask — candidates to ADD.
        # uncertain_lo: pixels in init_mask where U-Net was uncertain (0.35–0.65)
        # — candidates to REMOVE.
        if prob_map is not None:
            pm = prob_map.astype(np.float32)
            self._uncertain_hi = ((pm >= 0.35) & (pm <= 0.65) & (init_mask == 0)).astype(np.uint8)
            self._uncertain_lo = ((pm >= 0.35) & (pm <= 0.65) & (init_mask == 1)).astype(np.uint8)
        else:
            self._uncertain_hi = None
            self._uncertain_lo = None

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
        self.fail_thresh         = float(fail_thresh)
        self.fail_n              = int(fail_n)
        self.reward_step_penalty = float(reward_step_penalty)
        self.disable_auto_stop   = bool(disable_auto_stop)
        self.terminal_bonus_scale = float(terminal_bonus_scale)

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
        d0 = dice_score(self.mask, self.gt)
        self.dice_history = [d0]
        # Best-mask tracking: highest-Dice mask seen this episode. Eval can report
        # this as the achievable ceiling; the agent's job is to STOP at it.
        self.best_dice = d0
        self.best_mask = self.mask.copy()
        if self.reward_mode != 'dice_delta':
            self.hd95_history = [self._hd95_vs_gt(self.mask)]
        else:
            self.hd95_history = [float('nan')]
        return self._state()

    def get_best_mask(self) -> np.ndarray:
        """Highest-Dice mask seen during the episode (for diagnostics)."""
        return self.best_mask

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        a = int(action) if self.action_type == 'discrete' else action

        # ── Explicit stop action (action 13 for discrete) ────────────────────
        # The agent signals "satisfied". Episode terminates with the same reward
        # as hitting max_steps — encourages stopping as soon as the mask is good.
        if self.action_type == 'discrete' and a == self.STOP:   # a == 23
            self.t += 1
            new_dice = dice_score(self.mask, self.gt)
            new_hd95 = self._hd95_vs_gt(self.mask) \
                       if self.reward_mode == 'dice_hd_composite' else float('nan')
            dice_0 = self.dice_history[0]
            # STOP reward = current improvement, NO step penalty (it's the exit).
            raw_reward = self._compute_reward(new_dice, new_hd95)
            # Terminal bonus: STOP always ends the episode, so the terminal-state
            # bonus (see __init__ docstring on terminal_bonus_scale) always applies
            # here — it directly rewards stopping AT a final-Dice gain over baseline.
            if self.terminal_bonus_scale > 0.0:
                raw_reward += self.terminal_bonus_scale * (new_dice - dice_0)
            clip = self.reward_clip + self.terminal_bonus_scale
            reward = float(np.clip(raw_reward, -clip, clip))
            info = {'dice': new_dice, 'hd95': new_hd95, 'delta_dice': raw_reward,
                    'reward_clipped': reward != raw_reward, 'step': self.t,
                    'stop_action': True, 'best_dice': self.best_dice}
            return self._state(), reward, True, info

        # ── Apply action ─────────────────────────────────────────────────────
        if self.action_type == 'discrete':
            self.mask = self._apply_discrete(a)
        else:
            self.mask = self._apply_continuous(a)

        # Mask-empty guard: catastrophic erosion (especially on small targets)
        # can wipe the mask. Revert to init_mask so the episode can recover.
        if not self.mask.any():
            self.mask = self.init_mask.copy()

        self.t += 1
        new_dice = dice_score(self.mask, self.gt)

        if self.reward_mode == 'dice_hd_composite':
            new_hd95 = self._hd95_vs_gt(self.mask)
        else:
            new_hd95 = float('nan')

        self.dice_history.append(new_dice)
        self.hd95_history.append(new_hd95)

        # Track best mask seen this episode.
        if new_dice > self.best_dice:
            self.best_dice = new_dice
            self.best_mask = self.mask.copy()

        # ── Termination conditions ────────────────────────────────────────────
        done = self.t >= self.max_steps

        # Auto-stop heuristic (convergence / improvement-maintained). Disabled
        # when disable_auto_stop=True so the agent's explicit STOP action — not a
        # heuristic — decides when to end. The heuristic could otherwise terminate
        # the episode at a degraded state before the agent chooses to stop.
        if not done and not self.disable_auto_stop and len(self.dice_history) > self.stop_n:
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

        # Fail-fast (optional, primarily for small-target datasets like BRISC):
        # terminate if Dice has been ≥ fail_thresh below init for fail_n steps.
        if (not done and self.fail_thresh > 0.0
                and len(self.dice_history) > self.fail_n):
            dice_0 = self.dice_history[0]
            recent_bad = all(
                self.dice_history[-k-1] < dice_0 - self.fail_thresh
                for k in range(self.fail_n)
            )
            if recent_bad:
                done = True

        # Per-step penalty: subtracted from every non-STOP step so that idle
        # wandering at baseline costs reward and STOP becomes the optimal exit.
        # Computed AFTER `done` is known so the terminal bonus (if enabled) can
        # be attached to whichever transition actually ends the episode — not
        # just explicit STOP. This matters because most episodes currently end
        # via max_steps / fail-fast rather than STOP (see best-seen ≫ final
        # diagnostic), and those paths need the same path-independent signal.
        raw_reward = self._compute_reward(new_dice, new_hd95) - self.reward_step_penalty
        clip = self.reward_clip
        if done and self.terminal_bonus_scale > 0.0:
            dice_0 = self.dice_history[0]
            raw_reward += self.terminal_bonus_scale * (new_dice - dice_0)
            clip = self.reward_clip + self.terminal_bonus_scale
        reward = float(np.clip(raw_reward, -clip, clip))

        info = {
            'dice': new_dice,
            'hd95': new_hd95,
            'delta_dice': raw_reward,
            'reward_clipped': reward != raw_reward,
            'step': self.t,
            'best_dice': self.best_dice,
        }
        return self._state(), reward, done, info

    # ── private ──────────────────────────────────────────────────────────────

    def _apply_discrete(self, a: int) -> np.ndarray:
        """24-action discrete dispatcher — see class docstring for full layout."""
        sp = self.shift_px
        # 0–3: cardinal dilate
        if a == 0:  return ndi.binary_dilation(self.mask, SE_N).astype(np.uint8)
        if a == 1:  return ndi.binary_dilation(self.mask, SE_E).astype(np.uint8)
        if a == 2:  return ndi.binary_dilation(self.mask, SE_S).astype(np.uint8)
        if a == 3:  return ndi.binary_dilation(self.mask, SE_W).astype(np.uint8)
        # 4–7: cardinal erode
        if a == 4:  return ndi.binary_erosion(self.mask, SE_N).astype(np.uint8)
        if a == 5:  return ndi.binary_erosion(self.mask, SE_E).astype(np.uint8)
        if a == 6:  return ndi.binary_erosion(self.mask, SE_S).astype(np.uint8)
        if a == 7:  return ndi.binary_erosion(self.mask, SE_W).astype(np.uint8)
        # 8–11: diagonal dilate — corrects corner under-segmentation
        if a == 8:  return ndi.binary_dilation(self.mask, SE_NE).astype(np.uint8)
        if a == 9:  return ndi.binary_dilation(self.mask, SE_NW).astype(np.uint8)
        if a == 10: return ndi.binary_dilation(self.mask, SE_SE).astype(np.uint8)
        if a == 11: return ndi.binary_dilation(self.mask, SE_SW).astype(np.uint8)
        # 12–15: diagonal erode — corrects corner over-segmentation
        if a == 12: return ndi.binary_erosion(self.mask, SE_NE).astype(np.uint8)
        if a == 13: return ndi.binary_erosion(self.mask, SE_NW).astype(np.uint8)
        if a == 14: return ndi.binary_erosion(self.mask, SE_SE).astype(np.uint8)
        if a == 15: return ndi.binary_erosion(self.mask, SE_SW).astype(np.uint8)
        # 16–19: whole-mask shift
        if a == 16: return shifted(self.mask, -sp,   0)
        if a == 17: return shifted(self.mask, +sp,   0)
        if a == 18: return shifted(self.mask,   0, -sp)
        if a == 19: return shifted(self.mask,   0, +sp)
        # 20: smooth
        if a == 20:
            disk = ndi.generate_binary_structure(2, 1)
            smoothed = ndi.binary_closing(
                self.mask.astype(bool), structure=disk, iterations=1
            ).astype(np.uint8)
            return smoothed if smoothed.any() else self.mask
        # 21: add uncertain — add uncertain pixels not yet in current mask
        # Bug fix: use self.mask (not init_mask) to determine what's missing,
        # so repeated calls don't try to re-add already-added pixels.
        if a == 21:
            if self._uncertain_hi is not None:
                to_add = self._uncertain_hi & (1 - self.mask)   # only pixels not in current mask
                if to_add.any():
                    return np.clip(self.mask.astype(np.int16) + to_add, 0, 1).astype(np.uint8)
            return self.mask   # fallback: no prob_map or nothing to add
        # 22: remove uncertain — remove uncertain pixels still in current mask
        if a == 22:
            if self._uncertain_lo is not None:
                to_remove = self._uncertain_lo & self.mask       # only pixels in current mask
                if to_remove.any():
                    result = np.clip(self.mask.astype(np.int16) - to_remove, 0, 1).astype(np.uint8)
                    return result if result.any() else self.mask
            return self.mask   # fallback: no prob_map or nothing to remove
        # 23: stop — handled in step()
        if a == 23: return self.mask
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
