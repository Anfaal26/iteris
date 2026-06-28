"""Paradigm A global-morphology env (archived). See package __init__."""
from typing import Tuple, Dict
import numpy as np
import scipy.ndimage as ndi

from ...geometry import (STRUCT, SE_N, SE_S, SE_W, SE_E, SE_NE, SE_NW, SE_SE,
                        SE_SW, dice_score, _largest_cc, hd95_px, signed_dt, shifted)


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
    NUM_STATE_CHANNELS    = 5     # image, mask, sdt, init_mask, prob_map
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
        pbrs_gamma: float = 0.99,         # discount used inside potential-based shaping
                                          # (dice_pbrs / dice_hd_pbrs). MUST equal the
                                          # agent's γ for the telescoping identity
                                          # Σγ^t r_t = γ^T Φ_T − Φ_0 to hold exactly.
        reward_potential_scale: float = 1.0,
                                          # Multiplies the (baseline-centred) PBRS
                                          # potential Φ. CRITICAL at near-optimal
                                          # baselines (CAMUS ~0.94, BRISC ~0.88):
                                          #
                                          # The potential is Φ(s)=K·(Dice(s)−Dice_0)
                                          # — centred so Φ_0=0. Centring removes the
                                          # discount DRAG that an *un-centred* Φ=Dice
                                          # suffers: with Φ≈0.94 and γ=0.99, even
                                          # holding Dice constant pays (γ−1)·Φ≈−0.009
                                          # /step, and a real +0.005 Dice gain still
                                          # nets γΦ'−Φ<0 — so every realistic refine
                                          # step looks negative and the only non-
                                          # negative action is STOP-at-baseline. The
                                          # agent then provably collapses to "do
                                          # nothing", never beating baseline. Centring
                                          # makes holding-at-baseline reward exactly 0
                                          # (== STOP) and any genuine gain strictly
                                          # positive. K (≈10–20) then lifts the tiny
                                          # Dice deltas (±0.003–0.01) out of the
                                          # function-approximation noise floor.
                                          # Telescopes to γ^T·K·(Dice_T−Dice_0): same
                                          # "maximise final improvement, soonest"
                                          # objective, just properly scaled.
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
        assert reward_mode in ('dice_delta', 'dice_hd_composite', 'iou_delta',
                               'dice_pbrs', 'dice_hd_pbrs')
        assert image.shape == gt_mask.shape == init_mask.shape
        self.image     = image.astype(np.float32)
        self.gt        = gt_mask.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.H, self.W = image.shape
        # U-Net probability map for the target class — exposed as a STATE channel
        # so the policy can see *where the U-Net was unsure* (the single most
        # informative cue for where refinement is worthwhile). Falls back to the
        # binary init mask when no prob_map is supplied (older warm_start runs).
        self.prob_map = (prob_map.astype(np.float32) if prob_map is not None
                         else init_mask.astype(np.float32))
        # Modes whose reward is potential-based (need new vs. previous potential).
        self._is_pbrs = reward_mode in ('dice_pbrs', 'dice_hd_pbrs')
        # HD95 must be tracked whenever the potential / composite depends on it.
        self._needs_hd = reward_mode in ('dice_hd_composite', 'dice_hd_pbrs')
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
        self.pbrs_gamma          = float(pbrs_gamma)
        self.reward_potential_scale = float(reward_potential_scale)

        self.reset()

    def _potential(self, dice: float, hd: float) -> float:
        """State potential Φ(s) for potential-based reward shaping.

        BASELINE-CENTRED and SCALED (K = reward_potential_scale):

        dice_pbrs    : Φ = K·(Dice − Dice_0)
        dice_hd_pbrs : Φ = K·[α·(Dice − Dice_0) + β·(hd_term − hd_term_0)]
                       with hd_term = 1 − clip(HD95/hd_norm, 0, 1).

        Centring at the episode baseline (Dice_0, hd_term_0 — both fixed at
        reset()) makes Φ_0 = 0. This is essential at near-optimal baselines:
        an UN-centred Φ = Dice ≈ 0.94 makes the discount term (γ−1)·Φ ≈ −0.009
        dominate the genuine per-step Dice deltas (±0.003–0.01), so every
        realistic refine step scores negative and the agent collapses to STOP
        (Φ-invariance still holds, but the learnable signal is pure drag).
        Centred + scaled, holding at baseline pays exactly 0 (== STOP), real
        gains are strictly positive, and K lifts them above the FA noise floor.
        Telescopes to γ^T·Φ_T = γ^T·K·(Dice_T − Dice_0): identical objective.
        """
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
        if self._needs_hd:
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
            new_hd95 = self._hd95_vs_gt(self.mask) if self._needs_hd else float('nan')
            dice_0 = self.dice_history[0]
            if self._is_pbrs:
                # STOP is a pure "commit" action: it does not change the mask, so
                # the value of the current state was ALREADY credited by whichever
                # transition reached it. Crediting it again (γΦ−Φ) would penalise
                # committing at a good state. Reward 0 makes Q(STOP)=0 a clean
                # decision boundary: stop iff continuing has negative expected
                # value (dawdle cost + risk of degrading) — which is exactly the
                # desired "stop once no further gain is reachable" behaviour.
                raw_reward = 0.0
                clip = self.reward_clip
            else:
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

        if self._needs_hd:
            new_hd95 = self._hd95_vs_gt(self.mask)
        else:
            new_hd95 = float('nan')

        # Capture the PREVIOUS-state potential BEFORE appending the new state —
        # potential-based shaping needs Φ(s_t) and Φ(s_{t+1}).
        if self._is_pbrs:
            phi_prev = self._potential(self.dice_history[-1], self.hd95_history[-1])

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
        if self._is_pbrs:
            # Potential-based shaping: r = γ·Φ(s_{t+1}) − Φ(s_t) − step_penalty.
            # Summed over the episode this telescopes to γ^T·Φ_T − Φ_0, i.e. the
            # discounted FINAL-state improvement — the true deployment objective,
            # with dense per-action credit and no path-dependence. terminal_bonus
            # is intentionally NOT applied (it is redundant and would re-introduce
            # path-dependence).
            phi_new = self._potential(new_dice, new_hd95)
            raw_reward = (self.pbrs_gamma * phi_new - phi_prev) - self.reward_step_penalty
            clip = self.reward_clip
        else:
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
            self.prob_map,                      # U-Net confidence — where to refine
        ], axis=0)
