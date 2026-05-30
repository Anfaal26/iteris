"""
ARCHIVED — BRISC small-target refinement environment (DQN/DuelingDDQN family).

Superseded by the boundary-tracing paradigm (Paradigm 1). All BRISC discrete
agents now use ContourTracingEnv. This module is kept here as a historical
reference for the discrete refinement comparison; it is NOT imported by the
package or registered in ENV_REGISTRY.

To resurrect:
  - Move this file back to iteris/env_brisc.py
  - Re-add the import + ENV_REGISTRY['brisc_small_target'] entry to
    iteris/drl_training.py
  - Re-add the discrete refinement YAML blocks under configs/brisc_drl_tumor.yaml

Original location: iteris/env.py (SegmentationEnvBRISC class, lines 399–522)
Last active commit before archival: see git log for this file.
"""

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..env import SegmentationEnv, SE_N, SE_E, SE_S, SE_W


class SegmentationEnvBRISC(SegmentationEnv):
    """
    BRISC small-target environment for the DQN family.

    Why a separate class:
      The base 13-action env was designed around CAMUS-scale structures
      (~100 px linear extent at 256²).  A 1-px directional op there is ~1%
      of structure area — fine granularity.  On BRISC tumours (~30 px linear,
      ~1.7% image area), the same 1-px op is ~3% per step, and over a 20-step
      episode this compounds to ~60% worst-case Dice loss.  Empirically the
      base env on BRISC produces ΔDice ≈ -0.05 to -0.10 — the agent learns
      to grind through ~12 small bad moves per episode.

    Three structural changes specifically for small targets:

      1. ACTION SPACE CUT TO 9 (no whole-mask shifts):
         BRISC tumours are correctly *located* by the U-Net (mean init
         Dice 0.84) — the residual error is shape, not position.  Shifting a
         correctly-located tumour by 1 px is almost always net-negative
         because most of the boundary is already aligned.  Dropping the 4
         shift actions removes a tempting bad-action class and shrinks the
         exploration burden by ~30%.

      2. MAX_STEPS=5 (was 20):
         Each step is ~3% of tumour area in magnitude.  Capping at 5 steps
         caps worst-case damage at ~15% Dice instead of ~60%, and the
         agent now has 5 chances rather than 20 to compound bad moves.

      3. FAIL-FAST TERMINATION:
         If Dice has been below init by ≥ fail_thresh for fail_n consecutive
         steps, end the episode immediately.
    """

    NUM_DISCRETE_ACTIONS = 9
    DILATE_N, DILATE_E, DILATE_S, DILATE_W = 0, 1, 2, 3
    ERODE_N,  ERODE_E,  ERODE_S,  ERODE_W  = 4, 5, 6, 7
    NOOP                                    = 8

    DISCRETE_NAMES = [
        'dil-N', 'dil-E', 'dil-S', 'dil-W',
        'ero-N', 'ero-E', 'ero-S', 'ero-W',
        'no-op',
    ]

    def __init__(
        self,
        *args,
        fail_thresh: float = 0.05,
        fail_n:      int   = 2,
        **kwargs,
    ):
        kwargs.setdefault('max_steps',     5)
        kwargs.setdefault('stop_n',        2)
        kwargs.setdefault('stop_eps_dice', 0.005)
        super().__init__(*args, **kwargs)
        self.fail_thresh = float(fail_thresh)
        self.fail_n      = int(fail_n)

    def _apply_discrete(self, a: int) -> np.ndarray:
        if a == 0: return ndi.binary_dilation(self.mask, SE_N).astype(np.uint8)
        if a == 1: return ndi.binary_dilation(self.mask, SE_E).astype(np.uint8)
        if a == 2: return ndi.binary_dilation(self.mask, SE_S).astype(np.uint8)
        if a == 3: return ndi.binary_dilation(self.mask, SE_W).astype(np.uint8)
        if a == 4: return ndi.binary_erosion(self.mask, SE_N).astype(np.uint8)
        if a == 5: return ndi.binary_erosion(self.mask, SE_E).astype(np.uint8)
        if a == 6: return ndi.binary_erosion(self.mask, SE_S).astype(np.uint8)
        if a == 7: return ndi.binary_erosion(self.mask, SE_W).astype(np.uint8)
        if a == 8: return self.mask
        raise ValueError(f'Bad BRISC discrete action: {a}')

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = super().step(action)
        if done:
            return state, reward, done, info
        if len(self.dice_history) > self.fail_n:
            dice_0 = self.dice_history[0]
            recent_bad = all(
                self.dice_history[-k - 1] < dice_0 - self.fail_thresh
                for k in range(self.fail_n)
            )
            if recent_bad:
                done = True
                info['terminated_early'] = True
        return state, reward, done, info
