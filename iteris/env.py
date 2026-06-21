"""
Back-compat shim.

Geometry/metric helpers now live in iteris.geometry; the global-morphology
SegmentationEnv (Paradigm A) was archived to iteris.archive_paradigm_a (it is
structurally incapable of beating a strong baseline — kept for ablation only).
Both are re-exported here so existing `from .env import ...` keeps working.
New code should import geometry from iteris.geometry and the live env from
iteris.env_contour_refine.
"""
from .geometry import (  # noqa: F401  (re-export)
    STRUCT, SE_N, SE_S, SE_W, SE_E, SE_NE, SE_NW, SE_SE, SE_SW,
    dice_score, _largest_cc, hd95_px, signed_dt, shifted,
)
from .archive_paradigm_a.segmentation_env import SegmentationEnv  # noqa: F401
