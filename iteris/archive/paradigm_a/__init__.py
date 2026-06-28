"""
Paradigm A — GLOBAL morphological mask refinement (ARCHIVED, ablation only).

SegmentationEnv's whole-mask dilate/erode/shift action space cannot make
spatially-heterogeneous corrections, so it is structurally incapable of
beating a strong baseline (confirmed on CAMUS/BRISC: it degrades the mask).
The live refinement paradigm is the contour env (iteris.env_contour_refine).
Kept here so the failed global-morphology runs remain reproducible as an
ablation / negative control. Import explicitly:

    from iteris.archive.paradigm_a import SegmentationEnv
"""
from .segmentation_env import SegmentationEnv
