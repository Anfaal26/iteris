"""
Iteris — DRL-based medical image segmentation.

Top-level imports for convenience. Notebooks should import directly from
sub-modules where possible (clearer dependency graph).
"""

__version__ = "0.2.0"

from .config import load_config
from .models import AttentionResUNet
from .losses import build_loss
from .metrics import dice_score, hd95_batch
