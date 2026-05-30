"""
Iteris — DRL-based medical image segmentation.

Top-level imports for convenience. Notebooks should import directly from
sub-modules where possible (clearer dependency graph).
"""

__version__ = "0.4.0"

# Baseline (Week 1) exports
from .config import load_config, load_drl_config
from .models import AttentionResUNet
from .losses import build_loss
from .metrics import dice_score, hd95_batch

# DRL (Week 2+) exports
from .env          import SegmentationEnv
from .env_contour  import ContourTracingEnv, VectorisedContourEnv
from .buffer       import ReplayBuffer, ContourReplayBuffer
from .agents       import DQNAgent, DuelingDQNAgent, DDPGAgent
