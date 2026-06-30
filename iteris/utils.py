"""Misc helpers."""

import random
import numpy as np
import torch
from monai.utils import set_determinism


def seed_all(seed: int):
    """Seed Python, NumPy, PyTorch, and MONAI in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_suffix(cfg: dict) -> str:
    """Filename suffix that disambiguates artifacts sharing a dataset.

    Two independent, back-compatible components (both empty by default):

    1. Architecture: '' for the default attention Res-UNet (back-compat:
       camus_best.pt, camus_summary.json, camus_pred_masks/ …); '_lite_unet'
       for the lite baseline — so the two never overwrite each other.
    2. Data regime (Phase B/C low-data ablations): '' when label_frac >= 1.0
       (full data — Phase A, names unchanged), else '_lf<pct>' (e.g. '_lf10'
       for label_frac=0.10). Namespaces low-data checkpoints/summaries/masks so
       a Phase-B/C run never overwrites the Phase-A artifacts. Both
       training.py's checkpoint writer AND every DRL notebook's baseline-ckpt
       auto-detect derive the name through THIS function, so setting label_frac
       in the baseline config is the ONLY change needed to run a phase — the
       tag then propagates to every artifact name automatically.
    """
    model = cfg.get('model', 'attn_resunet')
    arch = '' if model == 'attn_resunet' else f'_{model}'
    lf = float(cfg.get('label_frac', 1.0))
    regime = '' if lf >= 1.0 else f'_lf{int(round(lf * 100)):02d}'
    return arch + regime
