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
    """Filename suffix that disambiguates architectures sharing a dataset.

    '' for the default attention Res-UNet (back-compat: camus_best.pt,
    camus_summary.json, camus_pred_masks/ …); '_lite_unet' for the lite baseline,
    so the two never overwrite each other's checkpoints / scores / masks / summary.
    """
    model = cfg.get('model', 'attn_resunet')
    return '' if model == 'attn_resunet' else f'_{model}'
