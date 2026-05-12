"""
Segmentation metrics — pure PyTorch.

We deliberately avoid `scipy.ndimage` and MONAI's `HausdorffDistanceMetric`
because Kaggle's bundled scipy breaks to import under numpy 2.x. Everything
here is implemented with torch ops only and runs on whichever device the
tensors live on.
"""

from typing import Optional
import torch
import torch.nn.functional as F


# ─── Dice ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-class Dice over a batch.

    Parameters
    ----------
    pred, target : (B, H, W) or (B, 1, H, W) integer class indices.
    Returns      : (B, C-1) tensor if `include_background=False`, else (B, C).
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_oh   = F.one_hot(pred.long(),   num_classes).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    start = 0 if include_background else 1
    inter = (pred_oh[:, start:] * target_oh[:, start:]).sum(dim=(2, 3))
    union = pred_oh[:, start:].sum(dim=(2, 3)) + target_oh[:, start:].sum(dim=(2, 3))
    return (2 * inter + eps) / (union + eps)


# ─── HD95 ────────────────────────────────────────────────────────────────────

# Cache the 3×3 ones kernel per device to avoid re-allocation every call.
_HD95_KERNELS: dict = {}


def _get_hd95_kernel(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _HD95_KERNELS:
        _HD95_KERNELS[key] = torch.ones(1, 1, 3, 3, device=device)
    return _HD95_KERNELS[key]


@torch.no_grad()
def hd95_single(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
    """
    HD95 in pixels for one (H, W) binary-mask pair. Pure torch.

    Returns
    -------
    0.0      if both masks are empty (perfect match by convention)
    NaN      if exactly one mask is empty (HD undefined)
    quantile otherwise
    """
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')

    kernel = _get_hd95_kernel(pred_bin.device)
    p = pred_bin.float().unsqueeze(0).unsqueeze(0)
    g = gt_bin.float().unsqueeze(0).unsqueeze(0)

    # Binary erosion via 3×3 sum-conv — a pixel survives iff all 9 neighbours are 1.
    p_eroded = (F.conv2d(p, kernel, padding=1) >= 9).float()
    g_eroded = (F.conv2d(g, kernel, padding=1) >= 9).float()
    edges_p  = (p - p_eroded).squeeze()
    edges_g  = (g - g_eroded).squeeze()

    if edges_p.sum() == 0 or edges_g.sum() == 0:
        return 0.0

    yp, xp = torch.nonzero(edges_p, as_tuple=True)
    yg, xg = torch.nonzero(edges_g, as_tuple=True)
    cp = torch.stack([yp.float(), xp.float()], dim=1)
    cg = torch.stack([yg.float(), xg.float()], dim=1)

    d     = torch.cdist(cp, cg)
    all_d = torch.cat([d.min(dim=1).values, d.min(dim=0).values])
    return float(torch.quantile(all_d, 0.95))


@torch.no_grad()
def hd95_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> torch.Tensor:
    """
    Per-class HD95 over a batch.

    Parameters
    ----------
    pred, target : (B, H, W) or (B, 1, H, W) integer class indices.
    Returns      : (B, C-1) tensor (or (B, C) if include_background=True).
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    B = pred.shape[0]
    start = 0 if include_background else 1
    out_C = num_classes - start
    out   = torch.full((B, out_C), float('nan'), device=pred.device)

    for b in range(B):
        for ci, c in enumerate(range(start, num_classes)):
            p_bin = (pred[b]   == c)
            g_bin = (target[b] == c)
            out[b, ci] = hd95_single(p_bin, g_bin)
    return out
