"""
Segmentation metrics — vendored from iteris/metrics.py (pure torch, no
scipy/MONAI) plus an IoU helper. Used only when the caller supplies a ground
truth mask; see inference.build_metrics.
"""

import torch
import torch.nn.functional as F

_HD95_KERNELS: dict = {}


def _get_hd95_kernel(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _HD95_KERNELS:
        _HD95_KERNELS[key] = torch.ones(1, 1, 3, 3, device=device)
    return _HD95_KERNELS[key]


@torch.no_grad()
def dice_iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """pred, target: (H, W) integer class indices. Returns (dice, iou) each (num_classes-1,) skipping background."""
    pred_oh = F.one_hot(pred.long(), num_classes).permute(2, 0, 1).float()
    target_oh = F.one_hot(target.long(), num_classes).permute(2, 0, 1).float()

    inter = (pred_oh[1:] * target_oh[1:]).sum(dim=(1, 2))
    union = pred_oh[1:].sum(dim=(1, 2)) + target_oh[1:].sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice, iou


@torch.no_grad()
def hd95_single(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')

    kernel = _get_hd95_kernel(pred_bin.device)
    p = pred_bin.float().unsqueeze(0).unsqueeze(0)
    g = gt_bin.float().unsqueeze(0).unsqueeze(0)

    p_eroded = (F.conv2d(p, kernel, padding=1) >= 9).float()
    g_eroded = (F.conv2d(g, kernel, padding=1) >= 9).float()
    edges_p = (p - p_eroded).squeeze()
    edges_g = (g - g_eroded).squeeze()

    if edges_p.sum() == 0 or edges_g.sum() == 0:
        return 0.0

    yp, xp = torch.nonzero(edges_p, as_tuple=True)
    yg, xg = torch.nonzero(edges_g, as_tuple=True)
    cp = torch.stack([yp.float(), xp.float()], dim=1)
    cg = torch.stack([yg.float(), xg.float()], dim=1)

    d = torch.cdist(cp, cg)
    all_d = torch.cat([d.min(dim=1).values, d.min(dim=0).values])
    return float(torch.quantile(all_d, 0.95))


@torch.no_grad()
def hd_single(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
    """Plain (max, not 95th-percentile) Hausdorff distance, same edge-extraction as hd95_single."""
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')

    kernel = _get_hd95_kernel(pred_bin.device)
    p = pred_bin.float().unsqueeze(0).unsqueeze(0)
    g = gt_bin.float().unsqueeze(0).unsqueeze(0)

    p_eroded = (F.conv2d(p, kernel, padding=1) >= 9).float()
    g_eroded = (F.conv2d(g, kernel, padding=1) >= 9).float()
    edges_p = (p - p_eroded).squeeze()
    edges_g = (g - g_eroded).squeeze()

    if edges_p.sum() == 0 or edges_g.sum() == 0:
        return 0.0

    yp, xp = torch.nonzero(edges_p, as_tuple=True)
    yg, xg = torch.nonzero(edges_g, as_tuple=True)
    cp = torch.stack([yp.float(), xp.float()], dim=1)
    cg = torch.stack([yg.float(), xg.float()], dim=1)

    d = torch.cdist(cp, cg)
    all_d = torch.cat([d.min(dim=1).values, d.min(dim=0).values])
    return float(all_d.max())
