"""
Loads the trained Attention Res-U-Net checkpoints (pushed to HF Hub from the
Kaggle output dataset) and runs inference + mask/metric building for the
FastAPI endpoints in main.py.

Preprocessing mirrors iteris/transforms.py / configs/{camus,brisc}.yaml:
greyscale, resize to 256x256, and dataset-specific intensity normalisation
(min-max for ultrasound, z-score over nonzero pixels for MRI). No MONAI
dependency — this Space stays lean.
"""

import base64
import io
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from .metrics import dice_iou_per_class, hd95_single, hd_single
from .models import AttentionResUNet
from .schemas import MaskLayer, Metrics, StructureMetrics

IMAGE_SIZE = 256

# Structure ids/labels/colors per dataset, in class-index order (index 1 = first
# entry, matching configs/*.yaml class_names[1:]). Colors mirror
# iteris_ui/src/tokens/index.ts maskColorsHex so overlays match the rest of the UI.
STRUCTURE_DEFS = {
    'camus': [
        ('lv_endo', 'LV Endocardium', '#00c9a7'),
        ('lv_epi', 'LV Epicardium', '#f59e0b'),
        ('la', 'Left Atrium', '#f87171'),
    ],
    # BRISC checkpoint is binary (background/tumor) — no tumor-type classifier
    # exists yet, so the single foreground channel is reported as 'glioma'
    # (first enum value) with an explicit "unclassified" label. Replace once a
    # tumor-type model or per-type checkpoints exist.
    'brisc': [
        ('glioma', 'Tumor (unclassified)', '#818cf8'),
    ],
}

DATASET_CFG = {
    'camus': dict(
        repo=os.environ.get('HF_REPO_CAMUS', 'anfaal26/iteris-attention-unet-camus'),
        filename=os.environ.get('HF_FILE_CAMUS', 'camus_best.pt'),
        in_channels=1,
        num_classes=4,
        normalize='minmax',
    ),
    'brisc': dict(
        repo=os.environ.get('HF_REPO_BRISC', 'anfaal26/iteris-attention-unet-brisc'),
        filename=os.environ.get('HF_FILE_BRISC', 'brisc_best.pt'),
        in_channels=1,
        num_classes=2,
        normalize='zscore',
    ),
}

_MODEL_CACHE: dict = {}


def get_model(dataset: str) -> AttentionResUNet:
    if dataset in _MODEL_CACHE:
        return _MODEL_CACHE[dataset]
    cfg = DATASET_CFG[dataset]
    path = hf_hub_download(repo_id=cfg['repo'], filename=cfg['filename'])
    model = AttentionResUNet(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[dataset] = model
    return model


def preload_models() -> None:
    """Called on startup so the first real request isn't slowed by a cold load."""
    for dataset in DATASET_CFG:
        try:
            get_model(dataset)
        except Exception as exc:  # noqa: BLE001 — log and keep serving /health
            print(f'[startup] failed to load {dataset} checkpoint: {exc}')


def _decode_b64_image(b64: str) -> Image.Image:
    if b64.startswith('data:'):
        b64 = b64.split(',', 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


def _to_normalized_array(img: Image.Image, normalize: str) -> np.ndarray:
    gray = img.convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if normalize == 'minmax':
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn + 1e-6)
    else:  # zscore, over nonzero pixels (mirrors NormalizeIntensityd(nonzero=True))
        nz = arr[arr != 0]
        if nz.size > 0:
            mean, std = float(nz.mean()), float(nz.std())
        else:
            mean, std = float(arr.mean()), float(arr.std())
        arr = (arr - mean) / (std + 1e-6)
    return arr


def run_inference(dataset: str, image_b64: str) -> torch.Tensor:
    """Returns the (256, 256) integer class-index prediction map."""
    model = get_model(dataset)
    cfg = DATASET_CFG[dataset]
    img = _decode_b64_image(image_b64)
    arr = _to_normalized_array(img, cfg['normalize'])
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).squeeze(0)
    return pred


def _mask_to_png_b64(binary_mask: np.ndarray, color_hex: str) -> str:
    color_hex = color_hex.lstrip('#')
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    h, w = binary_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = (binary_mask * 200).astype(np.uint8)
    out = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def build_masks(dataset: str, pred: torch.Tensor) -> list:
    pred_np = pred.cpu().numpy()
    masks = []
    for i, (structure, label, color) in enumerate(STRUCTURE_DEFS[dataset], start=1):
        binary = (pred_np == i).astype(np.uint8)
        masks.append(MaskLayer(
            structure=structure,
            label=label,
            color=color,
            imageB64=_mask_to_png_b64(binary, color),
        ))
    return masks


def _empty_metrics(dataset: str) -> Metrics:
    """No ground truth supplied — metrics are unavailable, return zeros.
    UI callers should treat dice == 0 with no gtMaskB64 sent as 'not computed'.
    """
    structures = [
        StructureMetrics(structure=s, label=label, dice=0.0, iou=0.0, hd=0.0, hd95=0.0)
        for s, label, _ in STRUCTURE_DEFS[dataset]
    ]
    return Metrics(dice=0.0, iou=0.0, hd=0.0, hd95=0.0, structures=structures, baselineDice=0.0)


def build_metrics(dataset: str, pred: torch.Tensor, gt_mask_b64: str | None) -> Metrics:
    cfg = DATASET_CFG[dataset]
    if not gt_mask_b64:
        return _empty_metrics(dataset)

    gt_img = _decode_b64_image(gt_mask_b64).convert('L').resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST,
    )
    gt_arr = np.asarray(gt_img)
    # Ground-truth masks are typically saved as 0/255 (binary) or as discrete
    # class indices already; if max > num_classes-1 treat it as 0/255 and
    # threshold at the midpoint, same convention as transforms.py.
    if gt_arr.max() > cfg['num_classes'] - 1:
        gt_arr = (gt_arr > 127).astype(np.int64)
        if cfg['num_classes'] > 2:
            # Binary GT can't disambiguate multi-class structures; fold all
            # foreground into class 1 so dice/iou still compute, but flag via
            # NaN-safe handling below for the remaining classes.
            gt_arr = np.clip(gt_arr, 0, cfg['num_classes'] - 1)
    gt = torch.from_numpy(gt_arr).long()

    dice, iou = dice_iou_per_class(pred, gt, cfg['num_classes'])

    structures = []
    for i, (structure, label, _) in enumerate(STRUCTURE_DEFS[dataset]):
        pred_bin = pred == (i + 1)
        gt_bin = gt == (i + 1)
        hd95 = hd95_single(pred_bin, gt_bin)
        hd = hd_single(pred_bin, gt_bin)
        structures.append(StructureMetrics(
            structure=structure,
            label=label,
            dice=round(float(dice[i]), 4),
            iou=round(float(iou[i]), 4),
            hd=0.0 if hd != hd else round(hd, 2),  # NaN guard
            hd95=0.0 if hd95 != hd95 else round(hd95, 2),
        ))

    mean_dice = round(float(dice.mean()), 4)
    mean_iou = round(float(iou.mean()), 4)
    mean_hd = round(float(np.mean([s.hd for s in structures])), 2)
    mean_hd95 = round(float(np.mean([s.hd95 for s in structures])), 2)

    return Metrics(
        dice=mean_dice, iou=mean_iou, hd=mean_hd, hd95=mean_hd95,
        structures=structures, baselineDice=mean_dice,
    )
