"""
MONAI transform pipelines.

Modality-aware: `cfg['normalize']` switches between min-max (ultrasound),
z-score (MRI), and HU windowing (CT).
"""

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, Resized,
    ScaleIntensityd, NormalizeIntensityd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandShiftIntensityd,
    RandAffined, RandGaussianNoised,
)


def build_intensity_transform(cfg: dict):
    """Return the modality-specific intensity normalisation transform."""
    mode = cfg['normalize']
    if mode == 'minmax':
        # Ultrasound, dermoscopy, fundus — scale [img_min, img_max] → [0, 1]
        return ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0)
    if mode == 'zscore':
        # MRI — zero-mean, unit-variance over non-zero voxels
        return NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)
    if mode == 'hu':
        # CT — clip to HU window, then linearly map to [0, 1]
        a_min, a_max = cfg['hu_window']
        return ScaleIntensityRanged(
            keys=['image'], a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    raise ValueError(f"Unknown normalize mode: {mode}")


def build_transforms(cfg: dict, split: str = 'train'):
    """
    Build the full MONAI transform pipeline for a given split.

    Train pipeline = base + augmentation.
    Val / test pipeline = base only (deterministic).
    """
    spacing = cfg.get('spacing', (1.0, 1.0))

    # Orientationd is only meaningful for 3D volumes (BraTS, full ACDC).
    # For 2D datasets (CAMUS / CHAOS / DRIVE / ISIC) it spams warnings and
    # does no useful work. Enable in YAML with `apply_orientation: true`.
    base = [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
    ]
    if cfg.get('apply_orientation', False):
        base.append(Orientationd(
            keys=['image', 'label'],
            axcodes=cfg.get('orientation_axcodes', 'RAS'),
        ))
    base += [
        Spacingd(
            keys=['image', 'label'],
            pixdim=(*spacing, -1),
            mode=('bilinear', 'nearest'),
        ),
        Resized(
            keys=['image', 'label'],
            spatial_size=cfg['image_size'],
            mode=('bilinear', 'nearest'),
        ),
        build_intensity_transform(cfg),
        EnsureTyped(keys=['image', 'label'], dtype=(torch.float32, torch.long)),
    ]

    if split != 'train':
        return Compose(base)

    aug = [
        RandFlipd(
            keys=['image', 'label'],
            prob=cfg.get('aug_flip_prob', 0.5),
            spatial_axis=1,
        ),
        RandRotate90d(keys=['image', 'label'], prob=0.3, max_k=3),
        RandAffined(
            keys=['image', 'label'],
            prob=cfg.get('aug_affine_prob', 0.6),
            rotate_range=(cfg.get('aug_rotate_range', 0.26),),
            scale_range=(cfg.get('aug_scale_range', 0.1),),
            translate_range=(cfg.get('aug_translate_range', 10),),
            mode=('bilinear', 'nearest'),
            padding_mode='border',
        ),
        RandShiftIntensityd(
            keys=['image'],
            offsets=cfg.get('aug_intensity_shift', 0.15),
            prob=cfg.get('aug_intensity_shift_prob', 0.4),
        ),
        RandGaussianNoised(
            keys=['image'],
            prob=cfg.get('aug_gauss_noise_prob', 0.3),
            std=cfg.get('aug_gauss_noise_std', 0.02),
        ),
    ]
    return Compose(base + aug)
