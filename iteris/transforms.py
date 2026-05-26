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
    # `loader_reader` lets datasets override the default ITKReader — DRIVE's
    # .gif vessel masks need PILReader (ITKReader treats unknowns as DICOM).
    #
    # DicomReader.verify_suffix() returns True for any file when pydicom is
    # installed, and it is registered last (= highest priority). To beat it we
    # subclass PILReader to also claim .gif and register it AFTER DicomReader.
    _loader_reader = cfg.get('loader_reader', None)
    if _loader_reader == 'PILReader':
        # `monai.utils.is_supported_format` is unstable across MONAI versions
        # (sometimes lives in monai.data.utils, sometimes monai.utils.misc).
        # Use a self-contained suffix check instead.
        from monai.data import PILReader as _PILReader

        _SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp',
                     '.tif', '.tiff', '.gif', '.npy', '.npz')

        class _GIFPILReader(_PILReader):
            @staticmethod
            def verify_suffix(filename):
                if isinstance(filename, (list, tuple)):
                    return all(_GIFPILReader.verify_suffix(f) for f in filename)
                return str(filename).lower().endswith(_SUFFIXES)

        _loader_reader = _GIFPILReader()
    elif isinstance(_loader_reader, type) or hasattr(_loader_reader, 'verify_suffix'):
        # Pre-instantiated reader passed in — use as-is.
        pass

    base = [
        LoadImaged(keys=['image', 'label'], reader=_loader_reader),
        EnsureChannelFirstd(keys=['image', 'label']),
    ]
    if cfg.get('apply_orientation', False):
        base.append(Orientationd(
            keys=['image', 'label'],
            axcodes=cfg.get('orientation_axcodes', 'RAS'),
        ))

    # Force consistent channel count on the image. Some datasets (e.g. BRISC)
    # have mixed grey/RGB JPGs which DataLoader can't batch together.
    target_c = cfg.get('in_channels', 1)
    from monai.transforms import Lambdad as _Lambdad

    def _to_target_channels(x, target=target_c):
        c = int(x.shape[0])
        if c == target:
            return x
        if target == 1 and c >= 1:
            # Average across channels (RGB → luma-style grayscale)
            if hasattr(x, 'mean'):
                return x.mean(dim=0, keepdim=True)
            import numpy as np
            return np.mean(x, axis=0, keepdims=True)
        if target == 3 and c == 1:
            # Replicate single channel to 3
            if hasattr(x, 'repeat'):
                return x.repeat(3, 1, 1)
            import numpy as np
            return np.repeat(x, 3, axis=0)
        if target == 3 and c == 4:
            # Drop alpha channel
            return x[:3]
        return x  # unknown — leave alone

    base.append(_Lambdad(keys=['image'], func=_to_target_channels))

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
    ]

    # Binarise labels for datasets stored as 0/255 PNGs (BRISC, HAM10000,
    # Kvasir, DRIVE).  Multi-class datasets like CAMUS leave this off.
    #
    # Threshold at midpoint (127), NOT zero.  BRISC labels were shipped with
    # JPEG-compression artifacts: pixel values 1-7 scattered through the
    # "background" region (decay 291→176→95→31→13 px) alongside the real
    # tumour values 249-251+, with a clean gap between them.  ``x > 0`` would
    # catch every compression-noise pixel as foreground, producing ~30 false
    # components per image (96.4% of CCs are ≤9 px speckle); ``x > 127``
    # cleanly separates the two clusters.  On the cleanly-binary datasets
    # (HAM10000, Kvasir, DRIVE) it's equivalent to ``x > 0``.
    if cfg.get('binarize_labels', False):
        from monai.transforms import Lambdad
        base.append(Lambdad(keys=['label'], func=lambda x: x > 127))

    base += [
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
