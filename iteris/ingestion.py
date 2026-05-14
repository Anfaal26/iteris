"""
Dataset file-list builders.

Each `build_*_dicts()` function walks a dataset's directory layout and returns
a list of `{image, label, patient, ...metadata}` dicts ready for MONAI
transforms. Adding a new dataset = one function here + one branch in
`build_dataset_dicts(cfg)`.
"""

from pathlib import Path
from typing import List


# Extensions tried in order — first match wins. `.nii.gz` first so it
# isn't shadowed by `.nii` as a substring.
NIFTI_EXTS = ['.nii.gz', '.nii', '.mhd']


def _has_patients(root: Path) -> bool:
    """True if any child directory of `root` is named like patientXXXX."""
    return any(
        c.is_dir() and c.name.lower().startswith('patient')
        for c in root.iterdir()
    )


def build_camus_dicts(data_root, views=('2CH', '4CH'), phases=('ED', 'ES')) -> List[dict]:
    """
    Walk CAMUS dataset and return a list of file pairs.

    Auto-descends one level if `data_root` contains a single wrapper folder
    (common in Kaggle Dataset uploads).

    Returns
    -------
    list of dicts with keys: image, label, patient, view, phase
    """
    root = Path(data_root)
    if not _has_patients(root):
        for sub in root.iterdir():
            if sub.is_dir() and _has_patients(sub):
                root = sub
                break
    print(f'[ingestion] Walking CAMUS at: {root}')

    records, missing = [], []
    for patient_dir in sorted(root.iterdir()):
        if not (patient_dir.is_dir() and patient_dir.name.lower().startswith('patient')):
            continue
        pid = patient_dir.name
        for view in views:
            for phase in phases:
                img = lbl = None
                for ext in NIFTI_EXTS:
                    cand_img = patient_dir / f'{pid}_{view}_{phase}{ext}'
                    cand_lbl = patient_dir / f'{pid}_{view}_{phase}_gt{ext}'
                    if cand_img.exists() and cand_lbl.exists():
                        img, lbl = cand_img, cand_lbl
                        break
                if img is not None:
                    records.append(dict(
                        image=str(img), label=str(lbl),
                        patient=pid, view=view, phase=phase,
                    ))
                else:
                    missing.append(f'{pid}_{view}_{phase}')

    if not records:
        raise RuntimeError(
            f'No CAMUS file pairs found under {root}. '
            f'Check the dataset path and folder layout.'
        )
    if missing:
        print(f'[ingestion] Skipped {len(missing)} missing combos '
              f'(first 3: {missing[:3]})')
    print(f'[ingestion] CAMUS: {len(records)} samples')
    return records


def build_chaos_ct_dicts(data_root) -> List[dict]:
    """
    Walk CHAOS CT dataset and return DICOM-slice + PNG-mask pairs.

    Folder layout assumed:
        data_root/CT/<patient_id>/DICOM_anon/*.dcm
        data_root/CT/<patient_id>/Ground/*.png
    """
    root = Path(data_root) / 'CT'
    records = []
    for patient_dir in sorted(root.iterdir()):
        dcm_dir = patient_dir / 'DICOM_anon'
        gt_dir  = patient_dir / 'Ground'
        if not (dcm_dir.exists() and gt_dir.exists()):
            continue
        dcm_files = sorted(dcm_dir.glob('*.dcm'))
        gt_files  = sorted(gt_dir.glob('*.png'))
        for dcm_f, gt_f in zip(dcm_files, gt_files):
            records.append(dict(
                image=str(dcm_f), label=str(gt_f),
                patient=patient_dir.name,
            ))
    print(f'[ingestion] CHAOS CT: {len(records)} slices')
    return records


def build_brisc_dicts(data_root) -> List[dict]:
    """
    Walk BRISC 2025 (brain tumour MRI) dataset.

    BRISC layout — parallel `images/` + `masks/` folders:
        <root>/[train|test]/images/<name>.jpg
        <root>/[train|test]/masks/<name>.png
    Files in `images/` and `masks/` share the same filename stem.

    Patient identity is extracted from the filename so multiple sequences/views
    of the same case stay in the same split:
        `brisc2025_train_00001_gl_ax_t1.jpg`  →  patient `train_00001`
    (split prefix kept so train/test 00001 don't collide).

    Note: BRISC's original train/test split is not honoured here — we pool all
    samples and apply our own patient-level split (consistent across all
    datasets in the iteris project).

    Falls back to filename-suffix pairing (`case42_mask.png`) if parallel folders
    aren't found, for compatibility with other BRISC mirrors.
    """
    import re

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f'BRISC data_root not found: {root}')

    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    PATIENT_RE = re.compile(r'brisc2025_(train|test)_(\d+)')
    records = []

    # ── Strategy 1: parallel images/ + masks/ folders (BRISC's actual layout) ─
    image_dirs = [p for p in root.rglob('images') if p.is_dir()]
    for img_dir in image_dirs:
        mask_dir = img_dir.parent / 'masks'
        if not mask_dir.is_dir():
            continue
        mask_index = {}
        for ext in exts:
            for m in mask_dir.glob(f'*{ext}'):
                mask_index[m.stem.lower()] = m
        for ext in exts:
            for img in img_dir.glob(f'*{ext}'):
                mask = mask_index.get(img.stem.lower())
                if mask is None:
                    continue
                m = PATIENT_RE.match(img.stem.lower())
                patient_id = f'{m.group(1)}_{m.group(2)}' if m else img.stem
                records.append(dict(
                    image=str(img), label=str(mask),
                    patient=patient_id,
                ))

    if records:
        n_patients = len({r['patient'] for r in records})
        print(f'[ingestion] BRISC: {len(records)} pairs across {n_patients} cases')
        return records

    # ── Strategy 2: filename-suffix pairing (fallback for other mirrors) ──────
    mask_kws  = ('mask', 'seg', 'segment', 'label', 'gt', 'annotation')
    all_files = [p for ext in exts for p in root.rglob(f'*{ext}')]
    masks     = [f for f in all_files if any(k in f.stem.lower() for k in mask_kws)]
    images    = [f for f in all_files if f not in set(masks)]
    mask_by_stem = {m.stem.lower(): m for m in masks}

    for img in images:
        s = img.stem.lower()
        matched = None
        for kw in mask_kws:
            for sep in ('_', '-', '.'):
                key = f'{s}{sep}{kw}'
                if key in mask_by_stem:
                    matched = mask_by_stem[key]
                    break
            if matched:
                break
        if matched:
            records.append(dict(
                image=str(img), label=str(matched),
                patient=img.stem,
            ))

    if not records:
        raise RuntimeError(f'No BRISC (image, mask) pairs found under {root}')
    print(f'[ingestion] BRISC: {len(records)} pairs (filename-suffix layout)')
    return records


def build_ham10000_dicts(image_roots, mask_root) -> List[dict]:
    """
    Pair HAM10000 dermoscopy images (kmader's dataset) with lesion
    segmentation masks (tschandl's dataset). Matched by ISIC ID.

    Args
    ----
    image_roots : list of directories with `ISIC_*.jpg` images.
                  Typically `HAM10000_images_part_1` and `_part_2`.
    mask_root   : directory with `ISIC_*_segmentation.png` masks.

    Patient identity = ISIC image ID. HAM10000 metadata also has lesion_id
    that can group multiple images per lesion, but we don't use it here —
    each image is segmented independently. Acceptable since per-image
    splits stay disjoint at the image level.
    """
    if isinstance(image_roots, (str, Path)):
        image_roots = [image_roots]
    image_roots = [Path(r) for r in image_roots]
    mask_root   = Path(mask_root)

    if not mask_root.exists():
        raise FileNotFoundError(f'HAM10000 mask_root not found: {mask_root}')

    mask_files = list(mask_root.glob('*_segmentation.png'))
    img_index  = {}
    for root in image_roots:
        if root.exists():
            for p in root.glob('ISIC_*.jpg'):
                img_index[p.stem] = p

    records = []
    for m in mask_files:
        key = m.stem.replace('_segmentation', '')
        if key in img_index:
            records.append(dict(
                image=str(img_index[key]),
                label=str(m),
                patient=key,   # ISIC ID
            ))

    if not records:
        raise RuntimeError(
            f'No HAM10000 pairs matched. Masks: {len(mask_files)}, '
            f'image index: {len(img_index)}. Check that both kmader/skin-cancer-mnist-ham10000 '
            f'and tschandl/ham10000-lesion-segmentations datasets are attached.'
        )
    print(f'[ingestion] HAM10000: {len(records)} pairs')
    return records


def build_dataset_dicts(cfg: dict) -> List[dict]:
    """Dispatch on cfg['dataset'] to the right builder."""
    name = cfg['dataset'].upper()
    if name == 'CAMUS':
        return build_camus_dicts(
            cfg['data_root'],
            views=tuple(cfg.get('views',  ('2CH', '4CH'))),
            phases=tuple(cfg.get('phases', ('ED', 'ES'))),
        )
    if name == 'CHAOS':
        return build_chaos_ct_dicts(cfg['data_root'])
    if name == 'BRISC':
        return build_brisc_dicts(cfg['data_root'])
    if name == 'HAM10000':
        return build_ham10000_dicts(
            image_roots=cfg['image_roots'],
            mask_root=cfg['mask_root'],
        )
    raise ValueError(f"Unknown dataset: {cfg['dataset']}")
