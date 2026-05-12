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
    raise ValueError(f"Unknown dataset: {cfg['dataset']}")
