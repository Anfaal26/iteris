"""
Run this as a single cell in a Kaggle notebook (with CAMUS, BRISC, and
iteris-pkg datasets attached) to export a handful of (image, mask) pairs as
flat PNGs, ready to download and upload through the live website to test
real Dice/IoU/HD metrics.

CAMUS ships as NIfTI (.nii/.nii.gz) — not browser-uploadable as-is — so this
converts each chosen slice to PNG. BRISC is already JPG/PNG and is just copied.
Output: /kaggle/working/sample_pairs/{camus,brisc}/*.png, zipped at the end.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# iteris/__init__.py imports monai/nibabel/etc — install before the iteris import below.
init_files = list(Path('/kaggle/input').rglob('iteris/__init__.py'))
if not init_files:
    raise RuntimeError('iteris-pkg not attached.')
PKG_ROOT = init_files[0].parent.parent
REQ = PKG_ROOT / 'requirements.txt'
subprocess.run(['pip', 'install', '-r', str(REQ), '--quiet'], check=True)
sys.path.insert(0, str(PKG_ROOT))

import nibabel as nib
import numpy as np
from PIL import Image

from iteris.ingestion import build_camus_dicts, build_brisc_dicts

OUT_DIR = Path('/kaggle/working/sample_pairs')
N_PAIRS = 3  # how many (image, mask) pairs to export per dataset

# ── CAMUS: convert chosen NIfTI slices to PNG ──────────────────────────────
camus_dirs = [p for p in Path('/kaggle/input').rglob('CAMUS') if p.is_dir()]
if camus_dirs:
    camus_out = OUT_DIR / 'camus'
    camus_out.mkdir(parents=True, exist_ok=True)
    records = build_camus_dicts(str(camus_dirs[0]))
    for rec in records[:N_PAIRS]:
        stem = f"{rec['patient']}_{rec['view']}_{rec['phase']}"

        img_arr = nib.load(rec['image']).get_fdata()
        img_arr = np.squeeze(img_arr)  # drop any singleton 3rd dim
        img_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-6)
        Image.fromarray((img_norm * 255).astype(np.uint8)).save(camus_out / f'{stem}_image.png')

        lbl_arr = nib.load(rec['label']).get_fdata()
        lbl_arr = np.squeeze(lbl_arr).astype(np.uint8)  # raw class indices 0-3, NOT scaled
        Image.fromarray(lbl_arr).save(camus_out / f'{stem}_mask.png')

        print(f'CAMUS  {stem}: image {img_arr.shape} -> PNG, mask classes {sorted(set(lbl_arr.flatten().tolist()))}')
else:
    print('CAMUS dataset not found under /kaggle/input — attach it and re-run.')

# ── BRISC: already JPG/PNG, just copy ──────────────────────────────────────
brisc_dirs = [p for p in Path('/kaggle/input').rglob('brisc2025') if p.is_dir()]
if brisc_dirs:
    brisc_out = OUT_DIR / 'brisc'
    brisc_out.mkdir(parents=True, exist_ok=True)
    records = build_brisc_dicts(str(brisc_dirs[0]))
    # prefer a few different tumor types if available, not 3x the same one
    seen_types = set()
    chosen = []
    for rec in records:
        t = rec.get('tumor_type', 'unknown')
        if t not in seen_types:
            seen_types.add(t)
            chosen.append(rec)
        if len(chosen) >= N_PAIRS:
            break
    for rec in chosen:
        stem = f"{rec['patient']}_{rec.get('tumor_type', 'unknown')}"
        shutil.copy(rec['image'], brisc_out / f'{stem}_image{Path(rec["image"]).suffix}')
        shutil.copy(rec['label'], brisc_out / f'{stem}_mask{Path(rec["label"]).suffix}')
        print(f'BRISC  {stem}: copied')
else:
    print('BRISC dataset not found under /kaggle/input — attach it and re-run.')

# ── Zip everything for one-click download from the notebook's Output tab ──
zip_path = '/kaggle/working/sample_pairs.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUT_DIR.rglob('*'):
        if f.is_file():
            zf.write(f, f.relative_to(OUT_DIR))
print(f'\nDone. Download {zip_path} from the notebook Output panel (right sidebar).')
