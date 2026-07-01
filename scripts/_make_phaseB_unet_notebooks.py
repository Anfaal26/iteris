"""One-off script: generate the Phase B U-Net notebook copies.
Not part of the package; run once from repo root, then delete or keep as a
reference for the eventual Phase C script. See docs/EXPERIMENTS.md.
"""
import json


def load(p):
    return json.load(open(p, encoding='utf-8'))


def save(nb, p):
    json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    json.load(open(p, encoding='utf-8'))


def replace_once(cell, old, new):
    joined = ''.join(cell['source'])
    n = joined.count(old)
    assert n == 1, f'expected 1 occurrence, got {n}: {old[:70]!r}'
    joined = joined.replace(old, new)
    cell['source'] = joined.splitlines(keepends=True)


def prepend_to_first_cell(nb, text):
    cell = nb['cells'][0]
    joined = ''.join(cell['source'])
    joined = joined.rstrip('\n') + '\n' + text
    cell['source'] = joined.splitlines(keepends=True)


EPOCH_LINE = 'print(f\'Epochs       : {cfg["epochs"]}  batch {cfg["batch_size"]}  lr {cfg["lr"]}\')\n'
EPOCH_LINE_NEW = (
    EPOCH_LINE +
    'print(f\'label_frac   : {cfg["label_frac"]}  (Phase B = low-data)\')\n'
)

# ============================================================================
# 1) UNet CAMUS LITE
# ============================================================================
nb = load('notebooks/unet/01_camus_lite.ipynb')
OLD = "cfg = load_config(str(PKG_ROOT / 'configs' / 'CAMUS' / 'camus_lite.yaml'))\n"
NEW = (
    OLD + "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- ~10% patient subset (~148/1500 training images).\n"
    "# See docs/EXPERIMENTS.md for sizing rationale. Val/test are UNAFFECTED --\n"
    "# patient_level_split() only shrinks the training pool. Do not remove this\n"
    "# line; it is what makes this notebook a Phase-B run instead of Phase A.\n"
    "cfg['label_frac'] = 0.10\n"
    "# ==========================================================================\n"
)
c = nb['cells'][4]
replace_once(c, OLD, NEW)
replace_once(c, EPOCH_LINE, EPOCH_LINE_NEW)
prepend_to_first_cell(nb, (
    "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (CAMUS)\n"
    "> Trains on a **~10% patient subset (~148 of 1500 training images)** instead of the full "
    "dataset (`label_frac` overridden in Cell 4 -- see docs/EXPERIMENTS.md). Val/test stay full "
    "size. Outputs auto-suffixed `_lf10` (via `utils.model_suffix`) so they never collide with "
    "the Phase-A checkpoints.\n"
    ">\n"
    "> **Phase guide** -- A: `notebooks/unet/01_camus_lite.ipynb` (unmodified, full data). "
    "**B: this notebook.** C: not yet available -- needs the archived MSA backbone wired into "
    "`AGENT_REGISTRY` first (docs/EXPERIMENTS.md section 4).\n"
))
save(nb, 'notebooks/phaseB/unet/01_camus_lite.ipynb')
print('OK: phaseB/unet/01_camus_lite.ipynb')

# ============================================================================
# 2) UNet CAMUS ATTENTION
# ============================================================================
nb = load('notebooks/unet/03_camus_attnunet.ipynb')
OLD = "cfg = load_config(str(PKG_ROOT / 'configs' / 'CAMUS' / 'camus.yaml'))\n"
NEW = (
    OLD + "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- ~10% patient subset (~148/1500 training images).\n"
    "# See docs/EXPERIMENTS.md. This is the low-data ATTENTION-U-Net competitor --\n"
    "# it must be trained on the SAME subset as the Phase-B lite U-Net for the\n"
    "# comparison to be fair (comparing low-data DRL against a full-data\n"
    "# attention net would not answer the Phase-B question). Val/test unaffected.\n"
    "cfg['label_frac'] = 0.10\n"
    "# ==========================================================================\n"
)
c = nb['cells'][4]
replace_once(c, OLD, NEW)
replace_once(c, EPOCH_LINE, EPOCH_LINE_NEW)
prepend_to_first_cell(nb, (
    "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (CAMUS)\n"
    "> Trains the **attention-U-Net competitor** on the **same ~10% patient subset "
    "(~148 of 1500 training images)** as the Phase-B lite U-Net -- required so the DRL-vs-"
    "attention comparison stays fair at this data scale (`label_frac` overridden in Cell 4 -- "
    "see docs/EXPERIMENTS.md). Val/test stay full size. Outputs auto-suffixed `_lf10`.\n"
    ">\n"
    "> **Phase guide** -- A: `notebooks/unet/03_camus_attnunet.ipynb` (unmodified, full data). "
    "**B: this notebook.** C: not yet available -- needs the archived MSA backbone wired into "
    "`AGENT_REGISTRY` first (docs/EXPERIMENTS.md section 4).\n"
))
save(nb, 'notebooks/phaseB/unet/03_camus_attnunet.ipynb')
print('OK: phaseB/unet/03_camus_attnunet.ipynb')

# ============================================================================
# 3) UNet BRISC LITE
# ============================================================================
nb = load('notebooks/unet/02_brisc_lite.ipynb')
OLD = "cfg = load_config(str(PKG_ROOT / 'configs' / 'BRISC' / 'brisc_lite.yaml'))\n"
NEW = (
    OLD + "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- ~5% subset (~155/~3092 training images).\n"
    "# See docs/EXPERIMENTS.md for sizing rationale. Val/test are UNAFFECTED --\n"
    "# patient_level_split() only shrinks the training pool. Do not remove this\n"
    "# line; it is what makes this notebook a Phase-B run instead of Phase A.\n"
    "cfg['label_frac'] = 0.05\n"
    "# ==========================================================================\n"
)
c = nb['cells'][4]
replace_once(c, OLD, NEW)
replace_once(c, EPOCH_LINE, EPOCH_LINE_NEW)
prepend_to_first_cell(nb, (
    "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (BRISC)\n"
    "> Trains on a **~5% subset (~155 of ~3092 training images)** instead of the full dataset "
    "(`label_frac` overridden in Cell 4 -- see docs/EXPERIMENTS.md). Val/test stay full size. "
    "Outputs auto-suffixed `_lf05` (via `utils.model_suffix`) so they never collide with the "
    "Phase-A checkpoints.\n"
    ">\n"
    "> **Phase guide** -- A: `notebooks/unet/02_brisc_lite.ipynb` (unmodified, full data). "
    "**B: this notebook.** C: not yet available -- needs the archived MSA backbone wired into "
    "`AGENT_REGISTRY` first (docs/EXPERIMENTS.md section 4).\n"
))
save(nb, 'notebooks/phaseB/unet/02_brisc_lite.ipynb')
print('OK: phaseB/unet/02_brisc_lite.ipynb')

# ============================================================================
# 4) UNet BRISC ATTENTION
# ============================================================================
nb = load('notebooks/unet/04_brisc_attnunet.ipynb')
OLD = "cfg = load_config(str(PKG_ROOT / 'configs' / 'BRISC' / 'brisc.yaml'))\n"
NEW = (
    OLD + "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- ~5% subset (~155/~3092 training images).\n"
    "# See docs/EXPERIMENTS.md. This is the low-data ATTENTION-U-Net competitor --\n"
    "# it must be trained on the SAME subset as the Phase-B lite U-Net for the\n"
    "# comparison to be fair. Val/test unaffected.\n"
    "cfg['label_frac'] = 0.05\n"
    "# ==========================================================================\n"
)
c = nb['cells'][4]
replace_once(c, OLD, NEW)
replace_once(c, EPOCH_LINE, EPOCH_LINE_NEW)
prepend_to_first_cell(nb, (
    "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (BRISC)\n"
    "> Trains the **attention-U-Net competitor** on the **same ~5% subset (~155 of ~3092 "
    "training images)** as the Phase-B lite U-Net -- required so the DRL-vs-attention "
    "comparison stays fair at this data scale (`label_frac` overridden in Cell 4 -- see "
    "docs/EXPERIMENTS.md). Val/test stay full size. Outputs auto-suffixed `_lf05`.\n"
    ">\n"
    "> **Phase guide** -- A: `notebooks/unet/04_brisc_attnunet.ipynb` (unmodified, full data). "
    "**B: this notebook.** C: not yet available -- needs the archived MSA backbone wired into "
    "`AGENT_REGISTRY` first (docs/EXPERIMENTS.md section 4).\n"
))
save(nb, 'notebooks/phaseB/unet/04_brisc_attnunet.ipynb')
print('OK: phaseB/unet/04_brisc_attnunet.ipynb')

print()
print('=== UNet phaseB notebooks done ===')
