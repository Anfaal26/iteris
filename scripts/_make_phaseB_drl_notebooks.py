"""One-off script: generate the Phase B DRL notebook copies.
Not part of the package; run once from repo root. See docs/EXPERIMENTS.md.
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


def find_config_cell(nb):
    for c in nb['cells']:
        if c.get('id') == 'code-config':
            return c
    raise AssertionError('code-config cell not found')


def prepend_to_first_cell(nb, text):
    cell = nb['cells'][0]
    joined = ''.join(cell['source'])
    joined = joined.rstrip('\n') + '\n' + text
    cell['source'] = joined.splitlines(keepends=True)


LABEL_FRAC_BLOCK_CAMUS = (
    "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- must match the retrained low-data lite/attention\n"
    "# checkpoints (~10% patient subset, ~148/1500 training images). This single\n"
    "# line does TWO things: (1) makes the checkpoint auto-detect below look for\n"
    "# the _lf10-suffixed checkpoint instead of the full-data one, and (2) makes\n"
    "# precompute_init_masks() below use the SAME low-data patient subset the\n"
    "# U-Net was trained on -- required for a fair, leak-free comparison (see\n"
    "# docs/EXPERIMENTS.md). Do not remove; do not set independently of the\n"
    "# baseline notebook's label_frac.\n"
    "baseline_cfg['label_frac'] = 0.10\n"
    "# ==========================================================================\n"
)

LABEL_FRAC_BLOCK_BRISC = (
    "\n"
    "# ==========================================================================\n"
    "# PHASE B override -- must match the retrained low-data lite/attention\n"
    "# checkpoints (~5% subset, ~155/~3092 training images). This single line\n"
    "# does TWO things: (1) makes the checkpoint auto-detect below look for the\n"
    "# _lf05-suffixed checkpoint instead of the full-data one, and (2) makes\n"
    "# precompute_init_masks() below use the SAME low-data subset the U-Net was\n"
    "# trained on -- required for a fair, leak-free comparison (see\n"
    "# docs/EXPERIMENTS.md). Do not remove; do not set independently of the\n"
    "# baseline notebook's label_frac.\n"
    "baseline_cfg['label_frac'] = 0.05\n"
    "# ==========================================================================\n"
)

CAMUS_NOTEBOOKS = [
    ('notebooks/camus/drl/03a_camus_drl_lv_endo.ipynb', 'notebooks/phaseB/camus/drl/03a_camus_drl_lv_endo.ipynb', 'LV_endo'),
    ('notebooks/camus/drl/03b_camus_drl_lv_epi.ipynb', 'notebooks/phaseB/camus/drl/03b_camus_drl_lv_epi.ipynb', 'LV_epi'),
    ('notebooks/camus/drl/03c_camus_drl_la.ipynb', 'notebooks/phaseB/camus/drl/03c_camus_drl_la.ipynb', 'LA'),
]

OLD_LOAD = "baseline_cfg = load_config(str(PKG_ROOT / 'configs' / cfg['baseline_cfg_name']))\n"

for src, dst, class_name in CAMUS_NOTEBOOKS:
    nb = load(src)
    c = find_config_cell(nb)
    NEW = OLD_LOAD + LABEL_FRAC_BLOCK_CAMUS
    replace_once(c, OLD_LOAD, NEW)
    prepend_to_first_cell(nb, (
        "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (CAMUS)\n"
        f"> Refines a lite U-Net that was itself **retrained on a ~10% patient subset "
        "(~148 of 1500 training images)** -- see `notebooks/phaseB/unet/01_camus_lite.ipynb`. "
        "`baseline_cfg['label_frac']` is overridden in the Configure cell below: it makes the "
        "checkpoint auto-detect find the `_lf10`-suffixed low-data checkpoint, and makes the "
        "warm-start use the same low-data patient subset the U-Net was trained on (see "
        "docs/EXPERIMENTS.md). **Attach the Phase-B checkpoint dataset, not the Phase-A one**, "
        "or the auto-detect raises `FileNotFoundError`.\n"
        ">\n"
        f"> **Phase guide** -- A: `notebooks/camus/drl/03{{a,b,c}}_camus_drl_*.ipynb` (unmodified, "
        "full data). **B: this notebook**, with the Phase-B lite/attention checkpoints attached. "
        "C: not yet available -- needs the archived MSA backbone wired into `AGENT_REGISTRY` "
        "first (docs/EXPERIMENTS.md section 4).\n"
    ))
    save(nb, dst)
    print(f'OK: {dst}')

# ============================================================================
# BRISC DRL
# ============================================================================
nb = load('notebooks/brisc/drl/04_brisc_drl.ipynb')
c = find_config_cell(nb)
NEW = OLD_LOAD + LABEL_FRAC_BLOCK_BRISC
replace_once(c, OLD_LOAD, NEW)
prepend_to_first_cell(nb, (
    "\n> ### PHASE B NOTEBOOK -- LOW-DATA REGIME (BRISC)\n"
    "> Refines a lite U-Net that was itself **retrained on a ~5% subset (~155 of ~3092 training "
    "images)** -- see `notebooks/phaseB/unet/02_brisc_lite.ipynb`. `baseline_cfg['label_frac']` "
    "is overridden in the Configure cell below: it makes the checkpoint auto-detect find the "
    "`_lf05`-suffixed low-data checkpoint, and makes the warm-start use the same low-data subset "
    "the U-Net was trained on (see docs/EXPERIMENTS.md). **Attach the Phase-B checkpoint dataset, "
    "not the Phase-A one**, or the auto-detect raises `FileNotFoundError`. This applies to all "
    "BRISC tumour subtypes too (glioma/meningioma/pituitary use the same pooled low-data lite/"
    "attention checkpoint, filtered by `tumor_type_filter` at warm-start time).\n"
    ">\n"
    "> **Phase guide** -- A: `notebooks/brisc/drl/04_brisc_drl.ipynb` (unmodified, full data). "
    "**B: this notebook**, with the Phase-B lite/attention checkpoints attached. C: not yet "
    "available -- needs the archived MSA backbone wired into `AGENT_REGISTRY` first "
    "(docs/EXPERIMENTS.md section 4).\n"
))
save(nb, 'notebooks/phaseB/brisc/drl/04_brisc_drl.ipynb')
print('OK: notebooks/phaseB/brisc/drl/04_brisc_drl.ipynb')

print()
print('=== DRL phaseB notebooks done ===')
