"""One-off script: add a Phase A/B/C guide banner to the ORIGINAL (Phase A)
notebooks. Documentation-only -- no code cells touched. See docs/EXPERIMENTS.md.
"""
import json


def load(p):
    return json.load(open(p, encoding='utf-8'))


def save(nb, p):
    json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    json.load(open(p, encoding='utf-8'))


def prepend_to_first_cell(nb, text):
    cell = nb['cells'][0]
    joined = ''.join(cell['source'])
    joined = joined.rstrip('\n') + '\n' + text
    cell['source'] = joined.splitlines(keepends=True)


def guide(dataset, kind, phaseB_path):
    """kind: 'lite' | 'attention' | 'drl'"""
    if kind == 'drl':
        what = "refines the lite U-Net's mask with a DRL agent (" + dataset + ")"
    elif kind == 'lite':
        what = "trains the lightweight RL warm-start baseline (" + dataset + ")"
    else:
        what = "trains the attention-U-Net competitor (" + dataset + ")"
    return (
        "\n> ### Phase guide -- which notebook/data for which phase\n"
        "> This notebook -- which " + what + " -- is the **Phase A** (full-dataset) version. "
        "Run it as-is, no changes needed.\n"
        ">\n"
        "> | Phase | Data | Notebook |\n"
        "> |---|---|---|\n"
        "> | **A** (full data) | full dataset | **this notebook**, unmodified |\n"
        "> | **B** (low-data regime) | " + ("~150" if dataset == "CAMUS" else "~155") +
        " training images | `" + phaseB_path + "` |\n"
        "> | **C** (MSA backbone, smaller subset) | " + ("~75" if dataset == "CAMUS" else "~93") +
        " training images | not yet available -- needs the archived MSA backbone wired into "
        "`AGENT_REGISTRY` first (see `docs/EXPERIMENTS.md` section 4) |\n"
        ">\n"
        "> Full methodology, exact sizes, and why: **`docs/EXPERIMENTS.md`**.\n"
    )


TARGETS = [
    ('notebooks/unet/01_camus_lite.ipynb', 'CAMUS', 'lite',
     'notebooks/phaseB/unet/01_camus_lite.ipynb'),
    ('notebooks/unet/02_brisc_lite.ipynb', 'BRISC', 'lite',
     'notebooks/phaseB/unet/02_brisc_lite.ipynb'),
    ('notebooks/unet/03_camus_attnunet.ipynb', 'CAMUS', 'attention',
     'notebooks/phaseB/unet/03_camus_attnunet.ipynb'),
    ('notebooks/unet/04_brisc_attnunet.ipynb', 'BRISC', 'attention',
     'notebooks/phaseB/unet/04_brisc_attnunet.ipynb'),
    ('notebooks/camus/drl/03a_camus_drl_lv_endo.ipynb', 'CAMUS', 'drl',
     'notebooks/phaseB/camus/drl/03a_camus_drl_lv_endo.ipynb'),
    ('notebooks/camus/drl/03b_camus_drl_lv_epi.ipynb', 'CAMUS', 'drl',
     'notebooks/phaseB/camus/drl/03b_camus_drl_lv_epi.ipynb'),
    ('notebooks/camus/drl/03c_camus_drl_la.ipynb', 'CAMUS', 'drl',
     'notebooks/phaseB/camus/drl/03c_camus_drl_la.ipynb'),
    ('notebooks/brisc/drl/04_brisc_drl.ipynb', 'BRISC', 'drl',
     'notebooks/phaseB/brisc/drl/04_brisc_drl.ipynb'),
]

for path, dataset, kind, phaseB_path in TARGETS:
    nb = load(path)
    prepend_to_first_cell(nb, guide(dataset, kind, phaseB_path))
    save(nb, path)
    print(f'OK: {path}')

print()
print('=== Phase guide added to all 8 original notebooks ===')
