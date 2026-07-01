"""One-off script: append a note about the new DRL-comparable metrics to the
'Test-set evaluation' markdown cell in all 8 U-Net notebooks (4 Phase A +
4 Phase B). Documentation-only -- the metrics themselves come from
iteris/evaluation.py, so no code cells need to change. See docs/EXPERIMENTS.md.
"""
import json

TARGETS = [
    'notebooks/unet/01_camus_lite.ipynb',
    'notebooks/unet/02_brisc_lite.ipynb',
    'notebooks/unet/03_camus_attnunet.ipynb',
    'notebooks/unet/04_brisc_attnunet.ipynb',
    'notebooks/phaseB/unet/01_camus_lite.ipynb',
    'notebooks/phaseB/unet/02_brisc_lite.ipynb',
    'notebooks/phaseB/unet/03_camus_attnunet.ipynb',
    'notebooks/phaseB/unet/04_brisc_attnunet.ipynb',
]

NOTE = (
    "\n\n**Metrics reported (per class):** `dice`, `hd95` (existing, torch-batched) plus "
    "`iou`, `boundary_iou` (`biou`), `precision`, `sensitivity`, `mean_surface_distance` (`msd`), "
    "and a connected-component-filtered `hd95geo` -- computed via the SAME `iteris.geometry` "
    "functions the DRL agents' test-set eval uses, so these numbers are directly comparable to "
    "the DRL results class-for-class. Use `hd95geo`, not `hd95`, when comparing HD95 against a "
    "DRL agent (`metrics.hd95_batch` has no connected-component filtering, so it isn't the same "
    "definition). All distances are in **pixels** -- no pixel-spacing metadata is plumbed through "
    "the pipeline, so millimetre distances aren't reported (would require fabricating a scale "
    "factor). Adds ~10ms/sample to this one-time end-of-training eval pass -- negligible."
)


def load(p):
    return json.load(open(p, encoding='utf-8'))


def save(nb, p):
    json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    json.load(open(p, encoding='utf-8'))


for path in TARGETS:
    nb = load(path)
    found = False
    for cell in nb['cells']:
        if cell.get('cell_type') != 'markdown':
            continue
        src = ''.join(cell.get('source', []))
        if 'Test-set evaluation' in src:
            joined = src.rstrip('\n') + NOTE + '\n'
            cell['source'] = joined.splitlines(keepends=True)
            found = True
            break
    assert found, f'{path}: "Test-set evaluation" markdown cell not found'
    save(nb, path)
    print(f'OK: {path}')

print()
print('=== metrics note added to all 8 U-Net notebooks ===')
