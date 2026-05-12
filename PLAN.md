# Iteris — Implementation Plan v2

Strategic document for the codebase. Read before adding new features, datasets, or notebooks.

---

## 1. Design Principles

**1. Notebooks are thin.** All logic lives in `iteris/`. Notebooks are ~50 lines: import, configure, call high-level functions, display results. When a bug is found, fix one file — not seven notebooks.

**2. Configs are YAML.** No hardcoded hyperparameters in Python. Every dataset has a YAML file in `configs/`. To switch datasets, point a notebook at a different YAML.

**3. Pure-torch metrics.** No `scipy.ndimage` anywhere — Kaggle's bundled scipy breaks under numpy 2.x. Dice and HD95 are computed entirely in PyTorch.

**4. Dataset-agnostic core.** `models.py`, `metrics.py`, `training.py`, `evaluation.py`, `losses.py` know nothing about CAMUS vs CHAOS. They consume `(image, label)` tensors and a `cfg` dict.

**5. Dataset-specific edges.** `ingestion.py` has per-dataset builder functions (`build_camus_dicts`, `build_chaos_ct_dicts`). `transforms.py` switches preprocessing pipeline based on `cfg['normalize']`. That's the entire surface area for adding a new dataset.

---

## 2. Module Responsibilities

| Module | Responsibility | Stable? |
|---|---|---|
| `config.py` | Load YAML, expand paths, validate keys | ✅ |
| `ingestion.py` | Walk filesystem → list of `{image, label, patient, ...}` dicts | One function per dataset |
| `transforms.py` | Modality-aware MONAI pipeline (minmax / zscore / HU) | ✅ |
| `splits.py` | Patient-level train/val/test split + label-fraction subsampling | ✅ |
| `models.py` | `AttentionResUNet` — configurable `in_channels`, `num_classes` | ✅ |
| `losses.py` | `DiceCELoss` (MONAI wrapper) — paper-standard combo | ✅ |
| `metrics.py` | Pure-torch `dice_score()` and `hd95_batch()` | ✅ |
| `training.py` | `train_epoch`, `eval_epoch`, `run_training()` orchestrator | ✅ |
| `evaluation.py` | Test-set per-patient CSV + predicted mask export | ✅ |
| `visualization.py` | Learning curves, qualitative overlays | ✅ |
| `utils.py` | Seeding, logging helpers | ✅ |

---

## 3. Adding a New Dataset — Step-by-Step

Let's add ACDC as worked example.

### Step 1 — Create config

`configs/acdc.yaml`:
```yaml
dataset: ACDC
modality: cardiac_mri
data_root: /kaggle/input/acdc
image_size: [256, 256]
num_classes: 4
class_names: [background, LV, RV, Myocardium]
class_colors: ['#000000', '#00C9A7', '#60A5FA', '#A78BFA']
spacing: [1.5, 1.5]
normalize: zscore        # MRI → z-score normalisation
label_frac: 1.0
val_split: 0.15
test_split: 0.10
batch_size: 8
epochs: 60
lr: 1.0e-4
weight_decay: 1.0e-5
patience: 10
seed: 42
checkpoint_dir: /kaggle/working
```

### Step 2 — Add ingestion function

`iteris/ingestion.py`:
```python
def build_acdc_dicts(data_root, ...):
    """Walks ACDC dataset → returns list of {image, label, patient, frame}."""
    ...
```

### Step 3 — Wire it up

Add a single line to `iteris/ingestion.py::build_dataset_dicts(cfg)`:
```python
if cfg['dataset'] == 'ACDC':
    return build_acdc_dicts(cfg['data_root'], ...)
```

### Step 4 — Copy notebook, swap config

`notebooks/03_acdc_baseline.ipynb` — duplicate `01_camus_baseline.ipynb`, change the config path. Done.

---

## 4. Kaggle Workflow

### One-time setup

1. Locally: edit code in `D:\iteris\iteris\*.py`
2. Locally: commit + push to GitHub
3. On Kaggle: upload the `iteris/` folder as a Kaggle Dataset named `iteris-pkg`
4. On Kaggle: when you update code locally, re-upload `iteris-pkg` as a new dataset version

### Each notebook starts the same way

```python
# Cell 1 — pull package
import sys
sys.path.insert(0, '/kaggle/input/iteris-pkg')

# Cell 2 — load config + override Kaggle paths
from iteris.config import load_config
cfg = load_config('/kaggle/input/iteris-pkg/configs/camus.yaml')
cfg['data_root'] = '/kaggle/input/datasets/anfaalhossain/camus/CAMUS'
cfg['checkpoint_dir'] = '/kaggle/working'

# Cell 3 — full pipeline
from iteris.training import run_training
from iteris.evaluation import evaluate_test_set
results = run_training(cfg)
test_results = evaluate_test_set(results['model'], results['test_loader'], cfg)
```

That's the full minimal notebook. Real notebooks add visualisation between steps.

---

## 5. Pinned Dependencies (Kaggle Compatibility)

In `requirements.txt`:
```
monai==1.4.0
numpy<2.0           # scipy.ndimage breaks under numpy 2.x on Kaggle
torch>=2.0
pyyaml
```

The numpy pin is the critical one. Notebooks install via `!pip install -r /kaggle/input/iteris-pkg/requirements.txt --quiet --force-reinstall` followed by a manual kernel restart.

---

## 6. What Stays in `archive/`

- `week1_camus_baseline.ipynb` — old 30-cell monolith
- `week2_00_env_validation.ipynb` — env validation (will be rewritten under new structure)

Kept for reference. Don't run them.

---

## 7. Next Steps After Baseline Lands

In order:

1. **CAMUS baseline** — verify the new structure produces ≥0.85 LV_endo Dice (matches old run)
2. **CHAOS ingestion + transform** — add `build_chaos_ct_dicts` + HU window
3. **CHAOS baseline** — same notebook pattern, swap config → train
4. **Env validation v2** — rewrite under new structure as `notebooks/04_env_validation.ipynb`
5. **Random-action baseline** — first DRL milestone
6. **DQN / DDQN / Dueling DQN** — Week 2 agents
7. **DDPG / MSA variants** — Week 3 agents
8. **Unified eval harness + ablations** — Week 4
9. **UI deployment** — Week 5

---

## 8. Decision Log

| Date | Decision | Why |
|---|---|---|
| 2026-05-08 | Restructure into `iteris/` package | Notebook-only structure didn't scale |
| 2026-05-08 | Pure-torch metrics, drop scipy.ndimage | Kaggle numpy 2.x breaks scipy |
| 2026-05-08 | Custom AttentionResUNet retained | 0.94 Dice on CAMUS, paper-credible |
| 2026-05-08 | Kaggle Dataset distribution | Keeps repo private, no token plumbing |
| 2026-05-08 | Per-class binary DRL agents | Literature-aligned, action space stays small |
