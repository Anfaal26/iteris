# Iteris — Implementation Plan

> Strategic document. Read before adding features, configs, or notebooks.
> **Last updated:** 2026-06-01. Supersedes all prior plan documents.

---

## 1. Engineering Principles

1. **Notebooks are thin.** All logic lives in `iteris/`. Notebooks are ~50 lines: import, configure, call, display. Fix bugs in one file, not seven notebooks.
2. **Configs are YAML.** No hardcoded hyperparameters in Python. Every dataset/agent has a YAML in `configs/`.
3. **Dataset-agnostic core.** `models.py`, `metrics.py`, `training.py` consume `(image, label)` tensors and a `cfg` dict — they know nothing about CAMUS vs BRISC.
4. **Modality-aware edges.** `ingestion.py` has per-dataset builders. `transforms.py` switches preprocessing on `cfg['normalize']` (minmax / zscore / hu). This is the entire surface area for a new dataset.
5. **Separate paradigm paths.** Tracing (`_run_contour_training`) and refinement (DDPG) dispatch at the top of `run_drl_training` based on `cfg['env_class']`. Neither path touches the other.

---

## 2. Current Status

### Complete ✓

| Item | Notes |
|---|---|
| CAMUS U-Net baseline | Dice 0.938/0.872/0.896 (c1/c2/c3) · checkpoint `camus_best.pt` |
| BRISC U-Net baseline | Dice 0.835 test · checkpoint `brisc_best.pt` |
| Boundary-tracing env | `ContourTracingEnv`, `VectorisedContourEnv`, GT-EDT precompute, 8-direction action |
| Region-aware reward | Terminal Dice (+10·D) dominant; coverage bonus; conditional closure gate |
| Seed Option A | Best-overlap CC + GT fallback at init_Dice < 0.30; centroid variant for BRISC |
| Agent + network | `DQNAgent`, `DuelingDQNAgent` with `patch=True`; `PatchQNetwork`, `PatchDuelingQNetwork` |
| Training loop | `_run_contour_training` — vectorised 16-env, prefill, ε-greedy, eval subset, checkpoint |
| Viz + diagnostics | `contour_viz.py`, `dryrun_viz.py` |
| BRISC 25k validation | DuelingDDQN: test Dice 0.806, HD95 8.76px, closure 87.6% — reward confirmed working |
| Phase B fixes | CAMUS steps bumped (50k/60k), centroid seed, smoothness penalty, T_max 300 for BRISC |
| BRISC per-type configs | Optional glioma/meningioma/pituitary configs with tuned T_max |

### In Progress

| Item | Blocker |
|---|---|
| BRISC DuelingDDQN at 50k | Upload latest Kaggle iteris-pkg dataset to `dfd9d54` → run §4 |
| BRISC DQN at 50k | Same — within-paradigm comparison |

### Pending (in order)

| Item | Notes |
|---|---|
| CAMUS c1 DuelingDDQN 50k | First cardiac test with region-aware reward |
| CAMUS c1 DQN 50k | Comparison |
| CAMUS c2 (LVepi) and c3 (LA) | Same protocol |
| DDPG baseline runs | Continuous baseline; mask morphology unchanged |

---

## 3. Kaggle Workflow

### Updating iteris-pkg

1. Locally: `git pull` (or `git push` after changes)
2. Download repo zip or zip `iteris/` + `configs/` + `requirements.txt`
3. Kaggle: `iteris-pkg` dataset → **New Version** → upload zip → Create
4. In notebook: sidebar → `iteris-pkg` → **Update to latest version** → restart kernel

### Running a training notebook

1. Attach: dataset + `iteris-pkg` + baseline outputs (`camus-baseline-outputs` or `brisc-baseline-outputs-v2`)
2. Settings: GPU T4, Persistence Files only, Internet On
3. §0 install → §1 config (`AGENT_NAME='DQN'` or `'DuelingDDQN'` or `'DDPG'`) → §2 warm-start → §4 train
4. Save Version → Save & Run All

### Expected wall-clock (T4)

| Run | Time |
|---|---|
| BRISC DQN/DuelingDDQN 50k steps | ~48 min |
| CAMUS DQN/DuelingDDQN 50k/60k steps | ~60–75 min |
| DDPG 100k–120k steps | ~3–4 hr |

---

## 4. Roadmap

### Phase C — Paper quality (implement after BRISC 50k confirms reward works)

| Item | What | Effort |
|---|---|---|
| **C1** | CAMUS `max_trace_length` reduction: c1 400→320, c2 480→360, c3 420→320 | YAML only |
| **C2** | n-step returns (n=5) in `DQNAgent.update()` + `ContourReplayBuffer.sample()` | ~60 LOC; bigger win on CAMUS than BRISC |
| **C3** | Full training matrix: DQN + DuelingDDQN × 3 CAMUS classes + BRISC | Kaggle runs |
| **C4** | Unified eval harness: load all checkpoints, same test split, results table | ~1 notebook |
| **C5** | Wilcoxon signed-rank tests vs U-Net baseline | `iteris/evaluation.py` extension |

### Phase D — Paper writing only

| Item | What |
|---|---|
| **D1** | LA results reported separately with mitral-valve-closure caveat |
| **D2** | Optional: morphological post-processing on jagged BRISC masks (close-then-open); report both |
| **D3** | Future work: learned First-P-Net seed; PPO comparison; skip-neighbourhood 16-action variant |

---

## 5. Adding a New Dataset

1. Add `configs/<dataset>.yaml` (set `normalize`, `num_classes`, `class_names`)
2. Add `build_<dataset>_dicts()` in `iteris/ingestion.py` → wire into `build_dataset_dicts(cfg)`
3. Add a U-Net baseline notebook (copy `01_camus_baseline.ipynb`, swap config)
4. For DRL: add `configs/<dataset>_drl_c<n>.yaml` with agent blocks

---

## 6. Critical Don'ts

- Do not add `scipy.ndimage` to the reward path — use the precomputed GT-EDT (`self._gt_edt[y,x]`)
- Do not restart the seeding discussion — best-overlap CC + GT fallback is settled (see CONTEXT.md §8)
- Do not create files in root — use `iteris/`, `configs/`, `notebooks/`, `docs/`
- Do not modify archived files in `iteris/archive/` unless resurrecting them intentionally
