# Iteris — Handoff Document

> **Use this as the seed context for a new chat.** Self-contained snapshot of project state, locked decisions, current progress, and next steps as of Week 2/3.

For deeper background, see also: [`CONTEXT.md`](./CONTEXT.md) (research scope + UI brief) and [`PLAN.md`](./PLAN.md) (architecture).

---

## 1. Project Summary

**Iteris** is a 5-week Taylor's University capstone (PRJ63504) producing a **research paper + UI demo** on DRL-based medical image segmentation refinement.

**The story:**
1. A baseline **U-Net** produces an initial segmentation mask.
2. A **DRL agent** iteratively refines the boundary over up to 20 steps.
3. Compare multiple DRL families (DDQN, Dueling, DDPG, MSA variants).
4. Demonstrate across **5 datasets** spanning 5 imaging modalities.

**Paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop · MedIA (prestige)
**Identity:** Iteris · tagline *"See how AI learns to see."* · **research demo, not a clinical tool**.

**Repo:** [github.com/Anfaal26/iteris](https://github.com/Anfaal26/iteris) (private)
**Local working dir:** `D:\iteris\`

---

## 2. Repo Structure

```
iteris/
├── HANDOFF.md                   This file
├── CONTEXT.md                   Research + UI seed doc
├── PLAN.md                      Strategic plan
├── README.md                    Repo map + quickstart
├── requirements.txt             Pinned minimal deps
├── .gitignore
│
├── configs/                     YAML configs (per dataset, per agent)
│   ├── camus.yaml               U-Net baseline configs
│   ├── brisc.yaml
│   ├── ham10000.yaml
│   ├── kvasir.yaml
│   ├── drive.yaml
│   ├── camus_dqn.yaml           DRL configs (per algorithm)
│   ├── camus_dueling.yaml
│   └── camus_ddpg.yaml
│
├── iteris/                      Python package — all reusable logic
│   ├── __init__.py
│   ├── config.py                load_config + load_drl_config (YAML loaders)
│   ├── ingestion.py             Dataset file-list builders (build_*_dicts per dataset)
│   ├── transforms.py            MONAI pipeline (modality-aware, RGB/grey, binarise labels)
│   ├── splits.py                Patient-level train/val/test
│   ├── models.py                AttentionResUNet — baseline architecture
│   ├── losses.py                DiceCELoss wrapper
│   ├── metrics.py               Pure-torch Dice + HD95 (no scipy dependency!)
│   ├── training.py              run_training() — full baseline pipeline
│   ├── evaluation.py            test eval + mask export + summary JSON
│   ├── visualization.py         Learning curves + qualitative grid
│   ├── utils.py                 seeding, device helpers
│   │── env.py                   ★ DRL — SegmentationEnv (per-class binary)
│   ├── buffer.py                ★ DRL — ReplayBuffer (memory-optimised)
│   ├── drl_networks.py          ★ DRL — CNN backbone + Q/Dueling/Actor/Critic heads
│   ├── agents.py                ★ DRL — DQN/DDQN/Dueling/DDPG with shared base
│   ├── drl_training.py          ★ DRL — run_drl_training() orchestrator
│   └── warm_start.py            ★ DRL — pre-compute U-Net init masks
│
├── notebooks/                   Thin Kaggle notebooks (~50 lines each)
│   ├── 01_camus_baseline.ipynb
│   ├── 02_brisc_baseline.ipynb
│   ├── 03_ham10000_baseline.ipynb
│   ├── 04_kvasir_baseline.ipynb
│   ├── 05_drive_baseline.ipynb
│   └── 06_camus_drl.ipynb       ★ Shared DRL notebook (CFG_NAME picks agent)
│
└── archive/                     Old monolithic notebooks (reference only)
```

---

## 3. Tech Stack

- **Training:** PyTorch · MONAI (transforms, losses, metrics for baseline)
- **Compute:** Kaggle T4 x2 GPU (P100 is dead — current PyTorch lacks Pascal kernels)
- **Repo distribution:** Code lives on GitHub. For Kaggle, zip `iteris/` + `configs/` + `requirements.txt` → upload as Kaggle Dataset `iteris-pkg` → notebooks attach this dataset
- **Logging:** CSV exports for paper figures (W&B optional)
- **Reproducibility:** YAML configs, fixed seeds (PyTorch + NumPy + Python random + MONAI)
- **Critical environment:** Kaggle's numpy 2.x breaks scipy.ndimage in some import paths. We replaced MONAI's HausdorffDistanceMetric with a pure-torch HD95. Don't pin numpy<2.0 (causes worse dependency conflicts).

---

## 4. Current Status

### Done ✓

| Item | Status |
|---|---|
| CAMUS baseline (U-Net) | ✓ Trained, Dice 0.9020 mean (LV_endo 0.9378, LV_epi 0.8723, LA 0.8958) |
| BRISC baseline | ✓ Notebook + config done; dry run passed |
| HAM10000 baseline | ✓ Notebook + config done |
| Kvasir-SEG baseline | ✓ Notebook + config done |
| DRIVE baseline | ✓ Notebook + config done (needs `loader_reader: PILReader` for .gif masks) |
| DRL infrastructure | ✓ env, buffer, networks, agents, training, warm_start — all committed |
| DDQN dry run on CAMUS | ✓ Pipeline works (600 steps, val Dice 0.9136 post-dry-run) |
| Visualisation cells | ✓ Stratified qualitative + Dice distribution + DRL playback |

### In progress / queued

| Item | Status |
|---|---|
| Full BRISC training | Dry run passed, ready for full commit (~2.5 hr) |
| Full HAM10000 training | Dry run passed, ready (~3 hr) |
| Full Kvasir-SEG training | Dry run passed, ready (~1.5 hr) |
| Full DRIVE training | Dry run pending after loader_reader fix |
| Full DDQN CAMUS LV_endo | Dry run passed, ready (~3 hr) |
| Dueling DQN + DDPG runs | Ready (notebooks + configs done) |
| MSA variants (MSA-Dueling, MSA-DDPG) | **Not yet implemented** — need `iteris/msa.py` |

### Not started

- Per-step mask trace export for UI's iteration playback page
- Unified eval harness (Week 4 — load all checkpoints, single test split)
- CHAOS cross-modal transfer experiment
- UI implementation
- Paper writing

---

## 5. Locked Decisions (env, agents, training)

### State design (4 channels, locked)

```
ch 0 : image                            (static within episode)
ch 1 : current binary mask              (dynamic — agent's prediction)
ch 2 : signed distance transform of ch1 (dynamic — derived)
ch 3 : U-Net init mask                  (static — fixed reference)
```

### Per-class binary agents

**One agent per (algorithm × structure × dataset).** CAMUS has 3 structures (LV_endo, LV_epi, LA) → 3 agents per algorithm. Other datasets have 1 structure each.

### Action spaces

- **Discrete (7 actions):** dilate, erode, shift ↑↓←→ (2 px), no-op
- **Continuous (2D):** (dy, dx) ∈ [−0.04, +0.04] (≈ ±10 px at 256, reduced from spec's ±0.1 which is too aggressive)

### Reward

```
r_t = Dice(mask_t, GT) − Dice(mask_{t−1}, GT)
clipped to [−1, +1] BEFORE storing in replay buffer
```

### Episode termination

`step ≥ 20` OR composite stop: `|ΔDice| < 0.001 AND |ΔHD95| < 0.5 px for 3 consecutive steps`

### Training tweaks (council-locked)

- **ε-decay:** linear 1.0 → 0.05 over 40k steps (vs 30k spec) — gives more exploration time given the dense low-magnitude reward signal
- **Buffer pre-fill:** 2000 random transitions before training starts
- **DDPG actor freeze:** first 2000 steps actor frozen, critic bootstraps from random rollouts first
- **DDPG mid-layer action injection:** in Critic, action projected to 128-d and concatenated with state embedding before the merge layers (NOT concatenated to final 256-d embedding — that drowns the action)
- **Dueling Q formula:** `Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]` (mean-centred, NOT max-centred)
- **Replay buffer:** memory-optimised — stores `(sample_idx, mask, action, reward, done)`. Static channels (image, init_mask) cached separately and looked up at sample time. Reduces 10k-transition buffer from ~20 GB to ~1.3 GB
- **Empty-boundary filter:** drop samples where GT structure area < 1% of image (degenerate for refinement)

### Algorithm scope

**In scope** for the 5-week project:
- DDQN (Double DQN)
- Dueling DQN (with Double DQN target)
- DDPG (continuous action)
- MSA-Dueling DQN (Multi-Head Self-Attention head on Dueling backbone) — NOT YET BUILT
- MSA-DDPG (MSA in actor) — NOT YET BUILT

**Out of scope (dropped):**
- PPO — different paradigm (on-policy), would need separate training loop. Drop unless time permits Week 4.
- Plain DQN (without Double DQN target) — strictly dominated by DDQN, skip.

---

## 6. The 5 Datasets

| Dataset | Modality | Channels | Samples | Classes | Notes |
|---|---|---|---|---|---|
| **CAMUS** | Cardiac ultrasound | 1 (grey) | ~2000 | 4 (bg, LV_endo, LV_epi, LA) | Primary. LV_epi class 2 is technically **myocardium** (region between endo and epi boundary) |
| **BRISC** | Brain tumour MRI | 1 (grey, some RGB) | 4793 | 2 (bg, tumor) | Parallel `images/`+`masks/` folders. Mixed JPG channel counts (RGB+grey) handled by `_to_target_channels` |
| **HAM10000** | Skin dermoscopy | 3 (RGB) | 10015 | 2 (bg, lesion) | Images: kmader/skin-cancer-mnist-ham10000 (2 folders). Masks: tschandl/ham10000-lesion-segmentations |
| **Kvasir-SEG** | GI endoscopy | 3 (RGB) | ~2976 (mirror-dependent) | 2 (bg, polyp) | Double-wrapped `Kvasir-SEG/Kvasir-SEG/` Kaggle path |
| **DRIVE** | Retinal fundus | 3 (RGB) | 20 labelled (only training split has labels) | 2 (bg, vessel) | **Hardest** — multi-blob vessels (450+ components per mask). Mask files are `.gif` → needs `loader_reader: PILReader` |

### Kaggle paths

| Dataset | Path |
|---|---|
| CAMUS | `/kaggle/input/datasets/anfaalhossain/camus/CAMUS` |
| BRISC | `/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/segmentation_task` |
| HAM10000 images | `/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_{1,2}` |
| HAM10000 masks | `/kaggle/input/datasets/tschandl/ham10000-lesion-segmentations/HAM10000_segmentations_lesion_tschandl` |
| Kvasir-SEG | `/kaggle/input/datasets/debeshjha1/kvasirseg` |
| DRIVE | `/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction` |
| iteris-pkg | `/kaggle/input/datasets/anfaalhossain/iteris-pkg/iteris-pkg` (note double wrapper) |

All notebooks auto-detect paths via `Path('/kaggle/input').rglob(...)` to avoid hardcoding.

---

## 7. Workflow — How to use the repo on Kaggle

### Updating the iteris-pkg dataset (after any code change)

1. **Locally:** `git pull` to get latest commits
2. **Zip:** select `iteris/` + `configs/` + `requirements.txt` → Send to → Compressed (zipped) folder
3. **Kaggle:** Go to `iteris-pkg` dataset page → **New Version** → drop the zip → write a version note → Create
4. **In notebook:** sidebar → kebab (⋯) next to `iteris-pkg` → **Update to latest version** → restart kernel

The dataset has a double `iteris-pkg/iteris-pkg/` wrapper because of how Kaggle uploads zipped folders. Notebooks use `rglob` to find the package regardless.

### Running a baseline notebook

1. Attach inputs: the dataset + `iteris-pkg`
2. Settings: GPU T4 x2, Persistence Files only, Internet On
3. Save Version → Save & Run All (Commit)
4. ~2.5 hr per run (some less, see expected times per dataset)
5. Output `/kaggle/working/` lands as a committed artifact

### Running a DRL notebook

1. Attach inputs: dataset + `iteris-pkg` + **`camus-baseline-outputs`** (a Kaggle Dataset created from the CAMUS baseline notebook's output, containing `camus_best.pt`)
2. In `06_camus_drl.ipynb` Cell 1, set `CFG_NAME = 'camus_dqn.yaml'` (or `_dueling`, `_ddpg`)
3. To change structure: override `cfg['target_class'] = 2` after loading (1=LV_endo, 2=LV_epi, 3=LA). Re-run warm-start cell.
4. Save Version → Save & Run All (Commit)
5. ~3 hr for DDQN/Dueling, ~5 hr for DDPG

### Dry-run pattern (before committing a real run)

Paste between Cell 1 and the training cell:

```python
# === DRY RUN — ~3 min ===
cfg['label_frac']          = 0.02
cfg['val_split']           = 0.02
cfg['test_split']          = 0.02
cfg['epochs']              = 2
cfg['patience']            = 99
cfg['save_every_n_epochs'] = 0
# For DRL also override: train_steps, buffer_size, prefill_steps
```

For DRL dry runs, also unpack `result['agent']` to enable visualisation cells:

```python
result      = run_drl_training(cfg, train_samples, val_samples)
agent       = result['agent']
history     = result['history']
best_dice   = result['best_dice']
```

**Always remove the dry-run cell before committing the real run** — its config overrides will carry through `cfg` to the training cell.

---

## 8. Known Issues / Gotchas

| Issue | Fix |
|---|---|
| Kaggle's numpy 2.x breaks scipy.ndimage in MONAI HD95 path | Already fixed: we use pure-torch `hd95_batch()` in `iteris/metrics.py`. Don't pin numpy<2.0 — causes worse conflicts. |
| `ModuleNotFoundError: load_drl_config` | iteris-pkg Kaggle Dataset is on an old version. Re-upload + update version + restart. |
| `InvalidDicomError` on DRIVE | Add `loader_reader: PILReader` to drive.yaml — fixed in `iteris/transforms.py` to honour this field. |
| Channel mismatch errors (`[3,256,256] vs [1,256,256]`) | `_to_target_channels` Lambdad in transforms.py coerces all images to `cfg['in_channels']` — handles mixed RGB/grey datasets like BRISC. |
| Stale module after re-upload | `importlib.reload(iteris.<module>)` OR restart kernel (cleaner). |
| DataLoader silent OOM on Kaggle | Use `dataloader_workers: 0` in YAML (already set). CacheDataset workers (`cache_workers: 2`) are fine — only DataLoader workers hit /dev/shm limits. |
| Visualisation cells slow during dry-run | The "stratified worst/median/best" version iterates all test samples. Use the **simpler** version that picks 3 random samples — provided in chat. |
| Kaggle Dataset path nesting (double folder) | Always use `rglob` to find files, never hardcode paths. |
| RAS warnings for 2D data | Set `apply_orientation: false` in YAML (already done for all 2D datasets). |

---

## 9. The 2-Week Training Plan (Week 3–4 of project)

### Scope decision

- Skip PPO (different paradigm, too much dev time)
- **5 algorithms × 3 CAMUS structures + 4 other datasets × 2 algorithms = 23 DRL agents**
- Plus 5 U-Net baselines = **28 total trained models**

### Distribution across 5 Kaggle accounts

| Account | Week 1 (CAMUS depth + breadth baselines) | Week 2 (MSA + transfer) |
|---|---|---|
| #1 — CAMUS LV_endo | DDQN, Dueling, DDPG on LV_endo | MSA-Dueling, MSA-DDPG on LV_endo |
| #2 — CAMUS LV_epi | Same algos on LV_epi | Same MSA on LV_epi |
| #3 — CAMUS LA | Same algos on LA | Same MSA on LA |
| #4 — Breadth #1 | BRISC baseline + Dueling, Kvasir baseline + Dueling | MSA-Dueling on BRISC, Kvasir |
| #5 — Breadth #2 | HAM10000 baseline + Dueling, DRIVE baseline + Dueling | MSA-Dueling on HAM10000, DRIVE |

Each Kaggle account has 30 GPU hours/week. 5 accounts × 2 weeks = 300 GPU hours budget. Actual usage: ~100 hours. Plenty of slack.

### Coordination

- Each account uploads outputs as a **Public Kaggle Dataset** so other accounts can attach them for the eval harness
- Account #1 runs the unified eval harness on Day 11

---

## 10. UI / Website Sequencing

The website backend needs three things from training:

1. **Checkpoints** — for `POST /predict` live inference (already exported)
2. **Per-step mask traces** — for the **Iteration Playback** page (the differentiator per CONTEXT.md §4). **NOT YET EXPORTED** — needs to be added to DRL training notebooks.
3. **Pre-computed demo samples** — 5–6 fixed test images, baked into "Demo Mode"

### Critical sequencing

```
Week 3:  CAMUS depth training (15 agents)           ┐
         + Add per-step trace export to DRL nb       │ in parallel
         UI dev: workspace + side-by-side pages      ┘

Week 4:  Other datasets training (8 agents)         ┐
         + MSA variants                              │ in parallel
         UI dev: iteration playback page             ┘ (needs traces by mid-week)

Week 5:  Paper writing + UI polish (no more training)
```

The **per-step trace export** is the long-lead UI dependency. Add it to `iteris/drl_training.py` (or as a post-training step in the notebook) before any heavy training runs — re-running all 15 CAMUS agents to add this would be expensive.

---

## 11. Next Logical Steps

In order of priority:

1. **Update iteris-pkg Kaggle Dataset** with latest code (most recently: `loader_reader` fix in transforms.py for DRIVE)
2. **Commit the 4 baseline trainings** (BRISC, HAM10000, Kvasir, DRIVE) — queue across multiple accounts
3. **Build `iteris/msa.py`** — Multi-Head Self-Attention module that sits between CNN backbone and head (4 heads, key_dim 64). Then `MSADuelingDQNAgent` and `MSADDPGAgent` subclasses.
4. **Add per-step trace export** to DRL training (writes `step_NN.npy` per sample for UI playback)
5. **Commit first 3 CAMUS DRL runs** (DDQN, Dueling, DDPG on LV_endo)
6. **Begin UI development** (FastAPI backend loading checkpoints) — can start with just CAMUS baseline + DDQN
7. **Week 4 evaluation harness** — single notebook loading all checkpoints, same test split, main results table

---

## 12. Quick Reference

### GitHub
- Repo: https://github.com/Anfaal26/iteris
- Latest commits relevant to current state: see `git log --oneline | head`

### Key file locations
- Baseline config: `configs/<dataset>.yaml`
- DRL config: `configs/camus_<agent>.yaml`
- Env locked spec: `iteris/env.py`
- Memory-optimised buffer: `iteris/buffer.py`
- Agent registry: `iteris/drl_training.py::AGENT_REGISTRY`

### Acceptance thresholds (paper)

| Dataset | Baseline Dice target | DRL refinement target |
|---|---|---|
| CAMUS LV_endo | ≥ 0.85 (✓ achieved 0.94) | +0.01–0.03 over baseline |
| CAMUS LV_epi | — | +0.01–0.03 |
| CAMUS LA | — | +0.01–0.03 |
| BRISC | ≥ 0.75 | TBD |
| HAM10000 | ≥ 0.85 | TBD |
| Kvasir-SEG | ≥ 0.80 | TBD |
| DRIVE | ≥ 0.65 (low ceiling, multi-blob) | TBD |

Paper acceptance bar: **>3pp Dice over baseline, p<0.05 (Wilcoxon)** for at least the primary CAMUS comparisons.

### Compute budget per run

| Run type | Wall time |
|---|---|
| U-Net baseline | ~2.5 hr |
| DDQN / Dueling DQN | ~3 hr |
| DDPG | ~5 hr |
| MSA variant | ×1.3 of base |
| Warm-start (U-Net inference over all CAMUS) | ~5 min |

---

## 13. Where to Pick Up

If a new chat is taking over from here:

1. Read this file + `CONTEXT.md` + `PLAN.md` first.
2. Check `git log --oneline -20` for the most recent changes.
3. The current immediate work item is **getting all 5 baselines committed** (CAMUS done, 4 pending dry runs / full runs).
4. Then the **MSA module** + **per-step trace export** before any heavy DRL training begins.
5. Do **not** start the 2-week training plan until both of those are in place — otherwise you'll need to re-run.

Open questions awaiting user decision:
- Rename `LV_epi` → `LV_myo` in configs for anatomical correctness?
- Build PPO as a Week 4 stretch (probably no — drop).
- MSA-DDPG on all 4 breadth datasets (+4 agents, +24 hr) or skip?
