# Iteris — Project Context

> Single source of truth. All other context files have been condensed into this document.
> **Last updated:** 2026-06-01. Prioritises recent decisions over earlier ones for conflicts.

---

## 1. What This Is

**Iteris** is a Taylor's University capstone (PRJ63504) producing a **research paper** on DRL-based medical image segmentation via sequential boundary tracing.

- **Research question:** Can a discrete DRL agent trace a structure's boundary point-by-point and produce a segmentation that improves on a U-Net baseline?
- **Paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop
- **Repo:** `github.com/Anfaal26/iteris` · local: `D:\iteris\`
- **UI work:** deferred — not in scope until training results are complete

---

## 2. Datasets

| Dataset | Modality | Classes | Baseline Dice | Notes |
|---|---|---|---|---|
| **CAMUS** | Cardiac ultrasound | LV-endo (c1), LV-epi (c2), LA (c3) | 0.938 / 0.872 / 0.896 | Primary; 3 per-class agents |
| **BRISC** | Brain tumour T1+Gd MRI | tumor (binary; glioma/meningioma/pituitary in data) | 0.835 (test) | Single binary agent + optional per-type configs |

**BRISC is brain MRI, not ultrasound.** Preprocessing: z-score normalisation. Per-type breakdown available via `tumor_type_filter` in the per-class configs (`brisc_drl_glioma.yaml`, `_meningioma.yaml`, `_pituitary.yaml`). Active runs use the binary `brisc_drl_tumor.yaml`.

---

## 3. Active Agent Set

| Selector key | Algorithm | Env / Paradigm |
|---|---|---|
| `DQN` | DQNAgent (patch CNN) | `ContourTracingEnv` — boundary tracing |
| `DuelingDDQN` | DuelingDQNAgent (V+A heads, double target) | `ContourTracingEnv` — boundary tracing |
| `DDPG` | DDPGAgent (OU noise, continuous) | `SegmentationEnv` — mask morphology (continuous baseline only) |

Retired / archived: `DDQNAgent`, `MSADuelingDQNAgent`, `msa.py`, `SegmentationEnvBRISC` — see `iteris/archive/` with resurrection notes.

---

## 4. The Boundary-Tracing Paradigm

The discrete agents (`DQN`, `DuelingDDQN`) **trace each structure's boundary one pixel at a time**. The rasterised polygon IS the segmentation. The U-Net's role is reduced to providing a seed point.

### State `(4, 64, 64)` — local patch centred on current point

| Channel | Content |
|---|---|
| 0 | Image patch |
| 1 | Position marker (1 at patch centre) |
| 2 | Visited mask (cumulative trace footprint) |
| 3 | U-Net init-mask boundary (prior) |

### Action space — 8 directional moves (Freeman chain code)
`N / NE / E / SE / S / SW / W / NW` — one pixel per step.

### Reward structure (region-aware, commit `5901de5`)

```
Per-step (dense):
  -0.01                         step cost (time pressure)
  +0.20 × n_new_covered         new GT-boundary pixels within 1.5px (one-shot per pixel)
  -0.05 × min(dist, 5)          soft distance gradient toward boundary

Terminal (dominant):
  +10.0 × Dice(rasterised_mask, GT)   ← direct objective alignment
  +2.0  × Dice  if closed AND Dice ≥ 0.20   (closure bonus, gated against empty loops)
  -10.0                         off-image penalty
  reward_smoothness_penalty     angular change penalty when |Δθ| > 45°
```

**Why this structure:** the old −distance + flat closure reward produced reward hacking (val Dice 0.018, agent closed minimum-length empty loops). Terminal Dice trains the agent on the exact metric used at evaluation.

### Seed selection

1. Compute `best_overlap_cc(init_mask, gt)` — CC with highest IoU against GT (not largest CC)
2. If `init_Dice < 0.30`, fall back to GT seed (catches catastrophic U-Net failures)
3. Select topmost boundary pixel of the chosen CC (CAMUS) or centroid-nearest pixel (BRISC)

---

## 5. Results to Date

### BRISC DuelingDDQN — 25k steps, region-aware reward

| Metric | Value | U-Net baseline |
|---|---|---|
| Test Dice | **0.806** | 0.835 |
| Test HD95 | **8.76 px** | 8.36 px |
| Closure rate | **87.6%** | — |
| Training time | **24 min** (T4) | — |

Learning curve was still rising at step 20k — 50k steps now configured. Projected final Dice: ~0.84 (at/above baseline).

### BRISC DuelingDDQN — 25k steps, OLD reward (pre-redesign, for reference)

Val Dice peaked at 0.018 (step 15k) then fell. Confirmed reward hacking. Old reward retired.

### CAMUS — not yet run with region-aware reward.

---

## 6. Key Config Values

| Config | train_steps | epsilon_decay | T_max | Notes |
|---|---|---|---|---|
| `brisc_drl_tumor.yaml` DQN/DuelingDDQN | 50k | 25k | 300 | Bumped after 25k run |
| `camus_drl_c1.yaml` (LVendo) | 50k | 30k | 400 | ~1.8×/image coverage |
| `camus_drl_c2.yaml` (LVepi) | 60k | 36k | 480 | Hardest class |
| `camus_drl_c3.yaml` (LA) | 50k | 30k | 420 | Report separately (artificial mitral closure) |
| All DDPG | 100k–120k | — | 20 steps | Mask morphology, unchanged |

All tracing: `num_envs=16`, `batch_size=64`, `embed_dim=128`, `eval_every=5000`, `val_subset=50`.

---

## 7. File Map

| Path | Purpose |
|---|---|
| `iteris/env_contour.py` | `ContourTracingEnv` (reward + dynamics) + `VectorisedContourEnv` |
| `iteris/contour_utils.py` | `DIRECTIONS`, `best_overlap_cc`, `seed_point_from_init_mask`, `gt_boundary_edt`, `rasterise_trajectory` |
| `iteris/drl_training.py` | `_run_contour_training`, `_evaluate_contour`, `ENV_REGISTRY`, `AGENT_REGISTRY` |
| `iteris/contour_viz.py` | Post-training viz (`plot_trace_comparison`, `plot_trajectory_playback`, etc.) |
| `iteris/dryrun_viz.py` | `dryrun_report()` — plumbing health-check after §3 |
| `iteris/agents.py` | `DQNAgent`, `DuelingDQNAgent`, `DDPGAgent` |
| `iteris/drl_networks.py` | `PatchQNetwork`, `PatchDuelingQNetwork` (tracing); `QNetwork`, `DuelingQNetwork` (DDPG fallback) |
| `iteris/buffer.py` | `ReplayBuffer` (DDPG), `ContourReplayBuffer` (tracing, float16 patches) |
| `iteris/env.py` | `SegmentationEnv` — kept for DDPG only |
| `iteris/warm_start.py` | U-Net inference → init masks; supports `tumor_type_filter` for BRISC |
| `iteris/ingestion.py` | Dataset builders; BRISC parses `tumor_type` from filename |
| `iteris/archive/` | Retired code with resurrection instructions |
| `configs/camus_drl_c{1,2,3}.yaml` | CAMUS per-class DRL configs |
| `configs/brisc_drl_tumor.yaml` | BRISC binary DRL config (primary) |
| `configs/brisc_drl_{glioma,meningioma,pituitary}.yaml` | Optional per-type BRISC configs |
| `notebooks/03{a,b,c}_camus_drl_*.ipynb` | CAMUS per-class training notebooks |
| `notebooks/04_brisc_drl.ipynb` | BRISC training notebook |

---

## 8. Settled Decisions (not for re-litigation)

| Decision | Rationale |
|---|---|
| DDPG kept as mask-morph baseline | User explicit: "keep DDPG as it is." Continuous comparison for the paper. |
| 8-direction discrete action | Validated by Edge-Sensitive LV + LV Contouring papers (Yin 2021, Mayy 2021). |
| Terminal Dice as dominant reward | Trains on exactly the eval metric. Empirically resolved reward hacking. |
| One-shot coverage invariant | Provably bounds coverage reward; can't be farmed. |
| `DQN`, `DuelingDDQN`, `DDPG` selectors (no `_TRACE` suffix, no DDQN) | User direction. |
| Dead code archived, not deleted | User direction. |
| Best-overlap CC seed (uses GT) | Curriculum aid; documented as paper limitation (Section 9). |
| BRISC is binary tumor segmentation | Active runs pool all tumor types. Per-type configs available for optional ablation. |

---

## 9. Known Limitations (paper disclosures)

1. GT used at training time for seed selection and terminal reward. Not deployment-grade without retraining the seed selector against a U-Net confidence map.
2. LA mitral-valve closure is artificial — report LA separately.
3. Multi-focal lesions (BRISC) not supported — largest-overlap CC only.
4. No on-policy comparison (PPO) — out of scope.
5. Reward magnitudes tuned heuristically — systematic grid search is future work.

---

## 10. Decision Log

| Date | Decision |
|---|---|
| 2026-05-29 | Paradigm 1 (boundary tracing) selected; refinement kept as DDPG-only paper baseline |
| 2026-05-30 | Discrete refinement (SegmentationEnvBRISC, DDQN) archived; tracing is the only discrete paradigm |
| 2026-05-30 | Selector set: `DQN`, `DuelingDDQN`, `DDPG` |
| 2026-05-30 | Seed Option A: best-overlap CC + GT fallback at init_Dice < 0.30 |
| 2026-05-30 | Region-aware reward redesign after BRISC reward-hacking confirmed empirically |
| 2026-05-31 | BRISC confirmed as brain MRI (T1+Gd); z-score normalisation already correct; per-type configs added |
| 2026-06-01 | BRISC T_max 200→300; centroid seed for BRISC; smoothness penalty added (Phase B complete) |
