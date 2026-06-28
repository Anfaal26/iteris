# Iteris — Project Context

> Single source of truth. **Last updated:** 2026-06-21 · **Semester week:** 10 of 14.

---

## 1. What This Is

**Iteris** is a Taylor's University capstone (PRJ63504) research paper on DRL-based medical-image segmentation via **contour refinement**.

- **Research question:** Can a DRL agent refine a U-Net segmentation mask by deforming its boundary contour, and does a **discrete** agent (DuelingDDQN) or a **continuous** agent (TD3) recover more of the gap toward a strong attention-U-Net baseline?
- **Design:** RL refines a deliberately **lightweight** U-Net (real headroom); the heavyweight **attention U-Net** is the upper-bound competitor.
- **Paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop
- **Repo:** `github.com/Anfaal26/iteris` · local: `D:\iteris\`

---

## 2. Paradigm — Contour Refinement (only)

The agent starts from a U-Net mask and deforms its **boundary contour** locally. The boundary is `N` ordered control points on a closed periodic spline; the agent pushes contiguous **angular sectors** of the contour along their outward normals.

- **DuelingDDQN (discrete):** 18 actions = push each of 8 angular sectors OUT/IN + smooth + stop.
- **TD3 (continuous):** per-sector signed displacement, `cont_sectors` wedges, each in `[-1,1]·disp_px`.

Angular (not index-based) sectoring keeps each action tied to a fixed spatial direction across samples → learnable.

**Why contour, not global morphology:** global dilate/erode/shift cannot fix masks that are over-segmented in one region and under-segmented in another → capped at baseline. Global morphology (the old `SegmentationEnv`) is **archived** (`iteris/archive/paradigm_a/`) and kept only as a negative-control ablation. Pixel-by-pixel boundary tracing was retired earlier (`iteris/archive/paradigm1_boundary_tracing/`).

---

## 3. Datasets & Baselines

RL **warm-starts from the lite U-Net**; the attention U-Net is the competitor.

| Dataset | Classes | Lite U-Net (RL start) | Attention U-Net (competitor) |
|---|---|---|---|
| **CAMUS** (cardiac US) | LV-endo (c1), LV-epi (c2), LA (c3) | ⏳ train (target ~0.80–0.85) | 0.938 / 0.872 / 0.896 Dice |
| **BRISC** (brain tumour MRI) | tumor (binary; +glioma/meningioma/pituitary) | ⏳ train (target ~0.78–0.82) | 0.835 Dice (test) |

Lite U-Net = `LiteUNet` (plain U-Net, no attention, ~0.48M params vs ~33M). Intentionally weaker so its errors are **systematic** (smooth over/under-segmentation) — exactly what contour refinement can correct.

---

## 4. Active Agent Set (two algorithms)

| Selector | Algorithm | Action space | Env |
|---|---|---|---|
| `DuelingDDQN` | Dueling **Double** DQN (V+A heads) | 18 discrete sector pushes | `ContourRefineEnv` |
| `TD3` | Twin Delayed DDPG (twin critics, target smoothing, delayed updates) | continuous per-sector displacement | `ContourRefineEnv` |

Archived (ablation only, `*_GLOBAL` / `DQN` / `DDPG` config blocks): global-morphology DQN / DuelingDDQN / DDPG on `SegmentationEnv`.

---

## 5. Reward — baseline-centred PBRS

Potential-based shaping (Ng et al. 1999): `r_t = γ·Φ(s_{t+1}) − Φ(s_t) − step_penalty`, with the potential **centred at the episode baseline and scaled**:

- `Φ(s) = K·(Dice(s) − Dice_0)` (`dice_pbrs`), or
- `Φ(s) = K·[α·(Dice − Dice_0) + β·(hd_term − hd_term_0)]` (`dice_hd_pbrs`).

Centring (`Φ_0 = 0`) removes the discount drag that made an un-centred `Φ=Dice≈0.94` punish every step; `K` (10 CAMUS / 15 BRISC) lifts the tiny per-step deltas above the Q-net noise floor. Telescopes to `γ^T·Φ_T` → maximise the final mask's improvement over baseline, soonest. STOP commits the contour (reward 0).

**State** `(5, 256, 256)`: image · current mask · SDT(mask) · U-Net init mask · U-Net prob-map.

---

## 6. File Map

| Path | Purpose |
|---|---|
| `iteris/geometry.py` | Shared helpers: `dice_score`, `hd95_px`, `signed_dt`, SEs |
| `iteris/env_contour_refine.py` | `ContourRefineEnv` — control-point + spline contour env (the live paradigm) |
| `iteris/agents.py` | `DuelingDQNAgent`, `TD3Agent` (+ archived `DQNAgent`, `DDPGAgent`) |
| `iteris/drl_networks.py` | `DuelingQNetwork`, `Actor`, `Critic` (5-channel input) |
| `iteris/buffer.py` | `ReplayBuffer` (memory-optimised, SDT-caching) |
| `iteris/drl_training.py` | `run_drl_training` — main loop; `AGENT_REGISTRY`, `ENV_REGISTRY` |
| `iteris/diagnostics.py` | `headroom_report` — oracle contour ceiling vs baseline (go/no-go) |
| `iteris/models.py` | `AttentionResUNet` (competitor) + `LiteUNet` (RL baseline); `build_model` |
| `iteris/warm_start.py` | U-Net inference → init masks + prob-maps |
| `iteris/refinement_viz.py` | Replays / comparison / playback / behaviour / test eval (discrete + continuous) |
| `iteris/archive/paradigm_a/` | Global-morphology `SegmentationEnv` — ablation only |
| `iteris/archive/paradigm1_boundary_tracing/` | Retired boundary-tracing paradigm |
| `configs/CAMUS/camus_lite.yaml`, `configs/BRISC/brisc_lite.yaml` | Lite-baseline training configs (`model: lite_unet`) |
| `configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml`, `configs/BRISC/DRL/brisc_drl_*.yaml` | DRL configs (DuelingDDQN + TD3 active; `*_GLOBAL`/DQN/DDPG = ablation) |
| `notebooks/unet/01_camus_lite.ipynb`, `unet/02_brisc_lite.ipynb` | Train the **lite** baselines (RL warm-start) |
| `notebooks/unet/03_camus_attnunet.ipynb`, `unet/04_brisc_attnunet.ipynb` | Train the **attention** baselines (competitor) |
| `notebooks/camus/drl/03{a,b,c}_camus_drl_*.ipynb`, `notebooks/brisc/drl/04_brisc_drl.ipynb` | DRL training (Kaggle) |
| `notebooks/local/local_{camus,brisc}_drl.ipynb` | DRL training (local fyp_env GPU) |

---

## 7. Current Status — Week 10 of 14

| Item | Status |
|---|---|
| `ContourRefineEnv` (discrete + continuous, angular sectors) | ✅ |
| DuelingDDQN + TD3 agents | ✅ implemented, smoke-tested |
| Baseline-centred PBRS reward + 5-channel state | ✅ |
| Lite U-Net model + configs (`LiteUNet`, `*_lite.yaml`) | ✅ |
| Ceiling diagnostic (`headroom_report`) | ✅ verified on synthetic (+0.097) |
| Paradigm A (global morph) archived | ✅ `archive/paradigm_a/` |
| Notebooks/configs cleaned to DuelingDDQN + TD3 only | ✅ |
| Lite U-Net baselines trained (CAMUS, BRISC) | ⏳ this week |
| DRL runs (DuelingDDQN + TD3 × CAMUS c1/c2/c3 + BRISC) | ⏳ this week |
| Evaluation + paper | ⬜ Weeks 12–14 |

---

## 8. Settled Decisions

| Decision | Rationale |
|---|---|
| Two algorithms only: DuelingDDQN (discrete) + TD3 (continuous) | Clean discrete-vs-continuous comparison; both on the contour env |
| Contour refinement is the only live paradigm | Global morphology is capped at baseline (confirmed on real runs) |
| RL refines the **lite** U-Net, not the attention net | Strong baseline has no headroom → RL cannot improve it |
| Baseline-centred scaled PBRS reward | Removes discount drag + path-dependence; lifts tiny signal above noise |
| TD3 over plain DDPG | Twin critics + target smoothing fix the overestimation that stalls DDPG |
| Dueling Double DQN over plain DQN | Dueling fits "many near-equal actions" refinement; Double kills overestimation |
| Run `headroom_report` before each full RL run | Cheap go/no-go: confirms reachable Dice > baseline before spending GPU |
| Paradigm A kept (archived) | Negative-control ablation for the paper |

---

## 9. Decision Log

| Date | Decision |
|---|---|
| 2026-06-02 | Pivoted from boundary tracing to mask refinement; `SegmentationEnv` v4 (global morph) |
| 2026-06-15 | Diagnosed reward path-dependence + discount drag; implemented **baseline-centred PBRS** + 5th (prob-map) state channel |
| 2026-06-18 | Added **TD3** on an **angular-sector contour** action; TD3 over DDPG (robustness) |
| 2026-06-20 | Real CAMUS/BRISC runs: global-morph + strong-baseline agents **fail to beat baseline** (best-seen ≈ baseline) → confirmed structural ceiling |
| 2026-06-21 | Strategy: RL refines a **lite U-Net** (headroom) vs **attention U-Net** competitor; added `LiteUNet` + `*_lite.yaml` + `headroom_report` diagnostic |
| 2026-06-21 | **Archived Paradigm A** (global morph → `archive_paradigm_a/`); extracted shared helpers to `geometry.py`; `env.py` is now a back-compat shim |
| 2026-06-21 | Renamed baselines 01/02 → lite, 03/04 → attention; **canonicalised configs/notebooks to DuelingDDQN + TD3 only** (global agents → `*_GLOBAL` ablation) |
