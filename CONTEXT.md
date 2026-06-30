# Iteris — Project Context

> Single source of truth. **Last updated:** 2026-06-30 · **Semester week:** ~11 of 14.

---

## 1. What This Is

**Iteris** is a Taylor's University capstone (PRJ63504) research paper on DRL-based medical-image segmentation via **contour refinement**.

- **Research question:** Can a DRL agent refine a U-Net segmentation mask by deforming its boundary contour, and does a **discrete** agent (DuelingDDQN) or a **continuous** agent (TD3) recover more of the gap toward a strong attention-U-Net baseline — and does that answer change with how much labelled data is available, or with a richer agent backbone? (See §3, Experimental Phases.)
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

## 3. Experimental Phases — A / B / C (data regime × architecture)

> **Naming note:** "Phase A/B/C" here is the *experimental design* axis (what data/architecture each run uses). This is a different thing from the *project timeline* stages (build → train → evaluate → write paper) — see SKILLS.md §4, which uses "Stage 1–4" to avoid colliding with these letters.

The research design runs the **same U-Net-vs-DRL comparison three times**, at decreasing data scale and increasing agent sophistication, so the paper can answer not just "does DRL refinement help" but "when does it help most."

| Phase | Dataset scope | Models compared | Question | Status |
|---|---|---|---|---|
| **A — Full data** | Whole dataset (CAMUS: all 500 patients / 2000 images; BRISC: full pooled set). `label_frac: 1.0`. | Lite U-Net · Attention U-Net · DRL-refined lite U-Net (DuelingDDQN + TD3) | Baseline comparison: how much of the lite→attention gap does contour-refinement DRL close when data is not a constraint? | ⏳ in progress (this is the current/default track — all DRL configs as of 2026-06-30) |
| **B — Low-data regime** | Small subset of the same dataset, simulating limited label availability. Implemented via the existing `label_frac` knob in `camus.yaml`/`camus_lite.yaml`/`brisc*.yaml` (`label_frac: 0.1` / `0.2`, already wired for few-shot ablations — **no new code needed**, just a config value + a fresh baseline/DRL training pass). | Same trio (Lite U-Net · Attention U-Net · DRL-refined lite U-Net), retrained from scratch on the smaller subset. | Does DRL's relative advantage over supervised baselines *grow* when labelled data is scarce? (Literature motivation: cold-start DRL segmentation beats supervised CNNs by a wide margin in small-data regimes — see the project's literature review, Verdict 2 — this phase tests whether the same effect holds for *warm-started contour refinement*, not just cold-start pixel RL.) | ⬜ not started |
| **C — MSA backbone adaptation** | An even smaller subset than Phase B (tightest data regime). | The best-performing DRL agent/config from Phases A/B, with its CNN backbone swapped for the **MSA (Multi-Head Self-Attention) backbone** — currently archived at `iteris/archive/msa.py` (`MSABackbone`, `MSADuelingQNetwork`; per its own archive note, "the planned 'run on the best base later' variant," not wired into any config/notebook/`AGENT_REGISTRY` entry right now). | Does giving the agent explicit cross-position spatial reasoning (self-attention over the 8×8 feature-map tokens, vs. CNN global-average-pool) help it learn faster / generalize better specifically in the most data-starved regime? | ⬜ not started — requires un-archiving `msa.py` + `agents_legacy.py::MSADuelingDQNAgent`, re-adding an `'MSA-DUELING'` (or `'MSA-TD3'`) `AGENT_REGISTRY` entry, and a config/notebook selector |

**Implementation notes carried over from existing infra (don't rebuild what's already there):**
- `label_frac` is already a config field (`configs/CAMUS/camus.yaml`, `camus_lite.yaml`, and the BRISC equivalents) with the comment `# 0.1 / 0.2 / 1.0 for few-shot ablations` — Phase B is mostly a matter of **using** this knob, not adding it.
- Patient-level splitting (`splits.py`) must stay patient-level when subsetting for Phase B/C, so the smaller train set still has disjoint val/test patients — don't naively slice the existing full split's image list.
- Phase C's MSA resurrection should target whichever of {DuelingDDQN, TD3} × {CAMUS class, BRISC class} came out strongest in Phase A/B — don't run MSA across the full agent×class matrix, that's not the point of Phase C (architecture ablation on the best base, not a full re-sweep).

---

## 4. Datasets & Baselines

RL **warm-starts from the lite U-Net**; the attention U-Net is the competitor.

| Dataset | Classes | Lite U-Net (RL start) | Attention U-Net (competitor) |
|---|---|---|---|
| **CAMUS** (cardiac US) | LV-endo (c1), LV-epi (c2), LA (c3) | 0.888 / 0.797 / 0.828 Dice (val, 2026-06-29 diagnostic) | 0.938 / 0.872 / 0.896 Dice |
| **BRISC** (brain tumour MRI) | tumor (binary; +glioma/meningioma/pituitary) | 0.836 Dice (val) | 0.835 Dice (test) |

Lite U-Net = `LiteUNet` (plain U-Net, no attention, ~0.48M params vs ~33M). Intentionally weaker so its errors are **systematic** (smooth over/under-segmentation) — exactly what contour refinement can correct.

**Confirmed realistic (non-GT-privileged) headroom, full-data (Phase A) regime** — attention Dice minus lite-baseline Dice, the honest ceiling DRL is chasing: LV_endo +0.050, LV_epi +0.073, LA +0.052. BRISC tumor (pooled, general class): **−0.0501 — no headroom**, lite baseline already exceeds the attention competitor on this class. BRISC subtypes (glioma/meningioma/pituitary): not yet individually diagnosed.

---

## 5. Active Agent Set (two algorithms, Phase A/B; +MSA variant in Phase C)

| Selector | Algorithm | Action space | Env |
|---|---|---|---|
| `DuelingDDQN` | Dueling **Double** DQN (V+A heads) | 18 discrete sector pushes | `ContourRefineEnv` |
| `TD3` | Twin Delayed DDPG (twin critics, target smoothing, delayed updates) | continuous per-sector displacement | `ContourRefineEnv` |

Archived (ablation only, `*_GLOBAL` / `DQN` / `DDPG` config blocks): global-morphology DQN / DuelingDDQN / DDPG on `SegmentationEnv`. Also archived, **newly relevant for Phase C**: `MSABackbone`/`MSADuelingQNetwork` (`iteris/archive/msa.py`) — see §3.

---

## 6. Reward — baseline-centred PBRS

Potential-based shaping (Ng et al. 1999): `r_t = γ·Φ(s_{t+1}) − Φ(s_t) − step_penalty`, with the potential **centred at the episode baseline and scaled**:

- `Φ(s) = K·(Dice(s) − Dice_0)` (`dice_pbrs`), or
- `Φ(s) = K·[α·(Dice − Dice_0) + β·(hd_term − hd_term_0)]` (`dice_hd_pbrs`).

Centring (`Φ_0 = 0`) removes the discount drag that made an un-centred `Φ=Dice≈0.94` punish every step; `K` (10 CAMUS / 15 BRISC) lifts the tiny per-step deltas above the Q-net noise floor. Telescopes to `γ^T·Φ_T` → maximise the final mask's improvement over baseline, soonest.

**DuelingDDQN-specific addition (2026-06-29):** a small `reward_step_penalty` (0.05) makes STOP the only penalty-free action, so the agent learns to stop once further edits stop paying for themselves — see SKILLS.md §3 for the full mechanism and why it doesn't apply to TD3.

**State** `(5, 256, 256)`: image · current mask · SDT(mask) · U-Net init mask · U-Net prob-map.

---

## 7. File Map

| Path | Purpose |
|---|---|
| `iteris/geometry.py` | Shared helpers: `dice_score`, `hd95_px`, `signed_dt`, SEs, + eval-only `iou_score`/`precision_recall`/`boundary_iou`/`mean_surface_distance_px` |
| `iteris/env_contour_refine.py` | `ContourRefineEnv` — control-point + spline contour env (the live paradigm) |
| `iteris/agents.py` | `DuelingDQNAgent`, `TD3Agent` (+ archived `DQNAgent`, `DDPGAgent`) |
| `iteris/drl_networks.py` | `DuelingQNetwork`, `Actor`, `Critic` (5-channel input) |
| `iteris/buffer.py` | `ReplayBuffer` (memory-optimised, SDT-caching) |
| `iteris/drl_training.py` | `run_drl_training` — main loop; `AGENT_REGISTRY`, `ENV_REGISTRY`; curriculum max_steps, milestone checkpoints |
| `iteris/diagnostics.py` | `headroom_report` — oracle contour ceiling vs baseline (go/no-go) |
| `iteris/models.py` | `AttentionResUNet` (competitor) + `LiteUNet` (RL baseline); `build_model` |
| `iteris/warm_start.py` | U-Net inference → init masks + prob-maps |
| `iteris/refinement_viz.py` | Replays / comparison / playback / behaviour / test eval (discrete + continuous); init+final literature-standard metrics |
| `iteris/archive/paradigm_a/` | Global-morphology `SegmentationEnv` — ablation only |
| `iteris/archive/paradigm1_boundary_tracing/` | Retired boundary-tracing paradigm |
| `iteris/archive/msa.py` | **Phase C target** — archived MSA (Multi-Head Self-Attention) backbone, not currently wired in |
| `configs/CAMUS/camus.yaml`, `configs/BRISC/brisc*.yaml` | Lite/attention baseline training configs; `label_frac` is the Phase B/C lever |
| `configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml`, `configs/BRISC/DRL/brisc_drl_*.yaml` | DRL configs (DuelingDDQN + TD3 active; `*_GLOBAL`/DQN/DDPG = ablation) |
| `notebooks/unet/01_camus_lite.ipynb`, `unet/02_brisc_lite.ipynb` | Train the **lite** baselines (RL warm-start) |
| `notebooks/unet/03_camus_attnunet.ipynb`, `unet/04_brisc_attnunet.ipynb` | Train the **attention** baselines (competitor) |
| `notebooks/camus/drl/03{a,b,c}_camus_drl_*.ipynb`, `notebooks/brisc/drl/04_brisc_drl.ipynb` | DRL training (Kaggle) |
| `notebooks/local/local_{camus,brisc}_drl.ipynb` | DRL training (local fyp_env GPU) |
| `notebooks/unet/00_free_diagnostics_all_classes.ipynb` | Headroom + prob_map informativeness diagnostic — run before any GPU round |

---

## 8. Current Status — Week ~11 of 14

| Item | Status |
|---|---|
| `ContourRefineEnv` (discrete + continuous, angular sectors) | ✅ |
| DuelingDDQN + TD3 agents | ✅ implemented, fully smoke-tested incl. continuous+BC+curriculum |
| Baseline-centred PBRS reward + 5-channel state | ✅ |
| Lite U-Net model + configs (`LiteUNet`, `*_lite.yaml`) | ✅ trained (CAMUS label-smoothing retrain done; prob_map still INERT for LV_endo/LA — see SKILLS.md) |
| Ceiling / headroom diagnostic | ✅ run on real data — confirmed positive headroom, all 3 CAMUS classes; confirmed negative for BRISC tumor (pooled) |
| Literature-standard metrics (IoU/Precision/Sensitivity/BIoU/MSD, init+final) | ✅ |
| DuelingDDQN STOP-learning fix (`reward_step_penalty` + curriculum `max_steps`) | ✅ landed, CAMUS + BRISC (all 4 BRISC classes) |
| TD3 `fail_thresh`/`fail_n` safety net + curriculum | ✅ landed, CAMUS + BRISC (all 4 BRISC classes) |
| **Phase A** (full-data DuelingDDQN + TD3 runs) | ⏳ in progress |
| **Phase B** (low-data-regime comparison) | ⬜ not started |
| **Phase C** (MSA backbone adaptation) | ⬜ not started |
| Evaluation + paper | ⬜ Weeks 12–14 |

---

## 9. Settled Decisions

| Decision | Rationale |
|---|---|
| Two algorithms only in Phase A/B: DuelingDDQN (discrete) + TD3 (continuous) | Clean discrete-vs-continuous comparison; both on the contour env |
| Three-phase experimental design (A: full data, B: low data, C: MSA backbone) | Answers not just "does DRL help" but "when" — data-scarcity and architecture as the two axes most likely to move the result, per literature review |
| Contour refinement is the only live paradigm | Global morphology is capped at baseline (confirmed on real runs) |
| RL refines the **lite** U-Net, not the attention net | Strong baseline has no headroom → RL cannot improve it |
| Baseline-centred scaled PBRS reward | Removes discount drag + path-dependence; lifts tiny signal above noise |
| TD3 over plain DDPG | Twin critics + target smoothing fix the overestimation that stalls DDPG |
| Dueling Double DQN over plain DQN | Dueling fits "many near-equal actions" refinement; Double kills overestimation |
| `reward_step_penalty` (DuelingDDQN only, never TD3) | Makes STOP the unique penalty-free action so the agent stops instead of overshooting; would push TD3's actor toward identity instead (no STOP action to be penalty-free relative to) |
| Curriculum `max_steps` (training-only, both agents) | GT-based difficulty can only be used where GT is available (training); deploy/eval always use the fixed `max_steps` |
| Run `headroom_report` before each full RL run | Cheap go/no-go: confirms reachable Dice > baseline before spending GPU |
| Paradigm A kept (archived) | Negative-control ablation for the paper |

---

## 10. Decision Log

| Date | Decision |
|---|---|
| 2026-06-02 | Pivoted from boundary tracing to mask refinement; `SegmentationEnv` v4 (global morph) |
| 2026-06-15 | Diagnosed reward path-dependence + discount drag; implemented **baseline-centred PBRS** + 5th (prob-map) state channel |
| 2026-06-18 | Added **TD3** on an **angular-sector contour** action; TD3 over DDPG (robustness) |
| 2026-06-20 | Real CAMUS/BRISC runs: global-morph + strong-baseline agents **fail to beat baseline** (best-seen ≈ baseline) → confirmed structural ceiling |
| 2026-06-21 | Strategy: RL refines a **lite U-Net** (headroom) vs **attention U-Net** competitor; added `LiteUNet` + `*_lite.yaml` + `headroom_report` diagnostic |
| 2026-06-21 | **Archived Paradigm A** (global morph → `archive/paradigm_a/`); extracted shared helpers to `geometry.py`; `env.py` is now a back-compat shim |
| 2026-06-21 | Renamed baselines 01/02 → lite, 03/04 → attention; **canonicalised configs/notebooks to DuelingDDQN + TD3 only** (global agents → `*_GLOBAL` ablation) |
| 2026-06-25 | Per-class diagnostics confirmed real headroom on all 3 CAMUS classes; BRISC tumor (pooled) confirmed **no headroom** (−0.0501) → BRISC paused at the time |
| 2026-06-26 | CAMUS lite U-Net retrained with label smoothing (`loss_label_smoothing: 0.1`) to address INERT prob_maps blocking the uncertainty gate |
| 2026-06-29 | Diagnosed DuelingDDQN's 0%-STOP-rate pathology; fixed via `reward_step_penalty` + training-only curriculum `max_steps`; extended fix to BRISC (all 4 classes); extended TD3 with curriculum + safety net (CAMUS, then BRISC) |
| 2026-06-30 | Re-tested CAMUS LV_endo/LA prob_map after a stronger retrain (`loss_label_smoothing: 0.25`) — **still INERT** (Dice-loss term likely cancels CE-only smoothing); `uncertainty_gate` re-disabled for those classes on both agents. Fixed a real bug where `refinement_env_kwargs` never carried `action_type`, crashing TD3's post-training replay/eval. Tightened BRISC TD3 `fail_n` 2→1 (tiny-target volatility breaches the safety net before a 2-step confirmation can catch it). Defined the three-phase experimental design (§3). |
