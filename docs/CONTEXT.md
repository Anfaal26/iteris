# Iteris — Project Context

> Single source of truth. **Last updated:** 2026-07-21 · **Semester week:** ~14 of 14 (final week).
> §1/§3/§4/§5/§9 corrected 2026-07-21 — a documentation audit found they still described the
> pre-2026-07-14 lite-vs-attention design (RL refining `LiteUNet`) after the project had already
> moved to refining `AttentionResUNet` directly (single-baseline design), and still listed
> **Phase C** (MSA backbone) as an open/planned item after it was formally abandoned. Both are
> fixed throughout this doc now; see §10 for the dated entries. §6/§10 were previously rewritten
> 2026-07-16 (reward-system sync) — still current, not touched by this pass.

---

## 1. What This Is

**Iteris** is a Taylor's University capstone (PRJ63504) research paper on DRL-based medical-image segmentation via **contour refinement**.

- **Research question:** Can a DRL agent refine a U-Net segmentation mask by deforming its boundary contour, and does a **discrete** agent (DuelingDDQN) or a **continuous** agent (TD3) recover more Dice — and does that answer change with how much labelled data is available? (See §3, Experimental Phases. A third axis — a richer, self-attention agent backbone — was originally planned as Phase C; **abandoned**, see §3.)
- **Design:** RL refines the **AttentionResUNet** baseline directly (single-baseline design, confirmed current as of `61097d0`, 2026-07-14 — see §9). A separate, deliberately **lightweight** `LiteUNet` is trained per phase alongside it purely as a weaker comparison point (architecture-headroom reference) — it is **not** an RL warm-start target.
- **Paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop
- **Repo:** `github.com/Anfaal26/iteris` · local: `D:\iteris\`
- **Datasets (only two, no others in scope):** CAMUS (cardiac ultrasound: LV_endo/LV_epi/LA) and BRISC (brain-tumour MRI: glioma/meningioma/pituitary/tumor-generic). Earlier exploratory baselines on HAM10000 (skin lesions), Kvasir-SEG, and DRIVE (retinal vessels) were built in May 2026 (`cf6cd6d`, `07edd34`) and archived/dropped well before the DRL/contour-refinement design existed — they are not part of the current project and aren't referenced elsewhere in this doc.

---

## 2. Paradigm — Contour Refinement (only)

The agent starts from a U-Net mask and deforms its **boundary contour** locally. The boundary is `N` ordered control points on a closed periodic spline; the agent pushes contiguous **angular sectors** of the contour along their outward normals.

- **DuelingDDQN (discrete):** 18 actions = push each of 8 angular sectors OUT/IN + smooth + stop.
- **TD3 (continuous):** per-sector signed displacement, `cont_sectors` wedges, each in `[-1,1]·disp_px`.

Angular (not index-based) sectoring keeps each action tied to a fixed spatial direction across samples → learnable.

**Why contour, not global morphology:** global dilate/erode/shift cannot fix masks that are over-segmented in one region and under-segmented in another → capped at baseline. Global morphology (the old `SegmentationEnv`) is **archived** (`iteris/archive/paradigm_a/`) and kept only as a negative-control ablation. Pixel-by-pixel boundary tracing was retired earlier (`iteris/archive/paradigm1_boundary_tracing/`).

---

## 3. Experimental Phases — A / B, data regime (Phase C abandoned)

> **Naming note:** "Phase A/B" here is the *experimental design* axis (what data regime each run uses). This is a different thing from the *project timeline* stages (build → train → evaluate → write paper) — see SKILLS.md §4, which uses "Stage 1–4" to avoid colliding with these letters.

The research design runs the **same U-Net-vs-DRL comparison at two data scales**, so the paper can answer not just "does DRL refinement help" but "does it help more when labelled data is scarce."

> A third phase was originally planned — see the "Phase C (abandoned)" row below — but was **dropped as a project decision (2026-07-21)** and never implemented. **Phase A and Phase B are the only phases that exist anywhere in this project**: the config, the checkpoints, the evaluation notebook, and the deployed UI all only ever handle these two.

> **Concrete data sizes, methodology, and the literature behind them live in [EXPERIMENTS.md](EXPERIMENTS.md).** Short version: Phase B uses the existing patient-level `label_frac` knob (train-only shrink; val/test stay full) at CAMUS `0.10` and BRISC `0.05` (≈150 training images each). **Both** the `AttentionResUNet` (the model DRL actually refines) **and** the separate `LiteUNet` comparison baseline are retrained on the same subset (leak-free by construction — identical `seed`+`label_frac`). A `label_frac < 1.0` auto-namespaces every artifact via `utils.model_suffix` (`_lf10` etc.), so running a phase is a config flip, not new code.

| Phase | Dataset scope | Models compared | Question | Status |
|---|---|---|---|---|
| **A — Full data** | Whole dataset (CAMUS: all 500 patients / 2000 images; BRISC: full pooled set). `label_frac: 1.0`. | AttentionResUNet · LiteUNet (comparison only) · DRL-refined AttentionResUNet (DuelingDDQN + TD3) | Baseline comparison: how much Dice does contour-refinement DRL recover over the deployed U-Net when data is not a constraint? | ✅ complete — all 8 (dataset×class×algo) runs done, wired into the deployed UI (§8) |
| **B — Low-data regime** | Small subset of the same dataset, simulating limited label availability, via the `label_frac` knob (`camus.yaml`/`camus_lite.yaml`/`brisc*.yaml`, `label_frac: 0.10`/`0.05`). | Same trio, retrained from scratch on the smaller subset. | Does DRL's relative advantage over the supervised baseline *grow* when labelled data is scarce? (Literature motivation: cold-start DRL segmentation beats supervised CNNs by a wide margin in small-data regimes — see the project's literature review, Verdict 2.) | ✅ complete — all 8 runs done, wired into the deployed UI (§8, `4825754`) |
| **C — MSA backbone adaptation** *(abandoned, historical only)* | Was planned: an even smaller subset than Phase B, on the best-performing Phase A/B agent, with its CNN backbone swapped for the archived **MSA (Multi-Head Self-Attention) backbone** (`iteris/archive/msa.py`). | *(never run)* | Was meant to test whether explicit cross-position spatial reasoning helps most in the most data-starved regime. | ❌ **abandoned (2026-07-21)** — a project decision, not a scheduling slip. `msa.py`/`agents_legacy.py::MSADuelingDQNAgent` remain archived and un-wired; no config, notebook, `AGENT_REGISTRY` entry, checkpoint, or evaluation-notebook logic for Phase C exists or will be built. Do not resurrect this without a fresh decision — see [EXPERIMENTS.md](EXPERIMENTS.md) §4 for the historical rationale it was scoped against. |

**Implementation notes carried over from existing infra (don't rebuild what's already there):**
- `label_frac` is already a config field (`configs/CAMUS/camus.yaml`, `camus_lite.yaml`, and the BRISC equivalents) with the comment `# 0.1 / 0.2 / 1.0 for few-shot ablations` — Phase B was mostly a matter of **using** this knob, not adding it.
- Patient-level splitting (`splits.py`) stays patient-level when subsetting for Phase B, so the smaller train set still has disjoint val/test patients — never naively slice the existing full split's image list.

---

## 4. Datasets & Baselines

Only two datasets are in scope, both listed below — no others (earlier May-2026 exploratory
work on HAM10000/Kvasir-SEG/DRIVE was archived/dropped before the DRL design existed; see §1).

**RL refines the AttentionResUNet baseline directly** (single-baseline design). `LiteUNet` is
trained per phase alongside it purely as a **weaker comparison baseline** — historically (before
the pivot to the single-baseline design, prior to 2026-07-14) it was the RL warm-start target and
`AttentionResUNet` was the fixed competitor; that arrangement is reversed now. See §9 for why.

| Dataset | Classes | LiteUNet (comparison baseline) | AttentionResUNet (RL warm-start target) |
|---|---|---|---|
| **CAMUS** (cardiac US) | LV-endo (c1), LV-epi (c2), LA (c3) | 0.888 / 0.797 / 0.828 Dice (val, 2026-06-29 diagnostic) | 0.938 / 0.872 / 0.896 Dice |
| **BRISC** (brain tumour MRI) | tumor (binary; +glioma/meningioma/pituitary) | 0.836 Dice (val) | 0.835 Dice (test) |

The historical Dice figures above predate the single-baseline pivot and are kept for the
architecture-headroom framing below (which is still a useful diagnostic even though DRL no longer
targets the weaker net) — for the actual DRL-vs-baseline results, see the evaluation notebook's
output (`notebooks/evaluation/`) and the Research page.

**Historical framing (architecture headroom, pre-pivot design)** — attention Dice minus lite-baseline
Dice, i.e. how much a heavier architecture alone buys over the lighter one: LV_endo +0.050, LV_epi
+0.073, LA +0.052. BRISC tumor (pooled, general class): **−0.0501**, the lite baseline already
exceeds the attention net on this class. BRISC subtypes (glioma/meningioma/pituitary): not
individually diagnosed. This number is no longer "the ceiling DRL is chasing" (that framing assumed
the old lite-RL-target design) — it's retained as an architecture-comparison data point only.

---

## 5. Active Agent Set (two algorithms — the only ones used, Phase A and Phase B alike)

| Selector | Algorithm | Action space | Env |
|---|---|---|---|
| `DuelingDDQN` | Dueling **Double** DQN (V+A heads) | 18 discrete sector pushes | `ContourRefineEnv` |
| `TD3` | Twin Delayed DDPG (twin critics, target smoothing, delayed updates) | continuous per-sector displacement | `ContourRefineEnv` |

Archived (ablation only, `*_GLOBAL` / `DQN` / `DDPG` config blocks): global-morphology DQN / DuelingDDQN / DDPG on `SegmentationEnv`. Also archived, **and staying that way**: `MSABackbone`/`MSADuelingQNetwork` (`iteris/archive/msa.py`) — this was scoped for the now-abandoned Phase C (§3) and is not wired into any config/notebook/`AGENT_REGISTRY` entry.

---

## 6. Reward — `contour_boundary` + optimal-stopping bonus (superseded PBRS 2026-07-08)

**This section describes the system actually in force.** PBRS (`dice_pbrs` /
`dice_hd_pbrs`, documented below for reference) was the design through
2026-06-30 but is no longer what any current notebook runs — every PhaseA/B
notebook calls `apply_refinement_config()` (`iteris/config.py`) right after
`resolve_agent_config()`, and that call **overrides** each class YAML's
`reward_mode` (and several other fields) unconditionally. The per-agent YAML
blocks still show `reward_mode: dice_pbrs` / `dice_hd_pbrs` and
`terminal_bonus_scale: 0.0` — those values are dead for any run that goes
through `apply_refinement_config` (which is all of them); do not read the raw
YAML as ground truth for what trained a checkpoint. `config.py`'s
`_REFINE_SHARED` dict is the actual source of truth.

**Reward — `contour_boundary` (dense, per-control-point):**
`raw_reward = (prev_mean_dist_to_GT_boundary − new_mean_dist_to_GT_boundary) − step_penalty`,
i.e. the reduction in mean Euclidean distance from the N control points to the
GT boundary this step. High-SNR (one signal per point, not one global Dice
scalar) and **self-regularising**: nudging a point that is already ON the
boundary *increases* its distance → negative reward, so the reward-optimal
policy is "leave correct points alone," not "keep pushing." `step_penalty=0.0`
under this mode (a nonzero DuelingDDQN step penalty here previously drove an
88%-STOP collapse — see §10, 2026-07-08).

**Optimal-stopping STOP bonus (2026-07-11, commit `2fd0d0b`) — DISCRETE ONLY:**
`contour_boundary`'s dense reward is ~0 and noisy near the peak, so
`Q(STOP)` vs `Q(keep-pushing)` sat below the Q-noise floor and the discrete
agent almost never chose STOP (12% CAMUS LV_endo, 0% BRISC tumor, confirmed on
real runs). Fix: a **chosen** STOP now earns
`terminal_bonus_scale · max(0, dice − dice_0)` — proportional to the gain
already captured, zero if stopped before any gain, and a max-steps timeout
(vs a deliberate STOP) earns nothing, so STOP strictly dominates drifting to
the cap on an unchanged mask. `terminal_bonus_scale` is per-class/dataset,
set 2026-07-11 (`69bc860`) and **still provisional** — no sweep was ever run
to calibrate it (see §10, 2026-07-16): CAMUS = 20 (default in
`apply_refinement_config`, unverified against real behaviour plots — no
CAMUS-DuelingDDQN images have been reviewed), BRISC tumor = **30** (raised
2026-07-16 from 10 — real PhaseA replay plots showed only ~1% STOP rate at
10, matching the same under-shoot pathology the Jul-11 fix targeted; 30 is
inside the 10/20/40 range the Jul-11 commit anticipated needing, biased high
since 10 measurably failed). **Explicitly inert for TD3** (`2fd0d0b`'s own
commit message: continuous action has no STOP to be strictly-preferred via a
terminal bonus; TD3 episodes always end via `max_steps` or `fail_thresh`).

**Continuous (TD3/DDPG) auto-stop (2026-07-16, new) — the TD3 analogue of
the discrete STOP fix above.** Because the STOP bonus is inert for TD3, every
continuous episode previously ran the FULL `max_steps` regardless of whether
the contour had already converged — this is the direct cause of the "wavy /
over-deformed" TD3 output seen in the UI (best Dice reached mid-episode, then
the actor keeps perturbing an already-good contour for the remaining steps
with no way to commit). Fix: `ContourRefineEnv` now tracks the actor's own
raw action magnitude and terminates once `mean(|action|) < auto_stop_action_eps`
for `auto_stop_patience` consecutive steps — GT-free (reads only the action
vector), valid at training AND deployment, and NOT a value/GT floor (it
doesn't substitute a different mask, it just stops feeding a converged actor
more steps). Defaults in `_REFINE_CONTINUOUS`: `auto_stop_action_eps=0.05`,
`auto_stop_patience=3`. `auto_stop_action_eps=0.0` fully disables it
(backward-compatible default on the raw `ContourRefineEnv` constructor).
Wired through `drl_training.ENV_OPTIONAL_KEYS` (so eval/replay never drifts
from training — see §10, 2026-07-09 `132bb1a`) and mirrored into
`server/app/env_contour_refine.py`'s copy for the deployed UI. Verified with
synthetic (torch-free) tests: near-zero actions stop at exactly `patience`
steps; large actions never trip it; `eps=0.0` is a full no-op. **Not yet
validated against a real trained TD3 actor** — no GPU access this session;
watch the §8 `plot_behaviour` panel on the first real PhaseB run.

**PBRS (reference — no longer the active default).** Potential-based shaping
(Ng et al. 1999): `r_t = γ·Φ(s_{t+1}) − Φ(s_t) − step_penalty`, baseline-
centred and scaled: `Φ(s) = K·(Dice(s) − Dice_0)` (`dice_pbrs`), or
`Φ(s) = K·[α·(Dice − Dice_0) + β·(hd_term − hd_term_0)]` (`dice_hd_pbrs`).
Centring removed the discount drag that made an un-centred `Φ=Dice≈0.94`
punish every step; `K` (10 CAMUS / 15 BRISC) lifted per-step deltas above the
Q-net noise floor. Still selectable via `reward_mode` and still what the raw
per-class YAMLs show, but superseded as the actual default 2026-07-08.

**State** `(5, 256, 256)`: image · current mask · SDT(mask) · U-Net init mask · U-Net prob-map.
Two known state-representation bugs were found and fixed 2026-07-09
(`37627be`, `132bb1a`) — debris-channel leak (channels 4–5 showed the raw
multi-blob U-Net mask instead of the largest-CC-only representation the
contour can actually reach) and SectorPool binning around the image centre
instead of the live contour centroid (54% cell mismatch for off-centre
masks). **Any checkpoint trained before 2026-07-09 ~03:39 with
`spatial_head: true` (the `apply_refinement_config` default since 2026-07-08)
is invalid and needs retraining** — this is a training-time bug baked into
the learned weights, not something a re-eval can correct. See §11 for whether
this applies to the currently-reported PhaseA results.

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
| `iteris/models.py` | `AttentionResUNet` (RL warm-start target, default `model`) + `LiteUNet` (weaker comparison-only baseline); `build_model` |
| `iteris/warm_start.py` | U-Net inference → init masks + prob-maps |
| `iteris/refinement_viz.py` | Replays / comparison / playback / behaviour / test eval (discrete + continuous); init+final literature-standard metrics; `refinement_env_kwargs()` is the single source of truth for train/eval env-kwarg parity (`ENV_OPTIONAL_KEYS`, promoted here 2026-07-09) |
| `iteris/config.py` | `resolve_agent_config()` + **`apply_refinement_config()`** — the actual reward/env config every current notebook runs under (`_REFINE_SHARED`/`_REFINE_CONTINUOUS`/`_REFINE_DISCRETE`); see §6. Overrides the raw per-class YAML — read this, not the YAML, for ground truth |
| `iteris/drl_reeval/re_eval_td3.py` | `reeval_checkpoint()` (added 2026-07-13, `17afaec`) — re-score an ALREADY-TRAINED checkpoint under current code (best-val reload, current env kwargs incl. the 2026-07-16 auto-stop) without retraining. Cheap (~test-set forward pass only); see §11 |
| `iteris/archive/paradigm_a/` | Global-morphology `SegmentationEnv` — ablation only |
| `iteris/archive/paradigm1_boundary_tracing/` | Retired boundary-tracing paradigm |
| `iteris/archive/msa.py` | Archived MSA (Multi-Head Self-Attention) backbone — scoped for the now-**abandoned** Phase C, not wired in, staying archived |
| `configs/CAMUS/camus.yaml`, `configs/BRISC/brisc*.yaml` | LiteUNet/AttentionResUNet baseline training configs; `label_frac` is the Phase B lever |
| `configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml`, `configs/BRISC/DRL/brisc_drl_*.yaml` | DRL configs — **env/reward fields here are overridden by `apply_refinement_config` for every current notebook** (DuelingDDQN + TD3 active; `*_GLOBAL`/DQN/DDPG = ablation) |
| `notebooks/phaseA/unet/01_camus_lite.ipynb`, `phaseA/unet/02_brisc_lite.ipynb` | Train the **LiteUNet** comparison baselines, full data (not an RL target) |
| `notebooks/phaseA/unet/03_camus_attnunet.ipynb`, `phaseA/unet/04_brisc_attnunet.ipynb` | Train the **AttentionResUNet** baselines (the RL warm-start target), full data |
| `notebooks/phaseA/camus/drl/03{a,b,c}_camus_drl_*.ipynb`, `notebooks/phaseA/brisc/04_brisc_drl.ipynb` | DRL training (Kaggle), full data |
| `notebooks/phaseB/unet/…`, `notebooks/phaseB/camus/drl/…`, `notebooks/phaseB/brisc/drl/…` | Same notebooks, low-data (`label_frac`) regime — mirrors the `phaseA/` layout 1:1 (reorganized 2026-07-08, `dd73405`) |
| `notebooks/local/local_{camus,brisc}_drl.ipynb` | DRL training (local fyp_env GPU) |
| `notebooks/unet/00_free_diagnostics_all_classes.ipynb` | Headroom + prob_map informativeness diagnostic — run before any GPU round |
| `notebooks/evaluation/comprehensive_model_evaluation.ipynb` | Cross-run evaluation notebook (added 2026-07-20) — ingests every U-Net/DRL/classifier output file it finds under `results/` or Kaggle input datasets, phase-aware throughout (Phase A/B never pooled; Phase C excluded+reported if ever seen), Plotly charts, Wilcoxon significance. Feeds `iteris_ui`'s Research page |
| `server/app/drl.py` | Deployed UI inference — its own hardcoded `_SHARED`/`REGISTRY` config, **not** derived from `apply_refinement_config`; **both** `'high'` (Phase A) and `'low'` (Phase B) regime entries now exist, 16 total (landed 2026-07-20, `4825754`) |
| `server/app/env_contour_refine.py` | Byte-identical duplicate of `iteris/env_contour_refine.py` for the deployed server — must be hand-kept in sync (no shared import) |

---

## 8. Current Status — Week ~14 of 14 (final week)

| Item | Status |
|---|---|
| `ContourRefineEnv` (discrete + continuous, angular sectors) | ✅ |
| DuelingDDQN + TD3 agents | ✅ implemented, fully smoke-tested incl. continuous+BC+curriculum |
| LiteUNet model + configs (comparison baseline only) | ✅ trained, both phases (CAMUS label-smoothing retrain done; prob_map still INERT for LV_endo/LA — see SKILLS.md) |
| AttentionResUNet model + configs (the RL warm-start target) | ✅ trained, both phases |
| Ceiling / headroom diagnostic | ✅ run on real data — confirmed positive headroom, all 3 CAMUS classes; confirmed negative for BRISC tumor (pooled), historical/architecture-comparison framing (§4) |
| Literature-standard metrics (IoU/Precision/Sensitivity/BIoU/MSD, init+final) | ✅ |
| Reward system: `contour_boundary` + optimal-stopping STOP bonus (discrete) | ✅ landed 2026-07-08/11, superseded PBRS as the active default — see §6 |
| Debris-channel leak + SectorPool centroid-binning fix | ✅ landed 2026-07-09 — invalidated any pre-fix `spatial_head=true` checkpoint (retrain needed, see §11) |
| Eval consistency: return/eval best-val checkpoint, not final-step | ✅ landed 2026-07-13 (`17afaec`) — TD3-specific (DuelingDDQN unaffected, monotonic); `reeval_checkpoint()` re-scores pre-fix runs without retraining |
| Continuous (TD3) GT-free action-magnitude auto-stop | ✅ landed 2026-07-16 — synthetic-verified only, not yet confirmed against a real GPU training pass |
| BRISC discrete `terminal_bonus_scale` 10→30 | ✅ landed 2026-07-16 — evidence-based (real Phase A behaviour plots showed ~1% STOP rate at 10), still provisional |
| Single-baseline design (RL refines AttentionResUNet directly, not LiteUNet) | ✅ confirmed current 2026-07-14 (`61097d0`) — every notebook already passed the correct `baseline_cfg_name`; that commit only fixed stale config comments/fallback values to match. §4/§9 of this doc were still describing the OLD design until this 2026-07-21 audit — now fixed |
| Server/UI wiring for DRL inference | ✅ **complete for both phases** — Phase A/high landed 2026-07-12/15 (`41b281b`, `5118e27`), Phase B/low landed 2026-07-20 (`4825754`); 16 entries total (2 datasets × classes × 2 algos × 2 regimes) |
| Cross-run evaluation notebook (`notebooks/evaluation/`) | ✅ added 2026-07-20 — phase-aware (Phase A/B never pooled; Phase C excluded+reported if present), Plotly-based, feeds the Research page's real results (`91fbcc4`) |
| **Phase A** (full-data DuelingDDQN + TD3 runs) | ✅ complete, results wired into the deployed UI and evaluation notebook |
| **Phase B** (low-data-regime comparison) | ✅ complete, results wired into the deployed UI and evaluation notebook (2026-07-20) |
| **Phase C** (MSA backbone adaptation) | ❌ **abandoned (2026-07-21)** — a project decision, not a delay. Not started, not planned. See §3 |
| Evaluation + paper | ⏳ school-submission pass now; full retrain-and-finalize pass deferred to later (see §11) |

---

## 9. Settled Decisions

| Decision | Rationale |
|---|---|
| Two algorithms only in Phase A/B: DuelingDDQN (discrete) + TD3 (continuous) | Clean discrete-vs-continuous comparison; both on the contour env |
| Two-phase experimental design (A: full data, B: low data) — a third phase (C: MSA backbone) was designed then **abandoned (2026-07-21)** | Answers not just "does DRL help" but "when" — data-scarcity was the axis that stayed in scope; the richer-agent-backbone axis (Phase C) was cut for time, not for a research reason — see §3 |
| Contour refinement is the only live paradigm | Global morphology is capped at baseline (confirmed on real runs) |
| RL refines the **AttentionResUNet** baseline directly, not a separate lite baseline (single-baseline design, confirmed current `61097d0`, 2026-07-14) | **Superseded decision, kept for history:** the project originally had RL refine a deliberately weak `LiteUNet` (real headroom by construction) with `AttentionResUNet` as a fixed, untouched competitor (decided 2026-06-21, `12948a5`) — reasoning being "the strong baseline has no headroom, so RL can't improve it." That arrangement was later reversed: every current notebook points `apply_refinement_config`'s `baseline_cfg_name` at the AttentionResUNet config, so RL now refines the strong baseline directly. `LiteUNet` is retrained per phase only as a separate, weaker comparison point (architecture-headroom reference), not touched by RL at all. The exact commit where the *behavior* flipped (as opposed to `61097d0`, which only fixed stale comments to match already-correct behavior) isn't cleanly identifiable from a single commit message — flagging this as a known gap in the historical record rather than asserting a false precise date. |
| Baseline-centred scaled PBRS reward | Removes discount drag + path-dependence; lifts tiny signal above noise |
| TD3 over plain DDPG | Twin critics + target smoothing fix the overestimation that stalls DDPG |
| Dueling Double DQN over plain DQN | Dueling fits "many near-equal actions" refinement; Double kills overestimation |
| `reward_step_penalty` (DuelingDDQN, PBRS-era) — **SUPERSEDED 2026-07-08** | Was the STOP mechanism under PBRS; under `contour_boundary` it's forced to 0.0 (a nonzero value drove an 88%-STOP collapse) and STOP is instead taught via `terminal_bonus_scale` — see §6 |
| `terminal_bonus_scale` optimal-stopping bonus (discrete only, `contour_boundary`) | Replaces `reward_step_penalty` as the STOP-teaching signal; proportional to captured gain, zero if stopped early, timeout earns nothing — makes a deliberate STOP strictly dominant. Per-class, still provisional (no sweep run) |
| Continuous (TD3) action-magnitude auto-stop (2026-07-16) | `terminal_bonus_scale` is inert for TD3 (no STOP action) — this is the GT-free analogue: stop once the actor's own action magnitude has settled near zero |
| Curriculum `max_steps` (training-only, both agents) | GT-based difficulty can only be used where GT is available (training); deploy/eval always use the fixed `max_steps` |
| Run `headroom_report` before each full RL run | Cheap go/no-go: confirms reachable Dice > baseline before spending GPU |
| Paradigm A kept (archived) | Negative-control ablation for the paper |
| Server (`server/app/drl.py`) config is NOT derived from `apply_refinement_config` | Deliberately separate, hand-maintained per-checkpoint config so a UI redeploy can never silently pick up an in-progress training-config change; costs manual sync discipline — see §11 |

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
| 2026-07-08 04:08 (`eb819c7`) | Fixed discrete BC oracle near-zero STOP rate: strict `>` comparison against current Dice let floating-point re-rasterisation jitter almost always beat the bar, so the oracle (used for BC warm-start demonstrations) almost never emitted STOP. |
| 2026-07-08 06:03–06:10 (`9f6155a`, `3e1b206`) | **Reward-system rewrite.** Introduced `apply_refinement_config()` (now in `iteris/config.py`) — ONE agent-aware call replacing the scattered per-notebook `cfg.update({...})` blocks. Switched the default `reward_mode` to **`contour_boundary`** (dense per-control-point distance-to-GT-boundary, self-regularising) for every notebook that calls it — this **overrides** the per-class YAML's `reward_mode`/`reward_step_penalty`/etc., which is why the YAML no longer reflects what actually trains. See §6. |
| 2026-07-08 06:30 (`dd73405`) | Reorganized `notebooks/{camus,brisc,unet}` → `notebooks/phaseA/{camus,brisc,unet}`, mirroring the existing `phaseB/` layout. §7's file map was stale on this until 2026-07-16. |
| 2026-07-09 01:50 (`37627be`) | **Two correctness bugs found from a real PhaseB LV_endo DUELING run** (agent held the initial shape / degraded, action histogram near sample-invariant): (1) debris-channel leak — state channels 4–5 showed the RAW multi-blob U-Net mask instead of the largest-CC-only representation the contour can actually reach, biasing the policy toward debris blobs; (2) `SectorPool` binned around the image centre while the env bins around the live contour centroid — 54% cell mismatch for off-centre masks, scrambling the local-error→correct-sector signal. **Invalidates any checkpoint trained before this fix with `spatial_head: true`** (default since the prior commit) — retrain required, no eval-time fix possible. |
| 2026-07-09 02:07 (`132bb1a`) | Follow-up audit for the same bug class (state acted-on ≠ state learned-on). Found two more: (1) the debris fix had been applied unconditionally, which was wrong for the default (non-contour) `SegmentationEnv`; gated by env type. (2) `refinement_viz` kept its own drifted copy of the env-kwarg key list, silently dropping `auto_smooth_lambda` and `uncertainty_gate`/`gate_*` at EVAL time only — so TD3 and the gated LV_epi class were tested on a different env than they trained on, corrupting reported test metrics. Fixed by promoting the key list to `drl_training.ENV_OPTIONAL_KEYS` as the single source of truth (this is the mechanism the 2026-07-16 auto-stop keys were added to). |
| 2026-07-09 02:53–03:39 (`fac2a4d`, `40f2091`) | Added boundary-aware (Tier 1) + adaptive (Tier 2) + uniform-floored (Tier 1.5) hard-sample-mining, replacing the flat exponential weighting that gave topology-broken (unreachable) samples the same aggressive boost as genuinely fixable ones. |
| 2026-07-09 19:48 (`b7a8207`) | `TRAIN_STEPS` set to 50000 across all 8 production DRL notebooks (Phase A + B). |
| 2026-07-11 11:02 (`2fd0d0b`) | **Diagnosed and fixed the discrete STOP-rate pathology under `contour_boundary`**: 12% STOP on CAMUS LV_endo, 0% on BRISC tumor. Added the optimal-stopping terminal bonus (`terminal_bonus_scale · max(0, dice−dice_0)` on a chosen STOP, nothing on timeout) — confirmed **inert for TD3** (no STOP action) in the same commit message. See §6. |
| 2026-07-11 11:12 (`69bc860`) | Set per-class/dataset `terminal_bonus_scale` (CAMUS=20, BRISC=10) — **no sweep was run**; the commit's own note anticipated validating via a 10/20/40 sweep that was never executed. BRISC's 10 turned out to under-shoot (see 2026-07-16). |
| 2026-07-12 10:06 (`41b281b`) | Workspace UI redesign; wired DuelingDDQN CAMUS-LV inference into the deployed server (`server/app/drl.py` REGISTRY, high-regime only). |
| 2026-07-13 22:51 (`17afaec`) | **Fixed eval consistency**: `run_drl_training` was returning the agent at its FINAL-step weights, not the saved best-val checkpoint. TD3-specific (actor drifts once `bc_lambda` decays) — DuelingDDQN unaffected (monotonic). Centralized the fix (every notebook's `evaluate_testset`/`build_replays` now measures the deployable model automatically) and added `iteris/drl_reeval/re_eval_td3.py::reeval_checkpoint()` to re-score checkpoints trained BEFORE this fix without retraining. **Any TD3 checkpoint trained before this timestamp may have reported understated test numbers — cheap to check via `reeval_checkpoint()`, see §11.** |
| 2026-07-14 (`61097d0`, `f3ea5dc`) | Documentation-consistency pass (single-U-Net design, no behaviour change) + landing-page redesign (DQN/DDQN/DDPG → Dueling DQN + TD3 in the UI model list, matching the actual active agent set). |
| 2026-07-15 22:15 (`5118e27`) | Wired the full Phase A high-regime registry into the server: all 6 CAMUS per-class agents (LV_endo/LV_epi/LA × DuelingDDQN/TD3) + BRISC tumor × {DuelingDDQN, TD3} — 8 entries, exact per-checkpoint hyperparameters. **No `'low'`-regime entries added** — see §11 before wiring Phase B checkpoints into the UI. |
| 2026-07-16 (this session) | Analyzed real PhaseA behaviour/comparison plots (CAMUS TD3 LV_epi/LA, BRISC DuelingDDQN/TD3 tumor) against a prior session's `dice_hd_pbrs` suggestion — found it moot (contour_boundary already superseded PBRS on 2026-07-08, unknown to that session). Root-caused the observed final≪best-seen drift and BRISC's ~1% STOP rate to two structural gaps: **(1)** TD3 has no stopping mechanism at all (confirmed by 2026-07-11's own commit message) → added GT-free action-magnitude auto-stop (`auto_stop_action_eps`/`auto_stop_patience`, `iteris/env_contour_refine.py` + `server/app/env_contour_refine.py`, wired through `ENV_OPTIONAL_KEYS`/`_REFINE_CONTINUOUS`); synthetic-verified, not yet GPU-validated. **(2)** BRISC's `terminal_bonus_scale=10` (2026-07-11) measurably under-shot → raised to 30 in both `phaseA/brisc/04_brisc_drl.ipynb` and `phaseB/brisc/drl/04_brisc_drl.ipynb` (CAMUS left untouched — no CAMUS-DuelingDDQN evidence reviewed). Rewrote §6/§7/§8/§9/§10 of this doc (previously last touched 2026-06-30, silently out of sync with the 2026-07-08–15 reward-system rewrite) and added §11 below. Structural audit of the low-data (Phase B) path — `splits.py` patient-level `label_frac` subsetting, `model_suffix()` checkpoint namespacing, hard-mining numerics, `refinable_gate` empty-set handling — found no new bugs (all defensively coded); found one live UI gap (§11.4). |
| 2026-07-16 (later same session) | **Fixed a real crash, found live**: `notebooks/phaseB/unet/02_brisc_lite.ipynb` failed at the `evaluate_test_set` import after training completed successfully (best val Dice 0.6863, checkpoint saved). Root cause: `17afaec` (2026-07-13) added the **package** `iteris/evaluation/` alongside the pre-existing **file** `iteris/evaluation.py` — same name, so the package silently shadowed the file for every `from iteris.evaluation import …`, and `evaluate_test_set`/`export_predicted_masks`/`save_summary_json` (all defined in the file) became unreachable. Affects **all 8** U-Net baseline notebooks (Phase A + B × {camus,brisc} × {lite,attention}) — any run after 2026-07-13 22:51 hits this at the same point. Fix: renamed the package `iteris/evaluation/` → `iteris/drl_reeval/` (no other code referenced the package path except its own usage docstring, now updated); `iteris/evaluation.py` is unshadowed again. **Not caused by, or related to, this session's earlier DRL changes.** Training itself is unaffected (`run_training` completes and saves the checkpoint before the crashed cell) and the DRL warm-start pipeline (`precompute_init_masks`) loads the checkpoint file directly — it does not depend on `export_predicted_masks`'s output — so this did not block or corrupt any DRL run. See §11.5 for what to check across already-run notebooks. |
| 2026-07-14 | `61097d0`, `f3ea5dc` — Doc/config-comment consistency pass confirming the single-baseline design (RL refines `AttentionResUNet` directly; every notebook already passed the correct `baseline_cfg_name`, only stale comments/fallback values were fixed) + landing-page model list updated to match the real active agent set (Dueling DQN + TD3, dropping the archived DQN/DDQN/DDPG). No training-behavior change. **This doc's own §4/§9 were NOT updated at the time** — found and fixed 2026-07-21 (see below), the direct trigger for this session's full doc audit. |
| 2026-07-15 | `5118e27` — Wired the full **Phase A** (high-regime) DRL registry into the deployed server: all 6 CAMUS per-class agents (LV_endo/LV_epi/LA × DuelingDDQN/TD3) + BRISC tumor × {DuelingDDQN, TD3}, 8 entries, exact per-checkpoint hyperparameters. Added TD3 support to the server's episode runner/checkpoint loader (previously discrete-only). |
| 2026-07-19 | Three independent changes, same day: (1) `f33f56e` — added CI/CD (GitHub Actions) for the FastAPI server's Hugging Face Space deploy + UI/Python sanity CI; confirmed Vercel's own integration already handles the frontend (no Action needed there) and Kaggle training is inherently manual (no deploy step applies). (2) `a0a0a00` — Research page wired to real project state: replaced the placeholder scaffold with all 9 spec sections behind a scroll-spy TOC, rewrote Abstract/Methods/Models for the current paradigm (contour env, angular sectors, `contour_boundary` reward, DuelingDDQN + TD3 vs LiteUNet/AttentionResUNet), and swapped invented Dice/p-values for honest pending states. (3) `ee87f6c` — new logo/favicon mark, navbar theme-toggle fix. |
| 2026-07-20 | `4825754` — Wired the full **Phase B** (low-regime) DRL registry into the deployed server, bringing the total to 16 entries (2 datasets × classes × 2 algos × 2 regimes, generated over the full combination instead of hand-copied). Confirmed Phase A and Phase B checkpoints for the same class share identical contour-env geometry — regime is purely a checkpoint-selection axis, not a hyperparameter change. **Phase A and Phase B are now both complete and fully wired end-to-end** (training → checkpoints → server → deployed UI). |
| 2026-07-20 | Built `notebooks/evaluation/comprehensive_model_evaluation.ipynb` from scratch, then iterated it substantially the same day: ingests every U-Net/DRL/classifier output file under `results/` or Kaggle input datasets (content-shape detection, sibling-JSON metadata matching), computes the master comparison table, significance tests (Wilcoxon + Bonferroni), and the qualitative gallery. Switched charting from matplotlib to Plotly, reframed the discrete-vs-continuous comparison to report **both** absolute deployed Dice (head-to-head) and delta-vs-baseline (learning behaviour) since they can disagree and neither subsumes the other. Fixed two real portability bugs found via actual Kaggle runs: (1) Plotly's default interactive-HTML renderer doesn't display on GitHub/nbviewer/Kaggle's Log console (no JS execution there) — switched every chart to render as a static PNG via `kaleido`; (2) Kaggle's preinstalled `plotly 5.24` + `kaleido 1.3` are mutually incompatible (kaleido 1.x needs plotly≥6.1.1) — the setup cell now pins the matched `plotly<6`+`kaleido==0.2.1` pair and probes it once up front (single clear failure message instead of one repeated per chart). |
| 2026-07-20 | **Added Phase A/B as a first-class grouping dimension to the evaluation notebook**, fixing a real correctness gap: every section was silently pooling Phase A and Phase B runs into single means/rankings/pairings (worst case, the discrete-vs-continuous head-to-head could pair a Phase A run against a Phase B run for the same class as if they'd trained under identical conditions). Detection: `label_frac` where saved (U-Net/classifier JSONs), else a `pa`/`pb` path-token fallback for DRL reeval JSON (which never saves `label_frac`) — matches the `pa-`/`pb-` Kaggle dataset-slug convention already in use. Verified with a synthetic fixture built to prove it matters: the per-phase head-to-head correctly reversed the winning agent between phases, which a pooled ranking would have missed entirely. **Known open gap, found the same day by a downstream consumer**: wiring the Research page (`91fbcc4`) to a real Kaggle run's `master_comparison.csv` found the U-Net baseline rows came out phase=`Unknown` (their `label_frac` apparently didn't survive to that run's summary JSON, and the U-Net Kaggle dataset slugs don't follow the `pa-`/`pb-` convention DRL runs do, so the path fallback also missed) — had to be disambiguated by Dice rank + cross-checked against DRL `init_dice_mean` by hand in the frontend. Not yet root-caused or fixed in the notebook itself — worth revisiting. |
| 2026-07-21 | **Phase C (MSA backbone adaptation) formally abandoned** — direct project decision, not a scheduling slip: "Phase C is dropped, not to be used." `iteris/archive/msa.py` and `agents_legacy.py::MSADuelingDQNAgent` stay archived; no config/notebook/`AGENT_REGISTRY` entry will be added. Do not resurrect without a fresh decision. |
| 2026-07-21 | **Full documentation audit** (this entry's session): analyzed the full commit log (189 commits) against every doc in `docs/` + `README.md` + `results/README.md` + `server/README.md` for staleness. Found and fixed three categories of drift: (1) the single-baseline pivot (RL refines AttentionResUNet, not LiteUNet — confirmed current since `61097d0`, 2026-07-14) had never been propagated into CONTEXT.md §1/§4/§9, EXPERIMENTS.md, PLAN.md, SKILLS.md, or README.md — all still described the old lite-RL-target design as current; (2) Phase C was listed as "not started"/planned everywhere despite being abandoned this session — updated to "abandoned" with the historical design record kept, not deleted; (3) confirmed no stale dataset mentions survived in current docs (HAM10000/Kvasir-SEG/DRIVE were archived/dropped in May 2026, well before the DRL design existed, and aren't referenced as in-scope anywhere) — added one explicit sentence to CONTEXT.md §1 noting this for anyone reading the git history and wondering. This transcript and SKILLS.md §4 were also brought current (previously frozen at 2026-06-30/07-01 respectively). |

---

## 11. Known Limitations — Accepted For This Submission

Decided 2026-07-16: PhaseA results are **already run and locked in** for the
UI and the school submission; there is no time to rerun them, and the items
below are either too small to matter for a school submission or require a
retrain that isn't happening until the actual paper push. Recorded here so
they aren't rediscovered as a surprise later, not as an active TODO.

**11.1 — Server UI config (`server/app/drl.py::_SHARED`) is stale relative to current training defaults.**
`disp_px=2.0`, `max_steps=10` match the OLD YAML-only training (pre-2026-07-08);
current `apply_refinement_config` defaults are `disp_px=0.5`, `max_steps=25`.
This is very likely *consistent*, not broken — `_SHARED`'s docstring cites the
raw per-class YAML paths as its source, suggesting the currently-deployed
HF-hosted checkpoints were trained under the old settings and the UI config
correctly matches them. **Not changed this session** — deliberately, since
flipping it without knowing which config actually trained the live checkpoint
risks a real train/deploy pixel-scale mismatch (worse than the status quo).
**Before ever redeploying a new checkpoint to the server:** confirm which
config trained it and update `_SHARED` (now including the 2026-07-16 auto-stop
keys, already added) to match exactly — this is deferred to the actual
paper-retrain pass, not this submission.

**11.2 — PhaseA vs PhaseB `evaluate_testset` call mismatch.**
`notebooks/phaseA/brisc/04_brisc_drl.ipynb` §10 doesn't pass `refinable_gate`
to `evaluate_testset`; PhaseB's does. PhaseA's reported test numbers therefore
don't use the GT-free routing-gate subset/routed reporting that's otherwise
standard. Accepted as-is per 2026-07-16 decision — the numeric effect is
small (the gate only changes which subset extra `*_refinable_*`/`routed_*`
keys are computed over; `init_dice_mean`/`final_dice_mean`/etc. are unaffected
either way) and PhaseA isn't being rerun. Worth reconciling before the paper
directly compares PhaseA vs PhaseB numbers side-by-side.

**11.3 — Do the already-run PhaseA checkpoints predate the 2026-07-09 or 2026-07-13 fixes?**
Unknown from this repo alone (checkpoints live on Kaggle `/kaggle/working` /
HF Hub, not locally) — this is genuinely worth 5 minutes to check before
finalizing submission numbers, split into two independent questions with very
different costs to answer:
  - **Cheap to fix, worth doing regardless:** if a TD3 checkpoint was trained
    before 2026-07-13 22:51 (`17afaec`), its reported test/replay numbers used
    the FINAL-step agent instead of the best-val checkpoint (DuelingDDQN is
    unaffected — monotonic). `iteris.drl_reeval.re_eval_td3.reeval_checkpoint()`
    re-scores an existing `*_best.pt` file under current code — no GPU
    training, just a test-set forward pass (minutes) — and will *also*
    automatically pick up the 2026-07-16 TD3 auto-stop fix for free, since
    both flow through the same `refinement_env_kwargs()`. This directly
    targets the "wavy" UI symptom without retraining. **Recommended**, even
    under the "no time to rerun" constraint — it is not a rerun.
  - **Not fixable without retraining:** if a `spatial_head: true` checkpoint
    (the default since 2026-07-08) was trained before 2026-07-09 ~03:39
    (`37627be`), the debris-channel/SectorPool bugs are baked into the learned
    weights — no eval-time correction exists. If this applies, it's the same
    bucket as 11.1/11.2: a known, accepted limitation for this submission,
    revisit at the paper retrain.

**11.4 — RESOLVED 2026-07-20 (`4825754`) — UI now has `'low'`-regime entries; verify the checklist below was actually followed.**
Written 2026-07-16 as a forward-looking checklist for "whenever Phase B gets
wired in" — that happened 2026-07-20, both sides now list all 16 combinations
(`server/app/drl.py::REGISTRY` and `iteris_ui/src/api/contract.ts::AVAILABLE_COMBINATIONS`).
Keeping the original checklist below as a **verification list**, not a TODO —
worth a five-minute check that the env_cfg-matching concern it raised was
actually respected when `4825754` landed, since that commit's own message says
"Phase A and Phase B checkpoints for the same class share the exact same
contour-env geometry" (i.e. it asserts the concern doesn't apply here — regime
is a checkpoint-selection axis only), which is a reasonable claim but hasn't
been independently re-verified against this doc's original wording:
  - `contract.ts::defaultRegime(modelId)` returns `'low'` unconditionally for
    any non-baseline model (`dueling-dqn`/`td3`) — it does **not** check
    `isCombinationAvailable`/`availableRegimes` first. Right now this is
    harmless only because whatever component drives the model/regime picker
    apparently gates on the availability table before falling back to
    `defaultRegime()` (the user's own successful UI test implies this — a
    naive `'low'` default against a `'high'`-only backend REGISTRY would 500).
    This has NOT been traced through the picker component to confirm; treat
    as unverified.
  - The moment a `{dataset, modelId, regime:'low'}` entry is added to
    `AVAILABLE_COMBINATIONS` (frontend), a matching entry with the CORRECT
    `env_cfg` (see 11.1 — must match whatever `apply_refinement_config`
    actually produced for that low-regime checkpoint, including
    `auto_stop_action_eps`/`patience`) must be added to `server/app/drl.py`'s
    `REGISTRY` in the same change. Adding one without the other reintroduces
    exactly the crash class 11.1 warns about, now for a config that doesn't
    exist at all yet rather than one that's merely stale.

**11.5 — `iteris.evaluation` package/file collision (fixed 2026-07-16), and what to check on already-run notebooks.**
`iteris/evaluation/` (package, added 2026-07-13) was renamed to `iteris/drl_reeval/`
to stop shadowing `iteris/evaluation.py` (file, pre-existing) — see §10. Action
items:
  - **Re-upload/re-version the `iteris-pkg` Kaggle Dataset** before re-running
    anything — the fix is only in this local repo; Kaggle notebooks pull from
    the uploaded dataset artifact, not this git tree directly.
  - `run_training()` has **no resume/skip-if-checkpoint-exists flag** (unlike
    the DRL notebooks' `RESUME`/`TRAIN_STEPS` pattern) — re-running a crashed
    U-Net notebook redoes the full training pass (~29 min for the BRISC-lite
    Phase B run that crashed). Not blocking, just budget for it.
  - **Check whether any of the other 7 U-Net notebooks** (Phase A lite/attention
    × CAMUS/BRISC, Phase B camus_lite/camus_attnunet/brisc_attnunet) were run
    after 2026-07-13 22:51 and hit the same import error — including the
    **already-locked-in Phase A baselines**. If so, their `evaluate_test_set`
    DataFrame / predicted-mask export / summary JSON never ran, though the
    per-epoch metrics are still recoverable from that run's printed log even
    without the JSON. This does **not** invalidate the trained checkpoints
    themselves (training completes before the crashed cell) — only the
    reporting artifacts are missing.
