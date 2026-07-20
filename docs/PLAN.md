# Iteris — Implementation Plan

> **Last updated:** 2026-07-21 · **Semester week:** ~14 of 14 (final week).
> Two algorithms — **DuelingDDQN** (discrete contour) and **TD3** (continuous contour) — refining
> the **AttentionResUNet** baseline directly (single-baseline design, confirmed current since
> `61097d0`, 2026-07-14). A separate, deliberately weaker **LiteUNet** is trained per phase as a
> comparison-only baseline, not an RL target. This doc previously (through 2026-06-21) described
> the reverse arrangement — RL refining LiteUNet, AttentionResUNet as a fixed untouched
> competitor — which is why anything below referencing that framing has been corrected.
>
> **Naming note:** this doc's own stages below ("Stage A–D") are the *project timeline*
> (build → train → evaluate → write paper) — a different axis from CONTEXT.md's **Phase A/B**
> (data regime: full vs. low). Earlier versions of this file called the timeline stages
> "Phase A–D" too, which collided with CONTEXT.md's naming once the data-regime design existed;
> renamed to **Stage 1–4** here to match the convention SKILLS.md already established.

---

## 1. Engineering Principles

1. **Notebooks are thin.** All logic in `iteris/`. Notebooks: import → configure → call → display.
2. **Configs are YAML.** Every dataset/agent has a block; `resolve_agent_config(load_drl_class_config(...), AGENT_NAME)`.
3. **Dense contour-boundary reward.** `contour_boundary` (per-control-point distance-to-GT-boundary, self-regularising) is the active default since 2026-07-08 — see CONTEXT.md §6 for the full history (this superseded the original baseline-centred PBRS, `r = γ·Φ(s') − Φ(s)`, `Φ = K·(Dice − Dice_0)`, which is kept in CONTEXT.md as a reference design, not what currently trains).
4. **Headroom first.** Run `headroom_report` before any full RL run — never train where the contour ceiling ≈ baseline.
5. **Single-baseline design.** RL refines `AttentionResUNet` directly; `LiteUNet` is a separate, weaker comparison baseline only. All artefacts are model-suffixed (`utils.model_suffix`) so they never collide, including across phases (`_lf10` etc. for Phase B).

---

## 2. The 14-Week Timeline

| Stage | Weeks | Goal | Status |
|---|---|---|---|
| **1 · Build & validate models** | Week 10 | Baselines trained; both agents confirmed running end-to-end on a dry-run + headroom check | ✅ done |
| **2 · Full training** | Week 10–11 | DuelingDDQN + TD3 trained on CAMUS c1/c2/c3 + BRISC tumour, **both Phase A (full data) and Phase B (low data)** | ✅ done — see CONTEXT.md §8 |
| **3 · Evaluation** | Week 12 | Unified eval harness, stats, ablations, figures | ✅ `notebooks/evaluation/comprehensive_model_evaluation.ipynb`, added 2026-07-20 |
| **4 · Paper** | Weeks 13–14 | Draft → revise → submit | ⏳ in progress — this is the current stage |

---

## 3. Stage 1 — Build & validate (done)

| Task | Done when |
|---|---|
| Train both baselines (`01_camus_lite`/`02_brisc_lite`, `03_camus_attnunet`/`04_brisc_attnunet`) | Checkpoints saved for both LiteUNet (comparison) and AttentionResUNet (RL target) |
| Save baseline checkpoints as Kaggle Datasets at the paths the DRL configs expect | `…/camus-baseline-outputs/camus_best.pt` (AttentionResUNet — what DRL actually loads) + the LiteUNet equivalents |
| **Headroom diagnostic** per dataset (`headroom_report` cell after warm-start) | Run — historical/architecture-comparison signal now (§4 of EXPERIMENTS.md), no longer a go/no-go gate for whether DRL targets the strong net (it always does) |
| **Dry-run** for DuelingDDQN + TD3 on CAMUS + BRISC | Both run clean end-to-end |
| Lock final per-config contour geometry (`n_points`, `cont_sectors`, `disp_px`, `spline_smooth`) | Configs frozen for the full runs |

If a dataset shows no reachable improvement (BRISC tumour is the historical risk case — irregular shapes cap the spline), either adjust contour params or record it as the "contour-expressivity ceiling" result for the paper.

---

## 4. Stage 2 — Full training (done)

Run matrix — **2 agents × 4 targets × 2 phases = 16 runs**:

| Target | Config | DuelingDDQN | TD3 |
|---|---|---|---|
| CAMUS LV-endo | `camus_drl_c1` | ✅ Phase A + B | ✅ Phase A + B |
| CAMUS LV-epi | `camus_drl_c2` | ✅ Phase A + B | ✅ Phase A + B |
| CAMUS LA | `camus_drl_c3` | ✅ Phase A + B | ✅ Phase A + B |
| BRISC tumour | `brisc_drl_tumor` | ✅ Phase A + B | ✅ Phase A + B |

All 16 runs are complete, wired into the deployed server (`server/app/drl.py::REGISTRY`), and ingested by the evaluation notebook. Wall-clock was ≈1.5–3h/run on Kaggle T4 (contour rasterisation, CPU-bound, was the bottleneck, not GPU).

**Kaggle workflow used**

```
git push origin main                         (local)
Kaggle iteris-pkg → New Version              (bump dataset)
Baseline notebooks → save checkpoints as Datasets
DRL notebook: warm-start → headroom + dry-run → full training
```

---

## 5. Stage 3 — Evaluation (done)

Grounded in current medical-segmentation reporting practice (Metrics Reloaded; Wilcoxon + Bonferroni), implemented in `notebooks/evaluation/comprehensive_model_evaluation.ipynb`:

- **Metrics:** Dice, IoU, BIoU, Precision, Sensitivity, **HD95** (+ MSD), per class, on the held-out **test** set.
- **Comparisons (per class, per phase — Phase A and Phase B never pooled):**
  - AttentionResUNet baseline → AttentionResUNet + DuelingDDQN → AttentionResUNet + TD3, both head-to-head on absolute Dice and on delta-vs-baseline (two distinct framings — see the notebook's Section 4/5).
  - LiteUNet kept as a secondary architecture-comparison line, not part of the headline DRL result.
- **Significance:** two-sided **Wilcoxon signed-rank** (paired, per-patient) on the U-Net per-patient CSVs, α = .05, **Bonferroni** across classes/models — plus a dedicated Phase A vs Phase B same-model test quantifying the low-data effect directly.
- **Ablations:** global-morph (`*_GLOBAL`/DQN/DDPG) as negative control; discrete vs continuous; `headroom_report` ceiling vs achieved.
- **Harness:** the evaluation notebook itself — ingests every run's output files (JSON/CSV/PNG/checkpoint) from `results/` or Kaggle input datasets, phase-aware throughout, Phase C excluded+reported if ever present.

---

## 6. Stage 4 — Paper (current stage, Weeks 13–14)

- **Methods:** contour-refinement MDP (state, angular-sector action, `contour_boundary` reward), DuelingDDQN vs TD3, single-baseline (AttentionResUNet) design.
- **Results:** main table (2 agents × 4 targets × 2 phases, Dice/IoU/HD95 + significance) + qualitative grid (best/median/worst) + learning curves — all sourced from the evaluation notebook.
- **Discussion:** discrete vs continuous; absolute-Dice vs delta-vs-baseline framing (they can disagree, neither subsumes the other); full-data vs low-data regime; the ceiling analysis; global-morphology negative control; the abandoned Phase C (MSA backbone) noted as future work, not a gap in this submission.

---

## 7. Critical Don'ts

- Don't re-introduce a separate lite-U-Net RL-warm-start / attention-competitor split — that was the pre-2026-07-14 design; RL refines AttentionResUNet directly now.
- Don't re-introduce global morphology or boundary tracing as a *contender* (ablation only).
- Don't deploy "best-seen" masks — no GT at test time; the agent must learn to STOP (discrete) or rely on the action-magnitude auto-stop (TD3).
- Don't add scipy SDT to the reward hot loop — use the `ReplayBuffer` SDT cache.
- Don't hardcode hyperparameters — everything in YAML.
- Don't resurrect Phase C (MSA backbone) without a fresh decision — it was formally abandoned 2026-07-21, not merely deferred.
