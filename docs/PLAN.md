# Iteris — Implementation Plan

> **Last updated:** 2026-06-21 · **Semester week:** 10 of 14.
> Two algorithms only — **DuelingDDQN** (discrete contour) and **TD3** (continuous contour) — refining a **lite U-Net** baseline, with the **attention U-Net** as competitor.

---

## 1. Engineering Principles

1. **Notebooks are thin.** All logic in `iteris/`. Notebooks: import → configure → call → display.
2. **Configs are YAML.** Every dataset/agent has a block; `resolve_agent_config(load_drl_class_config(...), AGENT_NAME)`.
3. **Baseline-centred PBRS reward.** `r = γ·Φ(s') − Φ(s)`, `Φ = K·(Dice − Dice_0)` — path-independent, no discount drag.
4. **Headroom first.** Run `headroom_report` before any full RL run — never train where the contour ceiling ≈ baseline.
5. **Lite vs attention.** RL improves the lite baseline; the attention net is the fixed upper-bound competitor. All artefacts are model-suffixed so they never collide.

---

## 2. The 14-Week Timeline (we are at the start of Week 10)

| Phase | Weeks | Goal |
|---|---|---|
| **A · Build & validate models** | **Week 10 (next 2–3 days)** | Lite baselines trained; both agents confirmed beating the lite baseline on a dry-run + headroom check |
| **B · Full training** | **Week 10–11 (~1 week)** | DuelingDDQN + TD3 trained on CAMUS c1/c2/c3 + BRISC tumour |
| **C · Evaluation** | **Week 12** | Unified eval harness, stats, ablations, figures |
| **D · Paper** | **Weeks 13–14** | Draft → revise → submit |

---

## 3. Phase A — Build & validate (next 2–3 days)

| Day | Task | Done when |
|---|---|---|
| 1 | Bump `iteris-pkg` (latest commit). Train **lite** baselines: `01_camus_lite`, `02_brisc_lite` | `camus_lite_unet_best.pt`, `brisc_lite_unet_best.pt` saved (Dice ~0.80–0.85 / ~0.78–0.82) |
| 1 | Save lite checkpoints as Kaggle Datasets at the paths the DRL configs expect | `…/camus-baseline-outputs/camus_lite_unet_best.pt`, `…/brisc-baseline-outputs/brisc_lite_unet_best.pt` |
| 2 | **Headroom diagnostic** per dataset (`headroom_report` cell after warm-start) | `headroom > 0.02` (GOOD) confirmed, else adjust contour params / lite capacity |
| 2 | **Dry-run** (`RUN_DRY_RUN=True`, 600 steps) for DuelingDDQN + TD3 on CAMUS c1 + BRISC | both run clean end-to-end; delta trending positive |
| 3 | Lock final per-config contour geometry (`n_points`, `cont_sectors`, `disp_px`, `spline_smooth`) from the dry-runs | configs frozen for the full runs |

If a dataset shows **no headroom** (BRISC tumour is the risk — irregular shapes cap the spline), either (a) lighten the lite baseline further so its errors are more systematic, or (b) record it as the "contour-expressivity ceiling" result for the paper.

---

## 4. Phase B — Full training (~1 week)

Run matrix — **2 agents × 4 targets = 8 runs** (BRISC subtypes optional if time permits):

| Target | Config | DuelingDDQN | TD3 |
|---|---|---|---|
| CAMUS LV-endo | `camus_drl_c1` | 60k | 60k |
| CAMUS LV-epi | `camus_drl_c2` | 60k | 60k |
| CAMUS LA | `camus_drl_c3` | 60k | 60k |
| BRISC tumour | `brisc_drl_tumor` | 60k | 60k |

Each run saves the best-val checkpoint to Drive/working (survives Kaggle disconnects). Wall-clock ≈ 1.5–3 h/run on T4; contour rasterisation is the bottleneck. Stagger across Kaggle sessions over the week.

**Kaggle workflow**

```
git push origin main                         (local)
Kaggle iteris-pkg → New Version              (bump dataset)
01/02 lite baselines → save checkpoints as Datasets
DRL notebook: §0–§2 (warm-start) → §3 (headroom + dry-run) → §4 (full)
```

---

## 5. Phase C — Evaluation (Week 12)

Grounded in current medical-segmentation reporting practice (Metrics Reloaded; Wilcoxon + Bonferroni):

- **Metrics:** Dice, IoU, **HD95** (+ ASD optional), per class, on the held-out **test** set.
- **Comparisons (per class):**
  - lite baseline → lite + DuelingDDQN → lite + TD3 → attention U-Net (competitor).
  - Report **Δ over lite baseline** and **gap closed toward attention** (%).
- **Significance:** two-sided **Wilcoxon signed-rank** (paired, per-sample) vs the lite baseline, α = .05, **Bonferroni** across classes/agents.
- **Uncertainty:** 95% **bootstrap CIs** on the mean Δ.
- **Ablations:** global-morph (`*_GLOBAL`/DQN/DDPG) as negative control; PBRS-centred vs un-centred; discrete vs continuous; `headroom_report` ceiling vs achieved.
- **Harness:** one script over all checkpoints → the main results table.

---

## 6. Phase D — Paper (Weeks 13–14)

- **Methods:** contour-refinement MDP (state, angular-sector action, baseline-centred PBRS), DuelingDDQN vs TD3, lite-vs-attention design.
- **Results:** main table (2 agents × 4 targets, Dice/IoU/HD95 + significance) + qualitative grid (best/median/worst) + learning curves.
- **Discussion:** discrete vs continuous; where contour refinement helps (smooth CAMUS) vs plateaus (irregular BRISC); the ceiling analysis; global-morphology negative control.

---

## 7. Critical Don'ts

- Don't refine the **attention** net with RL — no headroom (that was the failed direction).
- Don't re-introduce global morphology or boundary tracing as a *contender* (ablation only).
- Don't deploy "best-seen" masks — no GT at test time; the agent must learn to STOP.
- Don't add scipy SDT to the reward hot loop — use the `ReplayBuffer` SDT cache.
- Don't hardcode hyperparameters — everything in YAML.
