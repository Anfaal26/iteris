# Iteris — Implementation Plan

> **Last updated:** 2026-06-02. Paradigm restored to local mask refinement.

---

## 1. Engineering Principles

1. **Notebooks are thin.** All logic in `iteris/`. Notebooks: import → configure → call → display.
2. **Configs are YAML.** No hardcoded hyperparameters. Every dataset/agent has a YAML in `configs/`.
3. **Episode-start baseline reward.** `r_t = metric(t) - metric(0)` — eliminates the oscillation trap.
4. **Hard-sample mining.** Weighting ∝ `exp((1-init_dice) * scale)` — amplifies signal from hard samples.

---

## 2. Current Status

### ✅ Complete

| Item | Notes |
|---|---|
| CAMUS + BRISC baselines | `camus_best.pt`, `brisc_best.pt` |
| `SegmentationEnv` v4 | 14 actions: dil/ero ×4 + shift ×4 + **smooth** + **stop**; fail-fast; explicit stop reward |
| All configs rebuilt | CAMUS c1/c2/c3 + BRISC — clean refinement blocks, no tracing params |
| Notebooks cleaned | IS_TRACING branches removed; AGENT_NAME = DQN\|DuelingDDQN\|DDPG |
| Paradigm 1 archived | `iteris/archive/paradigm1_boundary_tracing/` with resurrection notes |

### ⏳ Next up

1. **Bump `iteris-pkg`** on Kaggle to current commit
2. **BRISC DQN dry-run** (`RUN_DRY_RUN = True`) — validate v4 env runs cleanly
3. **BRISC DQN 30k steps** — first real run with 14-action space
4. **CAMUS c1 DQN 50k** — cardiac validation

---

## 3. Kaggle Workflow

```
git push origin main          (local)
Kaggle iteris-pkg → New Version  (bump dataset)
§0 → §1 → §2 (warm-start) → §4 (train)
```

Expected wall-clock: BRISC 30k ~10 min · CAMUS 50k ~60 min.

---

## 4. Config Reference

| Config | DQN steps | DuelingDDQN steps | DDPG steps |
|---|---|---|---|
| BRISC | 30k | 30k | 100k |
| CAMUS c1 (LVendo) | 50k | 50k | 100k |
| CAMUS c2 (LVepi) | 60k | 60k | 120k |
| CAMUS c3 (LA) | 50k | 50k | 100k |

---

## 5. Roadmap

### Phase A — Re-establish baseline with v4 env

| Step | Target |
|---|---|
| BRISC DQN/DuelingDDQN | val Dice ≥ 0.86 (above baseline 0.835) |
| CAMUS c1 DQN/DuelingDDQN | val Dice ≥ 0.93 (above baseline 0.938) |
| All DDPG runs | Continuous comparison baseline |

### Phase B — Evaluation + stats

- Unified eval harness over all checkpoints
- Wilcoxon signed-rank vs U-Net baseline (per-class)
- 5-fold CV
- Main results table: DQN · DuelingDDQN · DDPG × CAMUS c1/c2/c3 + BRISC

### Phase C — Paper

- Methods: SegmentationEnv v4 description, reward rationale
- Results: main table + qualitative grid (Easy/Medium/Hard per dataset)
- Discussion: smooth action impact · stop action analysis · comparison with boundary tracing (archived)

---

## 6. Adding a New Dataset

1. Add `configs/<dataset>.yaml` (normalize, num_classes, class_names)
2. Add `build_<dataset>_dicts()` in `iteris/ingestion.py`
3. Add `configs/<dataset>_drl.yaml` agent blocks
4. Copy a training notebook, swap config path

---

## 7. Critical Don'ts

- Do not add scipy-based SDT to the reward hot loop — use the precomputed cache in `ReplayBuffer`
- Do not hardcode hyperparameters — everything goes in YAML
- Do not re-introduce boundary tracing without a clear literature justification
