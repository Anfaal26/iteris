# Iteris — Project Context

> Single source of truth. **Last updated:** 2026-06-02.

---

## 1. What This Is

**Iteris** is a Taylor's University capstone (PRJ63504) research paper on DRL-based medical image segmentation via **local mask refinement**.

- **Research question:** Can a discrete DRL agent refine a U-Net segmentation by applying local morphological operations, and does it outperform a continuous DDPG baseline?
- **Paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop
- **Repo:** `github.com/Anfaal26/iteris` · local: `D:\iteris\`

---

## 2. Paradigm — Local Mask Refinement

The agent starts from the U-Net's predicted mask and refines it by applying local operations at each step. **This is NOT pixel-by-pixel boundary tracing** (that approach was tried and retired — see `iteris/archive/paradigm1_boundary_tracing/`).

### Why refinement beats tracing (supervisor's analysis)
1. **No staircase artifact** — morphological operations are topology-preserving and globally smooth
2. **Reversible** — the agent can dilate then erode to undo a bad step; tracing cannot undo
3. **Short episodes** — 5-20 steps vs 200-400 steps for tracing → better credit assignment
4. **Literature grounded** — morphological RL refinement matches Sahba et al., Active Contour RL, rNCA

---

## 3. Datasets

| Dataset | Modality | Classes | U-Net baseline |
|---|---|---|---|
| **CAMUS** | Cardiac ultrasound | LV-endo (c1), LV-epi (c2), LA (c3) | 0.938 / 0.872 / 0.896 Dice |
| **BRISC** | Brain tumour T1+Gd MRI | tumor (binary) | 0.835 Dice (test) |

---

## 4. Active Agent Set

| Selector | Algorithm | Action space | Env |
|---|---|---|---|
| `DQN` *(default)* | DQNAgent | 14 discrete actions | `SegmentationEnv` |
| `DuelingDDQN` | DuelingDQNAgent (V+A) | 14 discrete actions | `SegmentationEnv` |
| `DDPG` | DDPGAgent (continuous) | 3-D continuous (morph, dy, dx) | `SegmentationEnv` |

---

## 5. SegmentationEnv v4 — Action Space (14 actions)

| Action | Name | What it does |
|---|---|---|
| 0–3 | dil-N/E/S/W | Expand boundary 1px outward in one direction (directional SE) |
| 4–7 | ero-N/E/S/W | Shrink boundary 1px inward from one direction |
| 8–11 | sh-↑/↓/←/→ | Translate entire mask by `shift_px` pixels |
| **12** | **smooth** | **Morphological closing (3×3) — fills holes, softens staircase** |
| **13** | **stop** | **Explicit terminal: agent signals "satisfied"** |

**State** `(4, 256, 256)`: image · current mask · SDT of mask · U-Net init mask.

**Reward modes** (per-class YAML):
- `dice_hd_composite` — α·ΔDice + β·ΔHD95 (CAMUS LV_endo, LV_epi)
- `dice_delta` — ΔDice only (CAMUS LA — shape variability makes HD95 unreliable)
- `iou_delta` — ΔIOU (BRISC — smoother for small targets)

All rewards use **episode-start baseline** (`r_t = metric(t) - metric(0)`) to eliminate the oscillation trap.

**New in v4 vs v3:**
- Action 12 (smooth): morphological closing. Agent learns when to call it.
- Action 13 (stop): explicit termination. Encourages stopping when mask is good.
- Optional `fail_thresh`/`fail_n` in constructor: fail-fast for small targets (BRISC).

---

## 6. File Map

| Path | Purpose |
|---|---|
| `iteris/env.py` | `SegmentationEnv` v4 — 14-action refinement environment |
| `iteris/agents.py` | `DQNAgent`, `DuelingDQNAgent`, `DDPGAgent` |
| `iteris/drl_networks.py` | `QNetwork`, `DuelingQNetwork`, `Actor`, `Critic` |
| `iteris/buffer.py` | `ReplayBuffer` (memory-optimised, SDT-caching) |
| `iteris/drl_training.py` | `run_drl_training` — main training loop |
| `iteris/env.py` | Helpers: `dice_score`, `hd95_px`, `signed_dt`, SEs |
| `iteris/warm_start.py` | U-Net inference → init masks |
| `iteris/archive/` | Retired code (MSA, DDQN, SegmentationEnvBRISC) |
| `iteris/archive/paradigm1_boundary_tracing/` | **Retired boundary-tracing paradigm** |
| `configs/camus_drl_c{1,2,3}.yaml` | CAMUS per-class DRL configs |
| `configs/brisc_drl_tumor.yaml` | BRISC DRL config |
| `notebooks/03{a,b,c}_camus_drl_*.ipynb` | CAMUS training notebooks |
| `notebooks/04_brisc_drl.ipynb` | BRISC training notebook |

---

## 7. Current Status

| Item | Status |
|---|---|
| CAMUS U-Net baseline | ✅ `camus_best.pt` — Dice 0.938/0.872/0.896 |
| BRISC U-Net baseline | ✅ `brisc_best.pt` — Dice 0.835 |
| SegmentationEnv v4 (14 actions) | ✅ Implemented |
| DRL training loop | ✅ Hard-sample mining, early stop, fail-fast |
| Configs / notebooks | ✅ Restored to refinement paradigm |
| First DRL training runs | ⏳ Pending (re-run with v4 env) |

---

## 8. Settled Decisions

| Decision | Rationale |
|---|---|
| 14-action space (not 13) | Adds `smooth` and `stop` per supervisor recommendation |
| DDPG kept as continuous baseline | Necessary for paper comparison: discrete vs continuous |
| Episode-start baseline reward | Eliminates per-step oscillation trap |
| Hard-sample mining (scale 5.0) | Prevents "easy majority" drowning the Q-signal |
| Boundary tracing paradigm archived | Staircase artifacts + credit-assignment failures; see archive/ |

---

## 9. Decision Log

| Date | Decision |
|---|---|
| 2026-05-29 | Paradigm 1 (boundary tracing) implemented and tested |
| 2026-05-30 | BRISC results: val Dice ~0.85, persistent staircase artifacts |
| 2026-06-02 | Supervisor: pivot back to local mask refinement with improved actions |
| 2026-06-02 | SegmentationEnv upgraded to v4: smooth + stop actions; fail-fast added |
| 2026-06-02 | Boundary tracing archived to `iteris/archive/paradigm1_boundary_tracing/` |
