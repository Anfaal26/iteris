# Iteris Boundary-Tracing — Council-Reviewed Engineering Context

**Companion to:** `CONTEXT_PARADIGM1.md` (design rationale, paradigm pivot), `PLAN_PARADIGM1.md` (day-by-day implementation log).
**This document is the active decision and action plan as of 2026-05-31.**

**Status:** Reward redesign complete (commit `5901de5`); CAMUS step-budget, smart-seed, smoothness term pending. **No empirical signal yet on the new reward — first re-run is the next blocker.**

---

## 1. Executive summary

A discrete-action DRL agent segments medical images by tracing each structure's boundary one pixel at a time using 8 directional actions. The rasterised polygon IS the segmentation. Trained per-class, per-dataset:

| Dataset | Class | Selector keys |
|---|---|---|
| CAMUS | LV-endo (c1), LV-epi (c2), LA (c3) | `DQN`, `DuelingDDQN` (tracing) + `DDPG` (continuous baseline) |
| BRISC | tumor (c1) | same |

Three live agents — `DQN`, `DuelingDDQN`, `DDPG`. Refinement-paradigm discrete agents (`DDQN`, `MSA-DUELING`, BRISC 9-action env) are archived under `iteris/archive/` with resurrection instructions.

The first run on BRISC with the old reward (per-step −distance + flat closure bonus) confirmed reward hacking — val Dice peaked at 0.018 (step 15k) then declined to 0.012 (step 25k) while closure rate climbed to 51% with bimodal trace lengths (30 or 200). The reward was redesigned (commit `5901de5`) but has not been re-run. **A council of five reviewers analyzed the redesigned spec and identified five further issues.** This document consolidates the work done, council findings, and the prioritized roadmap.

---

## 2. Work completed

### 2.1 Commits landed (most recent first)

| Commit | What changed | Why |
|---|---|---|
| `5901de5` | **Region-aware reward redesign.** Terminal Dice (+10·D) dominant; conditional closure bonus (Dice-gated at 0.20); one-shot coverage bonus (+0.20 per new GT boundary pixel reached, each rewards once); soft distance gradient (−0.05·min(d,5)); hard off-image penalty (−10). Old `−dist + flat closure` retired. | First BRISC run confirmed reward hacking on the old reward. New design trains on the same metric used at eval. |
| `1b158d9` | **Seed Option A.** `best_overlap_cc()` (highest-IoU CC against GT) replaces largest-CC selection. `init_Dice < 0.30` falls back to GT-seeded trace. `dryrun_viz.py` diagnostic module added. | Dry-run on BRISC showed seeds landing on stray U-Net FPs far from the GT tumor. |
| `c6b69f2` | Archived `DDQNAgent`, `MSADuelingDQNAgent`, `msa.py`. Cleaned import graph; fixed orphan `**msa_kwargs` in DDPG ctor (latent crash). | Dead code from paradigm shift; user direction to archive rather than delete. |
| `7365270` | Notebook conflicts after paradigm shift (CAMUS/BRISC §4 timing tables, BRISC §6 title, paradigm-aware §10 train cell). | Stale references to DDQN/MSA-DUELING in markdown. |
| `c353550` | Retired discrete mask-morph refinement; archived `SegmentationEnvBRISC`. Configs slimmed. | User direction: "remove mask morphology from discrete agents." |
| `d97add8` | Replaced `DDQN`/`MSA-DUELING` selectors with `DQN`/`DuelingDDQN`/`DDPG`; case-insensitive selector lookup in `config.py`. | User direction: "remove DDQN from defaults; final set is DQN, DuelingDDQN, DDPG." |
| `495b8a0` | Paradigm 1 full implementation: `ContourTracingEnv`, `VectorisedContourEnv`, patch networks, `ContourReplayBuffer`, `_run_contour_training`. Three speedups: vectorised envs (~10×), precomputed GT-EDT (O(1) reward lookup), reduced eval frequency with val subset. | Initial paradigm shift; pre-empirical-test scope. |

### 2.2 Empirical results to date

**BRISC DuelingDDQN, 25k steps, OLD reward (commit pre-`5901de5`):**

```
step  5000 → val Dice 0.0126  closure 46%
step 10000 → val Dice 0.0144  closure 54%
step 15000 → val Dice 0.0182  closure 42%   ← peak
step 20000 → val Dice 0.0131  closure 56%   ← declining
step 25000 → val Dice 0.0124  closure 50%
Best val final-Dice: 0.0182    Per-sample gain: -0.97 across the board.
```

**Diagnosis:** Reward hacking — agent learned to close minimum-length loops near the seed. Bimodal trace lengths (30 or 200), action distribution collapsed to 2 of 8 directions. Pipeline correct, reward broken.

**Empirical results on new reward (commit `5901de5`):** **NONE YET.** This is the next blocker.

---

## 3. Council critical findings (5 reviewers)

Ranked by severity. Reviewer in parentheses.

### 3.1 CRITICAL — CAMUS step budget under-provisioned (R1)

**Math:**
- 16 envs × 25k gradient steps × 1 grad/step = 400k env transitions
- Average episode at mid-training ≈ 200 steps → ~2000 distinct episodes
- CAMUS train set ≈ 450 patients × 4 views = ~1800 images
- **Coverage: 1.1 episodes per image. Insufficient for stable Q-value learning.**

BRISC is safer: 25k × 16 / ~100-step BRISC episodes = ~4000 episodes for ~3000 train samples = ~1.3×/image, but BRISC images are simpler (single small target, no view ambiguity).

**Fix:** CAMUS train_steps **30k → 50k** (LV-endo, LA); **35k → 60k** (LV-epi). Scale `epsilon_decay_steps` proportionally.

### 3.2 CRITICAL — Premature-closure exploit margin too thin (R4)

Council ran the reward math at three scenarios:

| Scenario | Partial close (t=80, D=0.60) | Full trace (t=300, D=0.75) | Full trace (t=300, D=0.92) |
|---|---|---|---|
| Total reward | 10·0.6 + 2·0.6 − 0.01·80 = **6.40** | 10·0.75 + 2·0.75 − 0.01·300 = **6.00** | 10·0.92 + 2·0.92 − 0.01·300 = **8.04** |

**Insight:** Full trace beats partial-close **only when full Dice > ~0.75**. If mid-training Dice is below this threshold, the agent prefers premature closure → exploit returns in a different form.

**Fix (after seeing first run):** if observed mid-training Dice < 0.75, either:
- Raise `reward_terminal_dice` 10 → 15 (raises full-trace floor), OR
- Drop `reward_step_cost` −0.01 → −0.005 (reduces partial-close advantage)

Don't apply preemptively — pick after seeing the first run's trajectory.

### 3.3 HIGH — Seed policy fragile on BRISC (R2, R3)

Topmost-leftmost seeding works on roughly-convex CAMUS LVs (the apex is structurally consistent across patients). **Predictably fails on BRISC irregular lesions** — the "topmost" pixel of a spiculated mass at the bottom-right is in a completely different anatomical position than the same pixel on a round mass at the top-center. Agent must learn a different local-context policy per shape → inflated state space.

Published precedent (Yin et al. 2021, Mayy 2021) uses a learned First-P-Net for seed selection. Adding a learned seed network doubles engineering scope; **smarter heuristic is the mid-ground:**

**Fix:** Add `seed_method='centroid'`. Compute centroid of best-overlap CC, return boundary pixel with minimum L2 distance to it. Rotation/translation-invariant; consistent across lesion locations. Make it default in `brisc_drl_tumor.yaml`; keep `topmost` default in CAMUS configs (works there, and changing breaks the council's CAMUS validation chain).

### 3.4 HIGH — No directional smoothness term (R4)

Current reward is direction-blind. A trace that zigzags (E→W→E→W) while slowly progressing clockwise still earns coverage rewards and can produce a valid closed polygon — but with clinically unusable jagged boundaries.

**Fix:** Add `reward_smoothness_penalty` (default −0.02). Compute angular change between current direction vector `cu.DIRECTIONS[a_t]` and previous `cu.DIRECTIONS[a_{t-1}]` via dot product → angle. Apply penalty only when `|Δθ| > 45°` (allows normal curvature).

```python
# Pseudo-code for env_contour.step()
if t > 1:
    d_curr = cu.DIRECTIONS[a_t]
    d_prev = cu.DIRECTIONS[self._prev_action]
    cos_dtheta = np.dot(d_curr, d_prev) / (np.linalg.norm(d_curr) * np.linalg.norm(d_prev))
    dtheta_deg = np.degrees(np.arccos(np.clip(cos_dtheta, -1, 1)))
    if dtheta_deg > 45:
        reward += self.reward_smoothness_penalty * ((dtheta_deg - 45) / 135)  # 0 at 45°, 1 at 180°
self._prev_action = a_t
```

### 3.5 MEDIUM — Credit assignment at early steps (R1)

For a 400-step CAMUS episode, action at t=1 has its Q-value updated via γ^399 · terminal_Dice ≈ 0.018·D. Functionally zero. The coverage bonus partially compensates (~0.40/step) but terminal:per-step ratio is ~25:1.

**Mitigation:** n-step returns (n=5). Instead of 1-step TD target, sum 5 discounted rewards before bootstrapping. Propagates terminal credit 5× faster (effective horizon 400/5 = 80 steps for the first action).

```python
# Modify ContourReplayBuffer.sample() to return n-step transitions
# Modify DQNAgent.update() target computation:
#   y = sum(γ^k · r_{t+k} for k in range(n)) + γ^n · Q_target(s_{t+n}, argmax Q_online(s_{t+n}, ·)) · (1 - done)
```

Helps CAMUS more than BRISC (BRISC episodes are short enough that 1-step credit reaches the start). **Phase C** — implement after Phase A+B prove the core paradigm works.

### 3.6 MEDIUM — BRISC non-convex shape handling (R3)

Spiculated/lobulated lesions force the agent to navigate concavities while preserving winding direction. With `min_perimeter_steps=30` and `closure_tolerance=4` on a 30-px lesion, closure requires returning within 13% of lesion diameter. Tight on irregular shapes — expect >50% timeout rate in early BRISC training.

**Mitigation:** Loosen `min_perimeter_steps` 30 → 20 for BRISC. Combined with the Dice-gated closure bonus, the empty-loop exploit stays dead.

### 3.7 LOW — LA artificial closure (R3)

CAMUS LA annotations close the mitral valve plane artificially. The "natural" structural boundary is open. Tracing agent will produce inconsistent results near the valve regardless of training quality.

**Action:** No code change. Document in the paper; report LA results separately from LV-endo/LV-epi with the artificial-closure caveat.

### 3.8 What the council missed

Two issues the council didn't surface but matter:

1. **`dryrun_viz` hasn't been re-run since reward redesign.** All theoretical analysis is operating without empirical signal from the new reward. The first BRISC run is the universal blocker — until it produces a Dice trajectory, every other priority is provisional.

2. **Reward magnitudes weren't stress-tested per dataset.** R4 found the partial-close margin breaks at Dice<0.75 with current values. A proper grid sweep over `(terminal_dice, coverage_bonus, step_cost)` would tighten the margin systematically. Future work; not pre-Phase-A.

---

## 4. Prioritized roadmap

### Phase A — MUST-DO before any other change (in order)

**A1. Re-run BRISC `DuelingDDQN` 25k steps with the new reward (commit `5901de5`).**
- Push Kaggle iteris-pkg dataset to `5901de5`.
- Run `04_brisc_drl.ipynb` end-to-end. Save the history DataFrame.
- Validate against Section 5 checklist.
- **No further code changes until this produces results.**

**A2. Stress-test reward magnitudes against R4's math.**
- If observed mid-training (step 10k–15k) Dice < 0.75:
  - Option a: raise `reward_terminal_dice` 10 → 15 in `env_contour.py` default and all 4 YAMLs.
  - Option b: drop `reward_step_cost` −0.01 → −0.005 in same.
- Pick whichever has cleaner empirical effect on the partial-vs-full margin.
- Re-run final 10k steps of A1; confirm Dice stable or rising.

### Phase B — Address before CAMUS run

**B1. Bump CAMUS train_steps.**
- `configs/camus_drl_c1.yaml`: `train_steps: 30000 → 50000`, `epsilon_decay_steps: 18000 → 30000` (tracing block only).
- `configs/camus_drl_c2.yaml`: `35000 → 60000`, `20000 → 36000`.
- `configs/camus_drl_c3.yaml`: `30000 → 50000`, `18000 → 30000`.
- DDPG blocks unchanged.

**B2. Centroid-anchored seed.**
- Add `centroid_boundary_seed(cc)` helper in `iteris/contour_utils.py`.
- Extend `seed_point_from_init_mask` to accept `method='centroid'`.
- BRISC config: `seed_method: 'centroid'` (default).
- CAMUS configs: leave at `'topmost'` (works; changing breaks council validation chain).

**B3. Smoothness reward term.**
- Add `reward_smoothness_penalty` parameter (default −0.02) to `ContourTracingEnv.__init__`.
- Add `self._prev_action` tracking in `reset()`/`step()`.
- Apply smoothness penalty in `step()` only when `|Δθ| > 45°`.
- Add to `_ENV_KEYS` in both `contour_viz.py` and `drl_training.py`.
- Add a corresponding line to all 4 YAML reward blocks.

**B4. BRISC `min_perimeter_steps` 30 → 20.**
- One-line YAML edit in `brisc_drl_tumor.yaml`.

### Phase C — Should-do for paper-quality results

**C1. CAMUS `max_trace_length` reduction.** `c1: 400→320`, `c2: 480→360`, `c3: 420→320`. Halves credit-assignment horizon; more episodes per training step.

**C2. n-step returns (n=5) in DQN update.** Buffer + agent.update() modification (~60 LOC). Bigger win on CAMUS than BRISC.

**C3. Re-run all 4 (3 CAMUS classes + BRISC tumor) for DQN and DuelingDDQN.** Same protocol; compare against U-Net baseline.

### Phase D — Paper-time only

**D1.** LA reported separately with mitral-valve disclosure.
**D2.** Optional post-processing fallback for highly jagged BRISC masks (morphological close-then-open); report both numbers.
**D3.** Future-work: learned First-P-Net seed; PPO on-policy comparison.

---

## 5. Validation checklist (next run — BRISC `DuelingDDQN`, new reward, 25k)

After A1's §4 completes:

| Signal | Healthy | Warning | Critical (re-tune) |
|---|---|---|---|
| Val Dice at step 25k | ≥ 0.30 | 0.10–0.30 | < 0.10 |
| Val Dice trajectory | Monotonic rise | Slight plateau last 5k | **Peak-then-fall** (reward hacking back) |
| Closure rate at step 25k | 15–40% | 40–60% | >70% with low Dice (empty-loop exploit) |
| Trace length histogram | Bell around ~100 (BRISC perimeter) | Slight bimodal | Hard bimodal at 30 and 200 |
| Action distribution (8 dirs) | All bars > 5% | 1–2 directions at 0% | 5+ directions at 0% (collapse) |
| Per-sample `init→final` Dice | Most: final ≥ 0.5×init | Mixed but >40% improve | Final < init on >90% samples |

**If any "Critical" trips:** pull history DataFrame → compute partial-vs-full margin at observed Dice → apply A2 → re-run last 10k.

---

## 6. Decisions NOT to revisit

Listed to prevent re-litigation.

| Decision | Why settled |
|---|---|
| DDPG paradigm retained (mask morphology) | User explicit: "keep DDPG as it is." It's the paper's continuous baseline. |
| 8-direction discrete action | Council unanimously validated; matches Edge-Sensitive LV + LV Contouring papers. |
| Polygon rasterization output | Topology-preserving; closed contour → simply-connected mask by construction. |
| One-shot coverage invariant | Provably bounds coverage reward at `0.20·|B*|`; can't be farmed. |
| Patch CNN 64×64 local state | Council validated; fits T4 memory/compute. |
| Vectorised 16-env training | Council validated; dominant speedup. |
| Best-overlap CC seed selection (uses GT) | Curriculum aid; documented as paper limitation. |
| `DQN`, `DuelingDDQN`, `DDPG` selector set (no `_TRACE` suffix, no DDQN) | User direction. |
| Dead code archived to `iteris/archive/` | User direction: "rather than rewiring from dead codes, archive the dead ones." |

---

## 7. Known limitations (paper disclosures)

1. **GT used at training time** for both seed selection (best-overlap CC) and terminal reward (`+10·Dice`). Not a deployment-grade method without retraining the seed selector against a U-Net confidence map.
2. **LA mitral-valve closure is artificial.** Tracing accuracy near the valve is bounded by annotation consistency, not policy quality.
3. **Multi-focal lesions (BRISC) not supported** — largest-overlap CC only. Documented v1 limitation.
4. **Reward magnitudes tuned, not searched.** A proper grid sweep over `(terminal_dice, coverage_bonus, step_cost)` would tighten the partial-vs-full margin. Future work.
5. **No on-policy comparison (PPO).** DQN and DuelingDDQN are the discrete agents; PPO scoped out.

---

## 8. File map

| Path | Purpose |
|---|---|
| `iteris/env_contour.py` | `ContourTracingEnv` (reward + dynamics) + `VectorisedContourEnv`. |
| `iteris/contour_utils.py` | `DIRECTIONS`, `best_overlap_cc`, `seed_point_from_init_mask`, `gt_boundary_edt`, `rasterise_trajectory`. |
| `iteris/drl_training.py` | `_run_contour_training`, `_evaluate_contour`, `ENV_REGISTRY`, `AGENT_REGISTRY`. |
| `iteris/contour_viz.py` | Post-training viz (`plot_trace_comparison`, `plot_trajectory_playback`, `plot_direction_behaviour`, `evaluate_trace_testset`). |
| `iteris/dryrun_viz.py` | `dryrun_healthcheck` + `plot_dryrun` + `dryrun_report` — plumbing-check after §3. |
| `iteris/agents.py` | `DQNAgent`, `DuelingDQNAgent`, `DDPGAgent`, `OUNoise`. |
| `iteris/drl_networks.py` | `QNetwork`, `DuelingQNetwork` (DDPG fallback); `PatchQNetwork`, `PatchDuelingQNetwork` (tracing). |
| `iteris/buffer.py` | `ReplayBuffer` (DDPG), `ContourReplayBuffer` (tracing, float16 patches). |
| `iteris/archive/` | Retired code (`DDQNAgent`, `MSADuelingDQNAgent`, `msa.py`, `SegmentationEnvBRISC`) with resurrection instructions in each file's header. |
| `configs/camus_drl_c{1,2,3}.yaml` | CAMUS configs (LV-endo, LV-epi, LA). Tracing + DDPG blocks per file. |
| `configs/brisc_drl_tumor.yaml` | BRISC configs. |
| `notebooks/03{a,b,c}_camus_drl_*.ipynb`, `notebooks/04_brisc_drl.ipynb` | Per-(dataset, class) training notebooks. |
| `docs/CONTEXT_PARADIGM1.md` | Original paradigm-pivot rationale (kept as historical record). |
| `docs/PLAN_PARADIGM1.md` | Day-by-day implementation log. |
| `docs/CONTEXT_COUNCIL.md` | **This file.** Active decision/action plan. |

---

## 9. Next action (the one-liner)

> **Push commit `5901de5` to Kaggle iteris-pkg dataset → run BRISC §4 with `DuelingDDQN` 25k steps → validate against Section 5 → if green, proceed to Phase B (CAMUS step bump + centroid seed + smoothness term + BRISC min_perimeter loosening). Each Phase-B item is a one-commit change.**

---

## 10. Decision log

| Date | Decision | Source |
|---|---|---|
| 2026-05-29 | Paradigm 1 (boundary tracing) selected over mask-morph refinement for discrete agents. | User (see `CONTEXT_PARADIGM1.md`). |
| 2026-05-30 | DDPG retained as continuous baseline; refinement code archived. | User: "keep DDPG as it is, change the discrete ones." |
| 2026-05-30 | Selector set: `DQN`, `DuelingDDQN`, `DDPG` (no `_TRACE` suffix, no DDQN). | User direction. |
| 2026-05-30 | Dead code archived rather than deleted (`DDQNAgent`, `MSADuelingDQNAgent`, `msa.py`, `SegmentationEnvBRISC`). | User: "archive the dead ones." |
| 2026-05-30 | Seed Option A (best-overlap CC + GT fallback at init_Dice < 0.30). | User selected from offered options. |
| 2026-05-30 | Region-aware reward redesign after BRISC reward-hacking observed. | User: "Strip off current reward system and reconfigure." |
| 2026-05-31 | This document supersedes `CONTEXT.md` as the active context; council findings folded in; phased roadmap committed. | This turn. |
