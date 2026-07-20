# Iteris — Decision & Change Transcript

> Append-only log: every decision, problem diagnosed, and change made on this project, in chronological order.
> **Maintenance rule (do this every time something new happens):** add a dated entry below *immediately* after making a decision, diagnosing a problem, or landing a change — whether via direct edit, commit, or experiment result. If the change also alters a convention, gotcha, or piece of standing knowledge, mirror that update into [SKILLS.md](SKILLS.md) in the same turn. Newest entries go at the bottom. Keep entries terse — one entry per decision/problem/change, not per file touched.

---

## 2026-06-11 — `37751f9` Fix DRL non-learning: baseline-centred + scaled PBRS potential

**Problem:** No agent could beat the U-Net baseline. Root cause: PBRS potential `Φ = Dice` at a ~0.94 baseline made *holding* pay `(γ−1)·Φ ≈ −0.009/step` — every agent collapsed to STOP-at-baseline (reward path-dependence + discount drag).
**Fix:** Baseline-centred scaled potential `Φ(s) = K·(Dice(s) − Dice_0)`, `K≈10` (CAMUS) / `K≈15` (BRISC), in `iteris/env.py`.

## 2026-06-20 — `9c27b3b` Add TD3 on angular-sector contour action

**Problem:** Global 3-D DDPG (SDT-threshold morph + translation) structurally capped at baseline — best response to a near-perfect mask is identity.
**Fix:** TD3 (twin critics, target-policy smoothing, delayed updates) on a new contour env, action = per-sector continuous displacement along outward normals. Angular (not index-based) sectoring keeps the action↔location map stable across samples → learnable.

## 2026-06-21 — `7da94b0`→`12948a5` Pivot to lite U-Net baseline; archive Paradigm A

**Problem:** Real CAMUS/BRISC runs confirmed RL can't beat a *strong* baseline (best-seen ≈ baseline = structural ceiling, not a bug).
**Decision:** RL now refines a deliberately weak `LiteUNet` (0.48M params, real headroom); `AttentionResUNet` becomes the fixed upper-bound competitor. Only two algorithms going forward: **DuelingDDQN** (discrete sectors) and **TD3** (continuous sectors), both on the contour env. Global morphology archived to `iteris/archive_paradigm_a/` (ablation only). Shared helpers moved to `iteris/geometry.py`; `iteris/env.py` is now a back-compat shim. Added `iteris/diagnostics.py::headroom_report` as a cheap go/no-go gate before any full training run.

## 2026-06-22 — `54a3a3a` Tune contour geometry from first 20k-step Kaggle runs

**Change:** `n_points=32`, `disp_px=2.0` matched to diagnostic for DuelingDDQN config; `spline_smooth: 0.0` for BRISC (smoothing was destroying irregular tumour boundaries); `max_steps` capped at 10 (was 20 — agent wandering past its peak Dice and degrading).

## 2026-06-23 — `39a296c` DRL fix batch P0+P1+P2 (10-agent adversarial review)

**Diagnosis (severity-ranked):**
- **Severity 4, confirmed:** GT-privileged "headroom" was a mirage — `oracle_greedy()` in `diagnostics.py` picked actions by reading ground truth, so prior "+0.061 Dice headroom" verdicts were unreachable by any GT-blind deployed policy.
- **Severity 4, confirmed:** Reward-clip trap — `reward_clip=1.0` floored both a catastrophic move and a mildly-bad move to the same −1.0, erasing the gradient between them; "do nothing" became the rational policy.
- **Severity 2–3, amplifier (not root cause):** Actor/Q-heads read all per-sector outputs from one spatially-collapsed global embedding (`AdaptiveAvgPool2d→Linear`) — can't cleanly localize "which sector is wrong."
- TD3's deterministic actor with small-init final layer starts near identity (a≈0) with no warm start — slow to escape the do-nothing basin.

**Fixes:**
- P0.1 — uncertainty gate (`env_contour_refine.py`): clamps edit magnitude by U-Net confidence band, bounds worst-case per-step Dice drop.
- P0.2/P0.4 — `reward_clip` 1.0→4.0; epsilon decay 30k→18k steps (TD3/DuelingDDQN blocks only).
- P0.3 — honest, non-GT-privileged headroom metric added to `diagnostics.py`.
- P1 — `iteris/bc_demo.py` (`collect_continuous_oracle_demos`, `DemoBuffer`) + `TD3Agent.pretrain_actor_bc()` for BC warm-start. Verified: 26.5× output-magnitude increase over a fresh actor (escapes identity basin).
- P2 — opt-in (`spatial_head: false` default) `SectorPool`/`SpatialActor`/`SpatialCritic`/`SpatialDuelingQNetwork` in `drl_networks.py`.

## 2026-06-24 — `ef39259` Multi-GPU runner; `02354fe` notebook viz fixes

- Added `scripts/run_drl_config.py` to run two configs concurrently across Kaggle's two T4s (`OMP_NUM_THREADS=1` to avoid CPU oversubscription — bottleneck is CPU-bound `scipy`/`skimage` in `env.step()`, not GPU).
- Fixed missing `plt` import and missing `env_cls` pass-through in `build_replays`/`evaluate_testset` across all 4 Kaggle DRL notebooks. Also fixed (same session, untracked in this log until now): `NameError: agent` in dry-run-only cells, `TypeError` on `SegmentationEnv.__init__()` from viz auto-detecting the wrong env class.

## 2026-06-25 — `c2d656b` Pillar 1 (value-floored deploy) + Pillar 4 (offline diagnostics)

**Context:** a 6-pillar fix plan was scoped after live runs showed CAMUS-LA TD3 flat at best-seen ≈0.833 for 58k/60k steps, `final` persistently below `init` — traced to TD3 having **no learned stop/commit action** and a reward-blind `_check_termination()` (only checks if Dice *stopped changing*, never if it converged to something *good*; `fail_thresh`/`fail_n` safety net defaults to disabled and no TD3 config overrides it).
**Decision:** implement only Pillars 1 & 4 now (free, no retrain needed); defer 2/3/5/6.
- Pillar 1 — `state_value()` added to `DQNAgent`/`DDPGAgent`/`TD3Agent` (GT-free value estimate); value-floored deploy selector wired into `replay_one`/`evaluate_testset` in `refinement_viz.py` — never deploys a state valued below the initial state.
- Pillar 4 — offline diagnostics appended to `diagnostics.py`: `pillar4_report` (prob_map informativeness), error-type audit (boundary vs topology vs interior), value-floored-delta sanity check.
- **Left open:** Pillar 2 (TD3 learned stop/commit head — the actual fix for the root cause above) and Pillar 3 (drop HD95 from the training reward, eval-only) — not implemented. User interrupted mid-implementation of Pillar 2 and asked for a plan instead.
- Last real numbers before this batch: BRISC tumour DuelingDDQN init 0.8304 → best Δ −0.0005 at step 50k/60k (closing the gap). CAMUS-LA TD3 init 0.8274 → best-seen flat ~0.8335, final Δ −0.0039 to −0.0082 (never crosses zero).

## 2026-06-26 — Session continuity: re-derived prior chat, created this transcript + SKILLS.md

**Context:** new session ("iteris.v3.") picked up from chat "Iteris reward path-dependence and contour refinement" (ended 2026-06-25, mid-investigation of mm-based/BIoU metrics when the session limit hit). Re-confirmed against live repo (`HEAD=c2d656b`, in sync with `origin/main`, two untracked notebooks `colab_brisc_drl.ipynb`/`colab_camus_drl.ipynb` present but not yet committed — purpose unconfirmed, likely an abandoned/parallel Colab multi-GPU attempt).
**Decision:** before further fixes, create durable per-project memory: this transcript (chronological decision/problem/change log) and [SKILLS.md](SKILLS.md) (living operational playbook). Both must be kept current going forward — see maintenance rule at the top of each file.
**Open threads carried forward, not yet picked:**
1. Pillar 2 — TD3 learned stop/commit head (fixes the confirmed root cause of TD3's below-baseline drift).
2. Pillar 3 — drop HD95 from the training reward (`dice_hd_pbrs` → `dice_pbrs`), keep HD95 eval-only.
3. mm-based metrics / BIoU — requested by user, citing literature; no `pixel_spacing`/`voxel_spacing` conversion helper exists yet in `transforms.py`/`config.py` despite `spacing` being referenced — this would be new plumbing.

## 2026-06-26 — Added `notebooks/00_free_diagnostics_all_classes.ipynb`

**Decision:** before any GPU training round, run the free (CPU-only) Pillar 4 + headroom diagnostics across every class in one notebook, instead of one-off per-notebook snippets. New notebook warm-starts the lite U-Net per class and runs `pillar4_report` + `headroom_report` for CAMUS (LV-endo/LV-epi/LA) and BRISC (generic tumour by default; glioma/meningioma/pituitary subtypes gated behind `RUN_BRISC_SUBTYPES=False` since each re-runs a full ~8–15 min U-Net pass and is explicitly optional per `PLAN.md`). Geometry params (`n_points`/`cont_sectors`/`disp_px`/`spline_smooth`) are read off each config's `TD3` block, which is the only block carrying the full locked geometry for every class — used purely as a geometry source, independent of which agent eventually trains. Ends with a summary table (`/kaggle/working/diagnostics_summary.csv`) and a plain-language go/no-go verdict per class. `attention_dice` reference values only cover CAMUS (3 classes) + BRISC generic tumour — subtypes report ceiling-only (no realistic-headroom number) until subtype-specific attention U-Net Dice is measured.
**Also confirmed (was flagged "unconfirmed" on 2026-06-26 in SKILLS.md):** `notebooks/colab_brisc_drl.ipynb` / `colab_camus_drl.ipynb` are dated 2026-06-20 — i.e. one day *before* the lite-U-Net pivot (`7da94b0`, 2026-06-21). They're stale/pre-pivot artefacts, not in-progress work — safe to ignore or delete, not a thread to pick up.

## 2026-06-26 — Diagnostics run (Kaggle) returned: results in, CAMUS prioritised over BRISC

**Results** (`00_free_diagnostics_all_classes.ipynb`, 2026-06-25 21:47 Kaggle run, saved `diagnostics_summary.csv`):

| Class | baseline Dice | prob_map | boundary_frac | realistic headroom | Pillar4 |
|---|---|---|---|---|---|
| CAMUS LV_endo | 0.8914 | INERT (band_frac 0.0073) | 0.715 | **+0.0466** | low payoff (gate) |
| CAMUS LV_epi | 0.7953 | USABLE (band_frac 0.0193) | 0.706 | **+0.0767** | worth training |
| CAMUS LA | 0.8474 | INERT (band_frac 0.0063) | 0.716 | **+0.0486** | low payoff (gate) |
| BRISC tumor | 0.8851 | INERT (band_frac 0.0007) | 0.798 | **−0.0501** | NO HEADROOM |

**Decision:** pause BRISC entirely (no realistic headroom — lite baseline 0.885 is already above the attention competitor's 0.835 on this class; training would chase a negative target). Focus on CAMUS, where all 3 classes have real, boundary-shaped headroom.

**Diagnosis (new, from this run):** verified the `uncertainty_gate` mechanism in `env_contour_refine.py:387-403` — gate weight is 1.0 only for prob_map values in `[gate_lo, gate_hi]=[0.35,0.65]`, ramping to 0.0 outside (fully zero below 0.25 / above 0.75). For LV_endo and LA, `band_frac` of 0.006-0.007 means ~99.3% of contour points fall outside the band — so `uncertainty_gate: true` (set in all 3 CAMUS DuelingDDQN/TD3 blocks since the P0.1 fix) isn't neutral for these two classes, it clamps edit magnitude to ~0 at nearly every point, blocking the confirmed real headroom. LV_epi's prob_map is graded enough (`band_frac` ~2.7× higher) that the gate functions as intended there.

**Fix applied (DuelingDDQN only, per user request — TD3 blocks untouched):**
- `configs/camus_drl_c1.yaml` (LV_endo) DuelingDDQN block: `uncertainty_gate: true` → `false`.
- `configs/camus_drl_c3.yaml` (LA) DuelingDDQN block: `uncertainty_gate: true` → `false`.
- `configs/camus_drl_c2.yaml` (LV_epi) DuelingDDQN: **unchanged** (`true`) — prob_map is usable, gate is functioning, Pillar4 already says train as-is.

**Trade-off accepted:** disabling the gate removes the P0.1 "bounded worst-case per-step Dice drop" safety property for LV_endo/LA. Judged acceptable because the gate currently provides zero benefit on these classes (can't distinguish real errors from confident-correct regions when prob_map is ~binary) while actively preventing the agent from reaching the confirmed headroom.

**Deferred (longer-term, costs a GPU round):** retrain the lite U-Net with label smoothing/temperature scaling to give LV_endo/LA a genuinely graded `prob_map`, then re-enable the gate with real benefit. Not done now — the free config fix unblocks the next training round immediately; this retrain is only worth it if the gate-off run underperforms.
**Same INERT diagnosis applies to TD3's LV_endo/LA blocks** (gate also `true` there) — not touched in this pass since the user scoped the ask to DuelingDDQN; revisit if/when TD3 training resumes on these classes.

## 2026-06-26 — Clarified: gate-disable unblocks, retrain would improve — decided success criterion

**User concern:** doesn't want training to "merely equate baseline" (i.e. just satisfy Pillar 1's value-floor safety net) — wants genuine improvement, and asked whether the lite-U-Net label-smoothing retrain would be more fruitful than the gate-disable fix already applied.

**Clarification (no code changed, decision/criterion only):**
- The `uncertainty_gate: false` fix (previous entry) only *removes an obstacle* — it makes improvement possible, it adds no learning signal.
- The deferred lite-U-Net label-smoothing retrain would do two things the gate fix can't: (1) let the gate actually function as designed (bound bad edits + focus learning on correctable regions) instead of being off; (2) fix `prob_map` as one of the network's 5 input state channels (`image, mask, sdt, init_mask, prob_map` — `env_contour_refine.py:117`) — currently near-redundant with `init_mask` for LV_endo/LA, so the agent gets no real "where am I uncertain" signal into its policy/value heads either. This is a stronger, more mechanistic case for actually improving over baseline, not just unblocking — but still not a guarantee (RL outcomes are empirical).
- **Decided success criterion for the upcoming DuelingDDQN run:** judge by how much of the realistic headroom is closed (LV_endo target ceiling 0.938 from 0.891 baseline; LA target ceiling 0.896 from 0.847 baseline), **not** by whether the value-floored deploy stays ≥ baseline (Pillar 1 guarantees that by construction — it's a safety floor, not a success signal).
- **Decided sequencing:** run the free gate-fixed DuelingDDQN training first (no added cost, a precondition either way). If raw/value-floored Dice closes a meaningful chunk of the headroom → gate fix alone was sufficient, skip the retrain. If it stays flat/marginal → that's evidence for the label-smoothing retrain, pursue it next (it costs a full GPU round: lite U-Net retrain + re-running the warm-start).

## 2026-06-26 — User chose to retrain first; added `loss_label_smoothing` support

**Decision:** user opted to skip the empirical sequencing above and retrain the CAMUS lite U-Net with label smoothing immediately, rather than waiting on a DuelingDDQN run first.

**Change:**
- `iteris/losses.py::build_loss` — added `label_smoothing` param (read from new `cfg['loss_label_smoothing']`, default `0.0` — fully backward-compatible, no other config affected). Wired straight into MONAI `DiceCELoss(label_smoothing=...)`, which passes it to the underlying `torch.nn.CrossEntropyLoss` — confirmed by reading MONAI's source (`monai/losses/dice.py`); only touches the CE term, not the Dice term's `smooth_nr`/`smooth_dr` (those are unrelated numerical-stability epsilons).
- `configs/camus_lite.yaml` — added `loss_label_smoothing: 0.1` (standard literature default for label-smoothing regularisation). `configs/brisc_lite.yaml` left untouched — BRISC is paused (no realistic headroom, see 2026-06-25 entry).

**Next steps (not yet done):**
1. Retrain on Kaggle via `notebooks/01_camus_lite.ipynb` (no notebook changes needed — it just calls `load_config('camus_lite.yaml')` and the new key flows through automatically). Produces a new `camus_lite_unet_best.pt`.
2. Re-run `notebooks/00_free_diagnostics_all_classes.ipynb` (CAMUS section) against the new checkpoint to confirm `prob_map_informativeness` now reports **USABLE** (not INERT) for LV_endo and LA, and check whether `baseline_dice` shifted (label smoothing can marginally move raw Dice, which would shift the realistic-headroom numbers too).
3. **Only if confirmed USABLE:** re-enable `uncertainty_gate: true` for LV_endo/LA DuelingDDQN in `camus_drl_c1.yaml`/`camus_drl_c3.yaml` — re-enabling on a still-INERT prob_map would reintroduce the original blocking bug.

## 2026-06-26 — Thorough TD3 review (CAMUS) + fixes; DuelingDDQN c3 run done

**Context:** user finished the DuelingDDQN run on CAMUS class 3 (LA) and asked for a thorough TD3 review + fix, CAMUS first. Reviewed the full TD3 path: `agents.py::TD3Agent`, `drl_training.py` (wiring, episode loop, eval), `env_contour_refine.py` (`_apply_continuous`, `_check_termination`, gate), `bc_demo.py`, `drl_networks.py` (Actor/SpatialActor init), and all 3 CAMUS TD3 config blocks.

**Findings (ranked):**
1. **CRITICAL — INERT gate poisons the *entire* TD3+BC pipeline for c1/c3, not just RL.** `_apply_continuous` multiplies the actor's per-point displacement by `_gate_weights` (`env_contour_refine.py:498`). With the INERT prob_map (band_frac 0.0073 c1 / 0.0063 c3) the gate is ~0 at ~99% of points, so (a) the RL actor physically can't move the contour, AND (b) `bc_demo.collect_continuous_oracle_demos` builds its env from the same `env_kwargs`+prob_map, so the oracle's trial steps are also gated to ~0 → all candidate actions look equal → oracle returns ~zero vectors → BC pretrains the actor toward identity. So the gate bug doesn't just stall RL, it makes the BC warm-start (the whole point of which is to escape the identity basin) teach identity. This is the dominant blocker for TD3 on c1/c3.
2. **CRITICAL (structural, pre-existing) — reward-blind termination commits sub-baseline masks.** TD3 has no STOP action; with `disable_auto_stop: false` the `dice_converged` branch of `_check_termination` (`env_contour_refine.py:364-374`) terminates whenever Dice stops *changing*, regardless of whether it's above baseline. `final_dice` (= `info['dice']` at the done step in `evaluate_agent`) is the deploy number AND the checkpoint-selection metric, so a drifting actor that converges below baseline gets committed. `fail_thresh`/`fail_n` (the safety net) defaulted to disabled and no TD3 config overrode them.
3. **NOTED (not changed) — checkpoint-selection metric ≠ deployment metric.** `evaluate_agent` selects the best checkpoint on raw `final_dice_mean` (can be < baseline). The value-floored "do-no-harm" selector (Pillar 1) exists only in `refinement_viz.evaluate_testset`, not in the training-time eval. So the checkpoint is chosen on a more permissive metric than the one used at deployment. Left unchanged deliberately — wiring the value floor into `evaluate_agent` changes discrete-agent eval semantics too and would make the just-completed DuelingDDQN c3 run non-comparable; flagged as a separate decision needing approval.
4. Checked and OK: eval uses `explore=False` (no noise leak); actor update uses critic1 only (standard TD3); BC demos regenerated per-run (no stale artifact to fix).

**Fixes applied (TD3-scoped, config-only, reversible):**
- `camus_drl_c1.yaml` (LV_endo) + `camus_drl_c3.yaml` (LA) TD3 blocks: `uncertainty_gate: true → false` (mirrors the DuelingDDQN decision; same re-enable-after-USABLE-confirmation gate). Comments cite the BC-poisoning mechanism.
- All 3 TD3 blocks (c1/c2/c3): added `fail_thresh: 0.02`, `fail_n: 2` — the documented interim Pillar-2 safety net so a drifting actor can't lock in a final mask >0.02 Dice below baseline.
- `camus_drl_c2.yaml` (LV_epi) gate KEPT `true` — its prob_map is USABLE (band_frac 0.0193).
- Verified all three YAMLs parse and resolve (`gate` c1=False/c2=True/c3=False; `fail_thresh=0.02`, `fail_n=2` on all three).

**Not done (deferred / needs decision):**
- Finding 3 (value-floored checkpoint metric in `evaluate_agent`) — needs approval; affects discrete agents + completed c3 run comparability.
- Pillar 2 proper (TD3 *learned* stop/commit head) — still the real fix; the `fail_thresh` net is only a backstop.
- Same INERT-gate fix is now applied to BOTH Dueling and TD3 for c1/c3 → when the label-smoothing retrain is confirmed USABLE, flip all four blocks (Dueling+TD3 × c1+c3) back on together.

## 2026-06-26 — Diagnosed the DuelingDDQN c3 (LA) run: it ran STALE code (none of our fixes)

**User panic:** 60k-step DuelingDDQN LA run ended with final Dice 0.82 < baseline 0.829 (Δ −0.009 to −0.012 every eval), best-seen only ~0.834 (+0.005 over baseline), and crashed at the end on `NameError: name 'plt' is not defined`.

**Diagnosis — the run contained NONE of this session's fixes; it executed stale code:**
1. **Gate was ON.** My `uncertainty_gate: false` edits are *uncommitted* (working tree only); the committed config (= what the Kaggle `iteris-pkg` dataset carries) still has `uncertainty_gate: true` for c3 DuelingDDQN. With the INERT prob_map (band_frac 0.0063), `_push_sector` (env_contour_refine.py:468) multiplies every OUT/IN edit by gate≈0, so ~99% of the agent's pushes were near-no-ops → the agent was paralyzed → best-seen barely cleared baseline (+0.005 vs the +0.049 realistic headroom). This is exactly the pre-fix failure mode, re-observed.
2. **The log shows raw final, not the deployable number.** `evaluate_agent` (training-time eval + checkpoint metric) reports raw final Dice with NO value floor (finding #3 from the 2026-06-26 TD3 review). best-seen 0.834 > baseline means the agent *passes through* above-baseline masks but overshoots and commits a worse one. The value-floored deploy (Pillar 1, in `refinement_viz.evaluate_testset`) would revert to init when the agent's own value says final < init → actual deployable ≈ baseline, not 0.82. The user never saw that number because…
3. **The notebook was stale → crashed before eval.** The `plt` crash at In[7] is impossible on the repo notebook (cell 12 imports plt and printed "Built 8 replays"). It can only happen if the executed notebook predates commit `02354fe`. So the Kaggle notebook (and almost certainly the iteris-pkg dataset) is behind repo HEAD. The crash also killed the `evaluate_testset` cell, hiding the value-floored number.

**Root cause umbrella:** stale Kaggle environment. The committed repo (c2d656b) and the Kaggle datasets/notebooks are behind the working tree; this run used the old gate-ON config, old env code, old notebook, and the old (overconfident, no-label-smoothing) lite checkpoint.

**Fix applied this turn:** hardened all 4 DRL notebooks (`03a/03b/03c/04`) — added a self-contained `import matplotlib.pyplot as plt` to the 3 plotting cells in each so they never depend on an earlier setup cell having run (defends against stale/partial runs). Validated all 4 still parse as JSON.

**Action plan handed to user (nothing pushed — awaiting their go):** commit + push the uncommitted fixes (configs gate-off + fail-safe, `losses.py` label smoothing, hardened notebooks, new diagnostics notebook) → bump the Kaggle `iteris-pkg` dataset to the new version → re-upload/sync the notebook → retrain the lite U-Net with label smoothing (their chosen first step) → re-run diagnostics to confirm USABLE → then the DRL runs will actually contain the fixes.

## 2026-06-26 — Chunked training + resume (don't burn 7h to diagnose a run)

**User ask:** the lite U-Net is retrained (label smoothing) and the c3 run *did* use the new lite outputs (so my "stale code" call was partly off — they bumped iteris; the residual issue was likely viz/notebook sync). They want to split a 60k run into a short diagnostic chunk (~20k) to catch a broken config without spending the full ~7h quota.

**Implemented:**
- `iteris/drl_training.py::_save_agent` — now persists optimizer state (generic: scans the agent for `torch.optim.Optimizer` attrs, so DQN `self.opt` and TD3 `actor_opt`/`critic_opt` all covered) + the `step` counter, alongside agent weights / best_dice / history. Backward-compatible (old checkpoints lack these keys → treated as missing).
- `run_drl_training` — opt-in `cfg['resume']`. Saves two checkpoints now: `*_best.pt` (best-val, deployment) and `*_last.pt` (latest full state, written every eval). On `resume=True` it loads `*_last.pt`: restores agent weights, optimizer moments, `step` (so epsilon + bc_lambda schedules continue — NOT reset to 1.0, which is the trap that would trash a warm policy), history, and best_dice; then `pbar`/loop run `start_step→train_steps`. Replay buffer is intentionally NOT persisted (too large) — it re-prefills cold, but eval is greedy so the resumed curve stays continuous; documented inline.
- All 4 DRL notebooks (`03a/03b/03c/04`) — prepended a "Diagnostic / quota controls" block to the **real** training cell (cell 10) exposing `TRAIN_STEPS` (default 20000 for the diagnostic chunk; None = full config run) and `RESUME` (False; True continues from `*_last.pt`). Recipe in-comment: run 20k → inspect §9 curve + §10 value-floored eval → if trending up, set TRAIN_STEPS=60000 + RESUME=True to finish the remaining 40k without redoing.
- **Gotcha hit + fixed during this edit:** `run_drl_training(` appears in BOTH the optional dry-run cell and the real-train cell; first attempt prepended the block to the dry-run cell. Corrected to target the cell that also assigns `result` and lacks `RUN_DRY_RUN`. Verified exactly one block per notebook, on cell 10, all code cells parse.

**Workflow now:** first pass `TRAIN_STEPS=20000, RESUME=False` (~2.3h) → diagnose → continue `TRAIN_STEPS=60000, RESUME=True` (~4.6h, no redo) instead of a blind 7h run. Still nothing committed/pushed — user must push + bump the Kaggle dataset for any of this to reach Kaggle.

## 2026-06-26 — Committed + pushed everything (`4bb93fe`)

Committed the session's CAMUS work and pushed to `origin/main` (`c2d656b`→`4bb93fe`): the 4 config edits (gate-off c1/c3 + fail-safe + label smoothing), `losses.py`, `drl_training.py` (resume), all 4 hardened DRL notebooks + quota block, the new diagnostics notebook, and TRANSCRIPT/SKILLS. Deliberately excluded the two stale `colab_*.ipynb` (left untracked). No `Co-Authored-By` trailer (`.claude/settings.json` has no `attribution.commit`; matches repo's existing commit style). Stayed on `main` per the user's solo-repo workflow + explicit "push everything" request.

## 2026-06-26 — Cleared notebook-version confusion + rewrote diagnostics into a full EDA report

**Version confusion:** user saw the Kaggle notebook's TOC "10 · Test-set evaluation" and thought "cell 10 isn't diagnostics → old version." Clarified: my quota block sits in the **§4 Full training** cell (JSON cell index 10, NOT TOC section 10 — the number collision caused the confusion). Reliable freshness check: open the **§4 Full training** cell; the new version's first line is `# ── Diagnostic / quota controls` with `TRAIN_STEPS`. The screenshot didn't prove staleness either way. Root reminder: notebooks are NOT part of `iteris-pkg` (which only carries `iteris/`+`configs/`+`requirements.txt`); the notebook reaches Kaggle only by importing/replacing the actual `.ipynb` file — editing on Kaggle directly is why the repo's notebook fixes kept not showing up.

**Diagnostics rewrite:** expanded `notebooks/00_free_diagnostics_all_classes.ipynb` from a thin Pillar-4+headroom runner into a complete **EDA + DRL-compatibility report**. New `full_report(label)` (reads cached warm-start samples, no re-warm-start) covers, per class: §3 dataset EDA (counts, GT area-fraction dist, topology/connected-component count, intensity, view/phase), §4 lite-U-Net baseline (init Dice/IoU distribution + percentiles + worst-5, prob_map band_frac/entropy → INERT/USABLE), §5 error decomposition (boundary vs topology/interior via `error_type_audit`), §6 DRL compatibility (`headroom_report`: contour-repr ceiling, GT-oracle ceiling, realistic headroom vs attention). Two figures/class (6-panel stats grid + 3-worst-case qualitative grid, saved to `/kaggle/working/eda_*.png`). §7 master table (`eda_diagnostic_summary.csv`) + a data-driven per-class **recommendation** (`recommend()`): SKIP / RISKY / TRAIN[STRONG|GOOD|MARGINAL] with gate on/off guidance, and names the single best bet by realistic headroom. Validated: all 13 code cells AST-parse; every diagnostics return-key (`uncertain_band_frac`, `boundary_frac`/`topology_frac`/`interior_frac`, `baseline_init_dice`/`contour_repr_ceiling`/`oracle_greedy_ceiling`/`realistic_headroom_estimate`) verified against source. No change to `diagnostics.py` itself — the notebook composes existing functions + inline EDA.

## 2026-06-29 — Config reorg; diagnosed + fixed DuelingDDQN's 0%-STOP pathology

**Config reorg:** flat `configs/*.yaml` → nested `configs/{CAMUS,BRISC}/{,DRL/}...`; all notebooks/scripts updated. Merged `iteris/archive_paradigm_a/` into `iteris/archive/paradigm_a/` (fixed the resulting relative-import depth bug in `segmentation_env.py`, `..geometry` → `...geometry`). Notebooks regrouped into `notebooks/{unet,camus/drl,brisc/drl,local}/`.

**STOP-learning diagnosis:** a real 50k-step CAMUS-LA DuelingDDQN run (`reward_step_penalty: 0.02`, the first attempted fix) still showed `STOP-action rate: 0%` and final Dice below baseline. Root cause confirmed in code: `_terminal_step` returns raw reward exactly 0.0 (no penalty), while the 17 move actions carry noisy ~0 Q-values that out-argmax a clean 0 almost every time. Raised `reward_step_penalty` to 0.05 and added training-only `curriculum_max_steps` (GT-based init-Dice buckets → per-episode max_steps, easy=4/medium=base/hard=15, to manufacture more near-peak terminal transitions without reducing hard-sample representation from `hard_mining_scale`). Landed on CAMUS c1/c2/c3 DuelingDDQN, then ported to all 4 BRISC DuelingDDQN configs (tumor/glioma/meningioma/pituitary, pituitary's curriculum scaled to its `max_steps=8`).

**New literature-standard metrics:** IoU, Precision, Sensitivity, Boundary IoU, Mean Surface Distance added to `evaluate_agent` (training) and `replay_one`/`evaluate_testset` (deploy) — eval-only, computed once per finished rollout, never in the env's per-step reward hot path. Initially final-mask-only; later (06-30) extended with `init_*` counterparts so every metric has a baseline to diff against, the same way Dice always did. Milestone checkpointing added (`checkpoint_every: 12500`, distinct step-stamped snapshots).

**Real diagnostic run (full data, post-fix):** confirmed real positive realistic headroom on all 3 CAMUS classes (LV_endo +0.050, LV_epi +0.073, LA +0.052) and confirmed *negative* headroom on BRISC tumor pooled (−0.0501, consistent with the earlier 2026-06-25 finding).

## 2026-06-30 — TD3 parity (CAMUS then BRISC); gate re-disabled after retrain still INERT; action_type crash fix; Phase A/B/C defined

**TD3 brought to the same standard as DuelingDDQN:** training-only `curriculum_max_steps` added to all CAMUS TD3 blocks (per-class hard-step caps: c1/c2=15, c3=12 — LA is ballooning-prone and TD3 has no STOP, so a more conservative excursion cap than its DuelingDDQN counterpart). `reward_step_penalty` deliberately left at 0.0 for TD3 (documented inline why: no STOP action to be penalty-free relative to, so a nonzero penalty would just depress every reward and fight the BC warm-start). Same treatment ported to all 4 BRISC TD3 blocks, which also turned up a real gap: BRISC TD3 had **no `fail_thresh`/`fail_n` at all** (CAMUS did) — added (`0.02`/`2` initially, see below).

**Gate re-enable → re-confirm INERT → re-disable, same day:** user reported the CAMUS lite U-Net was retrained with `loss_label_smoothing: 0.25` (up from 0.1) and asked to re-enable `uncertainty_gate` for TD3 c1/c3. Re-enabled, then the next diagnostic run showed LV_endo/LA **still INERT** (`band_frac` 0.0085/0.0077 — barely moved from the original 0.0073/0.0063, and LA actually moved backward vs the 0.1-smoothing checkpoint's 0.0100). Re-disabled gate for TD3 c1/c3 (and, on inspecting BRISC's even-more-extreme `band_frac=0.0007`, disabled it everywhere in BRISC too — DuelingDDQN and TD3, all 4 classes). Working theory for why CE-only label smoothing isn't moving the needle: `lambda_dice=0.5` means half the loss is still an un-smoothed Dice term actively pushing toward confident 0/1.

**Real-run anomaly → fix:** a live BRISC-tumor TD3 run showed final Dice landing −0.06 to −0.07 below baseline despite `fail_thresh: 0.02`. Diagnosed: `fail_n=2` requires 2 *consecutive* breaches before terminating, and BRISC's tiny targets (~1.7% area) can blow well past the threshold in a single edit — by the 2nd confirmation the damage is already 3x the nominal floor. Tightened BRISC `fail_n` to 1 (terminate on first breach); left CAMUS at 2 (larger structures, still a legitimate noise filter there).

**Crash fix:** the same BRISC-tumor TD3 run's post-training visualization cell crashed (`TypeError: only 0-dimensional arrays can be converted to Python scalars` inside `env.step()`). Root cause: `refinement_env_kwargs(cfg)` lists `'action_type'` in its key filter, but the resolved YAML `cfg` never actually contains that key (`run_drl_training` only computes it as a local variable, never writes it back) — so the env silently fell back to its `'discrete'` constructor default. Invisible for DuelingDDQN (matches the default by luck), fatal for TD3/DDPG once a continuous action array hits the discrete branch. Fixed by deriving `action_type` from `cfg['agent_type']` via `AGENT_REGISTRY` inside `refinement_env_kwargs` itself. Training/checkpoints were unaffected — only post-training replay/eval cells needed the re-run.

**Phase A/B/C defined (CONTEXT.md §3):** at the user's request, formalised a three-phase experimental design distinct from the existing build→train→eval→paper project timeline (renamed the latter to "Stage 1–4" in SKILLS.md to avoid a naming collision). Phase A = full-dataset DuelingDDQN+TD3 vs U-Net comparison (current default track). Phase B = same comparison on a small subset (`label_frac` knob, already implemented, no new code) to test whether DRL's advantage grows in a low-data regime — motivated by the project's own literature review (cold-start DRL beats supervised CNNs in scarce-data regimes). Phase C = swap in the archived MSA (Multi-Head Self-Attention) backbone (`iteris/archive/msa.py`, not currently wired into any config/registry) on the single best Phase A/B agent, tested on an even smaller subset than Phase B.

## 2026-07-01 — Phase B/C sizing research + protocol; phase-aware artifact naming; repo professionalisation

**Research (data sizes):** web-searched the few-shot / scarce-data medical-segmentation literature to size Phases B/C. Anchors: PixelDRL-MG (Liu 2025, the directly-comparable paper) benchmarks at **50- and 100-shot**; the Manis BraTS series shows the DRL-beats-DL effect strongest at **~30 images** (DQN 70% vs CNN 11%). Chosen levels (via the existing patient-level `label_frac`, train-only shrink, val/test stay full): CAMUS B=`0.10` (~37 pts/~148 img), C=`0.05` (~18 pts/~74 img); BRISC B=`0.05` (~155 img), C=`0.03` (~93 img) — fractions differ per dataset but absolute counts deliberately matched (~150 / ~75–95) for a clean cross-dataset story. Full rationale + sources written to new **`docs/EXPERIMENTS.md`**.

**Methodology decisions (answered the user's two questions):** (1) the lite U-Net **must** be retrained on the subset — reusing the full-data lite net would secretly leak all 1500 imgs into the warm-start, voiding the low-data claim; (2) the attention U-Net competitor **must also** be retrained on the same subset (DRL never refines the attention net — it's the fixed competitor; but comparing low-data DRL against a full-data attention net is not the question). Confirmed leak-free by construction: `training.py` and `warm_start.py` both call `patient_level_split` with the *same* baseline-config `seed`+`label_frac`, so agent and U-Net see identical patients. Noted BRISC may *gain* headroom in B/C (its 33M-param attention competitor overfits hard on ~150 imgs while lite+DRL is robust) — exactly where full-data BRISC had none.

**Phase mechanism (zero notebook edits):** made `utils.model_suffix` data-regime-aware — `label_frac < 1.0` appends `_lf<pct>` (e.g. `_lf10`) to every artifact name. Since `training.py`'s checkpoint writer AND all 4 DRL notebooks' baseline-ckpt auto-detect already route through `model_suffix`, setting `label_frac` in the baseline config is the *only* change needed to run a phase — the tag propagates everywhere, and low-data checkpoints never overwrite Phase-A ones. Phase A (label_frac 1.0 / absent) names are unchanged. Unit-tested incl. back-compat.

**Repo professionalisation (safe scope):** moved the 5 design docs (CONTEXT/PLAN/SKILLS/TRANSCRIPT/UI_PLAN) → `docs/`; new front-door `README.md` (overview diagram, phase table, repo map, doc index); added `pyproject.toml` (installable `iteris`, deps stay in requirements.txt as Kaggle's single source of truth); expanded `.gitignore` (`_tmp*/`, `*.log`, nbconvert exports, `node_modules/`); deleted stray `_tmp_final_ckpt/`. **Deliberately did NOT** touch the Kaggle-critical paths (`iteris/`, `configs/`, `notebooks/`) or relocate the dormant 90-file `iteris_ui/`+`server/` demo — flagged the latter as an opt-in `app/` move for later, to avoid churn during active Phase-A runs (lesson from this session's repeated path-staleness breakage). **MSA wiring for Phase C remains the one real code prerequisite** (un-archive `msa.py`+`agents_legacy.py`, register `MSA-DUELING`/`MSA-TD3`, add selector) — left as a focused follow-up. *(Note added 2026-07-21: Phase C was subsequently abandoned outright — see the 2026-07-21 entry below. This prerequisite is now moot, kept here only as the historical record of what was scoped.)*

## 2026-07-08 — Reward-system rewrite: `contour_boundary` supersedes PBRS; discrete BC oracle fix

**04:08 (`eb819c7`):** fixed the discrete behaviour-cloning oracle's near-zero STOP rate — a strict `>` comparison against current Dice let floating-point re-rasterisation jitter almost always beat the bar, so the oracle (used for BC warm-start demonstrations) almost never emitted STOP.

**06:03–06:10 (`9f6155a`, `3e1b206`):** introduced `apply_refinement_config()` in `iteris/config.py` — one agent-aware call replacing scattered per-notebook `cfg.update({...})` blocks. Switched the default `reward_mode` to **`contour_boundary`** (dense per-control-point distance-to-GT-boundary, self-regularising: nudging an already-correct point increases its distance, so "leave it alone" is reward-optimal) for every notebook that calls it. This **overrides** the per-class YAML's own `reward_mode`/`reward_step_penalty` — the raw YAML stopped being ground truth for what actually trains from this point on; `config.py`'s `_REFINE_SHARED` dict is now the source of truth. Full mechanism in CONTEXT.md §6.

**06:30 (`dd73405`):** reorganized `notebooks/{camus,brisc,unet}` → `notebooks/phaseA/{camus,brisc,unet}`, mirroring the (not-yet-existing at this point) `phaseB/` layout added later.

## 2026-07-09 — Two real state-representation bugs found from a live Phase-B run

**01:50 (`37627be`):** a real Phase-B LV_endo DuelingDDQN run held its initial shape / degraded, action histogram near sample-invariant. Root-caused to two bugs: (1) **debris-channel leak** — state channels 4–5 showed the RAW multi-blob U-Net mask instead of the largest-CC-only representation the contour can actually reach, biasing the policy toward debris blobs; (2) **SectorPool bin-centre bug** — `SectorPool` binned around the image centre while the env bins around the live contour centroid, a 54% cell mismatch for off-centre masks that scrambled the local-error→correct-sector signal. **Invalidates any checkpoint trained before this fix with `spatial_head: true`** (the default since the prior day's commit) — no eval-time fix, retrain required.

**02:07 (`132bb1a`):** follow-up audit for the same bug class (state acted-on ≠ state learned-on) found two more: the debris fix had been applied unconditionally (wrong for the default non-contour `SegmentationEnv`, now gated by env type); and `refinement_viz` kept its own drifted copy of the env-kwarg key list, silently dropping `auto_smooth_lambda`/`uncertainty_gate`/`gate_*` at EVAL time only — so TD3 and the gated LV_epi class were tested on a different env than they trained on. Fixed by promoting the key list to `drl_training.ENV_OPTIONAL_KEYS` as the single source of truth (later reused for the 2026-07-16 auto-stop keys).

**02:53–03:39 (`fac2a4d`, `40f2091`):** added boundary-aware + adaptive + uniform-floored hard-sample-mining tiers, replacing flat exponential weighting that gave topology-broken (unreachable) samples the same aggressive boost as genuinely fixable ones.

**19:48 (`b7a8207`):** `TRAIN_STEPS` set to 50000 across all 8 production DRL notebooks.

## 2026-07-11 — Optimal-stopping STOP bonus (discrete only)

**11:02 (`2fd0d0b`):** `contour_boundary`'s dense reward is ~0 and noisy near the peak, so the discrete agent almost never chose STOP (12% CAMUS LV_endo, 0% BRISC tumor, confirmed on real runs). Fix: a chosen STOP now earns `terminal_bonus_scale · max(0, dice − dice_0)` — proportional to captured gain, zero if stopped before any gain, nothing on a max-steps timeout — so STOP strictly dominates drifting to the cap. Confirmed **inert for TD3** (no STOP action) in the same commit.

**11:12 (`69bc860`):** set per-class/dataset `terminal_bonus_scale` (CAMUS=20, BRISC=10) — no sweep was run to calibrate it; BRISC's 10 later turned out to under-shoot (see 2026-07-16 below).

## 2026-07-13 — Fixed eval consistency: TD3 was reporting understated test numbers

**22:51 (`17afaec`):** `run_drl_training` was returning the agent at its FINAL-step weights, not the saved best-val checkpoint. TD3-specific (actor drifts once `bc_lambda` decays; DuelingDDQN unaffected — monotonic). Centralized the fix so every notebook's `evaluate_testset`/`build_replays` measures the deployable model automatically, and added `iteris/drl_reeval/re_eval_td3.py::reeval_checkpoint()` to re-score checkpoints trained before this fix without retraining (test-set forward pass only, no GPU training). Any TD3 checkpoint trained before this timestamp may have reported understated numbers.

## 2026-07-14 — Single-baseline design confirmed current; landing page matches the real agent set

**(`61097d0`, `f3ea5dc`):** documentation/config-comment consistency pass — every DRL notebook already passed `baseline_cfg_name='CAMUS/camus.yaml'` (or `BRISC/brisc.yaml`) to `apply_refinement_config`, which always overrides the per-class YAML's own field. That means **RL was already refining `AttentionResUNet` directly**, not the separate `LiteUNet` the project's 2026-06-21 decision had originally set up — this commit only fixed the stale comments/fallback checkpoint names in the 7 per-class DRL YAMLs to match, no behavior change. Also redesigned the landing page's model list to Dueling DQN + TD3 (dropping the archived DQN/DDQN/DDPG), removed unverified statistical claims. **This doc (TRANSCRIPT.md) never got an entry for the single-baseline design change itself** — the actual behavioral pivot (as opposed to this doc-consistency commit) isn't cleanly attributable to one commit; flagged as a known gap in the historical record (see the 2026-07-21 audit entry below).

## 2026-07-15 — Wired Phase A into the deployed server

**22:15 (`5118e27`):** extended `server/app/drl.py`'s DRL registry to the full Phase A (high-regime) set: all 6 CAMUS per-class agents (LV_endo/LV_epi/LA × DuelingDDQN/TD3) + BRISC tumor × {DuelingDDQN, TD3}, 8 entries, exact per-checkpoint hyperparameters (CAMUS-LA and BRISC use a different `spline_smooth`; BRISC's TD3 uses 12 continuous sectors vs 16 for CAMUS). Added TD3 (Actor/SpatialActor) support to the server's episode runner and checkpoint loader, which previously only handled the discrete DuelingDDQN path.

## 2026-07-16 — Continuous auto-stop; BRISC STOP-bonus retune; `iteris.evaluation` package/file collision fix

Analyzed real Phase-A behaviour/comparison plots (CAMUS TD3 LV_epi/LA, BRISC DuelingDDQN/TD3 tumor) and root-caused two structural gaps: **(1)** TD3 has no stopping mechanism at all — added a GT-free action-magnitude auto-stop (`auto_stop_action_eps`/`auto_stop_patience` in `env_contour_refine.py` + the server's duplicate copy), synthetic-verified but not yet run on a real GPU pass. **(2)** BRISC's `terminal_bonus_scale=10` (set 2026-07-11 without a sweep) measurably under-shot (~1% STOP rate) — raised to 30 in both Phase A and Phase B BRISC DRL notebooks; CAMUS left untouched (no CAMUS-DuelingDDQN evidence reviewed this session). Rewrote CONTEXT.md §6–§10 (previously last touched 2026-06-30, out of sync with the whole reward rewrite above).

**Separately, found and fixed a real crash:** `notebooks/phaseB/unet/02_brisc_lite.ipynb` failed at the `evaluate_test_set` import after training completed (checkpoint already saved, so no work lost). Root cause: the 2026-07-13 commit added a **package** `iteris/evaluation/` alongside the pre-existing **file** `iteris/evaluation.py` — same name, so the package silently shadowed the file's `evaluate_test_set`/`export_predicted_masks`/`save_summary_json` for every import. Affects all 8 U-Net baseline notebooks (Phase A+B × {camus,brisc} × {lite,attention}) for any run after 2026-07-13 22:51. Fixed by renaming the package `iteris/evaluation/` → `iteris/drl_reeval/`. Training itself was never affected — only the post-training reporting cell.

## 2026-07-19 — CI/CD, Research page wired to real state, rebrand

Three independent changes: **(1)** `f33f56e` — added GitHub Actions for the FastAPI server's Hugging Face Space deploy + UI/Python sanity CI (Vercel's own integration already auto-deploys the frontend independently; Kaggle training has no deploy step to automate). **(2)** `a0a0a00` — Research page (`iteris_ui`) rewritten to reflect the current paradigm (contour env, angular sectors, `contour_boundary` reward, DuelingDDQN + TD3 vs LiteUNet/AttentionResUNet) instead of the long-archived DQN/DDQN/DDPG framing, and honest pending states replaced invented Dice/p-value numbers. **(3)** `ee87f6c` — new logo/favicon mark, navbar theme-toggle fix.

## 2026-07-20 — Phase B wired end-to-end; evaluation notebook built and hardened; real results wired to the Research page

**`4825754`:** extended the server's DRL registry to the full Phase B (low-regime) set, generated over `{dataset, class, regime, algo}` instead of hand-copied — 16 entries total. Confirmed Phase A/B checkpoints for the same class share identical contour-env geometry (regime is a checkpoint-selection axis only, not a hyperparameter change). **Phase A and Phase B are now both complete and fully wired**, training through deployed UI.

**Built `notebooks/evaluation/comprehensive_model_evaluation.ipynb`** (new) and iterated it heavily the same day: ingests every U-Net/DRL/classifier output file it finds (content-shape + sibling-JSON metadata detection, never by folder naming), computes the master comparison table, Wilcoxon+Bonferroni significance, and a qualitative gallery. Switched all charting from matplotlib to Plotly; reframed the discrete-vs-continuous comparison to report both absolute deployed Dice (head-to-head) and delta-vs-baseline (learning behaviour), since the two can disagree and neither subsumes the other. Fixed two real portability bugs caught via actual Kaggle runs: Plotly's default interactive-HTML renderer doesn't display on GitHub/nbviewer/Kaggle's plain Log console (switched every chart to a static-PNG `kaleido` render); and Kaggle's preinstalled `plotly 5.24`+`kaleido 1.3` are mutually incompatible (pinned the matched `plotly<6`+`kaleido==0.2.1` pair, probed once up front instead of failing per-chart).

**Added Phase A/B as a first-class grouping dimension** to the evaluation notebook, fixing a real correctness gap: every section was silently pooling Phase A and Phase B into single means/rankings/pairings — worst case, the discrete-vs-continuous head-to-head could pair a Phase A run against a Phase B run for the same class as if they were comparable. Detection: `label_frac` where saved, else a `pa`/`pb` path-token fallback for DRL reeval JSON. Verified with a synthetic fixture engineered to prove it matters: the per-phase head-to-head correctly reversed the winning agent between phases, exactly the kind of result a pooled ranking would have hidden.

**`91fbcc4`:** wired the Research page to the real 2026-07-20 evaluation run's `master_comparison.csv`. Found, while doing so, that the U-Net baseline rows had come out phase=`Unknown` from the evaluation notebook (their `label_frac` apparently didn't survive to that run's summary JSON, and the U-Net Kaggle dataset slugs don't follow the `pa-`/`pb-` convention the DRL path-fallback relies on) — disambiguated by Dice rank, cross-checked 100% against every DRL run's `init_dice_mean`. **Not yet root-caused or fixed in the notebook itself** — a real, open gap, noted in SKILLS.md §4.

## 2026-07-21 — Phase C formally abandoned; full documentation audit

**Phase C (MSA backbone adaptation) formally abandoned** — direct instruction: "Phase C is dropped, not to be used." This is a project decision, not a scheduling slip. `iteris/archive/msa.py` and `agents_legacy.py::MSADuelingDQNAgent` remain archived; no config/notebook/`AGENT_REGISTRY` entry will be added without a fresh decision reopening this.

**Full documentation audit**, triggered by the observation that the repo's docs had drifted stale in three specific ways: analyzed all 189 commits against every doc in `docs/` + `README.md` + `results/README.md` + `server/README.md`. Found and fixed:
1. **The single-baseline pivot never fully propagated into the docs.** CONTEXT.md §1/§4/§9, EXPERIMENTS.md (especially §2.3, which directly answered "should DRL refine the attention net?" with "no" — the exact question this audit was triggered by), PLAN.md, SKILLS.md §1, and README.md were all still describing RL refining `LiteUNet` as current, three weeks after `61097d0` (2026-07-14) confirmed the opposite was already true in every notebook. All corrected, with the superseded design kept as an explicit, dated historical note (not silently deleted) since a paper needs to be able to say what changed and why.
2. **Phase C was listed as "not started"/planned everywhere**, despite this session's abandonment decision. Updated to "abandoned" throughout CONTEXT.md, EXPERIMENTS.md, PLAN.md, SKILLS.md, README.md — the design rationale is kept (EXPERIMENTS.md §4, CONTEXT.md §3) as a historical record, explicitly marked as not to be resurrected without a fresh decision.
3. **Confirmed no stale dataset scope survived** — HAM10000/Kvasir-SEG/DRIVE (added `cf6cd6d`/`07edd34`, May 2026) were archived/dropped before the DRL/contour-refinement design existed and aren't referenced as in-scope anywhere in the current docs; added one explicit sentence to CONTEXT.md §1 for anyone reading the git history cold.
4. **PLAN.md was rewritten wholesale** (last touched 2026-06-21, a month stale) — renamed its timeline "Phase A–D" to "Stage 1–4" to stop colliding with CONTEXT.md's data-regime Phase A/B naming (SKILLS.md had already made this renaming decision on 2026-06-30; PLAN.md itself never got the memo), and fixed a direct contradiction in its own "Critical Don'ts" ("Don't refine the attention net with RL — no headroom" — the literal opposite of the current design).
5. **`results/README.md`'s section numbers had drifted from the evaluation notebook's own restructuring** (this doc's own earlier-2026-07-20 changes bumped every section after 6 by 2) — corrected.
6. This TRANSCRIPT.md itself had a three-week gap (2026-07-01 → 2026-07-16, and 2026-07-16 → 2026-07-21) — backfilled with the entries above, consolidated from commit messages + CONTEXT.md §10 (which had been kept more current than this file during the gap — worth noting as a process gap: the maintenance rule at the top of this file says every decision goes here *and* SKILLS.md in the same turn; that discipline held for CONTEXT.md but not consistently for this file across the gap period).
