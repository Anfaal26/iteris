# Iteris — Project Skill

> The operational playbook for working on this codebase: how it's wired, where the landmines are, and what's currently open. Complements [CONTEXT.md](CONTEXT.md) (what/why, incl. the Phase A/B experimental design — Phase C was abandoned, see CONTEXT.md §3), [EXPERIMENTS.md](EXPERIMENTS.md) (Phase A/B data sizes + methodology), and [PLAN.md](PLAN.md) (roadmap) — this file is "how to not break it" + "what I already learned the hard way." All design docs now live in `docs/`; the front-door `README.md` stays at repo root.
>
> **Maintenance rule (this is the main skill of this file — apply it every time):** whenever a new architectural decision, fix, gotcha, or piece of standing knowledge is produced, update the relevant section below *in the same turn*, and add a matching dated entry to [TRANSCRIPT.md](TRANSCRIPT.md). TRANSCRIPT.md is the chronological "what happened, when" log; this file is the always-current "what you need to know" summary — the same fact often belongs in both, phrased differently (transcript = past tense event, skill = standing rule). Never let this file silently drift out of date relative to the transcript.
>
> **Naming note:** this file uses "**Stage 1–4**" for the project timeline (build → train → evaluate → write paper). CONTEXT.md separately defines "**Phase A/B**" for the experimental design (full data vs low-data). A third phase (**Phase C**, an MSA-backbone ablation) was designed but **formally abandoned 2026-07-21** — never implemented, not part of the active design; see CONTEXT.md §3. Don't conflate Stage and Phase — they're different axes that happen to both use letters/numbers for "phase."

---

## 1. What this project is (one paragraph)

DRL agent (`DuelingDDQN` discrete or `TD3` continuous) refines the **AttentionResUNet** segmentation mask toward ground truth by deforming its boundary contour in angular sectors, on CAMUS (cardiac ultrasound: LV-endo/epi/LA) and BRISC (brain tumour MRI: glioma/meningioma/pituitary/tumour-generic) — the only two datasets in scope. This is the **single-baseline design**, confirmed current since 2026-07-14 (`61097d0`) — a separate, deliberately weaker **LiteUNet** is trained per phase alongside it purely as an architecture-comparison baseline, not an RL target (this reverses the project's original 2026-06-21 decision, where RL refined the lite net and the attention net was the untouched competitor — see CONTEXT.md §9 for the full history). Reward is the dense `contour_boundary` signal (superseded PBRS 2026-07-08 — see CONTEXT.md §6) plus an optimal-stopping bonus (discrete) / action-magnitude auto-stop (TD3). The research design runs this comparison at two data scales — see CONTEXT.md §3 (Phase A: full data, Phase B: low-data regime via `label_frac`, both complete). A third phase (Phase C: archived MSA backbone) was scoped but abandoned — see CONTEXT.md §3/EXPERIMENTS.md §4.

## 2. Key files — where to make changes

| Concern | File |
|---|---|
| Contour env (the live paradigm) | `iteris/env_contour_refine.py` |
| Shared geometry helpers + eval-only metrics (IoU/Precision/Sensitivity/BIoU/MSD) | `iteris/geometry.py` |
| Agents (DuelingDQN, TD3 active; DQN/DDPG archived-use) | `iteris/agents.py` |
| Networks incl. optional spatial head | `iteris/drl_networks.py` |
| Archived MSA backbone (was scoped for the now-**abandoned** Phase C) | `iteris/archive/msa.py`, `iteris/archive/agents_legacy.py::MSADuelingDQNAgent` |
| Training loop / config resolution / curriculum / checkpointing | `iteris/drl_training.py` |
| Offline diagnostics (headroom, pillar4 prob_map informativeness) | `iteris/diagnostics.py` |
| BC warm-start demos | `iteris/bc_demo.py` |
| Deploy-time replay/eval + value-floored selector + init/final metrics | `iteris/refinement_viz.py` |
| Models (LiteUNet, AttentionResUNet) | `iteris/models.py` |
| Configs (one YAML block per dataset×agent) | `configs/{CAMUS,BRISC}/{,DRL/}{camus,brisc}_{lite,drl_*}.yaml` |
| Back-compat shim for old global-morph env | `iteris/env.py` (do not add new logic here) |
| Archived Paradigm A (global morphology, ablation only) | `iteris/archive/paradigm_a/` |
| Multi-GPU concurrent runner | `scripts/run_drl_config.py` |

Config resolution: `resolve_agent_config(load_drl_class_config(path), AGENT_NAME)`. Artefacts are model-suffixed via `utils.model_suffix` so lite/attention checkpoints never collide. **Configs live under `configs/{CAMUS,BRISC}/...` (nested), not flat `configs/*.yaml`** — reorganised 2026-06-29; if you see a stale flat path anywhere (old notebook, old doc, a Kaggle dataset that predates the reorg), that's the bug, not the code.

## 3. Standing gotchas (learned the hard way — don't re-discover these)

### DuelingDDQN STOP-learning (fixed 2026-06-29, verify it stays fixed)
- **Under pure PBRS, DQN-family agents can fail to ever learn the STOP action — even after 50k steps.** Mechanism: `_terminal_step()` (STOP) returns raw reward exactly `0.0` with no penalty, while the 17 other actions' Q-values are *noisy* ~0 near the Dice peak — the argmax of 17 noisy-near-zero values beats a clean 0 almost every time, so STOP never wins even though it should. Symptom: `STOP-action rate: 0%` in the post-run diagnostic, final Dice sitting visibly below `best-seen` with the gap not closing.
- **Fix: `reward_step_penalty` (e.g. 0.05) makes every *non*-STOP action cost a little.** STOP is the only action `_terminal_step` doesn't charge, so once a move's marginal Dice gain falls below the penalty, `Q(STOP)=0` beats `Q(move)<0`. A positive penalty can only make the agent stop *earlier* — bounded no-worse than the old overshoot, never worse.
- **`reward_step_penalty` must stay `0.0` for TD3 — do not copy the DuelingDDQN value over.** TD3 has no discrete STOP action; every step (no exception) goes through `_apply_continuous`, which *does* subtract the penalty. A nonzero value uniformly depresses every TD3 reward and pushes the BC-warm-started actor back toward identity — actively harmful, not neutral.
- **Even with the penalty, the buffer can still be starved of "near-peak, should-stop" transitions** if `hard_mining_scale` aggressively oversamples low-init-Dice cases (which rarely *reach* convergence within a short episode). **Fix: training-only `curriculum_max_steps`** — GT-based init-Dice buckets get a different per-episode `max_steps` (easy samples: short episode, manufactures more near-peak terminal transitions; hard samples: longer episode, more room to travel). GT-based difficulty is only legitimate during training (GT available); **eval/test always use the single fixed `max_steps`**, never the curriculum-adjusted one — `evaluate_agent`/`evaluate_testset` are called with the unmodified `env_kwargs`, only the main training loop builds a per-episode copy. Per-class hard-step caps should scale to each config's own `max_steps`, not be copy-pasted (e.g. BRISC pituitary's `max_steps=8` → `curriculum_hard_steps=13`, not CAMUS's 15).
- **One stronger penalty alone wasn't sufficient to confirm full closure** — 0.05 + curriculum is the current best-known config; re-verify STOP rate and the `best-seen`/`final` gap on the next full run before assuming this is fully solved.

### TD3 termination & safety net
- **TD3 has no STOP action and a reward-blind termination heuristic.** `_check_termination()` only checks whether Dice *stopped changing*, never whether it's good — a drifting actor can converge below baseline and commit it as both the deploy number and the checkpoint-selection metric.
- **`fail_thresh`/`fail_n` is the safety net, and it's a *delayed* circuit breaker, not instantaneous.** It terminates after `fail_n` *consecutive* steps below `dice_0 - fail_thresh`, not on the first breach — so the real damage by the time it fires can be well past the nominal threshold, especially on small/volatile targets where one bad edit swings Dice a lot in a single step.
- **`fail_n` should be tuned to target size/volatility, not copied across datasets.** CAMUS (`fail_n: 2`) — larger structures, slower per-step Dice impact, the 2-step wait is a legitimate noise filter. BRISC (`fail_n: 1`, tightened 2026-06-30) — tiny targets (~1.7% area; the configs' own header comments already warned "1 bad step can drop Dice significantly"), confirmed in a real run landing −0.06 to −0.07 below baseline with `fail_n: 2` despite `fail_thresh: 0.02`; a single breach is meaningful signal here, not noise.
- **BRISC's TD3 blocks were missing `fail_thresh`/`fail_n` entirely until 2026-06-30** — only CAMUS had it. If you add a new dataset/class's TD3 config, don't forget this; without it there's no downside floor at all.

### `refinement_env_kwargs` / post-training visualization (fixed 2026-06-30)
- **`refinement_env_kwargs(cfg)` never actually got `action_type` from `cfg`, even though it's listed in `_ENV_KEYS`.** Root cause: `action_type` is only ever computed as a *local variable* inside `run_drl_training()` (from `AGENT_REGISTRY[cfg['agent_type'].upper()]`), never written back into the resolved YAML `cfg` dict. So `refinement_env_kwargs` silently dropped the key, and `ContourRefineEnv` fell back to its constructor default (`'discrete'`). **Invisible for DuelingDDQN** (matches the default by coincidence) — **fatal for TD3/DDPG**: a continuous action array hits the discrete branch's `int(action)` → `TypeError: only 0-dimensional arrays can be converted to Python scalars`, crashing `build_replays`/`evaluate_testset` (notebook §5 onward), even though training itself completed fine and the checkpoint is valid.
- Now fixed by deriving `action_type` explicitly from `cfg['agent_type']` via `AGENT_REGISTRY` inside `refinement_env_kwargs` itself — don't revert to trusting `cfg` to carry it.
- **If a TD3/DDPG notebook crashes at the replay/visualization cell but training completed and saved a checkpoint, you don't need to retrain** — just re-run from that cell onward with the fix.

### Uncertainty gate — INERT prob_map is not "neutral," it's destructive
- **`uncertainty_gate: true` is only safe when the prob_map is graded, not binary.** The gate multiplies edit magnitude by 1.0 only inside `[gate_lo, gate_hi]`, ramping to 0.0 outside. If `band_frac` is near-zero (INERT), the gate clamps edits to ~0 at ~99% of contour points — it doesn't "do nothing," it paralyzes the agent exactly on the classes where headroom exists.
- **For TD3 it's worse:** `_apply_continuous` multiplies the actor's displacement by the gate, *and* `bc_demo.py` builds its oracle env from the same `env_kwargs`+prob_map — an INERT gate zeroes both the actor's moves and the BC oracle's demonstrations, training the actor *toward* identity via the very mechanism meant to escape it.
- **Always run the diagnostic (`00_free_diagnostics_all_classes.ipynb`) before enabling the gate for a new class/checkpoint.** Current confirmed status (2026-06-30, post-retrain-at-0.25-smoothing): CAMUS LV_epi USABLE (gate ON, both agents) — LV_endo and LA still **INERT** (gate OFF, both agents) — BRISC tumor INERT at `band_frac=0.0007`, the most binary of all diagnosed classes (gate OFF everywhere in BRISC).
- **Label smoothing alone did not fix it, even at 2.5× the original strength.** History: 0.0 → `band_frac` 0.0073/0.0063 (LV_endo/LA) → 0.1 retrain → 0.0079/0.0100 → 0.25 retrain → 0.0085/0.0077. Tripling the smoothing barely moved LV_endo and made LA *worse* (non-monotonic). Working theory: `DiceCELoss` only smooths the CE term (`lambda_ce=0.5`); the un-smoothed Dice term (`lambda_dice=0.5`) actively pushes probabilities toward confident 0/1 the whole time, likely cancelling most of the CE-smoothing effect. **Don't re-enable these gates on a future retrain without re-running the diagnostic first** — "we retrained with smoothing" is not sufficient evidence, check the actual `band_frac`/verdict.
- If pursuing this further, a different lever than CE label smoothing is probably needed (e.g. rebalancing `lambda_dice`/`lambda_ce`, or post-hoc temperature scaling) — not yet tried.

### Other standing gotchas
- **Kaggle runs use the pushed `iteris-pkg` dataset + uploaded notebook, NOT your working tree, NOT each other.** Three *separate* staleness failure modes, all seen this session: (1) the `iteris-pkg` dataset version pinned to a notebook can predate a repo push — re-attach/bump explicitly, don't assume a re-upload auto-propagates; (2) a dataset re-upload can be incomplete (e.g. the `iteris/` package folder silently missing while `configs/`+`requirements.txt` made it) — verify via the Kaggle file browser, not just "it ran without an attach error"; (3) the **notebook file itself** is a separate artifact from the dataset — re-uploading/version-bumping `iteris-pkg` does NOT update an already-imported `.ipynb`'s saved cell code. Before trusting any Kaggle result: confirm fixes are committed+pushed (`git status`/`git log` clean, 0 ahead/behind origin), the dataset is bumped to a version that actually contains them (spot-check the file browser), and the notebook itself reflects current cell code.
- **The training-log `final`/`Δ` is the RAW number, not the deployable one.** `evaluate_agent` reports raw final Dice with no value floor. `best-seen > baseline` while `final < baseline` means the agent reaches good masks but overshoots; the value-floored deploy (Pillar 1, `refinement_viz.evaluate_testset`) is closer to the honest number. Judge a run by `evaluate_testset`'s `value_floored_dice_mean`, not the training log's `final` — though note the value/Dice correlation itself should also be checked (a weak correlation, e.g. r≈0.12 seen on an early post-fix c3 run, means even the value-floored selector isn't fully trustworthy).
- **Windows console is cp1252.** Always set `PYTHONIOENCODING=utf-8`, or any print with `→`/`Φ`/non-ASCII crashes the run mid-training and looks like a silent hang.
- **`oracle_greedy()` in `diagnostics.py` is GT-privileged.** Its "headroom" numbers are an upper bound only — always report the honest non-GT-privileged variant (attention Dice − lite baseline Dice) as the architecture-comparison reference, not the oracle one. Note this is now a secondary diagnostic, not "the real target DRL is chasing" — DRL refines the attention net directly (single-baseline design), so its actual target is Dice gained over the attention baseline itself, not the lite→attention gap.
- **`reward_clip` must stay ≥4.0, not 1.0.** At 1.0 a catastrophic move and a mildly-bad move both floor to −1.0, erasing the gradient that tells the agent which mistakes are worse.
- **`spline_smooth` must be near-zero for BRISC.** Smoothing destroys irregular tumour boundaries; CAMUS (smooth anatomical structures) tolerates more (LA uses a *stronger* smoothing, 2.5, specifically to curb ballooning).
- **`max_steps` should stay capped (≈8–10), not 20.** Past ~10 steps agents wander past their Dice peak and degrade. Curriculum gives hard samples a *higher* cap during training only (§ above) — that's not the same as raising the base eval/deploy cap.
- **Notebooks must stay thin.** All logic lives in `iteris/`; notebooks only import → configure → call → display.
- **Don't deploy "best-seen" masks at test time** — no GT available then; the agent must learn to stop, or the value-floored selector must gate the deploy.
- **CPU, not GPU, is the training bottleneck.** `scipy`/`skimage` contour rasterisation in `env.step()` is CPU-bound — multi-GPU runs need `OMP_NUM_THREADS=1` or they oversubscribe.
- **Run `headroom_report` / the full diagnostic before any full training run.** Never train where the contour ceiling ≈ baseline — confirmed example: BRISC tumor (pooled) has *negative* realistic headroom, so "success" there is near-zero regression, not improvement; don't mistake that for a bug.
- **The new literature-standard metrics (IoU, Precision, Sensitivity, BIoU, MSD) are eval-only, computed once per finished rollout — never inside the env's per-step reward hot path.** Both `final_*` and `init_*` versions exist (the latter computed on the env's own rasterised init contour, `masks[0]`/`env.mask` right after `reset()` — not the raw `sample['init_mask']`, to avoid rasterisation mismatch) so every metric now has a baseline to diff against, the same way Dice always did.

## 4. Current state (keep this section current — see maintenance rule)

> This section was frozen at 2026-06-30 for three weeks (~90 commits) until a 2026-07-21
> documentation audit caught it. Rewritten to reflect current reality; see CONTEXT.md §10 for
> the full dated decision log this summarizes.

- **HEAD:** well past `fd9befa` — see `git log` for the exact commit; this doc tracks milestones, not a pinned SHA (the previous approach of naming one HEAD commit is exactly what let this section go stale for three weeks).
- **Major milestones since 2026-06-30 (see CONTEXT.md §10 for the full dated chain):**
  - **Reward-system rewrite (2026-07-08–11):** `contour_boundary` (dense per-control-point distance-to-GT-boundary) superseded PBRS as the active default; optimal-stopping STOP bonus added for the discrete agent. See CONTEXT.md §6.
  - **Two state-representation bugs fixed (2026-07-09):** debris-channel leak + SectorPool centroid-binning — invalidated any pre-fix `spatial_head=true` checkpoint.
  - **Eval consistency fix (2026-07-13):** TD3 now returns/evaluates the best-val checkpoint, not the final-step agent; `reeval_checkpoint()` added to re-score pre-fix checkpoints without retraining.
  - **Continuous auto-stop (2026-07-16):** GT-free action-magnitude auto-stop for TD3 (no learned STOP action, unlike the discrete agent).
  - **Single-baseline design confirmed current (2026-07-14, `61097d0`):** RL refines `AttentionResUNet` directly; `LiteUNet` is comparison-only. This reverses the 2026-06-21 decision (§1) — see CONTEXT.md §9 for the full history. **This doc's own §1 had drifted out of sync with that until the 2026-07-21 audit.**
  - **Phase A and Phase B both completed and wired end-to-end** — training (all 8 dataset×class×algo runs, both phases) → server registry (`5118e27` 2026-07-15 for Phase A, `4825754` 2026-07-20 for Phase B, 16 entries total) → deployed UI.
  - **Phase C (MSA backbone) formally abandoned (2026-07-21)** — a project decision, not a delay. Not pursued, not planned. See CONTEXT.md §3.
  - **Cross-run evaluation notebook added (2026-07-20):** `notebooks/evaluation/comprehensive_model_evaluation.ipynb` — ingests every run's output, phase-aware (Phase A/B never pooled), Plotly-based, feeds the Research page's real results (`91fbcc4`).
  - **CI/CD + Research page + rebrand (2026-07-19–20):** GitHub Actions for the HF Space deploy; Research page wired to real project state (dropped fabricated results); new logo/favicon.
- **BRISC tumor (pooled, general class): confirmed no realistic architecture-comparison headroom** (−0.0501, LiteUNet vs AttentionResUNet) — this is now a secondary diagnostic, not a go/no-go gate for DRL (which refines AttentionResUNet directly regardless). BRISC subtypes (glioma/meningioma/pituitary): not individually diagnosed on this metric.
- **CAMUS: all 3 classes have confirmed real architecture-comparison headroom** (LV_endo +0.050, LV_epi +0.073, LA +0.052), same caveat as above.
- **Open / not yet implemented:**
  1. **Pillar 2** — TD3 *learned* stop/commit head (the action-magnitude auto-stop, landed 2026-07-16, is a GT-free heuristic, not a learned head — still the eventual real fix).
  2. **Pillar 3** — strip HD95 out of any training reward variant that still includes it; compute HD95 only in eval/diagnostics.
  3. **Gate fix for CAMUS LV_endo/LA** — CE label smoothing alone (tried at 0.1 and 0.25) hasn't worked; needs a different lever (loss rebalancing or post-hoc calibration) before the uncertainty gate can be re-tried there.
  4. **mm-based metrics** — no `pixel_spacing`/`voxel_spacing` helper exists yet; would be new plumbing.
  5. **Evaluation-notebook phase-detection gap** (found 2026-07-20 by a downstream consumer, not yet root-caused): a real Kaggle run's U-Net baseline rows came out phase=`Unknown` in `master_comparison.csv` (their `label_frac` apparently didn't reach that run's summary JSON, and the U-Net dataset slugs don't follow the DRL runs' `pa-`/`pb-` convention the path-fallback relies on) — had to be disambiguated by hand in the frontend. Worth revisiting.
  6. **Phase C (MSA backbone)** — **abandoned, not open.** Do not pick this up without a fresh decision; see CONTEXT.md §3.
- **Timeline (project stages, not to be confused with CONTEXT.md's Phase A/B):** Stage 1 (build/validate) → Stage 2 (full training, **both phases done**) → Stage 3 (eval, **done** — the evaluation notebook) → Stage 4 (paper, current stage, Weeks 13–14).

## 5. Local environment

- Local Python: `F:\Anaconda\envs\fyp_env\python.exe` (torch/monai/scipy/skimage).
- Training actually runs on Kaggle (T4×2) via notebooks `03a_camus_drl_lv_endo.ipynb` / `04_brisc_drl.ipynb` (and siblings `03b`/`03c` for LV-epi/LA).
- Validation scripts in `scripts/` are CPU-only synthetic-disk checks; `_ckpt/` and `_*.txt` outputs are gitignored.
- When verifying fixes locally before pushing: leftover GPU memory from a prior crashed/interrupted Kaggle-style run can cause spurious local CUDA OOM/`cudaErrorUnknown` errors on small smoke tests — if `CUDA_VISIBLE_DEVICES=""` doesn't actually disable CUDA (it doesn't always, in this environment), monkeypatch `torch.cuda.is_available = lambda: False` at the top of the test script to force CPU.

## 6. Chunked training & resume (quota-saving)

Don't burn a full ~7h/50k Kaggle run to find out a config is broken. Each DRL notebook's training cell has a "Diagnostic / quota controls" block:
- `TRAIN_STEPS` — set to a fraction of the config's full `train_steps` for a cheap diagnostic chunk, or `None` to use the full budget. **Note:** the STOP-learning fix needs the full budget to show its effect (epsilon decays over 18k steps) — a short diagnostic chunk will under-represent how well STOP gets learned; don't judge the fix's success from a partial run.
- `RESUME` — `False` for a fresh run; `True` to continue from the `*_last.pt` checkpoint.

`run_drl_training` writes three kinds of checkpoint: `*_best.pt` (best-val, deploy), `*_last.pt` (latest full state — agent + optimizer moments + `step`, for resume), and `*_step<N>.pt` milestones every `checkpoint_every` steps (distinct snapshots, not overwritten). Resume restores the step counter so epsilon/bc_lambda continue (not reset to 1.0). The replay buffer is not persisted (re-prefills cold), but eval is greedy so the resumed curve stays continuous.
