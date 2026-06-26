# Iteris — Project Skill

> The operational playbook for working on this codebase: how it's wired, where the landmines are, and what's currently open. Complements [CONTEXT.md](CONTEXT.md) (what/why) and [PLAN.md](PLAN.md) (roadmap) — this file is "how to not break it" + "what I already learned the hard way."
>
> **Maintenance rule (this is the main skill of this file — apply it every time):** whenever a new architectural decision, fix, gotcha, or piece of standing knowledge is produced, update the relevant section below *in the same turn*, and add a matching dated entry to [TRANSCRIPT.md](TRANSCRIPT.md). TRANSCRIPT.md is the chronological "what happened, when" log; this file is the always-current "what you need to know" summary — the same fact often belongs in both, phrased differently (transcript = past tense event, skill = standing rule). Never let this file silently drift out of date relative to the transcript.

---

## 1. What this project is (one paragraph)

DRL agent (`DuelingDDQN` discrete or `TD3` continuous) refines a **lite U-Net** segmentation mask toward ground truth by deforming its boundary contour in angular sectors, on CAMUS (cardiac ultrasound: LV-endo/epi/LA) and BRISC (brain tumour MRI: glioma/meningioma/pituitary/tumour-generic). A separate **attention U-Net** is the fixed upper-bound competitor — RL never touches it. Reward is baseline-centred PBRS (`Φ = K·(Dice − Dice_0)`) so holding steady at baseline is reward-neutral, not penalized.

## 2. Key files — where to make changes

| Concern | File |
|---|---|
| Contour env (the live paradigm) | `iteris/env_contour_refine.py` |
| Shared geometry helpers | `iteris/geometry.py` |
| Agents (DuelingDQN, TD3 active; DQN/DDPG archived-use) | `iteris/agents.py` |
| Networks incl. optional spatial head | `iteris/drl_networks.py` |
| Training loop / config resolution | `iteris/drl_training.py` |
| Offline diagnostics (headroom, pillar4) | `iteris/diagnostics.py` |
| BC warm-start demos | `iteris/bc_demo.py` |
| Deploy-time replay/eval + value-floored selector | `iteris/refinement_viz.py` |
| Models (LiteUNet, AttentionResUNet) | `iteris/models.py` |
| Configs (one YAML block per dataset×agent) | `configs/{camus,brisc}_{lite,drl_*}.yaml` |
| Back-compat shim for old global-morph env | `iteris/env.py` (do not add new logic here) |
| Archived Paradigm A (global morphology, ablation only) | `iteris/archive_paradigm_a/` |
| Multi-GPU concurrent runner | `scripts/run_drl_config.py` |

Config resolution: `resolve_agent_config(load_drl_class_config(path), AGENT_NAME)`. Artefacts are model-suffixed via `utils.model_suffix` so lite/attention checkpoints never collide.

## 3. Standing gotchas (learned the hard way — don't re-discover these)

- **Kaggle runs use the pushed `iteris-pkg` dataset + uploaded notebook, NOT your working tree.** A fix that's only edited locally (uncommitted, or committed-but-not-pushed, or pushed-but-the-Kaggle-dataset-not-bumped) is NOT in the run. Symptom that you're running stale code: a bug you already fixed reappears (e.g. the `plt` NameError that `02354fe` fixed, or gate-paralysis after you set `uncertainty_gate: false`). Before trusting any Kaggle result, confirm: fixes committed+pushed → `iteris-pkg` dataset bumped to a new version → notebook re-uploaded. The 2026-06-26 DuelingDDQN c3 run wasted ~7h because it ran the old gate-ON config (the fix was uncommitted).
- **The training-log `final`/`Δ` is the RAW number, not the deployable one.** `evaluate_agent` reports raw final Dice with no value floor. `best-seen > baseline` while `final < baseline` means the agent reaches good masks but overshoots; the value-floored deploy (Pillar 1, `refinement_viz.evaluate_testset`) is the real number and never goes below baseline. Judge a run by `evaluate_testset`'s `value_floored_dice_mean`, NOT the training log's `final`.
- **Windows console is cp1252.** Always set `PYTHONIOENCODING=utf-8`, or any print with `→`/`Φ`/non-ASCII crashes the run mid-training and looks like a silent hang.
- **`oracle_greedy()` in `diagnostics.py` is GT-privileged.** Its "headroom" numbers are an upper bound only — never report them as an achievable target without the honest non-GT-privileged variant added in the P0.3 fix.
- **`reward_clip` must stay ≥4.0, not 1.0.** At 1.0 a catastrophic move and a mildly-bad move both floor to −1.0, erasing the gradient that tells the agent which mistakes are worse.
- **TD3 has no STOP action and a reward-blind termination heuristic.** `_check_termination()` in `env_contour_refine.py:364-374` only checks whether Dice *stopped changing*, never whether it's good — so a drifting actor can converge below baseline and commit it (`final_dice` = the deploy number AND the checkpoint-selection metric). DuelingDDQN doesn't have this problem because `disable_auto_stop: true` lets its learned STOP action (reward exactly 0) govern termination — **do not flip this to `false` for DuelingDDQN**, it would break the one agent that's actually working. **Interim mitigation (set 2026-06-26 on all 3 CAMUS TD3 blocks):** `fail_thresh: 0.02`, `fail_n: 2` terminate early if Dice sits >0.02 below baseline for 2 steps. This is a backstop, NOT the real fix — Pillar 2 (a TD3 *learned* stop/commit head) is still open.
- **`evaluate_agent` (training-time eval + checkpoint selection) does NOT apply the Pillar-1 value floor.** It selects the best checkpoint on raw `final_dice_mean`, which can be below baseline; the value-floored "do-no-harm" selector lives only in `refinement_viz.evaluate_testset` (deploy/test time). So the checkpoint is chosen on a more permissive metric than deployment uses. Known gap as of 2026-06-26 — wiring the floor into `evaluate_agent` would also change discrete-agent eval semantics, so it needs an explicit decision before touching.
- **`spline_smooth` must be near-zero for BRISC.** Smoothing destroys irregular tumour boundaries; CAMUS (smooth anatomical structures) tolerates more.
- **`uncertainty_gate: true` is only safe when the prob_map is graded, not binary.** The gate ([env_contour_refine.py:387-403](D:\iteris\iteris\env_contour_refine.py:387)) multiplies edit magnitude by 1.0 only inside `[gate_lo, gate_hi]`, ramping to 0.0 outside. If `pillar4_report`'s `prob_map` check comes back **INERT** (lite net overconfident, ~binary probabilities), `band_frac` is near-zero and the gate clamps edits to ~0 at nearly every contour point — it doesn't "do nothing," it paralyzes the agent on exactly the classes where real headroom exists. **For TD3 it's even worse:** `_apply_continuous` (env_contour_refine.py:498) multiplies the actor's displacement by the gate, AND `bc_demo.py` builds its oracle env from the same `env_kwargs`+prob_map — so an INERT gate zeroes both the RL actor's moves and the BC oracle's trial moves, meaning the BC warm-start (whose entire job is to escape the identity basin) trains the actor *toward* identity. **Always run Pillar 4 before enabling the gate for a new class**; if INERT, set `uncertainty_gate: false` for that class/agent until the lite net is retrained with label smoothing. As of 2026-06-26: OFF for CAMUS LV_endo + LA on **both DuelingDDQN and TD3** (all INERT); still ON for CAMUS LV_epi (USABLE, both agents). When the label-smoothing retrain is confirmed USABLE, flip all four blocks (Dueling+TD3 × c1+c3) back on together.
- **`cfg['loss_label_smoothing']` controls CE-term label smoothing in `build_loss()` ([losses.py](D:\iteris\iteris\losses.py)).** Default `0.0` (off, backward-compatible). Wired straight into MONAI `DiceCELoss(label_smoothing=...)` → `torch.nn.CrossEntropyLoss`; only affects the CE term, not the Dice term's `smooth_nr`/`smooth_dr` (unrelated numerical epsilons). This is the fix for an INERT prob_map — an overconfident net pins softmax near 0/1, which both starves the uncertainty gate above *and* makes `prob_map` near-redundant with `init_mask` as one of the contour env's 5 state channels. Set on `camus_lite.yaml` (`0.1`) as of 2026-06-26; not yet set on `brisc_lite.yaml` (BRISC paused).
- **`max_steps` should stay capped (≈10), not 20.** Past ~10 steps agents wander past their Dice peak and degrade — confirmed from the first 20k-step Kaggle runs.
- **Notebooks must stay thin.** All logic lives in `iteris/`; notebooks only import → configure → call → display (Engineering Principle in PLAN.md — violating this is how the `plt`/`agent`/`env_cls` notebook crashes happened).
- **Don't deploy "best-seen" masks at test time** — no GT available then; the agent must learn to stop or the value-floored selector (Pillar 1) must gate the deploy.
- **CPU, not GPU, is the training bottleneck.** `scipy`/`skimage` contour rasterisation in `env.step()` is CPU-bound — multi-GPU runs need `OMP_NUM_THREADS=1` (see `run_drl_config.py`) or they oversubscribe and slow each other down.
- **Run `headroom_report` before any full training run.** Never train where the contour ceiling ≈ baseline (no point) — it's a cheap go/no-go gate per PLAN.md.

## 4. Current state (keep this section current — see maintenance rule)

- **HEAD:** `4bb93fe` on `main`, pushed to `origin/main` (2026-06-26 — CAMUS gate-off + label smoothing + TD3 safety net + chunked resume + notebook hardening). The two stale `colab_*.ipynb` were deliberately left untracked, not committed.
- **`notebooks/colab_brisc_drl.ipynb`, `notebooks/colab_camus_drl.ipynb` are stale.** Dated 2026-06-20, one day before the lite-U-Net pivot (`7da94b0`, 2026-06-21) — pre-pivot artefacts, not in-progress work. Safe to ignore/delete.
- **`notebooks/00_free_diagnostics_all_classes.ipynb`** — the full **EDA + DRL-compatibility report** (rewritten 2026-06-26). Run before any GPU round. `full_report(label)` warm-starts the lite net once per class then reports: dataset EDA (area/topology/intensity/view-phase), lite baseline (init Dice/IoU dist + worst cases + prob_map INERT/USABLE), error decomposition (boundary vs topology/interior), DRL compatibility (repr ceiling, GT-oracle ceiling, realistic headroom vs attention). Outputs `/kaggle/working/eda_diagnostic_summary.csv` + per-class figures + a data-driven recommendation (SKIP/RISKY/TRAIN + gate guidance + the single best bet). CAMUS by default; `RUN_BRISC` flag for BRISC. Notebooks are NOT in `iteris-pkg` — upload the `.ipynb` to Kaggle directly.
- **Landed:** baseline-centred PBRS, TD3 + angular-sector contour, lite-U-Net pivot, P0/P1/P2 batch (uncertainty gate, reward de-clip, TD3 BC warm-start, optional spatial head), multi-GPU runner, notebook crash fixes, Pillar 1 (value-floored deploy) + Pillar 4 (offline diagnostics), per-class diagnostics run (2026-06-25) + `uncertainty_gate: false` fix for CAMUS LV_endo/LA DuelingDDQN.
- **BRISC: paused.** Diagnostics (2026-06-25) show NO realistic headroom — lite baseline 0.885 is already *above* the attention competitor's 0.835 on generic tumour. Not worth a GPU round until/unless re-diagnosed (e.g. after a different attention-Dice measurement or subtype-level diagnostics).
- **CAMUS: retraining the lite U-Net first (user's choice, 2026-06-26), before further GPU rounds.** All 3 classes have real, boundary-shaped headroom (LV_endo +0.047, LV_epi +0.077, LA +0.049). `camus_lite.yaml` now has `loss_label_smoothing: 0.1` — **not yet retrained**; do that next via `01_camus_lite.ipynb`, then re-run `00_free_diagnostics_all_classes.ipynb` to confirm LV_endo/LA's `prob_map` is USABLE before re-enabling `uncertainty_gate`.
- **DuelingDDQN c3 (LA) run: done** (user, ~2026-06-26). Results not yet logged here — fold in numbers when available.
- **CAMUS TD3: reviewed + config-fixed 2026-06-26, ready for a run after the retrain.** Per class: `uncertainty_gate` = false (c1 LV_endo), true (c2 LV_epi), false (c3 LA) — matching the DuelingDDQN gate state. `fail_thresh: 0.02`/`fail_n: 2` added to all 3 as the interim below-baseline safety net. `bc_warm_start: true` on all 3 (only meaningful once the gate is off / prob_map usable — see gate gotcha).
- **Open / not yet implemented:**
  1. **Pillar 2** — TD3 learned stop/commit head (or at minimum, set `fail_thresh`/`fail_n` > 0 in TD3 config blocks as a cheap interim safety net).
  2. **Pillar 3** — strip HD95 out of training reward (`dice_hd_pbrs` → `dice_pbrs` for training configs; compute HD95 only in eval/diagnostics).
  3. **Pillar 5/6** — explicitly deferred (action-space expansion; paper reframing).
  4. **mm-based metrics / BIoU** — requested by user; no `pixel_spacing`/`voxel_spacing` helper exists yet in `transforms.py`/`config.py`; would be new plumbing, not a wire-up.
  5. ~~TD3 + CAMUS LV_endo/LA `uncertainty_gate`~~ — DONE 2026-06-26 (gate off for c1/c3 TD3 + fail-safe on all 3).
  6. ~~Lite U-Net label-smoothing retrain~~ — DONE 2026-06-26 (`loss_label_smoothing: 0.1`, user retrained). Pending: re-run diagnostics on the new checkpoint to confirm LV_endo/LA prob_map is now USABLE, then re-enable the gates.
- **Timeline:** Semester week 10 of 14 (as of 2026-06-21). Phase A (build/validate) → B (full training, ~1 week) → C (eval, Wilcoxon + Bonferroni, Week 12) → D (paper, Weeks 13–14).

## 5. Local environment

- Local Python: `F:\Anaconda\envs\fyp_env\python.exe` (torch/monai/scipy/skimage).
- Training actually runs on Kaggle (T4×2) via notebooks `03a_camus_drl_lv_endo.ipynb` / `04_brisc_drl.ipynb` (and siblings `03b`/`03c` for LV-epi/LA).
- Validation scripts in `scripts/` are CPU-only synthetic-disk checks; `_ckpt/` and `_*.txt` outputs are gitignored.

## 6. Chunked training & resume (quota-saving)

Don't burn a full ~7h/60k Kaggle run to find out a config is broken. Each DRL notebook's training cell (cell 10) has a "Diagnostic / quota controls" block:
- `TRAIN_STEPS` — `20000` for a ~1/3-cost diagnostic chunk (default), or `None` to use the config's full `train_steps`.
- `RESUME` — `False` for a fresh run; `True` to continue from the `*_last.pt` checkpoint.

Workflow: run `TRAIN_STEPS=20000, RESUME=False` → inspect the §9 curve + §10 value-floored test eval → if `final`/`best-seen` trend up and beat baseline, set `TRAIN_STEPS=60000, RESUME=True` to run only the remaining 40k (no redo). `run_drl_training` writes two checkpoints: `*_best.pt` (best-val, deploy) and `*_last.pt` (latest full state — agent + optimizer moments + `step`, for resume). Resume restores the step counter so epsilon/bc_lambda continue (NOT reset to 1.0). The replay buffer is not persisted (re-prefills cold), but eval is greedy so the resumed curve stays continuous.
