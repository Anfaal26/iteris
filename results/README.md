# results/

Drop **whole run-output folders** here for
`notebooks/evaluation/comprehensive_model_evaluation.ipynb` — not just the JSON
summaries. Every file is identified by its content shape or, for files with no
metadata of their own (CSVs, PNGs, checkpoints), by matching it to a same-folder
sibling JSON with the same filename stem — never by which subfolder you use, so
you can organize things however you like (e.g. one folder per Kaggle output
dataset) without naming collisions.

**On Kaggle**: you don't need to copy anything into one place — attach every
output dataset (baselines, every DRL run, the classifier) to one notebook and the
evaluation notebook searches all of `/kaggle/input/` recursively on its own.
Outputs (figures, exported tables) go to `/kaggle/working/evaluation_outputs/`
there, since `/kaggle/input` is read-only.

## U-Net baselines (LiteUNet / AttentionResUNet)

Drop the whole `checkpoint_dir` output for each run — everything
`iteris.evaluation`/`iteris.visualization` write:

- `<dataset><suffix>_summary.json` (`save_summary_json`) — recognized by the `test_dice` key.
- `<dataset><suffix>_test_scores.csv` (`evaluate_test_set`) — **per-patient** Dice/IoU/HD95/etc,
  recognized by a `patient` column + `dice_<class>` columns. This is what makes
  Section 10's real (non-fabricated) Wilcoxon significance test possible — dropping
  the CSV alongside its sibling `_summary.json` (same folder, matching filename stem)
  lets LiteUNet vs AttentionResUNet be paired by patient ID and tested per class
  (within the same phase — see below).
- `<dataset><suffix>_learning_curves.png` (`plot_learning_curves`) and
  `<dataset><suffix>_qualitative.png` (`plot_qualitative_grid`) — displayed in the
  qualitative gallery (Section 9).

## Phase (data regime)

Every U-Net/DRL/classifier record is tagged **Phase A** (full-data) or **Phase B**
(~150-sample low-data regime) and never pooled across the two in any comparison.
Detection: `label_frac` in the summary JSON where present (U-Net/classifier always
have it — `>=1.0` is Phase A, `<1.0` is Phase B); DRL reeval JSON has no `label_frac`
at all, so those fall back to a `pa`/`pb` (or `phaseA`/`phaseB`) token anywhere in
the file's path — matching both this repo's `notebooks/phaseA`/`phaseB` layout and
a `pa-`/`pb-` Kaggle-dataset-slug naming convention. **Phase C is never used** —
if a file's path or label_frac implies Phase C, it's dropped and the count is
printed, not silently included. Anything with neither signal is tagged `Unknown`
and reported rather than guessed.

## DRL agents (DuelingDDQN / TD3)

Drop the whole `out_dir` output from `iteris.drl_reeval.reeval_checkpoint`:

- `<dataset>_<agent>_c<class>_reeval_test_results.json` — recognized by
  `init_dice_mean` + `final_dice_mean` keys. Any dict with those two keys plus
  `dataset` / `class_name` / `agent_type` fields works too, if you evaluated a
  checkpoint a different way (the shape `iteris.refinement_viz.evaluate_testset`
  returns, with those three fields added).
- `..._reeval_comparison.png` / `..._reeval_behaviour.png` — displayed in the
  qualitative gallery (Section 9).
- Checkpoints (`*_best.pt`, `*_stepN.pt`) — optional; if `torch` is importable in
  whatever runs the notebook, Section 11 reads scalar fields (`step`, `best_dice`)
  straight out of them with plain `torch.load` (never rebuilds a model, so this
  never needs `iteris`/`monai`). Skipped gracefully if `torch` isn't available.

### Optional — per-sample CSVs for DRL significance testing

Nothing in the pipeline exports this yet, but if you want a real Wilcoxon test on
a DRL run (not just the U-Net-vs-U-Net one described above), export a CSV with one
row per test patient and `init_dice` + `final_dice` columns (ideally also
`value_floored_dice`) — pulled from the `replays` list
`build_replays`/`evaluate_testset` (`iteris/refinement_viz.py`) already compute
internally (each replay dict has `init_dice`, `final_dice`, `value_floored_dice`
per sample).

## BRISC tumor-type classifier

Drop the classifier eval output:

- `brisc_tumor_classifier_summary.json` (`iteris.classifier.save_classifier_summary_json`)
  — recognized by `accuracy` + `per_class` keys inside `test_metrics`.
- `brisc_tumor_classifier_learning_curves.png` / `..._confusion_matrix.png` — displayed
  in Section 10.
- `brisc_tumor_classifier_best.pt` — optional, same lightweight metadata read as above.
