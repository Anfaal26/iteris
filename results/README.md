# results/

Drop run outputs here for `notebooks/evaluation/comprehensive_model_evaluation.ipynb`.
Files are identified by their **JSON content**, not their filename or folder, so you
can organize subfolders however you like (e.g. one per Kaggle run) without naming
collisions.

## U-Net baselines → `results/unet/` (or anywhere)

Drop the `<dataset><suffix>_summary.json` file `iteris.evaluation.save_summary_json`
writes to `cfg['checkpoint_dir']` after training (e.g. `camus_lite_summary.json`,
`camus_summary.json` for the attention model, `brisc_lite_summary.json`, ...).
Recognized by the `test_dice` dict key.

## DRL agents → `results/drl/` (or anywhere)

Drop the `<dataset>_<agent>_c<class>_reeval_test_results.json` file
`iteris.drl_reeval.reeval_checkpoint` writes, one per (dataset, class, agent) run —
e.g. `camus_td3_c1_reeval_test_results.json`, `brisc_dueling_c1_reeval_test_results.json`.
Recognized by the `init_dice_mean` + `final_dice_mean` keys.

If you evaluated a checkpoint a different way, any dict with those two keys plus
`dataset` / `class_name` / `agent_type` fields (the shape `evaluate_testset` in
`iteris/refinement_viz.py` returns, with those three fields added) works too.

## Optional — per-sample CSVs, for real significance testing

A CSV per run with one row per test patient and `init_dice` + `final_dice` columns
(ideally also `value_floored_dice`) enables the Wilcoxon signed-rank + Bonferroni
section. Without these, that section only reports what's missing — the JSON
summaries above are already aggregate means and can't support a real hypothesis test.

Pull these from the `replays` list `build_replays`/`evaluate_testset`
(`iteris/refinement_viz.py`) compute internally — each replay dict already has
`init_dice`, `final_dice`, `value_floored_dice` per sample; just write them to a
DataFrame and `.to_csv()`.
