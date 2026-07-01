"""
Test-set evaluation and predicted-mask export.

Produces the per-patient CSV (Dice + HD95 per class) and the per-sample
.npy masks that downstream DRL notebooks consume as initial state.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import dice_score, hd95_batch
from .geometry import (iou_score, precision_recall, boundary_iou,
                       mean_surface_distance_px, hd95_px)
from .utils import get_device, model_suffix


@torch.no_grad()
def evaluate_test_set(model, test_loader, cfg: dict) -> pd.DataFrame:
    """
    Run the model over the test set, compute per-class segmentation metrics
    per sample, and write a CSV. Returns the DataFrame.

    Assumes test_loader has batch_size=1 (which `run_training` configures).

    Two metric families, both eval-only (computed once per test sample, never
    per training step — same design principle as the DRL agents' eval-time
    metrics; adds negligible wall-clock to a one-time end-of-training pass):

    - `dice_<c>` / `hd95_<c>` (existing): torch-batched, from `metrics.py`.
      Kept exactly as before for backward compatibility with any existing
      consumer of this CSV/JSON.
    - `iou_<c>` / `biou_<c>` / `precision_<c>` / `sensitivity_<c>` /
      `msd_<c>` / `hd95geo_<c>` (new): computed via `iteris.geometry`, the
      SAME functions the DRL agents' `evaluate_agent`/`evaluate_testset` use
      (largest-connected-component filtered) — so these numbers are directly
      comparable to the DRL test-set metrics, class-for-class. `hd95geo_<c>`
      is included alongside the original `hd95_<c>` because `metrics.hd95_batch`
      does NOT do connected-component filtering (a stray false-positive pixel
      inflates it) while `geometry.hd95_px` does — the two can legitimately
      differ; use `hd95geo_<c>` when comparing against a DRL agent's HD95.

    Not included: millimetre-based distances. No pixel/voxel spacing metadata
    is plumbed through the ingestion/transform pipeline (images are resized to
    a fixed pixel grid with no recorded physical scale) — reporting "mm" here
    would mean fabricating a scale factor, which is worse than reporting px
    and saying so. All distances are in pixels, matching the DRL side exactly.
    """
    device = get_device()
    model.eval()
    rows = []

    csv_path = os.path.join(cfg['checkpoint_dir'],
                            f"{cfg['dataset'].lower()}{model_suffix(cfg)}_test_scores.csv")

    for batch in tqdm(test_loader, desc='Test eval'):
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)
        logits = model(imgs)
        preds  = logits.argmax(1)                      # (B, H, W)

        dice = dice_score(preds, labels, cfg['num_classes'])     # (B, C-1)
        hd95 = hd95_batch(preds, labels, cfg['num_classes'])     # (B, C-1)

        # Single-sample numpy binary masks for the geometry-based metrics
        # (batch_size=1 is a documented precondition of this function).
        preds_np  = preds[0].cpu().numpy()
        labels_np = labels[0, 0].cpu().numpy() if labels.dim() == 4 else labels[0].cpu().numpy()

        row = {
            'patient': batch['patient'][0] if 'patient' in batch else '',
            'view':    batch.get('view',  [''])[0],
            'phase':   batch.get('phase', [''])[0],
        }
        for c, name in enumerate(cfg['class_names'][1:]):
            d_val = dice[0, c]
            h_val = hd95[0, c]
            row[f'dice_{name}'] = float(d_val) if not torch.isnan(d_val) else np.nan
            row[f'hd95_{name}'] = float(h_val) if not torch.isnan(h_val) else np.nan

            class_idx = c + 1  # class_names[1:] skips background at index 0
            pred_bin  = (preds_np  == class_idx)
            gt_bin    = (labels_np == class_idx)
            precision, sensitivity = precision_recall(pred_bin, gt_bin)
            row[f'iou_{name}']         = iou_score(pred_bin, gt_bin)
            row[f'biou_{name}']        = boundary_iou(pred_bin, gt_bin)
            row[f'precision_{name}']   = precision
            row[f'sensitivity_{name}'] = sensitivity
            msd = mean_surface_distance_px(pred_bin, gt_bin)
            row[f'msd_{name}']         = msd if not np.isnan(msd) else np.nan
            h_geo = hd95_px(pred_bin, gt_bin)
            row[f'hd95geo_{name}']     = h_geo if not np.isnan(h_geo) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'[evaluation] Saved per-patient scores → {csv_path}')

    # Console summary
    print('\n── Test Results ────────────────────────────────────────────────')
    for name in cfg['class_names'][1:]:
        d_mean = df[f'dice_{name}'].mean()
        h_mean = df[f'hd95_{name}'].mean()
        iou_mean  = df[f'iou_{name}'].mean()
        biou_mean = df[f'biou_{name}'].mean()
        prec_mean = df[f'precision_{name}'].mean()
        sens_mean = df[f'sensitivity_{name}'].mean()
        msd_mean  = df[f'msd_{name}'].mean()
        hgeo_mean = df[f'hd95geo_{name}'].mean()
        print(f'  {name:15s}:  Dice {d_mean:.4f}  IoU {iou_mean:.4f}  BIoU {biou_mean:.4f}  '
              f'Prec {prec_mean:.4f}  Sens {sens_mean:.4f}')
        print(f'  {"":15s}   95HD {h_mean:.2f}px  95HD(geo) {hgeo_mean:.2f}px  MSD {msd_mean:.2f}px')
    overall = df[[c for c in df if c.startswith('dice_')]].mean().mean()
    print(f'  {"Mean":15s}:  Dice {overall:.4f}')
    print('────────────────────────────────────────────────────────────────')
    print('  (All distances in px — no pixel-spacing metadata plumbed, see docstring)')
    return df


@torch.no_grad()
def export_predicted_masks(model, test_loader, cfg: dict, out_dirname: str = None):
    """
    Save argmax-prediction masks as .npy files, one per test sample.
    Consumed by the DRL environment as `init_mask` for each episode.
    """
    device = get_device()
    model.eval()

    out_dirname = out_dirname or f"{cfg['dataset'].lower()}{model_suffix(cfg)}_pred_masks"
    out_dir = os.path.join(cfg['checkpoint_dir'], out_dirname)
    os.makedirs(out_dir, exist_ok=True)

    for batch in tqdm(test_loader, desc='Exporting masks'):
        imgs  = batch['image'].to(device)
        pred  = model(imgs).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        pid   = batch['patient'][0] if 'patient' in batch else 'sample'
        view  = batch.get('view',  [''])[0]
        phase = batch.get('phase', [''])[0]
        fname = (f'{pid}_{view}_{phase}_pred.npy' if view
                 else f'{pid}_pred.npy')
        np.save(os.path.join(out_dir, fname), pred)

    print(f'[evaluation] Masks saved → {out_dir}/')
    return out_dir


def save_summary_json(history, scores_df, cfg: dict, best_dice: float) -> str:
    """Write a small JSON snapshot that the next-week pipelines load."""
    path = os.path.join(cfg['checkpoint_dir'],
                        f"{cfg['dataset'].lower()}{model_suffix(cfg)}_summary.json")

    test_dice = {name: round(float(scores_df[f'dice_{name}'].mean()), 4)
                 for name in cfg['class_names'][1:]}
    test_hd95 = {name: round(float(scores_df[f'hd95_{name}'].mean()), 2)
                 for name in cfg['class_names'][1:]}

    # DRL-comparable metrics (geometry.py, same functions/CC-filtering the DRL
    # agents' evaluate_testset uses — see evaluate_test_set's docstring).
    _new_metric_cols = ('iou', 'biou', 'precision', 'sensitivity', 'msd', 'hd95geo')
    test_extra = {
        metric: {name: round(float(scores_df[f'{metric}_{name}'].mean()), 4)
                 for name in cfg['class_names'][1:]}
        for metric in _new_metric_cols
    }

    _model_names = {'attn_resunet': 'AttentionResUNet', 'lite_unet': 'LiteUNet'}
    summary = dict(
        dataset        = cfg['dataset'],
        model          = _model_names.get(cfg.get('model', 'attn_resunet'),
                                          cfg.get('model', 'attn_resunet')),
        num_classes    = cfg['num_classes'],
        image_size     = list(cfg['image_size']),
        label_frac     = cfg.get('label_frac', 1.0),   # Phase A=1.0; Phase B/C<1.0, see docs/EXPERIMENTS.md
        best_val_dice  = round(float(best_dice), 4),
        test_dice      = test_dice,
        test_dice_mean = round(float(np.mean(list(test_dice.values()))), 4),
        test_iou         = test_extra['iou'],
        test_iou_mean    = round(float(np.mean(list(test_extra['iou'].values()))), 4),
        test_biou        = test_extra['biou'],
        test_precision   = test_extra['precision'],
        test_sensitivity = test_extra['sensitivity'],
        test_msd_px      = test_extra['msd'],
        test_hd95geo_px  = test_extra['hd95geo'],
        test_hd95      = test_hd95,
    )
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[evaluation] Summary → {path}')
    return path
