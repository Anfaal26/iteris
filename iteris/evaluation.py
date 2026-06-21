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
from .utils import get_device, model_suffix


@torch.no_grad()
def evaluate_test_set(model, test_loader, cfg: dict) -> pd.DataFrame:
    """
    Run the model over the test set, compute per-class Dice + HD95 per sample,
    and write a CSV. Returns the DataFrame.

    Assumes test_loader has batch_size=1 (which `run_training` configures).
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
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'[evaluation] Saved per-patient scores → {csv_path}')

    # Console summary
    print('\n── Test Results ────────────────────────────────────────────────')
    for name in cfg['class_names'][1:]:
        d_mean = df[f'dice_{name}'].mean()
        h_mean = df[f'hd95_{name}'].mean()
        print(f'  {name:15s}:  Dice {d_mean:.4f}  |  95HD {h_mean:.2f} px')
    overall = df[[c for c in df if c.startswith('dice_')]].mean().mean()
    print(f'  {"Mean":15s}:  Dice {overall:.4f}')
    print('────────────────────────────────────────────────────────────────')
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

    _model_names = {'attn_resunet': 'AttentionResUNet', 'lite_unet': 'LiteUNet'}
    summary = dict(
        dataset        = cfg['dataset'],
        model          = _model_names.get(cfg.get('model', 'attn_resunet'),
                                          cfg.get('model', 'attn_resunet')),
        num_classes    = cfg['num_classes'],
        image_size     = list(cfg['image_size']),
        best_val_dice  = round(float(best_dice), 4),
        test_dice      = test_dice,
        test_dice_mean = round(float(np.mean(list(test_dice.values()))), 4),
        test_hd95      = test_hd95,
    )
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[evaluation] Summary → {path}')
    return path
