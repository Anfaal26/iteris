"""
Pre-compute U-Net baseline predictions for DRL warm-start.

The DRL environment requires per-sample init masks (the U-Net's prediction).
Running the U-Net inside the RL loop at every reset() would be wasteful and
slow — instead we pre-compute predictions for all training/val/test samples
upfront and feed them into the env as `init_mask`.

This module:
  1. loads the U-Net baseline checkpoint
  2. iterates the dataset (using the same MONAI transforms as Week 1)
  3. binarises image + GT mask + U-Net prediction for the target class
  4. filters samples with GT structure area < 1% (degenerate cases)
  5. returns three lists of {image, gt_mask, init_mask, patient, ...} dicts
"""

from typing import List, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm

from .config     import load_config
from .ingestion  import build_dataset_dicts
from .splits     import patient_level_split
from .transforms import build_transforms
from .models     import build_model


def precompute_init_masks(
    baseline_cfg: dict,
    baseline_checkpoint: str,
    target_class: int,
    min_area_fraction: float = 0.01,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Run the U-Net baseline and return (train, val, test) sample lists.

    Parameters
    ----------
    baseline_cfg        : the baseline YAML config (e.g. configs/camus.yaml).
    baseline_checkpoint : path to camus_best.pt produced by Week 1.
    target_class        : class index to binarise to (1=LV_endo, 2=LV_epi, 3=LA).
    min_area_fraction   : drop samples whose GT structure covers less than this
                          fraction of the image — these are degenerate for
                          boundary refinement.

    Each returned sample dict has:
        image     : (H, W) float32 in [0, 1]
        gt_mask   : (H, W) uint8 in {0, 1}
        init_mask : (H, W) uint8 in {0, 1} — U-Net prediction for target_class
        patient, view, phase : provenance metadata
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load baseline U-Net
    print(f'[warm_start] Loading baseline from {baseline_checkpoint}...')
    model = build_model(baseline_cfg).to(device)
    model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    model.eval()

    # Records → patient-level split
    records = build_dataset_dicts(baseline_cfg)
    train_r, val_r, test_r = patient_level_split(
        records,
        val_split  = baseline_cfg['val_split'],
        test_split = baseline_cfg['test_split'],
        label_frac = baseline_cfg['label_frac'],
        seed       = baseline_cfg['seed'],
    )

    eval_tfm = build_transforms(baseline_cfg, split='val')

    def _process(records, split_name):
        out, dropped = [], 0
        for r in tqdm(records, desc=f'warm_start[{split_name}]'):
            data = eval_tfm(r)
            image_tensor = data['image']                       # (C, H, W)
            label = data['label'][0].numpy().astype(np.int64)  # (H, W)

            gt_bin = (label == target_class).astype(np.uint8)
            if gt_bin.sum() < min_area_fraction * gt_bin.size:
                dropped += 1
                continue

            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0).to(device))
                pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            init_bin = (pred == target_class).astype(np.uint8)
            image_np = image_tensor[0].numpy().astype(np.float32)   # first channel

            out.append(dict(
                image     = image_np,
                gt_mask   = gt_bin,
                init_mask = init_bin,
                patient   = r.get('patient', ''),
                view      = r.get('view', ''),
                phase     = r.get('phase', ''),
            ))
        print(f'[warm_start] {split_name}: kept {len(out)}, dropped {dropped} (GT < {min_area_fraction*100:.0f}% area)')
        return out

    return _process(train_r, 'train'), _process(val_r, 'val'), _process(test_r, 'test')
