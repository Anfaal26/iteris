"""
Training orchestration.

`run_training(cfg)` does the full ingestion → split → dataloader → train →
checkpoint pipeline and returns a dict with the model, loaders, and history.
Notebooks call this directly.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import CacheDataset, Dataset

from .config import REQUIRED_KEYS  # noqa: F401  (re-export for convenience)
from .ingestion import build_dataset_dicts
from .transforms import build_transforms
from .splits import patient_level_split
from .models import build_model
from .losses import build_loss
from .metrics import dice_score
from .utils import seed_all, get_device, count_parameters, model_suffix


# ─── Epoch loops ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, num_classes):
    """One epoch of training. Returns (avg_loss, avg_train_dice)."""
    model.train()
    total_loss = 0.0
    dice_vals  = []
    for batch in loader:
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        with torch.no_grad():
            preds = logits.argmax(1)
            dice_vals.append(dice_score(preds, labels, num_classes).mean().item())

    return total_loss / len(loader), float(np.mean(dice_vals))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    """One epoch of validation. Returns (avg_loss, per_class_mean_dice_np)."""
    model.eval()
    total_loss = 0.0
    per_class_dices = []
    for batch in loader:
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(1)
        per_class_dices.append(dice_score(preds, labels, num_classes))  # (B, C-1)
    per_class = torch.cat(per_class_dices, dim=0).mean(dim=0).cpu().numpy()
    return total_loss / len(loader), per_class


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_training(cfg: dict, return_loaders: bool = True) -> dict:
    """
    End-to-end pipeline: ingest → split → build loaders → train → checkpoint.

    Returns
    -------
    {
      'model'       : trained nn.Module (best weights loaded),
      'history'     : dict of per-epoch metrics,
      'best_dice'   : float,
      'checkpoint'  : path to .pt file,
      'train_loader': DataLoader (if return_loaders),
      'val_loader'  : DataLoader,
      'test_loader' : DataLoader,
      'test_dicts'  : raw test records (for mask export, etc),
    }
    """
    device = get_device()
    seed_all(cfg['seed'])
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    # Tag non-default architectures (model_suffix) so the lite baseline and the
    # attention net don't overwrite each other's checkpoint. attn_resunet keeps
    # the bare name (back-compat: camus_best.pt / brisc_best.pt). Same suffix the
    # evaluation/export/summary outputs use, so all artefacts stay consistent.
    _tag = model_suffix(cfg)
    ckpt_path = os.path.join(cfg['checkpoint_dir'],
                             f"{cfg['dataset'].lower()}{_tag}_best.pt")

    # ── Data ─────────────────────────────────────────────────────────────────
    records = build_dataset_dicts(cfg)
    train_d, val_d, test_d = patient_level_split(
        records,
        val_split=cfg['val_split'],
        test_split=cfg['test_split'],
        label_frac=cfg['label_frac'],
        seed=cfg['seed'],
    )

    train_tfm = build_transforms(cfg, split='train')
    eval_tfm  = build_transforms(cfg, split='val')

    # CacheDataset uses workers ONLY during the one-time cache build, not at training time.
    # At training time, samples are already in RAM. DataLoader num_workers=0 avoids
    # /dev/shm exhaustion on Kaggle (~64MB by default) which causes silent OOM kills.
    cache_workers = cfg.get('cache_workers', 2)
    dl_workers    = cfg.get('dataloader_workers', 0)
    cache_rate    = cfg.get('cache_rate', 1.0)

    train_ds = CacheDataset(train_d, transform=train_tfm, cache_rate=cache_rate, num_workers=cache_workers)
    val_ds   = CacheDataset(val_d,   transform=eval_tfm,  cache_rate=cache_rate, num_workers=cache_workers)
    test_ds  = Dataset(test_d,       transform=eval_tfm)

    bs = cfg['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=dl_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=dl_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,  shuffle=False, num_workers=dl_workers, pin_memory=True)

    # ── Model / Loss / Optim ─────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-6
    )

    print(f'[training] device: {device}  |  params: {count_parameters(model)/1e6:.2f}M')
    print(f'[training] target: ckpt → {ckpt_path}')

    # ── Loop ──────────────────────────────────────────────────────────────────
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice_mean': []}
    best_dice, patience_ct = 0.0, 0

    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss, tr_dice         = train_epoch(model, train_loader, optimizer, criterion, device, cfg['num_classes'])
        vl_loss, vl_per_class    = eval_epoch(model, val_loader, criterion, device, cfg['num_classes'])
        scheduler.step()

        mean_dice = float(np.nanmean(vl_per_class))
        history['train_loss'].append(tr_loss)
        history['train_dice'].append(tr_dice)
        history['val_loss'].append(vl_loss)
        history['val_dice_mean'].append(mean_dice)

        improved = mean_dice > best_dice
        if improved:
            best_dice, patience_ct = mean_dice, 0
            torch.save(model.state_dict(), ckpt_path)
            marker = '✓'
        else:
            patience_ct += 1
            marker = ' '

        class_str = '  '.join(
            f'{cfg["class_names"][c+1]}:{vl_per_class[c]:.3f}'
            for c in range(len(vl_per_class))
        )
        print(f'Ep {epoch:03d} | tr_loss {tr_loss:.4f} tr_dice {tr_dice:.4f} | '
              f'vl_loss {vl_loss:.4f} Dice {mean_dice:.4f} {marker} | {class_str}')

        # Periodic safety checkpoint — survives kernel disconnects
        save_every = cfg.get('save_every_n_epochs', 10)
        if save_every and epoch % save_every == 0:
            safety_path = os.path.join(
                cfg['checkpoint_dir'],
                f"{cfg['dataset'].lower()}{_tag}_epoch{epoch:03d}.pt",
            )
            torch.save({
                'epoch':              epoch,
                'model_state':        model.state_dict(),
                'optimizer_state':    optimizer.state_dict(),
                'scheduler_state':    scheduler.state_dict(),
                'best_dice':          best_dice,
                'history':            history,
            }, safety_path)
            print(f'[training]   safety ckpt → {safety_path}')

        if patience_ct >= cfg['patience']:
            print(f'[training] Early stop at epoch {epoch}.')
            break

    print(f'[training] Best val Dice: {best_dice:.4f}')

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    result = {
        'model':       model,
        'history':     history,
        'best_dice':   best_dice,
        'checkpoint':  ckpt_path,
        'test_dicts':  test_d,
    }
    if return_loaders:
        result['train_loader'] = train_loader
        result['val_loader']   = val_loader
        result['test_loader']  = test_loader
    return result
