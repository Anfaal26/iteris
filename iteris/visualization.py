"""Plotting helpers — learning curves and qualitative overlays."""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

from .utils import get_device, model_suffix


def plot_learning_curves(history: dict, cfg: dict, target_dice: float = 0.85):
    """Two-panel figure: loss curves + Dice curves with target line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(history['train_loss'], label='Train loss')
    ax1.plot(history['val_loss'],   label='Val loss')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss curves')
    ax1.legend()

    ax2.plot(history['train_dice'],    label='Train Dice', linestyle='--', alpha=0.7)
    ax2.plot(history['val_dice_mean'], label='Val Dice')
    ax2.axhline(target_dice, color='red', linestyle='--', label=f'Target {target_dice}')
    ax2.set(xlabel='Epoch', ylabel='Dice', title=f"{cfg['dataset']} — Dice curves")
    ax2.legend()

    plt.suptitle(f"{cfg['dataset']} Attention Residual U-Net — baseline")
    plt.tight_layout()
    out_path = os.path.join(cfg['checkpoint_dir'],
                            f"{cfg['dataset'].lower()}{model_suffix(cfg)}_learning_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    return out_path


@torch.no_grad()
def plot_qualitative_grid(model, test_loader, cfg: dict, n_show: int = 4):
    """Three-column grid: input | ground truth | prediction."""
    device = get_device()
    model.eval()
    cmap = ListedColormap(cfg['class_colors'])

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))

    for row, batch in enumerate(test_loader):
        if row >= n_show:
            break
        img_t  = batch['image'][0].to(device)
        label  = batch['label'][0, 0].numpy()
        pred   = model(img_t.unsqueeze(0)).argmax(1)[0].cpu().numpy()
        img_np = img_t[0].cpu().numpy()

        axes[row, 0].imshow(img_np, cmap='gray')
        axes[row, 0].set_title('Input'); axes[row, 0].axis('off')

        axes[row, 1].imshow(img_np, cmap='gray')
        axes[row, 1].imshow(label, cmap=cmap, alpha=0.5,
                            vmin=0, vmax=cfg['num_classes'] - 1)
        axes[row, 1].set_title('Ground Truth'); axes[row, 1].axis('off')

        axes[row, 2].imshow(img_np, cmap='gray')
        axes[row, 2].imshow(pred, cmap=cmap, alpha=0.5,
                            vmin=0, vmax=cfg['num_classes'] - 1)
        axes[row, 2].set_title('Prediction'); axes[row, 2].axis('off')

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(cfg['class_colors'][1:], cfg['class_names'][1:])]
    fig.legend(handles=patches, loc='upper right')
    plt.suptitle(f"{cfg['dataset']} — Qualitative Results")
    plt.tight_layout()
    out_path = os.path.join(cfg['checkpoint_dir'],
                            f"{cfg['dataset'].lower()}{model_suffix(cfg)}_qualitative.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    return out_path
