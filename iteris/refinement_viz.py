"""
Visualisation + evaluation helpers for the local mask-refinement paradigm
(SegmentationEnv v4, 24 actions).

Kept in the package (not the notebook) so the per-notebook cells stay thin
one-liners and cannot break from stale inline branches. Every plot function
saves a PNG and returns the matplotlib Figure.

Usage in a notebook (§5 onward):
    from iteris.refinement_viz import (
        build_replays, plot_comparison, plot_playback,
        plot_behaviour, evaluate_testset, refinement_env_kwargs)
    ENV_KW   = refinement_env_kwargs(cfg)
    replays  = build_replays(agent, val_samples, ENV_KW, n_viz=8)
    plot_comparison(replays, baseline_cfg, cfg, out_path=...)
"""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .env import SegmentationEnv, dice_score, hd95_px

# Keys SegmentationEnv accepts — used to filter a flat cfg into env kwargs.
_ENV_KEYS = (
    'action_type', 'max_steps', 'shift_px', 'sdt_clip', 'reward_clip',
    'cont_morph_scale', 'cont_trans_scale',
    'reward_mode', 'reward_alpha', 'reward_beta', 'hd_norm',
    'stop_eps_dice', 'stop_eps_hd', 'stop_n', 'fail_thresh', 'fail_n',
    'reward_step_penalty', 'disable_auto_stop',
)


def refinement_env_kwargs(cfg: dict) -> dict:
    """Filter a flat resolved cfg down to SegmentationEnv constructor kwargs."""
    return {k: cfg[k] for k in _ENV_KEYS if k in cfg}


def _make_env(sample: dict, env_kwargs: dict) -> SegmentationEnv:
    return SegmentationEnv(
        image     = sample['image'],
        gt_mask   = sample['gt_mask'],
        init_mask = sample['init_mask'],
        prob_map  = sample.get('prob_map'),
        **env_kwargs,
    )


def replay_one(agent, sample: dict, env_kwargs: dict) -> Dict:
    """Greedy rollout of one sample. Records per-step masks/dices/actions."""
    env   = _make_env(sample, env_kwargs)
    state = env.reset()
    masks  = [env.mask.copy()]
    dices  = [env.dice_history[0]]
    acts   = []
    info   = {'dice': dices[0]}
    while True:
        a = agent.select_action(state, epsilon=0.0)
        acts.append(int(a))
        state, _, done, info = env.step(a)
        masks.append(env.mask.copy())
        dices.append(info['dice'])
        if done:
            break
    init_d  = env.dice_history[0]
    final_d = info['dice']
    return dict(
        sample     = sample,
        masks      = masks,
        dices      = dices,
        actions    = acts,
        final_mask = env.mask.copy(),
        best_mask  = env.get_best_mask(),
        init_dice  = init_d,
        final_dice = final_d,
        best_dice  = env.best_dice,
        final_hd95 = hd95_px(env.mask, sample['gt_mask']),
        gain       = final_d - init_d,
        n_steps    = len(acts),
        stopped    = bool(info.get('stop_action', False)),
    )


def build_replays(agent, samples: List[dict], env_kwargs: dict,
                  n_viz: int = 8, seed: int = 0) -> List[Dict]:
    """Replay ``n_viz`` random samples, sorted ascending by Dice gain
    (so replays[0]=worst, replays[-1]=best, replays[len//2]=median)."""
    rng = np.random.RandomState(seed)
    n = min(n_viz, len(samples))
    idx = rng.choice(len(samples), size=n, replace=False).tolist()
    replays = [replay_one(agent, samples[i], env_kwargs) for i in idx]
    replays.sort(key=lambda r: r['gain'])
    return replays


def plot_comparison(replays, baseline_cfg, cfg, class_idx=1, class_name='',
                    out_path=None):
    """§6: U-Net init | refined mask | GT, for best/median/worst-gain samples."""
    color = baseline_cfg['class_colors'][class_idx] if 'class_colors' in baseline_cfg else '#F43F5E'
    cmap  = ListedColormap([color])
    picks = [('BEST gain', replays[-1]),
             ('MEDIAN gain', replays[len(replays) // 2]),
             ('WORST gain', replays[0])]
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4 * len(picks)))
    for row, (label, r) in enumerate(picks):
        s = r['sample']
        cells = [('U-Net init', s['init_mask'], r['init_dice']),
                 (f'{cfg.get("agent_type","?")} refined', r['final_mask'], r['final_dice']),
                 ('Ground Truth', s['gt_mask'], 1.0)]
        for col, (title, mask, d) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(s['image'], cmap='gray')
            ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, alpha=0.5)
            tag = '' if col != 1 else f"  [{label}] {'stopped' if r['stopped'] else 'max-steps'}, {r['n_steps']} steps"
            ax.set_title(f'{title}\nDice {d:.3f}{tag}', fontsize=10)
            ax.axis('off')
    plt.suptitle(f'{cfg.get("dataset","")} {cfg.get("agent_type","")} — {class_name}', fontsize=13)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig


def plot_playback(replay, cfg, class_name='', out_path=None):
    """§7: per-step mask evolution for one episode (best-gain sample)."""
    masks = replay['masks']
    dices = replay['dices']
    s     = replay['sample']
    n     = len(masks)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    cmap = ListedColormap(['#F43F5E'])
    for t in range(n):
        ax = axes[t]
        ax.imshow(s['image'], cmap='gray')
        ax.imshow(np.ma.masked_where(masks[t] == 0, masks[t]), cmap=cmap, alpha=0.5)
        tag = 'init' if t == 0 else (f'step {t}: {cfg.get("__names__", ["?"]*99)[replay["actions"][t-1]]}'
                                     if '__names__' in cfg else f'step {t}')
        ax.set_title(f'{tag}\nDice {dices[t]:.3f}', fontsize=8)
        ax.axis('off')
    for t in range(n, len(axes)):
        axes[t].axis('off')
    plt.suptitle(f'{cfg.get("agent_type","")} refinement playback — {class_name} '
                 f'({replay["n_steps"]} steps, init {replay["init_dice"]:.3f} → '
                 f'final {replay["final_dice"]:.3f})', fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig


def plot_behaviour(replays, cfg, class_name='', out_path=None):
    """§8: per-episode Dice trajectories + action-usage histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Dice trajectory per replay (shows improve/degrade/stop behaviour)
    ax = axes[0]
    for r in replays:
        ax.plot(r['dices'], alpha=0.5, lw=1)
    ax.axhline(np.mean([r['init_dice'] for r in replays]), color='k',
               ls='--', lw=1, label='mean init Dice')
    ax.set(xlabel='Step', ylabel='Dice', title=f'{class_name} per-episode Dice trajectory')
    ax.legend(); ax.grid(alpha=0.3)

    # Right: action-usage histogram across all replays
    ax = axes[1]
    names = SegmentationEnv.DISCRETE_NAMES
    all_acts = [a for r in replays for a in r['actions']]
    counts = np.bincount(all_acts, minlength=len(names)).astype(float)
    counts = counts / max(counts.sum(), 1)
    ax.bar(range(len(names)), counts)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set(ylabel='Frequency', title=f'{class_name} action usage')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'{cfg.get("agent_type","")} behaviour — {class_name}')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)

    stop_rate = float(np.mean([r['stopped'] for r in replays]))
    print(f'\n── {cfg.get("agent_type","")} {class_name} behaviour ──')
    print(f'  Mean init Dice  : {np.mean([r["init_dice"] for r in replays]):.4f}')
    print(f'  Mean final Dice : {np.mean([r["final_dice"] for r in replays]):.4f}')
    print(f'  Mean best-seen  : {np.mean([r["best_dice"] for r in replays]):.4f}')
    print(f'  STOP-action rate: {stop_rate*100:.0f}%  (rest hit max_steps)')
    print(f'  Mean # steps    : {np.mean([r["n_steps"] for r in replays]):.1f}')
    return fig


def evaluate_testset(agent, test_samples: List[dict], env_kwargs: dict) -> Dict:
    """§10: greedy rollout over the test set → aggregate metrics."""
    init_d, final_d, best_d, final_h = [], [], [], []
    for s in test_samples:
        r = replay_one(agent, s, env_kwargs)
        init_d.append(r['init_dice']); final_d.append(r['final_dice'])
        best_d.append(r['best_dice']); final_h.append(r['final_hd95'])
    fh = np.asarray(final_h, dtype=float)
    fh = fh[~np.isnan(fh)]
    return dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),
        best_dice_mean  = float(np.mean(best_d)),
        final_hd95_mean = float(fh.mean()) if fh.size else float('nan'),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
    )
