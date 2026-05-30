"""
Visualisation + evaluation helpers for the tracing paradigm (Paradigm 1).

Kept out of the notebooks so the per-notebook tracing branches stay short: a
notebook sets ``IS_TRACING = cfg.get('env_class') == 'contour_tracing'`` and, when
true, calls these helpers instead of the refinement viz code.

All plotting functions save a PNG and return the matplotlib Figure.
"""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .env          import dice_score, hd95_px
from .env_contour  import ContourTracingEnv
from .contour_utils import DIRECTION_NAMES

# Keys ContourTracingEnv accepts — used to filter a flat cfg into env kwargs.
_ENV_KEYS = (
    'patch_size', 'max_trace_length', 'closure_tolerance', 'min_perimeter_steps',
    'boundary_bonus_distance', 'reward_boundary_bonus', 'reward_offimage',
    'reward_closure', 'reward_length_penalty', 'max_distance_penalty', 'seed_method',
)


def trace_env_kwargs(cfg: dict) -> dict:
    """Filter a flat resolved cfg down to ContourTracingEnv constructor kwargs."""
    return {k: cfg[k] for k in _ENV_KEYS if k in cfg}


def replay_trace(agent, sample: dict, env_kwargs: dict) -> Dict:
    """Greedy rollout of one sample. Returns trajectory + metrics for viz."""
    env = ContourTracingEnv(sample['image'], sample['gt_mask'],
                            sample['init_mask'], **env_kwargs)
    state = env.reset()
    dirs, info = [], {'closed': False}
    while True:
        a = agent.select_action(state, epsilon=0.0)
        dirs.append(int(a))
        state, _, done, info = env.step(a)
        if done:
            break
    final_mask = env.get_final_mask()
    init_d = dice_score(env.init_mask, env.gt)
    return dict(
        sample     = sample,
        trajectory = list(env.trajectory),
        seed_point = env.seed_point,
        final_mask = final_mask,
        dirs       = dirs,
        init_d     = init_d,
        final_d    = dice_score(final_mask, env.gt),
        final_h    = hd95_px(final_mask, env.gt),
        closed     = bool(info.get('closed', False)),
        gain       = dice_score(final_mask, env.gt) - init_d,
    )


def build_trace_replays(agent, samples: List[dict], env_kwargs: dict,
                        n_viz: int = 8, seed: int = 0) -> List[Dict]:
    """Replay ``n_viz`` random samples, sorted ascending by Dice gain."""
    np.random.seed(seed)
    n = min(n_viz, len(samples))
    idx = np.random.choice(len(samples), size=n, replace=False).tolist()
    replays = [replay_trace(agent, samples[i], env_kwargs) for i in idx]
    replays.sort(key=lambda r: r['gain'])
    return replays


def plot_trace_comparison(replays, baseline_cfg, cfg, class_idx, class_name,
                          out_path=None):
    """§6 (tracing): U-Net init | final rasterised trace | GT, for best/median/worst."""
    color = baseline_cfg['class_colors'][class_idx]
    cmap  = ListedColormap([color])
    picks = [('BEST gain', replays[-1]),
             ('MEDIAN gain', replays[len(replays) // 2]),
             ('WORST gain', replays[0])]
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4 * len(picks)))
    for row, (label, r) in enumerate(picks):
        s = r['sample']
        img, init, gt = s['image'], s['init_mask'], s['gt_mask']
        cells = [
            ('U-Net init (seed src)', init,            r['init_d']),
            (f'{cfg["agent_type"]}_TRACE mask', r['final_mask'], r['final_d']),
            ('Ground Truth',         gt,              1.0),
        ]
        for col, (title, mask, d) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, alpha=0.5)
            # overlay the trajectory polyline on the predicted-mask panel
            if col == 1:
                traj = np.asarray(r['trajectory'])
                ax.plot(traj[:, 1], traj[:, 0], '-', color='yellow', lw=0.8, alpha=0.9)
                sy, sx = r['seed_point']
                ax.plot(sx, sy, 'o', color='red', ms=4)
            tag = '' if col != 1 else (f"  [{label}] {'closed' if r['closed'] else 'OPEN'}")
            ax.set_title(f'{title}\nDice {d:.3f}{tag}', fontsize=10)
            ax.axis('off')
    plt.suptitle(f'{cfg["dataset"]} {cfg["agent_type"]}_TRACE — {class_name}', fontsize=13)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig


def plot_trajectory_playback(replay, cfg, class_name, out_path=None):
    """§7 (tracing): the trajectory drawn point-by-point, coloured by step index."""
    s = replay['sample']
    traj = np.asarray(replay['trajectory'])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(s['image'], cmap='gray')
    # GT boundary in green for reference
    from scipy import ndimage
    gt = s['gt_mask'].astype(bool)
    gt_edge = gt ^ ndimage.binary_erosion(gt)
    ax.imshow(np.ma.masked_where(~gt_edge, gt_edge), cmap='Greens', alpha=0.8)
    t = np.arange(len(traj))
    sc = ax.scatter(traj[:, 1], traj[:, 0], c=t, cmap='plasma', s=8)
    ax.plot(traj[:, 1], traj[:, 0], '-', color='white', lw=0.5, alpha=0.5)
    sy, sx = replay['seed_point']
    ax.plot(sx, sy, 'o', color='red', ms=8, label='seed')
    plt.colorbar(sc, ax=ax, label='step t', fraction=0.046)
    state = 'closed' if replay['closed'] else 'OPEN (timeout/offimage)'
    ax.set_title(f'{cfg["agent_type"]}_TRACE playback — {class_name}\n'
                 f'{len(traj)} steps, {state}, Dice {replay["final_d"]:.3f} '
                 f'(green = GT boundary)', fontsize=11)
    ax.legend(); ax.axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig


def plot_direction_behaviour(replays, cfg, class_name, out_path=None):
    """§8 (tracing): 8-direction preference histogram + trace-length distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_dirs = [d for r in replays for d in r['dirs']]
    counts = np.bincount(all_dirs, minlength=8).astype(float)
    counts = counts / max(counts.sum(), 1)
    axes[0].bar(range(8), counts)
    axes[0].set_xticks(range(8)); axes[0].set_xticklabels(DIRECTION_NAMES)
    axes[0].set(ylabel='Frequency', title=f'{class_name} direction distribution')
    axes[0].grid(alpha=0.3, axis='y')

    lengths = [len(r['trajectory']) for r in replays]
    axes[1].hist(lengths, bins=min(20, max(3, len(lengths))))
    axes[1].set(xlabel='Trajectory length (steps)', ylabel='Count',
                title=f'{class_name} trace-length distribution')
    axes[1].grid(alpha=0.3, axis='y')

    plt.suptitle(f'{cfg["agent_type"]}_TRACE behaviour — {class_name}')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)

    closed_rate = float(np.mean([r['closed'] for r in replays]))
    print(f'\n── {cfg["agent_type"]}_TRACE {class_name} behaviour ──')
    print(f'  Final Dice avg : {np.mean([r["final_d"] for r in replays]):.4f}')
    print(f'  Closure rate   : {closed_rate*100:.0f}%')
    print(f'  Avg trace len  : {np.mean(lengths):.1f} steps')
    return fig


def evaluate_trace_testset(agent, test_samples: List[dict], env_kwargs: dict) -> Dict:
    """§10 (tracing): rasterise each greedy trace, then Dice/HD95/closure aggregate."""
    init_d, final_d, final_h, closed = [], [], [], []
    for s in test_samples:
        r = replay_trace(agent, s, env_kwargs)
        init_d.append(r['init_d']); final_d.append(r['final_d'])
        final_h.append(r['final_h']); closed.append(r['closed'])
    return dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),
        final_hd95_mean = float(np.nanmean(final_h)),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
        closure_rate    = float(np.mean(closed)),
    )
