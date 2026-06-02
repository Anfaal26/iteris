"""
Dry-run diagnostics for the boundary-tracing paradigm.

A dry run trains for only ~600 steps — far too few to judge model quality.
Its real purpose is a **smoke test**: confirm every moving part of the tracing
pipeline is wired correctly (seeding → stepping → reward → rasterisation →
scoring) before committing to a multi-hour full run.

This module provides:
    dryrun_healthcheck(result, samples, cfg)  → runs assertions, prints a report,
                                                 returns a dict of pass/fail flags
    plot_dryrun(result, samples, cfg)          → a diagnostic figure (seed,
                                                 trajectory, rasterised mask, GT)
    dryrun_report(result, samples, cfg)        → convenience: healthcheck + plot

Usage in a notebook, right after the §3 dry-run cell:

    from iteris.dryrun_viz import dryrun_report
    dryrun_report(_dry_result, val_samples, cfg)

Nothing here assumes the agent is any good — it only checks the plumbing.
"""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .env           import dice_score
from .env_contour   import ContourTracingEnv
from .contour_utils import DIRECTION_NAMES
from .contour_viz   import trace_env_kwargs, replay_trace

NUM_DIRS = ContourTracingEnv.NUM_DISCRETE_ACTIONS   # 8


# ──────────────────────────────────────────────────────────────────────────────
# Internal: roll out a handful of samples and collect raw diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def _collect(agent, samples: List[dict], env_kwargs: dict, n: int) -> List[Dict]:
    """Greedy rollouts on the first ``n`` samples; also probes raw env signals."""
    diags = []
    for s in samples[:n]:
        # Probe the env directly for state-shape / reward finiteness on step 1.
        env   = ContourTracingEnv(s['image'], s['gt_mask'], s['init_mask'], **env_kwargs)
        state = env.reset()
        first_state_shape = tuple(state.shape)
        first_state_finite = bool(np.isfinite(state).all())
        # One greedy step to probe reward finiteness + that the point can advance.
        a0 = agent.select_action(state, epsilon=0.0)
        _, r0, _, _ = env.step(a0)
        reward_finite = bool(np.isfinite(r0))

        # Full greedy replay (reuses the shared contour_viz helper).
        r = replay_trace(agent, s, env_kwargs)

        diags.append(dict(
            seed_point        = r['seed_point'],
            traj_len          = len(r['trajectory']),
            dirs              = r['dirs'],
            final_mask_sum    = int(np.asarray(r['final_mask']).sum()),
            closed            = r['closed'],
            init_d            = r['init_d'],
            final_d           = r['final_d'],
            first_state_shape = first_state_shape,
            first_state_finite= first_state_finite,
            reward_finite     = reward_finite,
            sample            = s,
            replay            = r,
        ))
    return diags


# ──────────────────────────────────────────────────────────────────────────────
# Health check — the smoke test
# ──────────────────────────────────────────────────────────────────────────────
def dryrun_healthcheck(result: dict, samples: List[dict], cfg: dict,
                       n: int = 6) -> Dict:
    """Assert the tracing pipeline is wired correctly. Prints a report and
    returns a dict of {check_name: bool}. A dry run is healthy when every
    check passes — model *quality* is deliberately NOT assessed here."""
    agent      = result['agent']
    env_kwargs = trace_env_kwargs(cfg)
    patch      = cfg.get('patch_size', 64)
    diags      = _collect(agent, samples, env_kwargs, n)

    all_dirs   = [d for x in diags for d in x['dirs']]
    dir_counts = np.bincount(all_dirs, minlength=NUM_DIRS) if all_dirs else np.zeros(NUM_DIRS)

    checks = {}

    # 1. State tensor has the expected patch shape and is finite.
    checks['state_shape_ok'] = all(
        x['first_state_shape'] == (4, patch, patch) for x in diags)
    checks['state_finite'] = all(x['first_state_finite'] for x in diags)

    # 2. Rewards are finite numbers (no NaN/inf leaking from the EDT/reward).
    checks['reward_finite'] = all(x['reward_finite'] for x in diags)

    # 3. The agent actually walks: at least some trajectories advance past the seed.
    checks['trajectories_advance'] = any(x['traj_len'] > 1 for x in diags)

    # 4. Seeds are valid pixel coordinates inside the image.
    H, W = samples[0]['image'].shape[:2]
    checks['seeds_in_bounds'] = all(
        0 <= x['seed_point'][0] < H and 0 <= x['seed_point'][1] < W for x in diags)

    # 5. Rasterisation yields a non-empty mask whenever a trajectory is long
    #    enough to enclose area (≥3 vertices). (Short traces legitimately → empty.)
    long_enough = [x for x in diags if x['traj_len'] >= 3]
    checks['rasterise_nonempty'] = (
        all(x['final_mask_sum'] > 0 for x in long_enough) if long_enough else True)

    # 6. Dice scores are valid probabilities in [0, 1].
    checks['dice_in_range'] = all(
        0.0 <= x['final_d'] <= 1.0 and 0.0 <= x['init_d'] <= 1.0 for x in diags)

    # 7. Training history is populated with the expected columns.
    hist = result.get('history')
    needed = {'step', 'final_dice_mean', 'closure_rate'}
    checks['history_populated'] = (
        hist is not None and len(hist) > 0 and needed.issubset(set(hist.columns)))

    # 8. The action head isn't degenerate — greedy actions span >1 direction.
    #    (Soft: after only ~600 steps some collapse is normal, so warn not fail.)
    n_dirs_used = int((dir_counts > 0).sum())
    checks['actions_not_collapsed'] = n_dirs_used > 1

    # ── Report ────────────────────────────────────────────────────────────────
    hard = ['state_shape_ok', 'state_finite', 'reward_finite',
            'trajectories_advance', 'seeds_in_bounds', 'rasterise_nonempty',
            'dice_in_range', 'history_populated']
    soft = ['actions_not_collapsed']

    print('\n' + '═' * 64)
    print(f'  DRY-RUN HEALTH CHECK — {cfg.get("agent_type","?")} '
          f'(boundary tracing, {n} samples)')
    print('═' * 64)
    label = {
        'state_shape_ok':       f'State shape == (4,{patch},{patch})',
        'state_finite':         'State tensor all-finite',
        'reward_finite':        'Step reward finite (EDT/reward OK)',
        'trajectories_advance': 'Agent advances past the seed',
        'seeds_in_bounds':      'Seed points inside image',
        'rasterise_nonempty':   'Long traces rasterise to non-empty mask',
        'dice_in_range':        'Dice scores within [0, 1]',
        'history_populated':    'Training history has expected columns',
        'actions_not_collapsed':f'Greedy actions span >1 direction ({n_dirs_used}/{NUM_DIRS})',
    }
    for k in hard:
        print(f'  [{"PASS" if checks[k] else "FAIL"}]  {label[k]}')
    for k in soft:
        print(f'  [{"ok  " if checks[k] else "WARN"}]  {label[k]}'
              + ('' if checks[k] else '  (normal at 600 steps — recheck after §4)'))

    lengths = [x['traj_len'] for x in diags]
    print('─' * 64)
    print(f'  Trace lengths : min {min(lengths)}  max {max(lengths)}  '
          f'mean {np.mean(lengths):.1f}')
    print(f'  Closure rate  : {np.mean([x["closed"] for x in diags])*100:.0f}%  '
          f'(expected ~0% this early)')
    print(f'  Final Dice    : mean {np.mean([x["final_d"] for x in diags]):.3f}  '
          f'(meaningless at 600 steps — just checking it computes)')

    hard_pass = all(checks[k] for k in hard)
    print('─' * 64)
    if hard_pass:
        print('  ✓ PIPELINE HEALTHY — all structural checks passed. Safe to run §4.')
    else:
        failed = [label[k] for k in hard if not checks[k]]
        print('  ✗ PIPELINE PROBLEM — fix before §4. Failed:')
        for f in failed:
            print(f'      • {f}')
    print('═' * 64 + '\n')

    checks['_hard_pass'] = hard_pass
    checks['_diags']     = diags
    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plot — see the plumbing working
# ──────────────────────────────────────────────────────────────────────────────
def plot_dryrun(result: dict, samples: List[dict], cfg: dict,
                n_viz: int = 4, out_path: str = None):
    """Per-sample diagnostic grid: image+seed+trajectory | rasterised mask vs GT.
    Bottom row: action histogram + trace-length histogram across the samples."""
    agent      = result['agent']
    env_kwargs = trace_env_kwargs(cfg)
    diags      = _collect(agent, samples, env_kwargs, n_viz)
    pred_cmap  = ListedColormap(['#F59E0B'])   # amber predicted mask
    gt_cmap    = ListedColormap(['#10B981'])   # green GT

    fig = plt.figure(figsize=(4 * n_viz, 9))
    gs  = fig.add_gridspec(3, n_viz, height_ratios=[3, 3, 2])

    for j, x in enumerate(diags):
        s    = x['sample']
        traj = np.asarray(x['replay']['trajectory'])
        sy, sx = x['seed_point']

        # Row 0 — image + trajectory coloured by step + seed
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(s['image'], cmap='gray')
        if len(traj) > 1:
            ax.scatter(traj[:, 1], traj[:, 0], c=np.arange(len(traj)),
                       cmap='plasma', s=6)
        ax.plot(sx, sy, 'o', color='red', ms=7, label='seed')
        ax.set_title(f'trace #{j} — {x["traj_len"]} steps\n'
                     f'{"closed" if x["closed"] else "open"}', fontsize=9)
        ax.axis('off')
        if j == 0:
            ax.legend(loc='lower right', fontsize=7)

        # Row 1 — rasterised predicted mask (amber) vs GT (green outline)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(s['image'], cmap='gray')
        pred = np.asarray(x['replay']['final_mask'])
        gt   = s['gt_mask']
        ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=pred_cmap, alpha=0.5)
        ax.imshow(np.ma.masked_where(gt   == 0, gt),   cmap=gt_cmap,   alpha=0.35)
        ax.set_title(f'pred(amber) vs GT(green)\nDice {x["final_d"]:.3f}', fontsize=9)
        ax.axis('off')

    # Row 2, left half — action distribution
    all_dirs   = [d for x in diags for d in x['dirs']]
    dir_counts = (np.bincount(all_dirs, minlength=NUM_DIRS).astype(float)
                  if all_dirs else np.zeros(NUM_DIRS))
    axd = fig.add_subplot(gs[2, :max(1, n_viz // 2)])
    axd.bar(range(NUM_DIRS), dir_counts / max(dir_counts.sum(), 1))
    axd.set_xticks(range(NUM_DIRS))
    axd.set_xticklabels(DIRECTION_NAMES, fontsize=7)
    axd.set(ylabel='freq', title='Greedy action distribution')
    axd.grid(alpha=0.3, axis='y')

    # Row 2, right half — trace-length histogram
    axl = fig.add_subplot(gs[2, max(1, n_viz // 2):])
    lengths = [x['traj_len'] for x in diags]
    axl.hist(lengths, bins=min(10, max(2, len(lengths))))
    axl.set(xlabel='trajectory length', ylabel='count', title='Trace lengths')
    axl.grid(alpha=0.3, axis='y')

    fig.suptitle(f'Dry-run diagnostics — {cfg.get("agent_type","?")} '
                 f'(boundary tracing) — plumbing check, not quality', fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f'Saved → {out_path}')
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────
def dryrun_report(result: dict, samples: List[dict], cfg: dict,
                  n_viz: int = 4, out_path: str = None) -> Dict:
    """Run the health check, then draw the diagnostic figure. Returns the
    health-check dict (with '_hard_pass' indicating overall structural health)."""
    checks = dryrun_healthcheck(result, samples, cfg, n=max(n_viz, 6))
    fig = plot_dryrun(result, samples, cfg, n_viz=n_viz, out_path=out_path)
    plt.show()
    return checks
