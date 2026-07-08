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
from .geometry import iou_score, precision_recall, boundary_iou, mean_surface_distance_px
from .env_contour_refine import ContourRefineEnv


def refinement_env_cls(cfg: dict):
    """Resolve the env class from a flat cfg's ``env_class`` (default → global
    SegmentationEnv; 'contour' → control-point ContourRefineEnv)."""
    return ContourRefineEnv if cfg.get('env_class') == 'contour' else SegmentationEnv


# Map a discrete agent's action-head size → its env class. 24 (SegmentationEnv)
# and 18 (ContourRefineEnv) are distinct, so the viz can auto-match the env to
# the agent with no notebook change — the env_cls argument below is an override.
_NUM_ACTIONS_TO_ENV = {
    SegmentationEnv.NUM_DISCRETE_ACTIONS: SegmentationEnv,
    ContourRefineEnv.NUM_DISCRETE_ACTIONS: ContourRefineEnv,
}


def _resolve_env_cls(agent, explicit=None):
    """Pick the env class: explicit override, else infer from agent.num_actions
    (discrete), else fall back to SegmentationEnv (continuous / unknown)."""
    if explicit is not None:
        return explicit
    na = getattr(agent, 'num_actions', None)
    return _NUM_ACTIONS_TO_ENV.get(na, SegmentationEnv)

def refinement_env_kwargs(cfg: dict) -> dict:
    """Filter a flat resolved cfg down to the env constructor kwargs.

    Uses drl_training.ENV_OPTIONAL_KEYS — the SAME key set the training env is
    built from — so the eval/test/replay env is guaranteed identical to the one
    the agent trained on. Maintaining a second copy here previously let it drift
    (auto_smooth_lambda / uncertainty_gate were silently dropped at eval only,
    so TD3 and the gated class were tested on a different env than training).

    Also derives ``pbrs_gamma`` from the agent's ``gamma`` (same rule
    drl_training.py uses) so replay/visualisation envs reproduce the exact
    PBRS telescoping the agent was trained under, not the env's own default.
    """
    from .drl_training import ENV_OPTIONAL_KEYS
    kwargs = {k: cfg[k] for k in ENV_OPTIONAL_KEYS if k in cfg}
    kwargs['pbrs_gamma'] = cfg.get('gamma', 0.99)
    # action_type is NEVER present in the resolved YAML cfg -- run_drl_training
    # only ever computes it as a local variable from AGENT_REGISTRY, never
    # writes it back into cfg (and it is not in ENV_OPTIONAL_KEYS). Without the
    # explicit derivation below the env would fall back to its constructor
    # default ('discrete'): harmless for DQN/DuelingDDQN (matches by coincidence)
    # but fatal for TD3/DDPG — a continuous action (np.ndarray) hits the discrete
    # branch's `int(action)` in env_contour_refine.py's step() -> TypeError.
    # Derive it from agent_type instead of trusting cfg to carry it.
    if 'agent_type' in cfg:
        from .drl_training import AGENT_REGISTRY
        _, kwargs['action_type'] = AGENT_REGISTRY[cfg['agent_type'].upper()]
    return kwargs


def _make_env(sample: dict, env_kwargs: dict, env_cls=SegmentationEnv):
    return env_cls(
        image     = sample['image'],
        gt_mask   = sample['gt_mask'],
        init_mask = sample['init_mask'],
        prob_map  = sample.get('prob_map'),
        **env_kwargs,
    )


def replay_one(agent, sample: dict, env_kwargs: dict, env_cls=None) -> Dict:
    """Greedy rollout of one sample. Records per-step masks/dices/actions.
    ``env_cls`` defaults to auto-detection from the agent's action-head size."""
    env_cls = _resolve_env_cls(agent, env_cls)
    env   = _make_env(sample, env_kwargs, env_cls=env_cls)
    state = env.reset()
    masks  = [env.mask.copy()]
    dices  = [env.dice_history[0]]
    acts   = []
    info   = {'dice': dices[0]}
    # Discrete agents take an int action + epsilon-greedy; continuous agents
    # (DDPG / TD3) take a vector action + explore flag. Detect via action_type.
    is_discrete = getattr(agent, 'action_type', 'discrete') == 'discrete'
    # Pillar 1 (value-based "do no harm" floor): record the agent's OWN value
    # estimate V(s) of each visited state — GT-free, so usable at deployment.
    # values[t] is the value of masks[t]. We commit the highest-valued state and
    # fall back to init if nothing is valued above it (see below).
    has_value = hasattr(agent, 'state_value')
    values = [agent.state_value(state)] if has_value else None
    while True:
        if is_discrete:
            a = agent.select_action(state, epsilon=0.0)
            acts.append(int(a))
        else:
            a = agent.select_action(state, explore=False)
            acts.append(np.asarray(a, dtype=np.float32))
        state, _, done, info = env.step(a)
        masks.append(env.mask.copy())
        dices.append(info['dice'])
        if has_value:
            values.append(agent.state_value(state))
        if done:
            break
    init_d  = env.dice_history[0]
    final_d = info['dice']

    # ── Value-floored ("do no harm") deployable mask selection (Pillar 1) ──────
    # SELECTION uses only the agent's value estimates (no GT). We then score the
    # selected mask's Dice for REPORTING, exactly as we do for final/best — the
    # GT here measures the result, it does not drive the choice.
    if has_value:
        v = np.asarray(values, dtype=np.float64)
        best_t = int(v.argmax())
        vf_idx = best_t if v[best_t] > v[0] else 0   # else keep init = guaranteed parity
        vf_mask = masks[vf_idx]
        vf_dice = float(dice_score(vf_mask, sample['gt_mask']))
        vf_hd95 = hd95_px(vf_mask, sample['gt_mask'])
    else:
        values = None
        vf_idx, vf_mask = 0, masks[0]
        vf_dice, vf_hd95 = init_d, hd95_px(masks[0], sample['gt_mask'])

    # action_names only meaningful for discrete heads; None signals "continuous"
    # to plot_behaviour so it renders a per-sector magnitude bar instead.
    action_names = list(env_cls.DISCRETE_NAMES) if is_discrete else None

    # Extra literature-standard segmentation metrics — both on the FINAL mask
    # and on masks[0] (the env's own rasterised init contour, i.e. the same
    # mask init_dice is computed from — not the raw sample['init_mask'], to
    # avoid any rasterisation mismatch between the two). Computed once per
    # finished rollout, not per step; see geometry.py. The init_* versions are
    # what makes "does the agent beat baseline on IoU/BIoU/etc" answerable —
    # previously only Dice had a baseline figure to diff against.
    final_iou       = iou_score(env.mask, sample['gt_mask'])
    final_precision, final_sensitivity = precision_recall(env.mask, sample['gt_mask'])
    final_biou      = boundary_iou(env.mask, sample['gt_mask'])
    final_msd       = mean_surface_distance_px(env.mask, sample['gt_mask'])
    init_iou        = iou_score(masks[0], sample['gt_mask'])
    init_precision, init_sensitivity = precision_recall(masks[0], sample['gt_mask'])
    init_biou       = boundary_iou(masks[0], sample['gt_mask'])
    init_msd        = mean_surface_distance_px(masks[0], sample['gt_mask'])

    return dict(
        sample     = sample,
        masks      = masks,
        dices      = dices,
        actions    = acts,
        values     = values,
        final_mask = env.mask.copy(),
        best_mask  = env.get_best_mask(),
        value_floored_mask = vf_mask,
        value_floored_idx  = vf_idx,
        init_dice  = init_d,
        final_dice = final_d,
        best_dice  = env.best_dice,
        value_floored_dice = vf_dice,
        final_hd95 = hd95_px(env.mask, sample['gt_mask']),
        init_hd95  = hd95_px(masks[0], sample['gt_mask']),
        value_floored_hd95 = vf_hd95,
        final_iou         = final_iou,
        final_precision   = final_precision,
        final_sensitivity = final_sensitivity,
        final_biou        = final_biou,
        final_msd         = final_msd,
        init_iou          = init_iou,
        init_precision    = init_precision,
        init_sensitivity  = init_sensitivity,
        init_biou         = init_biou,
        init_msd          = init_msd,
        gain       = final_d - init_d,
        value_floored_gain = vf_dice - init_d,
        n_steps    = len(acts),
        stopped    = bool(info.get('stop_action', False)),
        action_names = action_names,
    )


def build_replays(agent, samples: List[dict], env_kwargs: dict,
                  n_viz: int = 8, seed: int = 0, env_cls=None) -> List[Dict]:
    """Replay ``n_viz`` random samples, sorted ascending by Dice gain
    (so replays[0]=worst, replays[-1]=best, replays[len//2]=median).

    ``env_cls`` defaults to auto-detection from the agent's action-head size
    (24→SegmentationEnv, 18→ContourRefineEnv), so the same notebook cell works
    for both paradigms. Pass refinement_env_cls(cfg) to override explicitly."""
    rng = np.random.RandomState(seed)
    n = min(n_viz, len(samples))
    idx = rng.choice(len(samples), size=n, replace=False).tolist()
    replays = [replay_one(agent, samples[i], env_kwargs, env_cls=env_cls) for i in idx]
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
        # Show the agent's ACTUAL starting mask (r['masks'][0] = the rasterised
        # largest-CC init contour), not the raw s['init_mask']. init_dice, gain
        # and every downstream metric are defined against this mask, so showing
        # the raw multi-blob U-Net output here while labelling it with the
        # single-contour init_dice was the source of the "which Dice is real?"
        # mismatch. Now panel and label refer to the same mask.
        cells = [('U-Net init (contour)', r['masks'][0], r['init_dice']),
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

    # Right: action usage. Discrete agents → action-frequency histogram.
    # Continuous agents (DDPG/TD3) have no discrete action vocabulary, so show
    # the mean per-component push magnitude instead (e.g. per angular sector).
    ax = axes[1]
    names = replays[0].get('action_names') if replays else None
    if names is not None:
        all_acts = [int(a) for r in replays for a in r['actions']]
        counts = np.bincount(all_acts, minlength=len(names)).astype(float)
        counts = counts / max(counts.sum(), 1)
        ax.bar(range(len(names)), counts)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set(ylabel='Frequency', title=f'{class_name} action usage')
    else:
        A = np.array([np.asarray(a) for r in replays for a in r['actions']])  # (T, K)
        if A.ndim == 2 and A.shape[0] > 0:
            mean_abs = np.abs(A).mean(axis=0)
            ax.bar(range(len(mean_abs)), mean_abs)
            ax.set_xticks(range(len(mean_abs)))
            ax.set(xlabel='action component (sector)', ylabel='mean |push|',
                   title=f'{class_name} mean per-sector push magnitude')
        else:
            ax.text(0.5, 0.5, 'no actions recorded', ha='center', va='center')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'{cfg.get("agent_type","")} behaviour — {class_name}')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)

    stop_rate = float(np.mean([r['stopped'] for r in replays]))
    print(f'\n── {cfg.get("agent_type","")} {class_name} behaviour ──')
    # NOTE: these means are over the {len(replays)} VISUALISATION replays only —
    # a tiny random subset for the qualitative plots, NOT a performance metric.
    # The reportable numbers are evaluate_testset()'s test-set JSON (init/final/
    # routed_dice_mean over all test samples). Small-sample noise here (one bad
    # debris case swings the mean) is expected — do not read these as results.
    print(f'  [{len(replays)} viz samples only — not the reported metric; see test JSON]')
    print(f'  Mean init Dice  : {np.mean([r["init_dice"] for r in replays]):.4f}')
    print(f'  Mean final Dice : {np.mean([r["final_dice"] for r in replays]):.4f}')
    print(f'  Mean best-seen  : {np.mean([r["best_dice"] for r in replays]):.4f}')
    print(f'  Mean final IoU  : {np.mean([r["final_iou"] for r in replays]):.4f}')
    print(f'  Mean final BIoU : {np.mean([r["final_biou"] for r in replays]):.4f}')
    print(f'  STOP-action rate: {stop_rate*100:.0f}%  (rest hit max_steps)')
    print(f'  Mean # steps    : {np.mean([r["n_steps"] for r in replays]):.1f}')
    return fig


def evaluate_testset(agent, test_samples: List[dict], env_kwargs: dict,
                     env_cls=None, refinable_gate: bool = False,
                     refinable_min_cc_frac: float = 0.004,
                     refinable_min_dominance: float = 0.5) -> Dict:
    """§10: greedy rollout over the test set → aggregate metrics.
    ``env_cls`` auto-detects from the agent's action-head size if not given.

    When ``refinable_gate=True`` the SAME GT-free init-quality gate used at
    training time (diagnostics.init_mask_refinable) is applied here, and the
    result additionally reports:
      * ``*_refinable_*`` — metrics on the in-regime subset only (the scientific
        claim: does refinement help where it is applicable?);
      * ``routed_*`` — the honest END-TO-END number over the FULL test set with a
        deployable fallback: refine when the gate passes, keep the U-Net (init)
        mask when it doesn't. No GT is used to decide routing, so nothing is
        cherry-picked — excluded cases are scored at their init Dice, not dropped.
    """
    from .diagnostics import init_mask_refinable
    init_d, final_d, best_d, final_h, vf_d = [], [], [], [], []
    final_iou, final_prec, final_sen, final_biou, final_msd = [], [], [], [], []
    init_h = []
    init_iou, init_prec, init_sen, init_biou, init_msd = [], [], [], [], []
    refinable_flags = []
    for s in test_samples:
        refinable_flags.append(bool(init_mask_refinable(
            s['init_mask'], refinable_min_cc_frac, refinable_min_dominance))
            if refinable_gate else True)
        r = replay_one(agent, s, env_kwargs, env_cls=env_cls)
        init_d.append(r['init_dice']); final_d.append(r['final_dice'])
        best_d.append(r['best_dice']); final_h.append(r['final_hd95'])
        vf_d.append(r['value_floored_dice'])
        final_iou.append(r['final_iou']); final_prec.append(r['final_precision'])
        final_sen.append(r['final_sensitivity']); final_biou.append(r['final_biou'])
        final_msd.append(r['final_msd'])
        init_h.append(r['init_hd95'])
        init_iou.append(r['init_iou']); init_prec.append(r['init_precision'])
        init_sen.append(r['init_sensitivity']); init_biou.append(r['init_biou'])
        init_msd.append(r['init_msd'])
    fh = np.asarray(final_h, dtype=float)
    fh = fh[~np.isnan(fh)]
    msd_arr = np.asarray(final_msd, dtype=float)
    msd_arr = msd_arr[~np.isnan(msd_arr)]
    ih = np.asarray(init_h, dtype=float)
    ih = ih[~np.isnan(ih)]
    init_msd_arr = np.asarray(init_msd, dtype=float)
    init_msd_arr = init_msd_arr[~np.isnan(init_msd_arr)]
    out = dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),       # raw last-state deploy (can go < init)
        best_dice_mean  = float(np.mean(best_d)),        # GT-privileged ceiling (NOT deployable)
        # Pillar 1: deployable value-floored deploy — GT-free selection, guaranteed
        # >= init by the agent's own value estimate. THIS is the honest deploy number.
        value_floored_dice_mean  = float(np.mean(vf_d)),
        value_floored_delta_mean = float(np.mean([v - i for v, i in zip(vf_d, init_d)])),
        final_hd95_mean = float(fh.mean()) if fh.size else float('nan'),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
        # Literature-standard extra metrics (final mask vs GT, full test set).
        final_iou_mean         = float(np.mean(final_iou)),
        final_precision_mean   = float(np.mean(final_prec)),
        final_sensitivity_mean = float(np.mean(final_sen)),
        final_biou_mean        = float(np.mean(final_biou)),
        final_msd_mean         = float(msd_arr.mean()) if msd_arr.size else float('nan'),
        # Same metrics on the U-Net baseline mask (init), so every extra metric
        # now has a baseline to diff against, the same way Dice already did.
        init_hd95_mean         = float(ih.mean()) if ih.size else float('nan'),
        init_iou_mean          = float(np.mean(init_iou)),
        init_precision_mean    = float(np.mean(init_prec)),
        init_sensitivity_mean  = float(np.mean(init_sen)),
        init_biou_mean         = float(np.mean(init_biou)),
        init_msd_mean          = float(init_msd_arr.mean()) if init_msd_arr.size else float('nan'),
        delta_iou_mean         = float(np.mean(final_iou) - np.mean(init_iou)),
        delta_biou_mean        = float(np.mean(final_biou) - np.mean(init_biou)),
    )

    # ── Refinable-gate reporting (GT-free routing) ──────────────────────────────
    if refinable_gate:
        flags = np.asarray(refinable_flags, dtype=bool)
        init_a = np.asarray(init_d, dtype=float)
        fin_a  = np.asarray(final_d, dtype=float)
        vf_a   = np.asarray(vf_d, dtype=float)
        n_ref  = int(flags.sum())
        # (a) subset means — refinement performance where it is APPLICABLE.
        if n_ref > 0:
            out.update(
                n_refinable                     = n_ref,
                n_total                         = int(flags.size),
                refinable_frac                  = float(flags.mean()),
                init_dice_refinable_mean        = float(init_a[flags].mean()),
                final_dice_refinable_mean       = float(fin_a[flags].mean()),
                value_floored_dice_refinable_mean = float(vf_a[flags].mean()),
                value_floored_delta_refinable_mean = float((vf_a[flags] - init_a[flags]).mean()),
            )
        # (b) routed full-set — deployable end-to-end: refine if gate passes,
        # else keep the U-Net (init) mask. GT never decides routing.
        routed = np.where(flags, vf_a, init_a)
        out.update(
            routed_dice_mean       = float(routed.mean()),
            routed_delta_mean      = float((routed - init_a).mean()),
        )
    return out
