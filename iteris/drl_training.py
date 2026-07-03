"""
DRL training loop.

Top-level entry point: `run_drl_training(cfg, train_samples, val_samples)`.
Each sample dict has keys {image, gt_mask, init_mask, patient, ...} produced
by `warm_start.precompute_init_masks()`.

Handles:
  - per-sample state caches (avoid recomputing image + init_mask channels)
  - buffer pre-fill with random rollouts
  - epsilon-greedy decay for DQN-family
  - periodic validation evaluation with greedy policy
  - best-checkpoint saving + training history return
"""

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from .env     import (SegmentationEnv, signed_dt, dice_score, hd95_px)
from .geometry import (iou_score, precision_recall, boundary_iou,
                       mean_surface_distance_px, sdt_direction_field)
from .diagnostics import sample_error_decomp, init_mask_refinable
from .buffer  import ReplayBuffer
from .agents  import DQNAgent, DuelingDQNAgent, DDPGAgent, TD3Agent
from .env_contour_refine import ContourRefineEnv
from . import bc_demo


# ─── Environment registry ────────────────────────────────────────────────────
# 'default' : SegmentationEnv — 24-action GLOBAL morphological mask refinement.
# 'contour' : ContourRefineEnv — control-point + spline LOCAL contour refinement
#             (DeepSnake/MARL-style sector push-out/in/smooth/stop). Same 4-channel
#             state and discrete-agent interface as 'default', so it is a drop-in
#             env swap selected via `env_class: contour` in the per-agent config.
# Archived : ContourTracingEnv (Paradigm 1, 8-dir boundary tracing) — retired for
#            staircase artefacts; see iteris/archive/paradigm1_boundary_tracing/.
ENV_REGISTRY = {
    'default': SegmentationEnv,
    'contour': ContourRefineEnv,
}

CONTINUOUS_ACTION_DIM = SegmentationEnv.CONTINUOUS_ACTION_DIM   # 3


AGENT_REGISTRY = {
    'DQN':     (DQNAgent,        'discrete'),   # 14-action local mask refinement
    'DUELING': (DuelingDQNAgent, 'discrete'),   # same, with V+A dueling heads
    'DDPG':    (DDPGAgent,       'continuous'), # 3-D global morph+shift (baseline-capped)
    'TD3':     (TD3Agent,        'continuous'), # robust DDPG on angular-sector contour action
}


def _build_state_caches(samples: List[dict], image_size: int) -> dict:
    """Stack image + init_mask (+ optional prob_map) arrays for vectorised lookup."""
    caches = dict(
        image     = np.stack([s['image']     for s in samples]).astype(np.float32),
        gt_mask   = np.stack([s['gt_mask']   for s in samples]).astype(np.uint8),
        init_mask = np.stack([s['init_mask'] for s in samples]).astype(np.uint8),
    )
    # prob_map is present when warm_start was run with the updated code.
    # Stored as float16 to save RAM; env converts to float32 on use.
    if all('prob_map' in s for s in samples):
        caches['prob_map'] = np.stack([s['prob_map'] for s in samples]).astype(np.float16)
    return caches


def _make_state_builder(caches: dict, sdt_clip: float, directional: bool = False):
    """
    Return a builder (idx, current_mask, cached_sdt=None) → torch (C, H, W).

    If `cached_sdt` is provided (e.g. from ReplayBuffer with cache_sdt=True),
    skip the expensive scipy.ndimage SDT recomputation. This is the single
    biggest training-speed optimisation — without it, agent.update() spends
    ~80% of its time recomputing SDTs that were already computed during
    env.step().

    `directional`: when True, append the 2 DeepSnake-style SDT-gradient
    direction channels (5→7 total), derived from the SAME sdt via the shared
    geometry.sdt_direction_field — so this MUST stay identical to the append in
    ContourRefineEnv._state(). Off by default (5 channels, unchanged).
    """
    def build(idx, current_mask, cached_sdt=None):
        image = caches['image'][idx]
        init  = caches['init_mask'][idx].astype(np.float32)
        cur   = current_mask.astype(np.float32)
        if cached_sdt is not None:
            sdt = cached_sdt.astype(np.float32)
        else:
            sdt = signed_dt(current_mask, sdt_clip)
        # 5th channel: U-Net probability map (static per sample). Falls back to
        # the binary init mask when warm_start produced no prob_map. MUST match
        # the channel order in ContourRefineEnv._state().
        if 'prob_map' in caches:
            prob = caches['prob_map'][idx].astype(np.float32)
        else:
            prob = init
        chans = [image, cur, sdt, init, prob]
        if directional:
            field = sdt_direction_field(sdt)     # (2, H, W): dy, dx — appended at end
            chans.extend([field[0], field[1]])
        return torch.from_numpy(np.stack(chans, axis=0))
    return build


def _subset_caches(caches: dict, idx: np.ndarray) -> dict:
    """Return a new caches dict keeping only rows in `idx` (int index array).
    All cache arrays share the sample axis (axis 0), so a single fancy-index
    keeps image/gt/init/prob aligned."""
    return {k: v[idx] for k, v in caches.items()}


def _make_env(caches: dict, idx: int, env_kwargs: dict,
              env_cls=SegmentationEnv) -> SegmentationEnv:
    # Pass prob_map if warm_start produced one (enables ADD_UNCERTAIN/REMOVE_UNCERTAIN)
    prob = caches['prob_map'][idx].astype(np.float32) \
           if 'prob_map' in caches else None
    return env_cls(
        image     = caches['image'][idx],
        gt_mask   = caches['gt_mask'][idx],
        init_mask = caches['init_mask'][idx],
        prob_map  = prob,
        **env_kwargs,
    )


def _extract_loss(upd: dict) -> float:
    """Pull one representative scalar loss out of agent.update()'s return dict.
    DQN-family returns {'loss': ...}; DDPG/TD3 return {'critic_loss':...,
    'actor_loss':...} — critic_loss is computed every call (actor_loss is 0.0
    on non-policy-delay steps for TD3, which would bias a mean toward 0)."""
    if 'loss' in upd:
        return float(upd['loss'])
    if 'critic_loss' in upd:
        return float(upd['critic_loss'])
    return float('nan')


def _save_agent(agent, history, best_dice, path, step=None):
    # Persist optimizer state generically (DQN: self.opt; DDPG/TD3: actor_opt +
    # critic_opt) so a resumed run continues with warm Adam moments, plus `step`
    # so epsilon / bc_lambda schedules pick up where they left off. Older
    # checkpoints without these keys still load (the resume path treats them as
    # missing → fresh optimizer/step).
    opt_states = {name: o.state_dict()
                  for name, o in vars(agent).items()
                  if isinstance(o, torch.optim.Optimizer)}
    torch.save({
        'agent':      agent.state_dict(),
        'optimizers': opt_states,
        'best_dice':  best_dice,
        'history':    history,
        'step':       step,
    }, path)


@torch.no_grad()
def evaluate_agent(agent, samples_or_caches, env_kwargs, env_cls=SegmentationEnv) -> dict:
    """Greedy policy evaluation over a sample list. Returns aggregate metrics.

    ``env_cls`` must match the env class used at training time — otherwise the
    agent's num_actions head will mismatch the env's discrete action space.
    """
    if isinstance(samples_or_caches, dict):
        caches = samples_or_caches
        n = len(caches['image'])
        samples = None
    else:
        samples = samples_or_caches
        n = len(samples)
        caches = _build_state_caches(samples, samples[0]['image'].shape[0])

    init_d, final_d, final_h, best_d = [], [], [], []
    # Extra literature-standard segmentation metrics (Dice/IoU/PPV/SEN/BIoU/HD95
    # is the table shape used by most DRL-segmentation papers — see project
    # research notes). Computed ONCE per finished rollout (final mask vs GT),
    # never per env.step(), so this adds no hot-loop cost. init_* versions use
    # the env's own rasterised init contour (right after reset(), before any
    # step) so every extra metric has a baseline to diff against, same as Dice.
    final_iou, final_prec, final_sen, final_biou, final_msd = [], [], [], [], []
    init_h = []
    init_iou, init_prec, init_sen, init_biou, init_msd = [], [], [], [], []
    # Value-floored ("do no harm") deploy Dice — GT-FREE selection (the agent's
    # own value estimate picks which visited mask to commit, falling back to init
    # if it never values an edit above the start), then scored vs GT for REPORTING
    # only. This is the DEPLOYABLE number and what checkpoint selection optimises,
    # so the saved model is the one that deploys best. Parity is relative to the
    # agent's OWN value estimate (not a hard GT floor): a well-trained, calibrated
    # value fn ⇒ genuine do-no-harm; a noisy one can still commit a worse edit —
    # which is precisely why we keep the checkpoint with the best val-vfloor.
    # Mirrors refinement_viz's Pillar-1 selector so train-time val and final test
    # agree. Agents without a state_value method (none currently) fall back to init.
    vf_d = []
    has_value = hasattr(agent, 'state_value')
    for i in range(n):
        env = _make_env(caches, i, env_kwargs, env_cls=env_cls)
        state = env.reset()
        init_d.append(env.dice_history[0])
        init_mask0 = env.mask.copy()
        masks  = [env.mask.copy()]
        values = [agent.state_value(state)] if has_value else None
        while True:
            if agent.action_type == 'discrete':
                a = agent.select_action(state, epsilon=0.0)
            else:
                a = agent.select_action(state, explore=False)
            state, _, done, info = env.step(a)
            masks.append(env.mask.copy())
            if has_value:
                values.append(agent.state_value(state))
            if done:
                break
        final_d.append(info['dice'])
        final_h.append(info['hd95'])
        # best_dice = highest Dice seen during the episode (the achievable ceiling
        # if STOP timing were perfect). final_dice = the deployment number.
        best_d.append(info.get('best_dice', info['dice']))

        gt = caches['gt_mask'][i]
        # Value-floored selection: commit the highest-VALUED visited mask (GT-free);
        # keep init if nothing is valued above it → guaranteed ≥ baseline parity.
        if has_value:
            v = np.asarray(values, dtype=np.float64)
            bt = int(v.argmax())
            vf_idx = bt if v[bt] > v[0] else 0
            vf_d.append(dice_score(masks[vf_idx], gt))
        else:
            vf_d.append(env.dice_history[0])
        final_iou.append(iou_score(env.mask, gt))
        p, r = precision_recall(env.mask, gt)
        final_prec.append(p); final_sen.append(r)
        final_biou.append(boundary_iou(env.mask, gt))
        final_msd.append(mean_surface_distance_px(env.mask, gt))

        init_h.append(hd95_px(init_mask0, gt))
        init_iou.append(iou_score(init_mask0, gt))
        ip, isen = precision_recall(init_mask0, gt)
        init_prec.append(ip); init_sen.append(isen)
        init_biou.append(boundary_iou(init_mask0, gt))
        init_msd.append(mean_surface_distance_px(init_mask0, gt))

    final_h_arr = np.asarray(final_h, dtype=float)
    valid_h = final_h_arr[~np.isnan(final_h_arr)]
    msd_arr = np.asarray(final_msd, dtype=float)
    valid_msd = msd_arr[~np.isnan(msd_arr)]
    init_h_arr = np.asarray(init_h, dtype=float)
    valid_init_h = init_h_arr[~np.isnan(init_h_arr)]
    init_msd_arr = np.asarray(init_msd, dtype=float)
    valid_init_msd = init_msd_arr[~np.isnan(init_msd_arr)]
    return dict(
        init_dice_mean  = float(np.mean(init_d)),
        final_dice_mean = float(np.mean(final_d)),
        best_dice_mean  = float(np.mean(best_d)),    # diagnostic ceiling
        # Deployable do-no-harm number: what checkpoint selection optimises and
        # what should be reported as "DRL deploy Dice" (never below baseline).
        value_floored_dice_mean  = float(np.mean(vf_d)),
        value_floored_delta_mean = float(np.mean([v - i for v, i in zip(vf_d, init_d)])),
        final_hd95_mean = float(valid_h.mean()) if valid_h.size else float('nan'),
        delta_dice_mean = float(np.mean([f - i for f, i in zip(final_d, init_d)])),
        final_iou_mean         = float(np.mean(final_iou)),
        final_precision_mean   = float(np.mean(final_prec)),
        final_sensitivity_mean = float(np.mean(final_sen)),
        final_biou_mean        = float(np.mean(final_biou)),
        init_hd95_mean         = float(valid_init_h.mean()) if valid_init_h.size else float('nan'),
        init_iou_mean          = float(np.mean(init_iou)),
        init_precision_mean    = float(np.mean(init_prec)),
        init_sensitivity_mean  = float(np.mean(init_sen)),
        init_biou_mean         = float(np.mean(init_biou)),
        init_msd_mean          = float(valid_init_msd.mean()) if valid_init_msd.size else float('nan'),
        final_msd_mean         = float(valid_msd.mean()) if valid_msd.size else float('nan'),
    )


def run_drl_training(
    cfg: dict,
    train_samples: List[dict],
    val_samples:   List[dict],
) -> dict:
    """Train one DRL agent for one (dataset, structure) combination."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = cfg.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent_cls, action_type = AGENT_REGISTRY[cfg['agent_type'].upper()]
    image_size = cfg['image_size']
    H = image_size

    # ── Environment class selection ──────────────────────────────────────────
    # All agents use SegmentationEnv (14-action local mask refinement).
    # env_class is always 'default' — kept as a config key for forward compat.
    # Both discrete and continuous agents may select the env via env_class.
    # TD3 in particular runs on env_class: contour (angular-sector continuous
    # action). DDPG defaults to 'default' (SegmentationEnv 3-D global action).
    env_class_name = cfg.get('env_class', 'default')
    if env_class_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env_class '{env_class_name}'. Available: {list(ENV_REGISTRY)}")
    env_cls = ENV_REGISTRY[env_class_name]
    num_actions_for_cls = env_cls.NUM_DISCRETE_ACTIONS
    print(f"[drl] env_class={env_class_name} ({env_cls.__name__}) "
          f"| discrete actions={num_actions_for_cls} | action_type={action_type}")

    # ── Caches ───────────────────────────────────────────────────────────────
    train_caches = _build_state_caches(train_samples, image_size)
    val_caches   = _build_state_caches(val_samples,   image_size)

    # ── Hard sample mining ────────────────────────────────────────────────────
    # Problem: most training samples already have high init Dice (e.g. 0.93 for
    # CAMUS LV_endo). On those, every action except no-op degrades performance.
    # The Q-network is overwhelmed by "everything is bad" signal from the easy
    # majority and never properly learns to refine the rare hard samples where
    # improvement is actually achievable.
    #
    # Fix: exponential weighting that preferentially samples lower-init-Dice
    # cases. Weight ∝ exp((1-init_dice) * hard_mining_scale).
    #   scale=0  → uniform (off)
    #   scale=5  → mild: dice=0.75 gets ~2.5× more training than dice=0.93
    #   scale=10 → strong: dice=0.75 gets ~7× more training than dice=0.93
    #
    # Keeps easy samples in the mix (they teach "no-op when already good"),
    # but amplifies learning signal from samples where improvement is possible.
    _hard_mining_scale = cfg.get('hard_mining_scale', 0.0)
    # Curriculum needs per-sample init Dice too — compute it once and share
    # with hard-mining instead of duplicating the pass over train_caches.
    _curriculum_on = bool(cfg.get('curriculum_max_steps', False))
    _init_dices = None
    if _hard_mining_scale > 0 or _curriculum_on:
        _init_dices = np.array([
            dice_score(train_caches['init_mask'][i], train_caches['gt_mask'][i])
            for i in range(len(train_samples))
        ])

    # ── Topology/interior training-sample reweighting (opt-in, off by default) ──
    # At low-data scale the lite net makes structural errors (whole missing/false
    # components) that contour nudging CANNOT fix — training on them is gradient
    # noise. When `topology_filter` is on, samples whose (topology+interior)
    # error fraction exceeds `topology_max_frac` are down-weighted (mode
    # 'downweight', factor `topology_downweight`) or excluded (mode 'drop').
    # Uses GT — legitimate at TRAINING time only, exactly like hard_mining above;
    # the val/test sets are never filtered (no cherry-picking the eval). Default
    # 'downweight' (not 'drop') so the agent still sees enough broken cases to
    # learn to STOP/abstain on them at deployment.
    _topology_filter = bool(cfg.get('topology_filter', False))
    _topo_weights = None
    if _topology_filter:
        _cap = np.array([
            sum(sample_error_decomp(train_caches['init_mask'][i],
                                    train_caches['gt_mask'][i])[1:])   # topology+interior
            for i in range(len(train_samples))
        ])
        _thr  = float(cfg.get('topology_max_frac', 0.5))
        _mode = cfg.get('topology_filter_mode', 'downweight')
        _dw   = float(cfg.get('topology_downweight', 0.1))
        over  = _cap > _thr
        _topo_weights = np.ones(len(train_samples))
        _topo_weights[over] = 0.0 if _mode == 'drop' else _dw
        n_over = int(over.sum())
        print(f"[drl] Topology filter ON: mode={_mode} thresh={_thr} | "
              f"{n_over}/{len(train_samples)} samples over threshold "
              f"(topo+interior>{_thr}) -> weight {'0 (dropped)' if _mode=='drop' else _dw}")
        if _topo_weights.sum() == 0:
            raise ValueError("topology_filter dropped ALL training samples — "
                             "lower topology_max_frac or use mode='downweight'.")

    # ── Combined sampling weights (hard-mining × topology filter) ───────────────
    _weights = None
    if _hard_mining_scale > 0:
        _raw = (1.0 - _init_dices) * _hard_mining_scale
        _raw -= _raw.max()   # numerical stability before exp
        _weights = np.exp(_raw)
        hard_pct = (_init_dices < 0.90).mean() * 100
        print(f'[drl] Hard sample mining ON: scale={_hard_mining_scale:.1f} | '
              f'init_dice mean={_init_dices.mean():.4f} | '
              f'hard (<0.90): {hard_pct:.1f}% of train')
    if _topo_weights is not None:
        _weights = _topo_weights if _weights is None else (_weights * _topo_weights)

    # ── Refinable gate (GT-FREE, opt-in) ────────────────────────────────────────
    # Train ONLY on cases whose init mask is in the contour-refinable regime — a
    # single dominant CC of plausible size (the U-Net actually localised the
    # structure). Uses ONLY the init mask, never GT, so it is a deployable routing
    # gate, not test cherry-picking: at inference you refine when the gate passes
    # and keep the raw U-Net mask when it doesn't. This is the regime published
    # contour-refinement methods assume (a competent initialisation); feeding the
    # agent structurally-broken inits (missed/fragmented objects) is pure gradient
    # noise it cannot act on. Val/test are gated the SAME way below (subset report
    # + routed fallback), so the comparison stays honest.
    _refinable_gate = bool(cfg.get('refinable_gate', False))
    _rf_min_area = float(cfg.get('refinable_min_cc_frac', 0.004))
    _rf_min_dom  = float(cfg.get('refinable_min_dominance', 0.5))
    if _refinable_gate:
        _rf_train = np.array([
            init_mask_refinable(train_caches['init_mask'][i], _rf_min_area, _rf_min_dom)
            for i in range(len(train_samples))], dtype=bool)
        n_ref = int(_rf_train.sum())
        print(f"[drl] Refinable gate ON (GT-FREE): {n_ref}/{len(train_samples)} train init masks "
              f"refinable (min_cc_frac={_rf_min_area}, min_dominance={_rf_min_dom}) "
              f"-> training only on these")
        if n_ref == 0:
            raise ValueError("refinable_gate excluded ALL training samples — loosen "
                             "refinable_min_cc_frac / refinable_min_dominance.")
        _rf_w = _rf_train.astype(float)
        _weights = _rf_w if _weights is None else (_weights * _rf_w)

    # Gate the VALIDATION set the same way (GT-free) so the training curve,
    # checkpoint selection, and reported val metrics all reflect the refinable
    # regime the agent is actually being asked to handle. Non-refinable val cases
    # would be routed to the U-Net mask at deployment (identity), so including
    # them here would just add a constant band of init==final noise.
    _eval_caches = val_caches
    if _refinable_gate:
        _rf_val = np.array([
            init_mask_refinable(val_caches['init_mask'][i], _rf_min_area, _rf_min_dom)
            for i in range(len(val_caches['image']))], dtype=bool)
        n_val_ref = int(_rf_val.sum())
        print(f"[drl] Refinable gate: val {n_val_ref}/{len(_rf_val)} refinable "
              f"-> eval + checkpoint selection on this subset")
        if n_val_ref == 0:
            raise ValueError("refinable_gate excluded ALL validation samples — loosen thresholds.")
        _eval_caches = _subset_caches(val_caches, np.nonzero(_rf_val)[0])

    if _weights is not None:
        _weights = _weights / _weights.sum()
        def _sample_idx():
            return int(np.random.choice(len(train_samples), p=_weights))
    else:
        def _sample_idx():
            return np.random.randint(len(train_samples))

    # cont_action_scale is the legacy single-scale key (old per-agent configs).
    # New per-class configs supply cont_morph_scale / cont_trans_scale separately.
    _legacy_scale = cfg.get('cont_action_scale', 0.02)
    # action_type is always passed; everything else only goes through if the
    # cfg explicitly sets it.  This lets the chosen env class (e.g. BRISC)
    # apply its own defaults for max_steps / stop_n / stop_eps_dice rather
    # than being clobbered by hardcoded base-env defaults from this dict.
    env_kwargs = {'action_type': action_type}
    _env_optional_keys = (
        'max_steps', 'shift_px', 'sdt_clip', 'reward_clip',
        'cont_morph_scale', 'cont_trans_scale',
        'reward_mode', 'reward_alpha', 'reward_beta', 'hd_norm',
        'stop_eps_dice', 'stop_eps_hd', 'stop_n',
        'fail_thresh', 'fail_n',                       # small-target extras
        'reward_step_penalty', 'disable_auto_stop',    # STOP-incentive shaping
        'terminal_bonus_scale',                        # path-independent terminal reward
        'reward_potential_scale',                      # PBRS potential scale (baseline-centred Φ)
        'n_points', 'disp_px', 'spline_smooth', 'smooth_lambda', 'cont_sectors',  # contour env
        'uncertainty_gate', 'gate_lo', 'gate_hi', 'gate_margin',  # U-Net confidence gate
        'directional_state',                           # +2 SDT-gradient direction channels
    )
    for _k in _env_optional_keys:
        if _k in cfg:
            env_kwargs[_k] = cfg[_k]
    # Potential-based reward shaping (dice_pbrs / dice_hd_pbrs) needs the SAME
    # discount the agent uses, so the shaped return telescopes exactly to
    # γ^T·Φ_T − Φ_0. Always pass it; the env ignores it for non-pbrs modes.
    env_kwargs['pbrs_gamma'] = cfg.get('gamma', 0.99)
    # cont_trans_scale legacy fallback (old configs use cont_action_scale)
    if 'cont_trans_scale' not in env_kwargs and 'cont_action_scale' in cfg:
        env_kwargs['cont_trans_scale'] = cfg['cont_action_scale']

    # ── Per-sample curriculum max_steps (TRAINING ONLY) ───────────────────────
    # GT-based difficulty (init Dice) is only knowable because GT is available
    # during training — never at deployment/eval, so the EVAL/TEST env always
    # uses the single fixed `max_steps` from env_kwargs unchanged (see
    # evaluate_agent / evaluate_testset, which never see this override).
    # Rationale: hard_mining oversamples low-init-Dice cases, which rarely
    # *reach* a near-converged state within a short episode, so the buffer was
    # starved of "already near peak, should stop" terminal transitions — the
    # likely reason reward_step_penalty alone didn't induce STOP usage. Forcing
    # SHORT episodes on easy samples manufactures many more such transitions
    # without reducing hard-sample representation; hard samples get a LONGER
    # episode so they have more room to travel toward the boundary.
    _sample_max_steps = None
    if _curriculum_on:
        _base_max_steps = int(env_kwargs.get('max_steps', 20))
        _easy_dice  = float(cfg.get('curriculum_easy_dice', 0.90))
        _hard_dice  = float(cfg.get('curriculum_hard_dice', 0.80))
        _easy_steps = int(cfg.get('curriculum_easy_steps', max(2, _base_max_steps // 2)))
        _hard_steps = int(cfg.get('curriculum_hard_steps', _base_max_steps + 5))
        _sample_max_steps = np.where(
            _init_dices >= _easy_dice, _easy_steps,
            np.where(_init_dices < _hard_dice, _hard_steps, _base_max_steps)
        ).astype(int)
        n_easy = int((_init_dices >= _easy_dice).sum())
        n_hard = int((_init_dices < _hard_dice).sum())
        n_med  = len(_init_dices) - n_easy - n_hard
        print(f'[drl] Curriculum max_steps ON (train-only; eval always uses '
              f'max_steps={_base_max_steps}): '
              f'easy(d>={_easy_dice})→{_easy_steps} steps [{n_easy} samples]  |  '
              f'medium→{_base_max_steps} steps [{n_med} samples]  |  '
              f'hard(d<{_hard_dice})→{_hard_steps} steps [{n_hard} samples]')

    # Directional-state ablation: +2 SDT-gradient channels (5→7). Read once here
    # so the state builder, the env (via env_kwargs above), and in_channels below
    # all agree. Off by default → 5 channels, backward-compatible with all
    # existing checkpoints. Turning it ON invalidates 5-channel checkpoints
    # (first conv shape changes) — it's an ablation arm, not a drop-in resume.
    _directional = bool(cfg.get('directional_state', False))
    _n_state_ch = 7 if _directional else 5
    # state_builder needs sdt_clip even if not in cfg (env class default applies).
    state_builder = _make_state_builder(train_caches, cfg.get('sdt_clip', 20.0),
                                        directional=_directional)

    # ── Continuous action dim (env-dependent) ─────────────────────────────────
    #   contour env → cont_sectors angular wedges (TD3, action in [-1,1]^K)
    #   default env → 3-D global (morph, dy, dx)  (DDPG)
    is_contour = (env_cls is ContourRefineEnv)
    cont_action_dim = None
    if action_type == 'continuous':
        cont_action_dim = (int(cfg.get('cont_sectors', 16)) if is_contour
                           else CONTINUOUS_ACTION_DIM)

    # ── Agent ─────────────────────────────────────────────────────────────────
    common = dict(in_channels=_n_state_ch, gamma=cfg.get('gamma', 0.99),
                  tau=cfg.get('tau', 0.005),
                  embed_dim=cfg.get('embed_dim', 256), device=device)
    if action_type == 'discrete':
        # num_actions is env-class-dependent — agent's output head must match
        # exactly or argmax will produce invalid indices.
        agent = agent_cls(num_actions=num_actions_for_cls,
                          lr=cfg.get('lr', 1e-4),
                          spatial=cfg.get('spatial_head', False), **common)
    else:
        common.pop('lr', None)
        if is_contour:
            action_scale = [1.0] * cont_action_dim   # env multiplies by disp_px
        else:
            _legacy_scale = cfg.get('cont_action_scale', 0.02)
            morph_scale   = cfg.get('cont_morph_scale', 0.25)
            trans_scale   = cfg.get('cont_trans_scale', _legacy_scale)
            action_scale  = [morph_scale, trans_scale, trans_scale]

        if cfg['agent_type'].upper() == 'TD3':
            agent = agent_cls(
                action_dim   = cont_action_dim,
                action_scale = action_scale,
                actor_lr     = cfg.get('actor_lr',  1e-4),
                critic_lr    = cfg.get('critic_lr', 1e-3),
                policy_delay = cfg.get('policy_delay', 2),
                expl_noise   = cfg.get('expl_noise', 0.2),
                target_noise = cfg.get('target_noise', 0.2),
                noise_clip   = cfg.get('noise_clip', 0.5),
                spatial      = cfg.get('spatial_head', False),
                **common,
            )
        else:   # DDPG
            ou_sigma_raw = cfg.get('ou_sigma', None)
            agent = agent_cls(
                action_dim         = cont_action_dim,
                action_scale       = action_scale,
                actor_lr           = cfg.get('actor_lr',  1e-4),
                critic_lr          = cfg.get('critic_lr', 1e-3),
                ou_theta           = cfg.get('ou_theta', 0.15),
                ou_sigma           = ou_sigma_raw,
                actor_freeze_steps = cfg.get('actor_freeze_steps', 2000),
                **common,
            )

    # ── TD3+BC: oracle-demonstration actor warm-start (opt-in) ─────────────────
    # Fujimoto & Gu 2021. Vanilla TD3's deterministic actor starts near-identity
    # (small-uniform final-layer init) and the reward landscape around zero is
    # the only "safe" region, so the actor never escapes it. Pretraining on
    # GT-privileged oracle demonstrations (train-time only — never used at
    # deployment) gives it a competent starting policy instead; `bc_lambda`
    # regularisation during RL fine-tuning (see below) keeps it from drifting
    # back to identity. Only applies to TD3 on the continuous contour env.
    demo_buffer = None
    bc_warm_start = (cfg['agent_type'].upper() == 'TD3' and is_contour
                     and cfg.get('bc_warm_start', False))
    if bc_warm_start:
        bc_demo_episodes = cfg.get('bc_demo_episodes', 150)
        bc_demo_max_steps = cfg.get('bc_demo_max_steps', 8)
        bc_pretrain_epochs = cfg.get('bc_pretrain_epochs', 30)
        print(f'[drl] BC warm-start: collecting {bc_demo_episodes} oracle demo '
              f'episodes (max_steps={bc_demo_max_steps})...')
        demos = bc_demo.collect_continuous_oracle_demos(
            train_samples, env_kwargs, cont_action_dim,
            bc_demo_episodes, bc_demo_max_steps, seed=cfg.get('seed', 42))
        demo_buffer = bc_demo.DemoBuffer(demos, mask_shape=(H, H))
        bc_loss = agent.pretrain_actor_bc(
            demo_buffer, state_builder,
            epochs=bc_pretrain_epochs, batch_size=cfg.get('batch_size', 64))
        print(f'[drl] BC warm-start: {len(demos)} demo transitions, '
              f'pretrain loss={bc_loss:.4f}')

    bc_lambda_start = cfg.get('bc_lambda_start', 0.5)
    bc_lambda_end = cfg.get('bc_lambda_end', 0.05)
    bc_lambda_decay_steps = cfg.get('bc_lambda_decay_steps', 20000)

    # ── Buffer ────────────────────────────────────────────────────────────────
    # cache_sdt=True is the key speed optimisation: skips ~80% of SDT recomputation
    # in agent.update() by reusing the SDT computed during env.step().
    buffer = ReplayBuffer(
        capacity   = cfg.get('buffer_size', 10000),
        mask_shape = (H, H),   # BRISC/CAMUS are square (256×256); extend to (H,W) if needed
        # continuous action width: 3 (global DDPG) or cont_sectors (contour TD3).
        action_dim = cont_action_dim if action_type == 'continuous' else None,
        discrete   = (action_type == 'discrete'),
        # cache_sdt FORCED ON: reusing the SDT computed during env.step() skips
        # ~80% of SDT recomputation in agent.update() — the single biggest
        # speed optimisation in the training loop.
        cache_sdt  = True,
    )

    # ── Pre-fill buffer with random rollouts ──────────────────────────────────
    # Scoped out of curriculum deliberately: prefill is pure random exploration
    # to warm the buffer, not where the STOP/terminal-density problem lives —
    # keeping it on the fixed env_kwargs keeps this change minimal and low-risk.
    prefill_steps = cfg.get('prefill_steps', 2000)
    print(f'[drl] Pre-filling buffer with {prefill_steps} random transitions...')
    # Build per-component ranges for random continuous actions
    _legacy_scale = cfg.get('cont_action_scale', 0.02)
    _morph_scale  = cfg.get('cont_morph_scale', 0.25)
    _trans_scale  = cfg.get('cont_trans_scale', _legacy_scale)

    while len(buffer) < prefill_steps:
        idx = _sample_idx()
        env = _make_env(train_caches, idx, env_kwargs, env_cls=env_cls)
        prev_state = env.reset()           # state = stack([img, mask, SDT, init])
        while True:
            if action_type == 'discrete':
                a = np.random.randint(num_actions_for_cls)
            elif is_contour:
                # angular-sector action in [-1, 1]^K
                a = np.random.uniform(-1.0, 1.0, size=cont_action_dim).astype(np.float32)
            else:
                # 3-component global: (morph, dy_norm, dx_norm)
                a = np.array([
                    np.random.uniform(-_morph_scale,  _morph_scale),
                    np.random.uniform(-_trans_scale,  _trans_scale),
                    np.random.uniform(-_trans_scale,  _trans_scale),
                ], dtype=np.float32)
            prev_mask = env.mask.copy()
            prev_sdt  = prev_state[2]      # SDT slot of state tensor
            next_state, r, done, _ = env.step(a)
            buffer.push(idx, prev_mask, a, r, env.mask.copy(), done,
                        current_sdt=prev_sdt, next_sdt=next_state[2])
            prev_state = next_state
            if done or len(buffer) >= prefill_steps:
                break
    print(f'[drl] Buffer size: {len(buffer)}  (SDT cache: {buffer.cache_sdt})')

    # ── Main training loop ────────────────────────────────────────────────────
    train_steps         = cfg.get('train_steps', 50000)
    eval_every          = cfg.get('eval_every',   2000)
    batch_size          = cfg.get('batch_size',   64)
    eps_start           = cfg.get('epsilon_start',       1.0)
    eps_end             = cfg.get('epsilon_end',         0.05)
    eps_decay_steps     = cfg.get('epsilon_decay_steps', 40000)
    target_class        = cfg.get('target_class', 1)

    history    = []
    best_dice  = 0.0
    ckpt_dir   = Path(cfg.get('checkpoint_dir', '/kaggle/working'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _stem      = f"{cfg.get('dataset','camus').lower()}_{cfg['agent_type'].lower()}_c{target_class}"
    ckpt_path  = ckpt_dir / f"{_stem}_best.pt"     # best-val mask, for deployment
    last_path  = ckpt_dir / f"{_stem}_last.pt"     # latest full state, for resume

    # ── Resume (opt-in) ───────────────────────────────────────────────────────
    # Split a long run into chunks WITHOUT redoing earlier steps: run e.g.
    # train_steps=20000 once, inspect the curve, then re-run with resume=True and
    # a larger train_steps to continue. Restores agent weights, optimizer moments,
    # the step counter (so epsilon/bc_lambda continue), history, and best_dice.
    # The replay buffer is NOT persisted (too large) — it re-prefills cold, but
    # eval uses the greedy policy directly so the resumed curve stays continuous.
    start_step = 0
    if cfg.get('resume', False) and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        agent.load_state_dict(ck['agent'])
        for name, st in ck.get('optimizers', {}).items():
            opt = getattr(agent, name, None)
            if isinstance(opt, torch.optim.Optimizer):
                opt.load_state_dict(st)
        start_step = int(ck.get('step') or 0)
        best_dice  = float(ck.get('best_dice', 0.0))
        _h = ck.get('history')
        history = _h.to_dict('records') if hasattr(_h, 'to_dict') else (list(_h) if _h else [])
        print(f'[drl] RESUMED from {last_path} at step {start_step} '
              f'(best_dice={best_dice:.4f}) → continuing to {train_steps}')
    elif cfg.get('resume', False):
        print(f'[drl] resume=True but no {last_path.name} found — starting fresh.')

    # Milestone checkpoints — periodic snapshots distinct from ckpt_path (best)
    # and last_path (resume), kept by step number so any milestone can be
    # rolled back to, not just the most recent one.
    checkpoint_every = cfg.get('checkpoint_every')

    print(f'[drl] Training: steps {start_step}→{train_steps}  →  {ckpt_path}')
    if checkpoint_every:
        print(f'[drl] Milestone checkpoints every {checkpoint_every} steps  →  {ckpt_dir}/{_stem}_step<N>.pt')
    pbar = trange(start_step, train_steps, desc=f"{cfg['agent_type']} c{target_class}")

    # RL training diagnostics accumulated between evals (no extra rollouts —
    # just the loss/reward already produced by the existing update()/step()
    # calls), reset after each eval so each history row reports its own window.
    _loss_buf, _ep_reward_buf = [], []
    _ep_reward = 0.0

    step = start_step
    while step < train_steps:
        idx = _sample_idx()
        # Curriculum override: a per-episode max_steps copy of env_kwargs for
        # this training rollout only. env_kwargs itself is never mutated, so
        # evaluate_agent(val_caches, env_kwargs, ...) below always sees the
        # single fixed max_steps — eval/deploy behaviour is unaffected.
        if _sample_max_steps is not None:
            ep_env_kwargs = dict(env_kwargs, max_steps=int(_sample_max_steps[idx]))
        else:
            ep_env_kwargs = env_kwargs
        env = _make_env(train_caches, idx, ep_env_kwargs, env_cls=env_cls)
        state = env.reset()

        if action_type == 'continuous':
            # DDPG keeps temporally-correlated OU noise that must reset per
            # episode; TD3 uses i.i.d. Gaussian exploration and has no such state.
            if hasattr(agent, 'noise'):
                agent.noise.reset()
            epsilon = None
        else:
            epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)

        episode_steps = 0
        while True:
            if action_type == 'discrete':
                a = agent.select_action(state, epsilon)
            else:
                a = agent.select_action(state, explore=True)

            prev_mask = env.mask.copy()
            prev_sdt  = state[2]           # cached SDT from previous state
            next_state, r, done, _ = env.step(a)
            buffer.push(idx, prev_mask, a, r, env.mask.copy(), done,
                        current_sdt=prev_sdt, next_sdt=next_state[2])
            _ep_reward += float(r)

            if len(buffer) >= batch_size:
                if demo_buffer is not None and len(demo_buffer) > 0:
                    bc_batch = demo_buffer.sample(batch_size)
                    bc_lambda = max(bc_lambda_end, bc_lambda_start
                                    - (bc_lambda_start - bc_lambda_end) * step / bc_lambda_decay_steps)
                    upd = agent.update(buffer.sample(batch_size), state_builder,
                                      bc_batch=bc_batch, bc_lambda=bc_lambda)
                else:
                    upd = agent.update(buffer.sample(batch_size), state_builder)
                _loss_buf.append(_extract_loss(upd))

            state = next_state
            step += 1
            episode_steps += 1
            pbar.update(1)

            if done:
                _ep_reward_buf.append(_ep_reward)
                _ep_reward = 0.0

            # Periodic eval — also the cadence at which the resume/best
            # checkpoints refresh (eval_every controls both).
            if step % eval_every == 0:
                metrics = evaluate_agent(agent, _eval_caches, env_kwargs, env_cls=env_cls)
                metrics['step']    = step
                metrics['epsilon'] = epsilon if epsilon is not None else None
                # RL training diagnostics for this window (since the last eval),
                # not re-derived from anything — just the loss/reward already
                # produced above. NaN-safe: empty windows (e.g. eval_every <
                # one episode) report NaN rather than crashing.
                metrics['loss_mean'] = (float(np.nanmean(_loss_buf))
                                        if _loss_buf else float('nan'))
                metrics['episode_reward_mean'] = (float(np.mean(_ep_reward_buf))
                                                  if _ep_reward_buf else float('nan'))
                _loss_buf, _ep_reward_buf = [], []
                history.append(metrics)
                # Select the best checkpoint on the DEPLOYABLE value-floored Dice
                # (do-no-harm, GT-free selection) rather than raw final Dice: the
                # saved model is then the one that deploys best, never one that
                # degrades good masks. Falls back to final_dice_mean for agents
                # without a value head (backward-compatible).
                _deploy = metrics.get('value_floored_dice_mean', metrics['final_dice_mean'])
                improved = _deploy > best_dice
                if improved:
                    best_dice = _deploy
                    _save_agent(agent, history, best_dice, ckpt_path, step=step)
                # Always refresh the resume checkpoint (latest full state), even
                # when val didn't improve, so a chunked run can continue from here.
                _save_agent(agent, history, best_dice, last_path, step=step)
                pbar.write(
                    f"step {step:6d} | init {metrics['init_dice_mean']:.4f} "
                    f"→ final {metrics['final_dice_mean']:.4f} "
                    f"| deploy(vfloor) {metrics.get('value_floored_dice_mean', float('nan')):.4f} "
                    f"| best-seen {metrics.get('best_dice_mean', float('nan')):.4f} "
                    f"(Δfinal {metrics['delta_dice_mean']:+.4f}, "
                    f"Δdeploy {metrics.get('value_floored_delta_mean', float('nan')):+.4f}, "
                    f"HD95 {metrics['final_hd95_mean']:.2f}px, "
                    f"IoU {metrics['final_iou_mean']:.4f}, BIoU {metrics['final_biou_mean']:.4f})"
                    f"{' ✓' if improved else ''}"
                )

            # Milestone checkpoint — independent of eval_every / improvement,
            # a distinct snapshot every `checkpoint_every` steps.
            if checkpoint_every and step % checkpoint_every == 0:
                milestone_path = ckpt_dir / f"{_stem}_step{step}.pt"
                _save_agent(agent, history, best_dice, milestone_path, step=step)
                pbar.write(f"  [checkpoint] milestone saved → {milestone_path}")

            if done or step >= train_steps:
                break

    pbar.close()
    print(f'[drl] Done. Best val final-Dice: {best_dice:.4f}')

    return dict(
        agent      = agent,
        history    = pd.DataFrame(history),
        best_dice  = best_dice,
        checkpoint = str(ckpt_path),
    )
