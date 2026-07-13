"""Re-evaluate an already-trained DRL checkpoint on the test set.

WHY THIS EXISTS
---------------
``run_drl_training`` trains the agent IN PLACE and (historically) returned it at
its FINAL-step weights. The notebooks then fed that returned agent straight into
``evaluate_testset`` / ``build_replays``. For an agent that PEAKS EARLY then
DRIFTS — most notably TD3, whose actor drifts once the BC regulariser
(``bc_lambda``) decays — the final-step weights can sit well below the best-val
checkpoint that is actually saved and deployed (``*_best.pt``). So the reported
test number understated the deployable result.

``run_drl_training`` now reloads the best-val checkpoint before returning, so new
runs are already consistent. This module re-scores checkpoints that were trained
BEFORE that fix (or any checkpoint you want to evaluate in isolation): it rebuilds
the network with ``drl_training.build_agent`` — the SAME constructor training
uses, so architecture can never drift — loads the checkpoint's weights, and runs
the identical ``evaluate_testset``.

USAGE (Kaggle notebook cell) — you already have ``cfg`` + warm-started samples
--------------------------------------------------------------------------------
    from iteris.config import (load_drl_class_config, resolve_agent_config,
                               load_config, apply_refinement_config)
    from iteris.evaluation import reeval_checkpoint

    cfg = resolve_agent_config(load_drl_class_config('.../camus_drl_c1.yaml'), 'TD3')
    apply_refinement_config(cfg, baseline_cfg_name='CAMUS/camus.yaml', uncertainty_gate=False)
    # ... warm-start cell already ran -> train_samples, val_samples, test_samples ...

    res = reeval_checkpoint(cfg, '/kaggle/input/<your-outputs>/camus_td3_c1_best.pt',
                            test_samples=test_samples)
    print(res['metrics']['value_floored_dice_mean'])

USAGE — let it run warm-start for you (only baseline_cfg + baseline_checkpoint)
------------------------------------------------------------------------------
    res = reeval_checkpoint(cfg, '.../camus_td3_c1_best.pt',
                            baseline_cfg=baseline_cfg,
                            baseline_checkpoint=cfg['baseline_checkpoint'])

Works for any agent (named for the TD3 use case that motivated it).
"""

from pathlib import Path
import json

import torch

from ..drl_training import build_agent
from ..refinement_viz import (refinement_env_kwargs, refinement_env_cls,
                              evaluate_testset, build_replays,
                              plot_comparison, plot_behaviour)


def reeval_checkpoint(
    cfg: dict,
    drl_checkpoint: str,
    *,
    test_samples=None,
    baseline_cfg: dict = None,
    baseline_checkpoint: str = None,
    device=None,
    n_viz: int = 8,
    make_plots: bool = True,
    out_dir: str = '/kaggle/working',
    verbose: bool = True,
) -> dict:
    """Re-score ``drl_checkpoint`` on the test set with the deployable weights.

    Args:
        cfg: the SAME resolved cfg used to train the checkpoint (after
            ``resolve_agent_config`` + ``apply_refinement_config``). Its
            architecture keys (agent_type, cont_sectors, spatial_head,
            directional_state, embed_dim, env_class) must match training or the
            state_dict will not load.
        drl_checkpoint: path to a ``*_best.pt`` (or any milestone ``*_stepN.pt``)
            saved by ``run_drl_training`` — a dict with an ``'agent'`` state_dict.
        test_samples: precomputed test samples (from ``precompute_init_masks``).
            If None, they are computed here from ``baseline_cfg`` +
            ``baseline_checkpoint`` (both then required).
        baseline_cfg, baseline_checkpoint: used only when ``test_samples`` is None.
        device: torch device; defaults to cuda if available.
        n_viz, make_plots, out_dir: replay/plot controls (mirrors the notebook).
        verbose: print the before-vs-after style summary line.

    Returns:
        dict with ``metrics`` (evaluate_testset output), ``agent`` (weights
        loaded), ``checkpoint_step`` and ``checkpoint_best_dice`` (from the ckpt),
        and ``json_path`` (saved corrected metrics), when applicable.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_cls = refinement_env_cls(cfg)

    # ── Test samples (given, or warm-started here) ────────────────────────────
    if test_samples is None:
        if baseline_cfg is None or baseline_checkpoint is None:
            raise ValueError(
                'reeval_checkpoint: pass test_samples, OR both baseline_cfg and '
                'baseline_checkpoint so they can be warm-started here.')
        from ..warm_start import precompute_init_masks
        if verbose:
            print('[reeval] warm-starting test samples from baseline...')
        _, _, test_samples = precompute_init_masks(
            baseline_cfg=baseline_cfg,
            baseline_checkpoint=baseline_checkpoint,
            target_class=cfg['target_class'],
            min_area_fraction=cfg.get('min_area_fraction', 0.01),
        )

    # ── Rebuild the SAME network training used, then load the checkpoint ──────
    agent = build_agent(cfg, env_cls, device)
    ckpt_path = Path(drl_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ck = torch.load(str(ckpt_path), map_location=device)
    state = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck
    agent.load_state_dict(state)   # raises loudly if cfg architecture mismatches the ckpt
    ck_step = ck.get('step') if isinstance(ck, dict) else None
    ck_best = ck.get('best_dice') if isinstance(ck, dict) else None
    if verbose:
        print(f"[reeval] loaded {ckpt_path.name} "
              f"(saved step={ck_step}, saved val deploy-Dice={ck_best})")

    # ── Evaluate exactly as the notebook does ─────────────────────────────────
    env_kwargs = refinement_env_kwargs(cfg)
    metrics = evaluate_testset(
        agent, test_samples, env_kwargs, env_cls=env_cls,
        refinable_gate=cfg.get('refinable_gate', False),
        refinable_min_cc_frac=cfg.get('refinable_min_cc_frac', 0.004),
        refinable_min_dominance=cfg.get('refinable_min_dominance', 0.5),
    )

    if verbose:
        _i = metrics['init_dice_mean']
        _f = metrics['final_dice_mean']
        _d = metrics.get('value_floored_dice_mean', float('nan'))
        _b = metrics.get('best_dice_mean', float('nan'))
        print(f"[reeval] test: init {_i:.4f} -> final {_f:.4f} "
              f"(Δ {metrics.get('delta_dice_mean', float('nan')):+.4f}) | "
              f"deploy(vfloor) {_d:.4f} "
              f"(Δ {metrics.get('value_floored_delta_mean', float('nan')):+.4f}) | "
              f"best-seen {_b:.4f}")

    out = dict(metrics=metrics, agent=agent,
               checkpoint_step=ck_step, checkpoint_best_dice=ck_best)

    # ── Save corrected metrics + optional plots ───────────────────────────────
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.get('dataset', 'ds').lower()}_{cfg['agent_type'].lower()}_c{cfg.get('target_class', 1)}"
    json_path = out_root / f"{stem}_reeval_test_results.json"
    summary = {**metrics,
               'agent_type': cfg['agent_type'], 'target_class': cfg.get('target_class'),
               'class_name': cfg.get('class_name'), 'dataset': cfg.get('dataset'),
               'reeval': True, 'source_checkpoint': str(ckpt_path),
               'checkpoint_step': ck_step, 'checkpoint_saved_val_deploy': ck_best}
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    out['json_path'] = str(json_path)
    if verbose:
        print(f"[reeval] saved -> {json_path}")

    if make_plots:
        class_name = cfg.get('class_name', '')
        replays = build_replays(agent, test_samples, env_kwargs, n_viz=n_viz,
                                seed=0, env_cls=env_cls)
        plot_comparison(replays, baseline_cfg or {}, cfg,
                        class_idx=cfg.get('target_class', 1), class_name=class_name,
                        out_path=str(out_root / f"{stem}_reeval_comparison.png"))
        plot_behaviour(replays, cfg, class_name=class_name,
                       out_path=str(out_root / f"{stem}_reeval_behaviour.png"))
        out['replays'] = replays

    return out
