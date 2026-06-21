"""
Headroom / ceiling diagnostics for the DRL refinement task.

Answers the go/no-go question BEFORE spending GPU on full RL training:
"starting from this baseline mask, what is the BEST Dice the contour action
space can reach?" If that ceiling is barely above the baseline, no RL algorithm
can help (there is no positive reward signal); if it is well above, RL has room.

Two numbers per dataset:

  repr_ceiling  : how faithfully the N-point spline contour can represent GT at
                  all (fit GT's own boundary -> spline -> rasterise -> Dice vs GT).
                  This is the hard cap of the contour paradigm, independent of RL.
                  If repr_ceiling < the strong U-Net's Dice, contour refinement
                  can never match it (the smooth spline cannot encode the detail).

  oracle_greedy : from the baseline contour, an ORACLE (which can see GT) greedily
                  picks the single best discrete contour action each step. This is
                  the best a perfect contour policy could reach -> the practical
                  ceiling for the discrete agent. headroom = oracle_greedy - init.

Usage (notebook, on warm-start samples):
    from iteris.diagnostics import headroom_report
    headroom_report(val_samples, n_points=32, cont_sectors=8, disp_px=2.0,
                    spline_smooth=2.0, max_steps=20, n_samples=60, label='CAMUS')
"""
from copy import deepcopy
from typing import List, Dict
import numpy as np

from .env_contour_refine import ContourRefineEnv
from .geometry import dice_score


def _base_env_kwargs(n_points, cont_sectors, disp_px, spline_smooth, max_steps):
    # reward_mode is irrelevant to the geometry ceiling; use the cheap one.
    return dict(action_type='discrete', reward_mode='dice_delta',
                n_points=n_points, cont_sectors=cont_sectors, disp_px=disp_px,
                spline_smooth=spline_smooth, max_steps=max_steps,
                disable_auto_stop=True)


def repr_ceiling(gt_mask: np.ndarray, n_points: int, spline_smooth: float) -> float:
    """Best Dice the N-point spline contour can represent for this GT.

    Initialise a contour env FROM the GT itself: env.reset() resamples the GT
    boundary to n_points and rasterises the spline, so dice_history[0] is exactly
    Dice(spline-of-GT, GT) — the representational ceiling.
    """
    env = ContourRefineEnv(
        image=np.zeros_like(gt_mask, dtype=np.float32), gt_mask=gt_mask,
        init_mask=gt_mask, prob_map=None,
        **_base_env_kwargs(n_points, 8, 1.0, spline_smooth, 1))
    return float(env.dice_history[0])


def oracle_greedy(sample: dict, env_kwargs: dict, max_steps: int) -> Dict[str, float]:
    """Greedy oracle rollout: each step pick the discrete action that most raises
    Dice vs GT (excluding STOP). Returns init / best-reachable / final Dice."""
    env = ContourRefineEnv(
        image=sample['image'], gt_mask=sample['gt_mask'],
        init_mask=sample['init_mask'],
        prob_map=sample.get('prob_map'), **env_kwargs)
    init_d = float(env.dice_history[0])
    best_d = init_d
    n_act = ContourRefineEnv.NUM_DISCRETE_ACTIONS
    stop = ContourRefineEnv.STOP
    for _ in range(max_steps):
        best_a, best_after = None, -1.0
        for a in range(n_act):
            if a == stop:
                continue
            trial = deepcopy(env)
            _, _, _, info = trial.step(a)
            if info['dice'] > best_after:
                best_after, best_a = info['dice'], a
        if best_a is None:
            break
        _, _, done, info = env.step(best_a)
        best_d = max(best_d, info['dice'])
        if done or best_after <= best_d - 1e-4:   # no further oracle gain
            break
    return dict(init=init_d, oracle_best=best_d, final=float(info['dice']))


def headroom_report(samples: List[dict], n_points: int = 32, cont_sectors: int = 8,
                    disp_px: float = 2.0, spline_smooth: float = 2.0,
                    max_steps: int = 20, n_samples: int = 60,
                    label: str = '') -> Dict[str, float]:
    """Aggregate ceiling diagnostic over a sample subset. Prints + returns means."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    sub = [samples[i] for i in idx]
    env_kwargs = _base_env_kwargs(n_points, cont_sectors, disp_px, spline_smooth, max_steps)

    init, repr_c, oracle = [], [], []
    for s in sub:
        init.append(dice_score(s['init_mask'], s['gt_mask']))
        repr_c.append(repr_ceiling(s['gt_mask'], n_points, spline_smooth))
        oracle.append(oracle_greedy(s, env_kwargs, max_steps)['oracle_best'])

    out = dict(
        label=label, n=len(sub),
        baseline_init_dice = float(np.mean(init)),
        contour_repr_ceiling = float(np.mean(repr_c)),
        oracle_greedy_ceiling = float(np.mean(oracle)),
        headroom = float(np.mean(oracle) - np.mean(init)),
    )
    print(f"[ceiling:{label}] n={out['n']} | baseline {out['baseline_init_dice']:.4f} "
          f"| contour-repr cap {out['contour_repr_ceiling']:.4f} "
          f"| oracle-greedy {out['oracle_greedy_ceiling']:.4f} "
          f"| HEADROOM {out['headroom']:+.4f}")
    verdict = ('GOOD: RL has room to improve' if out['headroom'] > 0.02 else
               'MARGINAL: little reachable gain' if out['headroom'] > 0.005 else
               'NO HEADROOM: contour cannot beat this baseline')
    print(f"[ceiling:{label}] verdict -> {verdict}")
    return out
