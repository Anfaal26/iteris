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

IMPORTANT CAVEAT on oracle_greedy / oracle_greedy_ceiling: the oracle reads GROUND
TRUTH (`info['dice']` vs GT) to choose every action. No deployed RL policy has
access to GT, so this ceiling is GT-PRIVILEGED -- it is an upper bound on the
contour ACTION SPACE's expressiveness (can the discrete moves reach a
high-Dice contour at all), not a target any image-only policy can actually
reach. Treat `oracle_greedy_ceiling` / `headroom` as "is the action space rich
enough to be worth training on", never as "RL should get close to this number".
To get a realistic, non-GT-privileged estimate of achievable headroom, pass
`attention_dice` (a fully-supervised competitor's Dice on the same class/
dataset) to `headroom_report` -- it derives `realistic_headroom_estimate` from
that instead.

Usage (notebook, on warm-start samples):
    from iteris.diagnostics import headroom_report
    headroom_report(val_samples, n_points=32, cont_sectors=8, disp_px=2.0,
                    spline_smooth=2.0, max_steps=20, n_samples=60, label='CAMUS',
                    attention_dice=0.938)
"""
from copy import deepcopy
from typing import List, Dict
import numpy as np

import scipy.ndimage as ndi

from .env_contour_refine import ContourRefineEnv
from .geometry import dice_score, _largest_cc, STRUCT


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
                    label: str = '', attention_dice: float = None) -> Dict[str, float]:
    """Aggregate ceiling diagnostic over a sample subset. Prints + returns means.

    `attention_dice`, if given, is a fully-supervised competitor's (e.g. an
    attention U-Net) test Dice on this same class/dataset. When provided, it is
    used to derive `realistic_headroom_estimate` -- a non-GT-privileged estimate
    of achievable headroom, since `oracle_greedy_ceiling` itself is GT-privileged
    (see module docstring) and should not be read as an achievable RL target.
    """
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
        oracle_is_gt_privileged = True,
    )
    print(f"[ceiling:{label}] n={out['n']} | baseline {out['baseline_init_dice']:.4f} "
          f"| contour-repr cap {out['contour_repr_ceiling']:.4f} "
          f"| oracle-greedy (GT-PRIVILEGED, NOT achievable at deployment) {out['oracle_greedy_ceiling']:.4f} "
          f"| HEADROOM {out['headroom']:+.4f}")

    if attention_dice is not None:
        realistic = float(attention_dice) - out['baseline_init_dice']
        out['attention_dice'] = float(attention_dice)
        out['realistic_headroom_estimate'] = realistic
        print(f"[ceiling:{label}] realistic (non-GT) headroom estimate = "
              f"attention({attention_dice:.4f}) - baseline({out['baseline_init_dice']:.4f}) "
              f"= {realistic:+.4f}")
        verdict = ('GOOD: RL has room to improve' if realistic > 0.02 else
                   'MARGINAL: little reachable gain' if realistic > 0.005 else
                   'NO HEADROOM: contour cannot beat this baseline')
        print(f"[ceiling:{label}] verdict (realistic, non-GT-privileged) -> {verdict}")
    else:
        verdict = ('GOOD: RL has room to improve' if out['headroom'] > 0.02 else
                   'MARGINAL: little reachable gain' if out['headroom'] > 0.005 else
                   'NO HEADROOM: contour cannot beat this baseline')
        print(f"[ceiling:{label}] verdict [GT-PRIVILEGED ESTIMATE, may not reflect "
              f"deployment-achievable improvement] -> {verdict}")
    return out


# ─── Pillar 4: offline pipeline diagnostics (no GPU, no training) ─────────────
# Three cheap checks to run on warm-start samples BEFORE spending a GPU round, so
# you don't full-train a class whose ceiling is structurally capped or whose
# inputs can't even support the gate. None of these use the agent; all are static
# analyses of (init_mask, gt_mask, prob_map).

def prob_map_informativeness(samples, gate_lo=0.35, gate_hi=0.65,
                             n_samples=80, label=''):
    """Is the U-Net prob_map usable, or is the lite net overconfident?

    The uncertainty gate and the 5th state channel both rely on the prob_map
    carrying a graded confidence signal. If the lite net is overconfident
    (probabilities pinned near 0/1), the prob_map is effectively binary, the gate
    is inert, and the agent is blind to where it is wrong -- a genuine
    preprocessing-stage failure, fixable only by retraining the lite net with
    label smoothing / temperature.

    Reports, averaged over a sample subset:
      uncertain_band_frac : mean fraction of pixels with gate_lo <= prob <= gate_hi
                            (the band the gate actually lets the agent edit)
      mean_entropy        : mean per-pixel binary entropy of prob (0=binary, 1=max)
      has_real_probmap    : whether prob_map differs from the binary init mask at all
    """
    rng = np.random.RandomState(0)
    idx = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    band, ent, real = [], [], 0
    for i in idx:
        s = samples[int(i)]
        pm = s.get('prob_map')
        if pm is None:
            continue
        pm = np.asarray(pm, dtype=np.float32)
        # is it a genuine soft map, or just the binary init mask cast to float?
        if not np.array_equal(np.unique(pm), np.array([0.0, 1.0], dtype=np.float32)):
            real += 1
        band.append(float(((pm >= gate_lo) & (pm <= gate_hi)).mean()))
        p = np.clip(pm, 1e-6, 1 - 1e-6)
        ent.append(float((-p * np.log2(p) - (1 - p) * np.log2(1 - p)).mean()))
    n = max(len(band), 1)
    out = dict(label=label, n=len(band),
               uncertain_band_frac=float(np.mean(band)) if band else 0.0,
               mean_entropy=float(np.mean(ent)) if ent else 0.0,
               real_probmap_frac=real / n)
    verdict = ('USABLE: prob_map carries graded confidence' if out['uncertain_band_frac'] > 0.01
               else 'INERT: prob_map ~binary -> gate does nothing, agent is blind to error '
                    'location (retrain lite net with label smoothing/temperature)')
    print(f"[probmap:{label}] band_frac={out['uncertain_band_frac']:.4f} "
          f"entropy={out['mean_entropy']:.4f} real_probmap={out['real_probmap_frac']:.2f} "
          f"-> {verdict}")
    return out


def sample_error_decomp(init_mask, gt_mask, band_px=4):
    """Per-sample init-vs-GT error decomposition → (boundary, topology, interior)
    fractions that sum to 1.0 (or (1,0,0) when there is no error).

    Single source of truth for the boundary/topology/interior split — used both
    by `error_type_audit` (aggregate diagnostic) and by drl_training's optional
    topology-based training-sample reweighting. Definitions:
      boundary : error within `band_px` of the init-mask boundary — the ONLY
                 error a contour-nudging agent can address.
      topology : whole connected components in GT with no counterpart in init
                 (missed objects), or in init with none in GT (false blobs) —
                 UNreachable by boundary nudging.
      interior : the remainder (interior holes / far-from-boundary error).
    """
    init = np.asarray(init_mask).astype(bool)
    gt   = np.asarray(gt_mask).astype(bool)
    err  = init ^ gt
    tot  = int(err.sum())
    if tot == 0:
        return 1.0, 0.0, 0.0
    bnd  = init ^ ndi.binary_erosion(init, STRUCT)
    band = ndi.binary_dilation(bnd, STRUCT, iterations=int(band_px))
    topo = np.zeros_like(err)
    for src, other in ((gt, init), (init, gt)):
        lab, ncc = ndi.label(src)
        for c in range(1, ncc + 1):
            comp = lab == c
            if not (comp & other).any():     # this whole object has no counterpart
                topo |= comp
    topo_err  = int((err & topo).sum())
    band_only = int((err & band & ~topo).sum())   # don't double-count topo∩band
    inter_err = max(tot - band_only - topo_err, 0)
    return band_only / tot, topo_err / tot, inter_err / tot


def init_mask_refinable(init_mask, min_cc_area_frac=0.004, min_dominance=0.5):
    """GT-FREE deployable gate: is this U-Net init mask in the regime a
    contour-deformation agent can actually improve?

    True iff the init mask has a single DOMINANT connected component of plausible
    size — i.e. the U-Net genuinely localised the structure, so local boundary
    nudging has something coherent to refine. Uses ONLY the init mask (never GT),
    so it is a DEPLOYABLE routing gate: at inference you refine when this is True
    and fall back to the raw U-Net mask when False.

    This is the honest way to scope refinement to fixable cases — exactly the
    regime published contour-refinement methods (DeepSnake, MARL-MambaContour)
    operate in, since their initialisation always comes from a competent detector.
    It is NOT cherry-picking the test set: because the gate never sees GT, the
    same decision is available at deployment, so excluded cases are ROUTED
    (kept as the U-Net mask), not silently deleted.

    Params:
      min_cc_area_frac : largest CC must cover at least this fraction of the image
                         (rejects near-total misses / tiny specks).
      min_dominance    : largest CC must be at least this fraction of the total
                         init foreground (rejects fragmented multi-blob masks).
    """
    m = np.asarray(init_mask).astype(bool)
    tot = int(m.sum())
    if tot == 0:
        return False                                  # U-Net found nothing
    lab, ncc = ndi.label(m)
    if ncc == 0:
        return False
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    if largest < min_cc_area_frac * m.size:
        return False                                  # only a tiny speck localised
    if largest / tot < min_dominance:
        return False                                  # too fragmented, no dominant CC
    return True


def error_type_audit(samples, band_px=4, n_samples=80, label=''):
    """What fraction of the lite-mask error is contour-fixable vs structural?

    Aggregate wrapper over `sample_error_decomp` (see it for the per-category
    definitions). A high topology_frac/interior_frac means the action space caps
    achievable Dice no matter how good the RL is — the case for Pillar 5
    (richer actions).
    """
    rng = np.random.RandomState(0)
    idx = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    b_frac, t_frac, i_frac = [], [], []
    for i in idx:
        s = samples[int(i)]
        b, t, it = sample_error_decomp(s['init_mask'], s['gt_mask'], band_px=band_px)
        b_frac.append(b); t_frac.append(t); i_frac.append(it)
    out = dict(label=label, n=len(b_frac),
               boundary_frac=float(np.mean(b_frac)),
               topology_frac=float(np.mean(t_frac)),
               interior_frac=float(np.mean(i_frac)))
    cap = out['topology_frac'] + out['interior_frac']
    verdict = ('GOOD: most error is boundary-shaped (contour-fixable)' if out['boundary_frac'] > 0.7
               else f'CAPPED: {cap:.0%} of error is topology/interior -> contour nudging '
                    f'structurally cannot fix it (needs richer actions)')
    print(f"[errtype:{label}] boundary={out['boundary_frac']:.2f} "
          f"topology={out['topology_frac']:.2f} interior={out['interior_frac']:.2f} -> {verdict}")
    return out


def pillar4_report(samples, gate_lo=0.35, gate_hi=0.65, band_px=4,
                   n_samples=80, label=''):
    """Run all Pillar-4 offline checks and print a combined go/no-go.

    Cheap (CPU, no agent, no training). Run on val_samples before committing a
    GPU training round for a class. Returns the two sub-reports as a dict."""
    print(f"\n=== Pillar 4 diagnostics: {label} (n<={n_samples}) ===")
    pm = prob_map_informativeness(samples, gate_lo, gate_hi, n_samples, label)
    et = error_type_audit(samples, band_px, n_samples, label)
    go = pm['uncertain_band_frac'] > 0.01 and et['boundary_frac'] > 0.5
    print(f"[pillar4:{label}] OVERALL -> "
          f"{'worth a training round' if go else 'low expected payoff — reconsider before spending GPU'}")
    return dict(prob_map=pm, error_type=et, go=go)
