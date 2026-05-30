"""
Geometry helpers for the sequential boundary-tracing paradigm (Paradigm 1).

These functions are pure NumPy / SciPy / scikit-image and have no knowledge of
the RL machinery — they are exercised directly by ``ContourTracingEnv`` and by
the Day-2 synthetic-circle validation tests.

The 8-direction action lookup ``DIRECTIONS`` is the single source of truth for
the discrete action space: action ``a`` moves the current point by
``DIRECTIONS[a]`` (one pixel, in (dy, dx) image coordinates).
"""

from typing import List, Tuple, Sequence

import numpy as np
import scipy.ndimage as ndi
from skimage.draw import polygon as _sk_polygon


# 8-direction action lookup, (dy, dx).  Index == discrete action id.
#   0=N  1=NE  2=E  3=SE  4=S  5=SW  6=W  7=NW
DIRECTIONS = np.array([
    [-1,  0],   # 0: N
    [-1, +1],   # 1: NE
    [ 0, +1],   # 2: E
    [+1, +1],   # 3: SE
    [+1,  0],   # 4: S
    [+1, -1],   # 5: SW
    [ 0, -1],   # 6: W
    [-1, -1],   # 7: NW
], dtype=np.int32)

DIRECTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# 4-connectivity cross for boundary extraction (matches env.STRUCT).
_STRUCT = ndi.generate_binary_structure(2, 1)


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Binary mask keeping only the largest connected component.

    Mirrors ``env._largest_cc``: stray U-Net fragments far from the structure
    would otherwise produce a bogus seed point or distance field.
    """
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                     # ignore background
    return labeled == sizes.argmax()


def best_overlap_cc(init_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """CC of ``init_mask`` with the highest IoU against ``gt_mask``.

    Why: ``largest_cc`` fails on datasets where the U-Net produces a stray
    false-positive blob (e.g. skull / scalp in BRISC MRI) that is larger
    than the true-positive detection — the seed then lands far from the
    GT structure and the agent cannot recover. Picking the CC with the
    highest GT-overlap is a one-line curriculum aid that keeps training
    honest while avoiding catastrophic seeding errors.

    Falls back to ``largest_cc`` when no CC overlaps GT at all (an empty
    U-Net + empty-overlap case — the env then falls back to GT seeding).
    """
    labeled, n = ndi.label(init_mask.astype(bool))
    if n == 0:
        return init_mask.astype(bool)
    gt_bool   = gt_mask.astype(bool)
    best_iou  = 0.0
    best_cc   = None
    for k in range(1, n + 1):
        cc    = (labeled == k)
        inter = int((cc & gt_bool).sum())
        if inter == 0:
            continue
        union = int((cc | gt_bool).sum())
        iou   = inter / max(union, 1)
        if iou > best_iou:
            best_iou, best_cc = iou, cc
    return best_cc if best_cc is not None else largest_cc(init_mask)


def boundary_edge_pixels(mask: np.ndarray) -> np.ndarray:
    """(N, 2) array of (y, x) coordinates on the boundary of ``mask``.

    The boundary is the set of foreground pixels lost by a 1-px erosion
    (the inner edge band), consistent with the HD95 edge definition in env.py.
    Returns an empty (0, 2) array for an empty mask.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.int32)
    edge = m ^ ndi.binary_erosion(m, _STRUCT)
    ys, xs = np.nonzero(edge)
    return np.stack([ys, xs], axis=1).astype(np.int32)


def seed_point_from_init_mask(
    init_mask: np.ndarray,
    method:    str             = 'topmost',
    gt_mask:   np.ndarray      = None,
) -> Tuple[int, int]:
    """(y, x) seed point on the boundary of a CC of ``init_mask``.

    Connected-component selection:
      - If ``gt_mask`` is provided, use the CC with highest IoU against GT
        (``best_overlap_cc``). This is the recommended setting for training
        and eval on datasets with U-Net false positives (e.g. BRISC) — it
        prevents a stray skull/scalp blob from anchoring the trace.
      - Otherwise use the largest CC (legacy behaviour, kept for any caller
        that does not have a GT reference).

    Boundary-pixel choice:
      - 'topmost' (deterministic): the boundary pixel with the smallest row
        index, tie-broken by smallest column.  Single-line heuristic from
        the design doc — no training required.
    """
    cc = best_overlap_cc(init_mask, gt_mask) if gt_mask is not None \
         else largest_cc(init_mask)
    edge = boundary_edge_pixels(cc)
    if edge.shape[0] == 0:
        raise ValueError('seed_point_from_init_mask: init_mask has no foreground')
    if method == 'topmost':
        # lexicographic min on (y, x): smallest row, then smallest col
        order = np.lexsort((edge[:, 1], edge[:, 0]))
        y, x = edge[order[0]]
        return int(y), int(x)
    raise ValueError(f'Unknown seed method: {method!r}')


def gt_boundary_edt(gt_mask: np.ndarray) -> np.ndarray:
    """(H, W) float32 EDT to the nearest GT-boundary pixel.

    Computed ONCE per episode in ``env.reset()``.  The per-step distance reward
    is then the O(1) lookup ``edt[y, x]`` instead of an O(N_boundary)
    nearest-pixel scan (speedup option 2).  Uses the largest-CC boundary so a
    stray GT speckle cannot warp the field.
    """
    edge_px = boundary_edge_pixels(largest_cc(gt_mask))
    H, W = gt_mask.shape
    if edge_px.shape[0] == 0:
        # No boundary → uniform large distance so every step is penalised.
        return np.full((H, W), float(H + W), dtype=np.float32)
    edge_mask = np.zeros((H, W), dtype=bool)
    edge_mask[edge_px[:, 0], edge_px[:, 1]] = True
    return ndi.distance_transform_edt(~edge_mask).astype(np.float32)


def distance_to_boundary(
    point: Sequence[int], boundary_pixels: np.ndarray
) -> float:
    """Euclidean distance from ``point`` (y, x) to nearest boundary pixel.

    Reference / test path.  The training env uses ``gt_boundary_edt`` for the
    O(1) lookup; this O(N) scan backs the Day-2 synthetic-circle validation.
    """
    if boundary_pixels.shape[0] == 0:
        return float('inf')
    d = boundary_pixels.astype(np.float64) - np.asarray(point, dtype=np.float64)
    return float(np.sqrt((d * d).sum(axis=1)).min())


def is_off_image(point: Sequence[int], H: int, W: int) -> bool:
    """True if (y, x) lies outside [0, H) × [0, W)."""
    y, x = point
    return not (0 <= y < H and 0 <= x < W)


def is_closed(
    trajectory: List[Tuple[int, int]],
    seed_point: Tuple[int, int],
    closure_tolerance: float,
    min_steps: int,
) -> bool:
    """True if the trace has returned to the seed and is long enough.

    Both conditions are required: distance(current, seed) < tolerance AND the
    trajectory has at least ``min_steps`` points.  The length floor blocks the
    degenerate "step once, return home, claim closure" exploit.
    """
    if len(trajectory) < min_steps:
        return False
    cur = np.asarray(trajectory[-1], dtype=np.float64)
    seed = np.asarray(seed_point, dtype=np.float64)
    return float(np.hypot(*(cur - seed))) < closure_tolerance


def rasterise_trajectory(
    trajectory: List[Tuple[int, int]], H: int, W: int
) -> np.ndarray:
    """Fill the closed polygon described by ``trajectory`` → (H, W) uint8 mask.

    Uses ``skimage.draw.polygon`` (scanline fill).  Points outside the image
    are clipped into bounds by ``polygon`` itself.  A trajectory with < 3
    vertices cannot enclose area and yields an empty mask.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(trajectory) < 3:
        return mask
    traj = np.asarray(trajectory, dtype=np.int32)
    rr, cc = _sk_polygon(traj[:, 0], traj[:, 1], shape=(H, W))
    mask[rr, cc] = 1
    return mask
