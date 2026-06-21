"""
Shared geometry / metric helpers — dataset- and paradigm-agnostic.

Extracted from the original env.py so both the contour env and the archived
global-morphology env (Paradigm A) can share them without either importing
the other. iteris.env re-exports everything here for back-compat.
"""
import numpy as np
import scipy.ndimage as ndi


# 4-connectivity cross — kept for HD95 boundary extraction
STRUCT = ndi.generate_binary_structure(2, 1)

# ── Directional structuring elements ─────────────────────────────────────────
# Cardinal (3-element): 1-px move in one direction, affects only boundary facing that way.
SE_N = np.array([[1], [1], [0]], dtype=np.uint8)
SE_S = np.array([[0], [1], [1]], dtype=np.uint8)
SE_W = np.array([[1, 1, 0]],    dtype=np.uint8)
SE_E = np.array([[0, 1, 1]],    dtype=np.uint8)

# Diagonal (2×2 corner): 1-px move along a diagonal, expands/contracts the corner.
# Crucial for irregular BRISC tumors — cardinal ops alone can't correct diagonal errors.
SE_NE = np.array([[0, 1], [1, 1]], dtype=np.uint8)  # northeast corner
SE_NW = np.array([[1, 0], [1, 1]], dtype=np.uint8)  # northwest corner
SE_SE = np.array([[1, 1], [0, 1]], dtype=np.uint8)  # southeast corner
SE_SW = np.array([[1, 1], [1, 0]], dtype=np.uint8)  # southwest corner


def dice_score(m1: np.ndarray, m2: np.ndarray, eps: float = 1e-6) -> float:
    m1 = m1.astype(bool); m2 = m2.astype(bool)
    inter = (m1 & m2).sum()
    return (2.0 * inter + eps) / (m1.sum() + m2.sum() + eps)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return a binary mask keeping only the largest connected component.

    Stray isolated U-Net / GT pixels far from the main structure inflate HD95
    catastrophically (a single pixel in the image corner → ~200 px distance to
    the real boundary).  Keeping only the largest CC removes these artefacts
    before edge / EDT computation.
    """
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0             # ignore background
    return (labeled == sizes.argmax()).astype(bool)


def hd95_px(m1: np.ndarray, m2: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in pixels.

    Applies largest-connected-component filtering before edge extraction to
    prevent stray isolated pixels (U-Net FP fragments far from the structure)
    from inflating HD95 to hundreds of pixels.
    """
    m1b = _largest_cc(m1)
    m2b = _largest_cc(m2)
    if not m1b.any() and not m2b.any():
        return 0.0
    if not m1b.any() or not m2b.any():
        return float('nan')
    edges_1 = m1b ^ ndi.binary_erosion(m1b, STRUCT)
    edges_2 = m2b ^ ndi.binary_erosion(m2b, STRUCT)
    if not edges_1.any() or not edges_2.any():
        return 0.0
    dt_2 = ndi.distance_transform_edt(~edges_2)
    dt_1 = ndi.distance_transform_edt(~edges_1)
    d_12 = dt_2[edges_1]
    d_21 = dt_1[edges_2]
    return float(np.percentile(np.concatenate([d_12, d_21]), 95))


def signed_dt(mask: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Signed distance transform, clipped and normalised to [-1, +1]."""
    pos = ndi.distance_transform_edt(mask.astype(bool))
    neg = ndi.distance_transform_edt(~mask.astype(bool))
    sdt = pos - neg
    return (np.clip(sdt, -clip, clip) / clip).astype(np.float32)


def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate by (dy, dx) px with zero-fill, no wraparound."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    y_src = slice(max(0, -dy), H - max(0, dy))
    y_dst = slice(max(0, dy),  H - max(0, -dy))
    x_src = slice(max(0, -dx), W - max(0, dx))
    x_dst = slice(max(0, dx),  W - max(0, -dx))
    out[y_dst, x_dst] = mask[y_src, x_src]
    return out

