"""Stroke-preserving denoising (§4.1).

Scan noise is typically salt-and-pepper specks much smaller than pen strokes. We
remove it with a small median filter (excellent at killing isolated specks while
keeping edges) followed by a light non-local-means pass when available. Both are
no-ops without OpenCV.
"""

from __future__ import annotations

import numpy as np

from ._cv import get_cv2


def denoise(gray: np.ndarray, strength: int = 7) -> np.ndarray:
    """Return a denoised copy of ``gray``.

    ``strength`` controls the non-local-means filter strength; the median window
    is fixed at 3 px so thin strokes survive. Falls back to the input unchanged
    when OpenCV is missing.
    """
    cv2 = get_cv2()
    if cv2 is None:
        return gray
    # 3x3 median removes single-pixel specks without eroding 1-2px strokes.
    out = cv2.medianBlur(gray, 3)
    # Edge-preserving smoothing of remaining grain; guarded because it is the
    # most expensive step and only helps on genuinely noisy scans.
    if strength > 0:
        out = cv2.fastNlMeansDenoising(out, None, float(strength), 7, 21)
    return out
