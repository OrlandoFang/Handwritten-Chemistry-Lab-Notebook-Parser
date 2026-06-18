"""Skew estimation and correction (§4.1).

Uses a projection-profile objective: a page is "straight" when rotating it makes
the horizontal projection (row-sum) of ink as peaky as possible, i.e. text lines
collapse into high-variance bands. We search angles on a downsampled binary image
for speed, then rotate the full-resolution grayscale once at the chosen angle.
"""

from __future__ import annotations

import numpy as np

from ._cv import get_cv2


def _projection_score(binary: np.ndarray, angle: float) -> float:
    """Score how well ``binary`` aligns to horizontal text lines at ``angle``.

    Rotates the (already small) binary image by ``angle`` and returns the
    variance of its row sums. Higher variance => text rows are well separated =>
    better alignment. Returns 0.0 if rotation is unavailable.
    """
    cv2 = get_cv2()
    if cv2 is None:
        return 0.0
    h, w = binary.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    row_sums = rotated.sum(axis=1, dtype=np.float64)
    return float(np.var(row_sums))


def estimate_skew(gray: np.ndarray, max_skew_deg: float = 15.0) -> float:
    """Estimate the page skew angle in degrees within ``±max_skew_deg``.

    A positive return value means the page is rotated counter-clockwise and
    should be rotated clockwise by that amount to correct it. The search is
    coarse-to-fine to stay cheap on large scans.
    """
    cv2 = get_cv2()
    if cv2 is None or max_skew_deg <= 0:
        return 0.0

    # Downsample for the angle search; the optimum is scale-invariant and this
    # keeps the (many) rotations cheap on multi-megapixel scans.
    long_side = max(gray.shape[:2])
    scale = min(1.0, 800.0 / long_side) if long_side > 0 else 1.0
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    # Inverted binary so ink = 1 and the projection measures ink density.
    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(np.uint8)

    # Coarse search at 1° resolution, then refine ±1° at 0.2° resolution.
    coarse = np.arange(-max_skew_deg, max_skew_deg + 0.5, 1.0)
    best = max(coarse, key=lambda a: _projection_score(binary, a))
    fine = np.arange(best - 1.0, best + 1.0 + 0.01, 0.2)
    best = max(fine, key=lambda a: _projection_score(binary, a))
    return float(round(best, 2))


def deskew(gray: np.ndarray, max_skew_deg: float = 15.0) -> tuple[np.ndarray, float]:
    """Return a deskewed copy of ``gray`` and the angle that was corrected.

    If OpenCV is unavailable or no meaningful skew is found, the input is
    returned unchanged with angle 0.0.
    """
    cv2 = get_cv2()
    angle = estimate_skew(gray, max_skew_deg)
    if cv2 is None or abs(angle) < 0.1:
        return gray, 0.0
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Border filled with the page's bright background (estimated as the median)
    # so rotation does not introduce dark wedges that confuse later stages.
    border = float(np.median(gray))
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return rotated, angle
