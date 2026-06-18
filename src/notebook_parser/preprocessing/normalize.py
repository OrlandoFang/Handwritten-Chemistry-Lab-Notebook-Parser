"""Illumination/contrast normalization and auxiliary binarization (§4.1).

Two products come out of this module:

* a cleaned **grayscale** image with flattened illumination and boosted local
  contrast, suitable for feeding recognition models;
* an auxiliary **binary** mask (Sauvola if scikit-image is present, otherwise
  adaptive thresholding) used by the heuristic layout stage.
"""

from __future__ import annotations

import numpy as np

from ._cv import get_cv2, get_skimage_filters


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven page lighting/shading.

    Estimates the bright background with a large morphological closing, then
    divides it out so faint strokes on shaded areas become comparable to strokes
    on well-lit areas. Returns the input unchanged without OpenCV.
    """
    cv2 = get_cv2()
    if cv2 is None:
        return gray
    # Kernel scales with image size so the background estimate ignores text but
    # tracks slow shading gradients.
    k = max(15, (max(gray.shape[:2]) // 30) | 1)  # odd, >= 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=k / 3.0)
    # Divide and rescale to 0-255; add 1 to avoid divide-by-zero.
    norm = (gray.astype(np.float32) + 1.0) / (background.astype(np.float32) + 1.0)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Boost local contrast with CLAHE to recover faint pencil strokes."""
    cv2 = get_cv2()
    if cv2 is None:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Produce a foreground (ink=255) binary mask.

    Prefers Sauvola local thresholding (robust to residual shading); falls back
    to OpenCV adaptive Gaussian thresholding, and finally to a global Otsu-like
    split implemented in NumPy when neither library is present.
    """
    filters = get_skimage_filters()
    if filters is not None:
        window = max(15, (min(gray.shape[:2]) // 20) | 1)
        thresh = filters.threshold_sauvola(gray, window_size=window, k=0.2)
        return ((gray < thresh).astype(np.uint8)) * 255

    cv2 = get_cv2()
    if cv2 is not None:
        block = max(15, (min(gray.shape[:2]) // 20) | 1)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            10,
        )

    # Pure-NumPy fallback: threshold at the midpoint between the two dominant
    # intensity modes approximated by the image mean.
    threshold = float(gray.mean())
    return ((gray < threshold).astype(np.uint8)) * 255
