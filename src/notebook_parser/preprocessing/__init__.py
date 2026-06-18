"""Image normalization stage (§4.1).

Exposes :func:`preprocess`, which runs resize -> deskew -> denoise ->
illumination/contrast normalization -> auxiliary binarization, returning both a
grayscale copy (for recognition models) and a binary mask (for layout
heuristics) plus provenance about what was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PreprocessingConfig
from ._cv import get_cv2
from .deskew import deskew
from .denoise import denoise
from .normalize import binarize, enhance_contrast, normalize_illumination

__all__ = ["PreprocessResult", "preprocess", "to_grayscale"]


@dataclass
class PreprocessResult:
    """Outputs of preprocessing plus provenance.

    ``scale`` is the factor applied to the original image (``<1`` means it was
    downscaled); multiply processed-space coordinates by ``1/scale`` to map back
    to the original page if needed.
    """

    gray: np.ndarray
    binary: np.ndarray
    skew_angle: float = 0.0
    scale: float = 1.0
    original_size: tuple[int, int] = (0, 0)  # (height, width)
    steps: list[str] = field(default_factory=list)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an HxWx3/4 image to single-channel uint8 grayscale.

    Already-grayscale inputs are returned (as uint8). Uses OpenCV when present,
    otherwise a luminance-weighted NumPy reduction so preprocessing still works
    headless.
    """
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    cv2 = get_cv2()
    if cv2 is not None:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    rgb = image[:, :, :3].astype(np.float32)
    gray = rgb @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights
    return np.clip(gray, 0, 255).astype(np.uint8)


def _resize_long_side(gray: np.ndarray, max_long_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``<= max_long_side``; return (img, scale)."""
    if max_long_side <= 0:
        return gray, 1.0
    long_side = max(gray.shape[:2])
    if long_side <= max_long_side:
        return gray, 1.0
    scale = max_long_side / float(long_side)
    cv2 = get_cv2()
    if cv2 is None:
        return gray, 1.0  # cannot resize without cv2; keep full res
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess(
    image: np.ndarray, config: PreprocessingConfig | None = None
) -> PreprocessResult:
    """Run the full preprocessing chain on a page image.

    Steps are individually gated by ``config`` and degrade gracefully when the
    optional vision libraries are absent. The chain is deterministic: the same
    input and config always yield the same output.
    """
    config = config or PreprocessingConfig()
    original_size = (int(image.shape[0]), int(image.shape[1]))
    steps: list[str] = []

    gray = to_grayscale(image)
    steps.append("grayscale")

    if not config.enabled:
        return PreprocessResult(
            gray=gray,
            binary=binarize(gray),
            scale=1.0,
            original_size=original_size,
            steps=steps + ["binarize"],
        )

    gray, scale = _resize_long_side(gray, config.max_long_side)
    if scale != 1.0:
        steps.append(f"resize(scale={scale:.3f})")

    angle = 0.0
    if config.deskew:
        gray, angle = deskew(gray, config.max_skew_deg)
        steps.append(f"deskew(angle={angle})")

    if config.denoise:
        gray = denoise(gray)
        steps.append("denoise")

    if config.normalize_illumination:
        gray = normalize_illumination(gray)
        gray = enhance_contrast(gray)
        steps.append("normalize_illumination+clahe")

    binary = binarize(gray)
    steps.append("binarize")

    return PreprocessResult(
        gray=gray,
        binary=binary,
        skew_angle=angle,
        scale=scale,
        original_size=original_size,
        steps=steps,
    )
