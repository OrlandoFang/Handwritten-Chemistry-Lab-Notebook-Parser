"""Image conditioning before sending a page to the model (§4.2).

Keeps local processing light: load, optional projection-profile deskew, downscale
to a cost/legibility budget, and encode to a base64 data URL. Deskew is
implemented with NumPy + Pillow so the core path needs no OpenCV.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ImagingConfig


@dataclass
class ConditionedImage:
    """A page ready for the model plus conditioning provenance."""

    data_url: str
    width: int
    height: int
    skew_angle: float = 0.0
    steps: list[str] = field(default_factory=list)


def load_image(path: str | os.PathLike) -> Image.Image:
    """Load an image file as an RGB Pillow image.

    Raises ``FileNotFoundError`` if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        return im.convert("RGB")


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Coerce an array or Pillow image into an RGB Pillow image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")


def _projection_variance(gray: np.ndarray) -> float:
    """Variance of row-sum ink density; peaks when text lines are horizontal."""
    return float(np.var(gray.sum(axis=1, dtype=np.float64)))


def estimate_skew(image: Image.Image, max_skew_deg: float = 15.0) -> float:
    """Estimate page skew (degrees) by maximizing horizontal projection variance.

    Searches coarse-to-fine on a downscaled, inverted binary image. A positive
    return means Pillow should rotate by that angle to straighten the page. Pure
    NumPy + Pillow, so no OpenCV dependency.
    """
    if max_skew_deg <= 0:
        return 0.0
    gray = image.convert("L")
    # Downscale for a cheap search; the optimum is scale-invariant.
    long_side = max(gray.size)
    if long_side > 800:
        scale = 800.0 / long_side
        gray = gray.resize((max(1, int(gray.width * scale)), max(1, int(gray.height * scale))))
    arr = np.asarray(gray, dtype=np.float64)
    # Inverted binary (ink = 1) via mean threshold.
    binary = (arr < arr.mean()).astype(np.uint8)
    base = Image.fromarray(binary * 255)

    def score(angle: float) -> float:
        rotated = base.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return _projection_variance(np.asarray(rotated, dtype=np.float64) / 255.0)

    coarse = np.arange(-max_skew_deg, max_skew_deg + 0.5, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 0.01, 0.2)
    best = max(fine, key=score)
    return float(round(best, 2))


def _resize(image: Image.Image, max_long_side: int) -> tuple[Image.Image, bool]:
    """Downscale so the longest side is <= ``max_long_side``; report if changed."""
    long_side = max(image.size)
    if long_side <= max_long_side:
        return image, False
    scale = max_long_side / float(long_side)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS), True


def _encode(image: Image.Image, fmt: str, quality: int) -> str:
    """Encode a Pillow image to a base64 data URL."""
    buffer = io.BytesIO()
    fmt = fmt.lower()
    if fmt == "png":
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        image.save(buffer, format="JPEG", quality=quality)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def condition_image(
    image: Image.Image | np.ndarray | str | os.PathLike,
    config: ImagingConfig | None = None,
) -> ConditionedImage:
    """Run the conditioning chain and return a model-ready data URL + metadata.

    Steps (each optional/gated by config): load -> deskew -> resize -> encode.
    Deterministic for a fixed input and config.
    """
    config = config or ImagingConfig()
    steps: list[str] = []

    if isinstance(image, (str, os.PathLike)):
        pil = load_image(image)
        steps.append("load")
    else:
        pil = _to_pil(image)

    angle = 0.0
    if config.deskew:
        angle = estimate_skew(pil, config.max_skew_deg)
        if abs(angle) >= 0.1:
            pil = pil.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
            steps.append(f"deskew(angle={angle})")

    pil, resized = _resize(pil, config.max_long_side)
    if resized:
        steps.append(f"resize({pil.width}x{pil.height})")

    data_url = _encode(pil, config.encode_format, config.jpeg_quality)
    steps.append(f"encode({config.encode_format})")
    return ConditionedImage(
        data_url=data_url,
        width=pil.width,
        height=pil.height,
        skew_angle=angle,
        steps=steps,
    )
