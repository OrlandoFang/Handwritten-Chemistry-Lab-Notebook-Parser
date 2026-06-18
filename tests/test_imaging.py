"""Tests for image conditioning (§4.2)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from notebook_parser.config import ImagingConfig
from notebook_parser.imaging import condition_image, estimate_skew


def test_condition_image_returns_data_url(synthetic_page):
    """Conditioning yields a base64 image data URL and pixel dimensions."""
    result = condition_image(synthetic_page, ImagingConfig(deskew=False))
    assert result.data_url.startswith("data:image/")
    assert ";base64," in result.data_url
    assert result.width > 0 and result.height > 0
    assert any(s.startswith("encode") for s in result.steps)


def test_condition_image_resizes_large_pages():
    """A page larger than the cap is downscaled to the configured long side."""
    big = Image.new("RGB", (4000, 2000), (255, 255, 255))
    result = condition_image(big, ImagingConfig(deskew=False, max_long_side=1600))
    assert max(result.width, result.height) == 1600


def test_condition_image_accepts_numpy():
    """A NumPy array input is accepted and conditioned."""
    arr = np.full((100, 120, 3), 255, np.uint8)
    result = condition_image(arr, ImagingConfig(deskew=False))
    assert result.width == 120 and result.height == 100


def test_estimate_skew_recovers_rotation():
    """A page rotated by a known angle is detected within tolerance."""
    page = np.full((400, 600), 255, np.uint8)
    for i in range(6):
        page[60 + i * 50 : 64 + i * 50, 60:540] = 0
    rotated = Image.fromarray(page).rotate(5, resample=Image.BILINEAR, fillcolor=255).convert("RGB")
    angle = estimate_skew(rotated, max_skew_deg=15.0)
    assert abs(abs(angle) - 5.0) < 1.5


def test_condition_image_deterministic(synthetic_page):
    """Same input + config yields an identical data URL (§10)."""
    a = condition_image(synthetic_page, ImagingConfig())
    b = condition_image(synthetic_page, ImagingConfig())
    assert a.data_url == b.data_url
