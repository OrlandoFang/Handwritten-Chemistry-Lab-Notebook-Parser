"""Tests for the image-normalization stage (§4.1, §10)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from notebook_parser.config import PreprocessingConfig
from notebook_parser.preprocessing import preprocess, to_grayscale
from notebook_parser.preprocessing.deskew import estimate_skew


def test_to_grayscale_reduces_channels():
    """A 3-channel image becomes a single-channel uint8 image."""
    color = np.zeros((10, 12, 3), dtype=np.uint8)
    gray = to_grayscale(color)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray.shape == (10, 12)


def test_preprocess_outputs_and_provenance(synthetic_page):
    """Preprocessing yields aligned gray+binary images and records its steps."""
    result = preprocess(synthetic_page, PreprocessingConfig())
    assert result.gray.ndim == 2
    assert result.binary.shape == result.gray.shape
    assert result.binary.max() > 0  # some ink detected
    assert "binarize" in result.steps
    assert result.original_size == synthetic_page.shape[:2]


def test_preprocess_on_noisy_scan(synthetic_page):
    """Salt-and-pepper noise does not prevent ink recovery after denoising."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 100, size=synthetic_page.shape, dtype=np.int16)
    mask = rng.random(synthetic_page.shape) < 0.05
    noisy = synthetic_page.astype(np.int16)
    noisy[mask] = noise[mask]
    noisy = noisy.clip(0, 255).astype(np.uint8)
    result = preprocess(noisy, PreprocessingConfig())
    assert result.binary.max() > 0


def test_estimate_skew_recovers_rotation():
    """A page rotated by a known angle is detected within tolerance."""
    page = np.full((400, 600), 255, np.uint8)
    # Draw several horizontal text-like bars.
    for i in range(6):
        page[60 + i * 50 : 64 + i * 50, 60:540] = 0
    rotated = np.asarray(Image.fromarray(page).rotate(5, resample=Image.BILINEAR, fillcolor=255))
    angle = estimate_skew(rotated, max_skew_deg=15.0)
    assert abs(abs(angle) - 5.0) < 1.5


def test_preprocess_deterministic(synthetic_page):
    """Same input + config produces identical output (§11 determinism)."""
    a = preprocess(synthetic_page, PreprocessingConfig())
    b = preprocess(synthetic_page, PreprocessingConfig())
    assert np.array_equal(a.gray, b.gray)
    assert np.array_equal(a.binary, b.binary)
