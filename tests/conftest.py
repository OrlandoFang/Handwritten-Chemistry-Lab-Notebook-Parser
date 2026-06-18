"""Shared test fixtures.

Provides a synthetic notebook page renderer so tests exercise the real pipeline
deterministically without depending on an OCR engine or sample scans.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

# The canonical lines rendered onto the synthetic page; also used as the scripted
# recognizer output so downstream stages get realistic text.
SAMPLE_LINES = [
    "Goal: synthesize aspirin",
    "Add 10 mL of 0.5 M HCl",
    "Heat to 80 oC for 30 min",
    "Observed white precipitate",
    "Yield: 75%",
]


def render_page(lines: list[str], size: tuple[int, int] = (820, 460)) -> np.ndarray:
    """Render text lines onto a white page and return a grayscale NumPy array."""
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 25 + i * 70), line, fill=0)
    return np.asarray(img)


@pytest.fixture
def sample_lines() -> list[str]:
    """The text content rendered on the synthetic page."""
    return list(SAMPLE_LINES)


@pytest.fixture
def synthetic_page(sample_lines) -> np.ndarray:
    """A synthetic grayscale notebook page image."""
    return render_page(sample_lines)
