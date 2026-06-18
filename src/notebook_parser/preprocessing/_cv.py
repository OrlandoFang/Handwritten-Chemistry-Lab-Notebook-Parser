"""Lazy, optional OpenCV/scikit-image access.

Preprocessing benefits from OpenCV, but the package must still import (and the
non-vision stages still run) in environments without it. These helpers import the
heavy libraries on first use and report availability so callers can degrade
gracefully instead of crashing at import time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_cv2() -> Any | None:
    """Return the ``cv2`` module if installed, else ``None`` (cached)."""
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:  # pragma: no cover - exercised only without opencv
        return None


@lru_cache(maxsize=1)
def get_skimage_filters() -> Any | None:
    """Return ``skimage.filters`` if installed, else ``None`` (cached)."""
    try:
        from skimage import filters  # type: ignore

        return filters
    except Exception:  # pragma: no cover - exercised only without skimage
        return None


def have_cv2() -> bool:
    """Whether OpenCV is importable."""
    return get_cv2() is not None
