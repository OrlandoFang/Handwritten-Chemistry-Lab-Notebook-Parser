"""Heuristic layout detection and text-line segmentation (§4.2).

The default detector uses connected-component analysis on the binary ink mask:

1. characters are merged into blocks via morphological dilation,
2. each block becomes a candidate region,
3. blocks are classified (table/drawing/heading/annotation/text) from cheap
   geometric and ink-density features.

These heuristics are deliberately conservative and serve as the swappable
fallback the spec calls for when a trained layout model is unavailable or low
confidence. Everything is deterministic.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import LayoutConfig
from ..preprocessing._cv import get_cv2
from ..types import BoundingBox, LayoutRegion, RegionType


class LayoutDetector(Protocol):
    """Interface every layout backend must implement."""

    def detect(self, binary: np.ndarray) -> list[LayoutRegion]:
        """Return unordered typed regions for an ink=255 binary mask."""
        ...


def _connected_component_boxes(mask: np.ndarray, min_area: int) -> list[BoundingBox]:
    """Return bounding boxes of connected components in ``mask`` >= ``min_area``.

    Uses OpenCV's connected-components when available; otherwise returns a single
    box covering all ink (a safe, if coarse, fallback).
    """
    cv2 = get_cv2()
    if cv2 is None:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return []
        return [
            BoundingBox(
                x=int(xs.min()),
                y=int(ys.min()),
                width=int(xs.max() - xs.min() + 1),
                height=int(ys.max() - ys.min() + 1),
            )
        ]
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes: list[BoundingBox] = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append(BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)))
    return boxes


def _grid_line_mask(binary: np.ndarray) -> np.ndarray:
    """Return a mask of long horizontal+vertical rules (table grid candidates)."""
    cv2 = get_cv2()
    if cv2 is None:
        return np.zeros_like(binary)
    h, w = binary.shape[:2]
    src = (binary > 0).astype(np.uint8)
    # Structuring elements long enough to keep only ruled lines, not glyph strokes.
    hori_len = max(20, w // 12)
    vert_len = max(20, h // 12)
    hori = cv2.morphologyEx(
        src, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hori_len, 1)),
    )
    vert = cv2.morphologyEx(
        src, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len)),
    )
    return ((hori + vert) > 0).astype(np.uint8)


def _count_text_rows(block_mask: np.ndarray, line_gap_frac: float) -> int:
    """Estimate the number of text rows in a block via horizontal projection.

    Rows are runs of nonzero row-sums separated by gaps; used to distinguish
    multi-line text blocks from drawings (few/no regular rows).
    """
    row_ink = block_mask.sum(axis=1) > 0
    if not row_ink.any():
        return 0
    rows = 0
    prev = False
    for v in row_ink:
        if v and not prev:
            rows += 1
        prev = bool(v)
    return rows


class HeuristicLayoutDetector:
    """Connected-component + geometry based layout backend (default)."""

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Store config; all thresholds derive from it for reproducibility."""
        self.config = config or LayoutConfig()

    def _classify(
        self,
        binary: np.ndarray,
        grid: np.ndarray,
        bbox: BoundingBox,
        median_h: float,
    ) -> tuple[RegionType, float]:
        """Classify a single block and return (type, confidence in [0,1])."""
        page_h, page_w = binary.shape[:2]
        sub = binary[bbox.y : bbox.y1, bbox.x : bbox.x1]
        if sub.size == 0:
            return RegionType.UNKNOWN, 0.1
        ink_density = float((sub > 0).mean())
        aspect = bbox.width / max(1, bbox.height)
        n_rows = _count_text_rows(sub, self.config.line_gap_frac)
        cx, _ = bbox.center

        grid_sub = grid[bbox.y : bbox.y1, bbox.x : bbox.x1]
        if grid_sub.size and float(grid_sub.mean()) > 0.02:
            return RegionType.TABLE, 0.6

        # Drawings: compact, dense ink without a regular multi-row text pattern.
        if (
            n_rows <= 2
            and ink_density > 0.12
            and 0.4 < aspect < 2.5
            and bbox.area > 0.01 * page_h * page_w
        ):
            return RegionType.DRAWING, 0.5

        # Headings: tall single rows near the top of the page.
        if (
            bbox.y < 0.15 * page_h
            and bbox.height > 1.4 * median_h
            and n_rows <= 1
        ):
            return RegionType.HEADING, 0.5

        # Marginal annotations: small blocks hugging the left/right margins.
        if (cx < 0.08 * page_w or cx > 0.92 * page_w) and bbox.area < 0.02 * page_h * page_w:
            return RegionType.ANNOTATION, 0.4

        return RegionType.TEXT, 0.55

    def detect(self, binary: np.ndarray) -> list[LayoutRegion]:
        """Detect typed regions from an ink=255 binary mask."""
        page_h, page_w = binary.shape[:2]
        min_area = int(self.config.min_region_area_frac * page_h * page_w)

        cv2 = get_cv2()
        if cv2 is None:
            # Without morphology we cannot group; treat the whole inked area as
            # one text region so the pipeline still produces output.
            boxes = _connected_component_boxes(binary, min_area)
            return [
                LayoutRegion(
                    id=f"r{i}", type=RegionType.TEXT, bbox=b, confidence=0.3
                )
                for i, b in enumerate(boxes)
            ]

        # Merge glyphs -> blocks. Horizontal dilation joins characters/words on a
        # line; modest vertical dilation joins lines within a paragraph/block.
        kx = max(5, page_w // 40)
        ky = max(3, page_h // 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        merged = cv2.dilate((binary > 0).astype(np.uint8), kernel, iterations=1)
        boxes = _connected_component_boxes(merged * 255, min_area)

        if not boxes:
            return []

        median_h = float(np.median([b.height for b in boxes]))
        grid = _grid_line_mask(binary)

        regions: list[LayoutRegion] = []
        for i, bbox in enumerate(boxes):
            rtype, conf = self._classify(binary, grid, bbox, median_h)
            regions.append(
                LayoutRegion(id=f"r{i}", type=rtype, bbox=bbox, confidence=conf)
            )
        return regions


def detect_text_lines(binary: np.ndarray, region: BoundingBox) -> list[BoundingBox]:
    """Segment a text region into line bounding boxes, top-to-bottom (§ stage 3).

    Crops the region, joins glyphs horizontally so each text line becomes one
    component, and returns line boxes in original page coordinates. Falls back to
    a single line covering the region when OpenCV is unavailable.
    """
    cv2 = get_cv2()
    sub = binary[region.y : region.y1, region.x : region.x1]
    if sub.size == 0:
        return []
    if cv2 is None:
        return [region]

    width = sub.shape[1]
    # Aggressively connect along x so a whole line collapses to one component,
    # but keep vertical extent so adjacent lines stay separate.
    kx = max(10, width // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    merged = cv2.dilate((sub > 0).astype(np.uint8), kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

    lines: list[BoundingBox] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 0.2 * width:  # ignore stray marks, not full lines
            continue
        lines.append(
            BoundingBox(
                x=region.x + int(x),
                y=region.y + int(y),
                width=int(w),
                height=int(h),
            )
        )
    lines.sort(key=lambda b: b.y)
    return lines or [region]
