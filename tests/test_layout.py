"""Tests for layout detection and reading order (§4.2, §10)."""

from __future__ import annotations

from notebook_parser.config import LayoutConfig
from notebook_parser.layout import (
    HeuristicLayoutDetector,
    assign_reading_order,
    detect_text_lines,
)
from notebook_parser.preprocessing import preprocess
from notebook_parser.types import BoundingBox, LayoutRegion, RegionType


def _region(rid: str, x: int, y: int, w: int = 50, h: int = 20) -> LayoutRegion:
    """Build a text region with the given position for ordering tests."""
    return LayoutRegion(id=rid, type=RegionType.TEXT, bbox=BoundingBox(x=x, y=y, width=w, height=h))


def test_reading_order_single_column_top_to_bottom():
    """In one column, regions are ordered by vertical position."""
    regions = [_region("b", 10, 200), _region("a", 10, 10), _region("c", 10, 100)]
    ordered = assign_reading_order(regions, page_width=300, config=LayoutConfig())
    assert [r.id for r in ordered] == ["a", "c", "b"]
    assert [r.reading_order for r in ordered] == [0, 1, 2]


def test_reading_order_two_columns_left_before_right():
    """Left-column regions precede right-column regions regardless of y."""
    regions = [
        _region("r_top", 700, 10),
        _region("l_bottom", 20, 300),
        _region("l_top", 20, 10),
        _region("r_bottom", 700, 300),
    ]
    ordered = assign_reading_order(regions, page_width=1000, config=LayoutConfig())
    assert [r.id for r in ordered] == ["l_top", "l_bottom", "r_top", "r_bottom"]


def test_detector_finds_regions(synthetic_page):
    """The heuristic detector returns at least one region for a text page."""
    pre = preprocess(synthetic_page)
    regions = HeuristicLayoutDetector().detect(pre.binary)
    assert len(regions) >= 1
    assert all(r.bbox.area > 0 for r in regions)


def test_detect_text_lines_orders_top_to_bottom(synthetic_page):
    """Line detection inside a full-page region returns ordered line boxes."""
    pre = preprocess(synthetic_page)
    h, w = pre.binary.shape
    full = BoundingBox(x=0, y=0, width=w, height=h)
    lines = detect_text_lines(pre.binary, full)
    assert len(lines) >= 2
    ys = [b.y for b in lines]
    assert ys == sorted(ys)
