"""Reading-order estimation for detected regions (§4.2).

Notebook pages may be single- or multi-column. We detect columns by clustering
region x-centers, then read columns left-to-right and regions within a column
top-to-bottom. The function mutates and returns the regions with
``reading_order`` populated; ordering is stable and deterministic.
"""

from __future__ import annotations

import numpy as np

from ..config import LayoutConfig
from ..types import LayoutRegion


def _assign_columns(regions: list[LayoutRegion], page_width: int, split_frac: float) -> list[int]:
    """Group regions into columns; return a column index per region.

    Sorts regions by x-center and starts a new column whenever the gap to the
    previous center exceeds ``split_frac`` of the page width. This handles the
    common 1- and 2-column notebook layouts without a clustering library.
    """
    if not regions:
        return []
    centers = [r.bbox.center[0] for r in regions]
    order = np.argsort(centers)
    threshold = split_frac * max(1, page_width)
    columns = [0] * len(regions)
    current = 0
    prev_center = centers[order[0]]
    for idx in order:
        if centers[idx] - prev_center > threshold:
            current += 1
        columns[idx] = current
        prev_center = centers[idx]
    return columns


def assign_reading_order(
    regions: list[LayoutRegion],
    page_width: int,
    config: LayoutConfig | None = None,
) -> list[LayoutRegion]:
    """Populate ``reading_order`` on each region and return them sorted by it."""
    config = config or LayoutConfig()
    if not regions:
        return regions
    columns = _assign_columns(regions, page_width, config.column_split_frac)
    # Sort key: column first (left-to-right), then vertical, then horizontal
    # position to break ties deterministically.
    indexed = sorted(
        range(len(regions)),
        key=lambda i: (columns[i], regions[i].bbox.y, regions[i].bbox.x),
    )
    ordered: list[LayoutRegion] = []
    for order_value, region_idx in enumerate(indexed):
        regions[region_idx].reading_order = order_value
        ordered.append(regions[region_idx])
    return ordered
