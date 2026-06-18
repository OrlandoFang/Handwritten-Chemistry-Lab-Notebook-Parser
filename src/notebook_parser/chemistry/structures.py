"""Hand-drawn chemical structure extraction (§4.5).

A full system would vectorize strokes and infer atoms/bonds/rings with a trained
model. Absent that model here, the default :class:`HeuristicStructureExtractor`
still produces a *partial, explicitly uncertain* representation: it confirms a
drawing region, estimates bond/junction counts via line detection, and records
them as notes. This honors the spec's requirement to emit a partial structured
result with clear uncertainty rather than nothing. The extractor is swappable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..preprocessing._cv import get_cv2
from ..types import BoundingBox, ChemicalStructure, LayoutRegion, RegionType


class StructureExtractor(Protocol):
    """Interface for structure-recognition backends."""

    def extract(self, gray: np.ndarray, region: LayoutRegion) -> ChemicalStructure:
        """Infer a (possibly partial) structure from a drawing region."""
        ...


def _estimate_bonds(crop_binary: np.ndarray) -> int | None:
    """Estimate the number of straight bond strokes via the Hough transform.

    Returns ``None`` when OpenCV is unavailable. This is a rough proxy used only
    to annotate uncertainty, not to claim an exact structure.
    """
    cv2 = get_cv2()
    if cv2 is None or crop_binary.size == 0:
        return None
    min_len = max(8, crop_binary.shape[0] // 8)
    lines = cv2.HoughLinesP(
        (crop_binary > 0).astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=min_len,
        maxLineGap=3,
    )
    return 0 if lines is None else int(len(lines))


class HeuristicStructureExtractor:
    """Default backend: confirms drawings and annotates rough estimates."""

    def __init__(self, binary: np.ndarray | None = None) -> None:
        """Optionally take the page binary mask for line-count estimates."""
        self._binary = binary

    def extract(self, gray: np.ndarray, region: LayoutRegion) -> ChemicalStructure:
        """Return an uncertain structure record for one drawing region."""
        bbox: BoundingBox = region.bbox
        bonds = None
        if self._binary is not None:
            crop = self._binary[bbox.y : bbox.y1, bbox.x : bbox.x1]
            bonds = _estimate_bonds(crop)
        note = "hand-drawn structure detected; full recognition unavailable"
        if bonds is not None:
            note += f"; ~{bonds} bond-like strokes estimated"
        return ChemicalStructure(
            region_id=region.id,
            smiles=None,
            atoms=[],
            bonds=[],
            confidence=0.2,
            uncertain=True,
            notes=note,
        )


def extract_structures(
    gray: np.ndarray,
    regions: list[LayoutRegion],
    binary: np.ndarray | None = None,
    extractor: StructureExtractor | None = None,
) -> list[ChemicalStructure]:
    """Run structure extraction over all drawing regions on the page."""
    extractor = extractor or HeuristicStructureExtractor(binary)
    return [
        extractor.extract(gray, region)
        for region in regions
        if region.type == RegionType.DRAWING
    ]
