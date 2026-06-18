"""Chemistry extraction stage (§4.5).

Exposes :func:`extract_chemistry`, which combines text-based extraction
(reagents, formulas, concentrations) with drawing-based structure inference, then
normalizes/deduplicates the result into a
:class:`~notebook_parser.types.ChemistrySection`.
"""

from __future__ import annotations

import numpy as np

from ..types import ChemistrySection, LayoutRegion, TranscriptLine
from .extract import extract_text_chemistry
from .normalize import canonical_smiles, normalize_chemistry
from .structures import HeuristicStructureExtractor, extract_structures

__all__ = [
    "extract_chemistry",
    "extract_text_chemistry",
    "extract_structures",
    "normalize_chemistry",
    "canonical_smiles",
    "HeuristicStructureExtractor",
]


def extract_chemistry(
    lines: list[TranscriptLine],
    gray: np.ndarray | None = None,
    regions: list[LayoutRegion] | None = None,
    binary: np.ndarray | None = None,
) -> ChemistrySection:
    """Build the chemistry section from transcript text and drawing regions.

    Text extraction runs on the lines; structure extraction runs on drawing
    regions when an image is supplied. The combined section is normalized and
    deduplicated before return.
    """
    reagents, formulas, concentrations = extract_text_chemistry(lines)
    structures = []
    if gray is not None and regions:
        structures = extract_structures(gray, regions, binary=binary)
    section = ChemistrySection(
        reagents=reagents,
        formulas=formulas,
        structures=structures,
        concentrations=concentrations,
    )
    return normalize_chemistry(section)
