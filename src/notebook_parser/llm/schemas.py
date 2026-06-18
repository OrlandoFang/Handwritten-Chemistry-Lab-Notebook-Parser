"""Structured-output response models, one group per pass (§7).

These mirror the canonical schema but are tuned for OpenAI Structured Outputs:
every field is required and optional values are expressed as nullable (no Python
defaults, which Structured Outputs disallow). The passes map these into the
canonical :mod:`notebook_parser.types` models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class RegionTypeOut(str, Enum):
    """Region categories the model may assign."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    DRAWING = "drawing"
    ANNOTATION = "annotation"
    LABEL = "label"
    UNKNOWN = "unknown"


class BoxOut(BaseModel):
    """Pixel bounding box in the conditioned image's coordinate space."""

    x: int
    y: int
    width: int
    height: int


class RegionOut(BaseModel):
    """A detected layout region."""

    id: str
    type: RegionTypeOut
    bbox: BoxOut
    reading_order: int
    confidence: float


class LineOut(BaseModel):
    """A transcribed line with alternatives."""

    id: str
    region_id: str
    text: str
    confidence: float
    alternatives: list[str]


class SymbolKindOut(str, Enum):
    """Kinds of recovered scientific token."""

    UNIT = "unit"
    SYMBOL = "symbol"
    NOTATION = "notation"


class SymbolOut(BaseModel):
    """A recovered scientific symbol/unit and the raw form it replaced."""

    raw: str
    normalized: str
    kind: SymbolKindOut
    line_id: Optional[str]
    confidence: float


class TranscriptionResponse(BaseModel):
    """Output of the transcription pass."""

    regions: list[RegionOut]
    lines: list[LineOut]
    symbols: list[SymbolOut]


class RoleOut(str, Enum):
    """Reagent roles."""

    REAGENT = "reagent"
    SOLVENT = "solvent"
    CATALYST = "catalyst"
    PRODUCT = "product"
    UNKNOWN = "unknown"


class QuantityOut(BaseModel):
    """A numeric value with a unit."""

    value: Optional[float]
    unit: Optional[str]
    raw: str


class ConcentrationOut(BaseModel):
    """A concentration mention."""

    value: Optional[float]
    unit: Optional[str]
    species: Optional[str]
    raw: str
    evidence: list[str]
    confidence: float


class ReagentOut(BaseModel):
    """A reagent mention linked to evidence."""

    name: str
    normalized_name: Optional[str]
    role: RoleOut
    quantity: Optional[QuantityOut]
    concentration: Optional[ConcentrationOut]
    evidence: list[str]
    confidence: float


class FormulaOut(BaseModel):
    """A chemical formula token."""

    raw: str
    normalized: str
    valid: Optional[bool]
    evidence: list[str]
    confidence: float


class StructureOut(BaseModel):
    """A hand-drawn structure as SMILES + name with an uncertainty flag."""

    region_id: Optional[str]
    name: Optional[str]
    smiles: Optional[str]
    uncertain: bool
    confidence: float
    notes: Optional[str]


class ChemistryResponse(BaseModel):
    """Output of the chemistry pass."""

    reagents: list[ReagentOut]
    formulas: list[FormulaOut]
    structures: list[StructureOut]
    concentrations: list[ConcentrationOut]


class EvidenceOut(BaseModel):
    """A semantic statement linked to transcript evidence."""

    text: str
    evidence: list[str]
    inferred: bool
    confidence: float


class ExperimentResponse(BaseModel):
    """Output of the experiment pass."""

    goal: Optional[str]
    conditions: list[EvidenceOut]
    procedure: list[EvidenceOut]
    observations: list[EvidenceOut]
    results: list[EvidenceOut]
