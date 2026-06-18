"""Typed data model for the parser.

Every intermediate and final artifact is a Pydantic model so that:

* outputs validate against a single source of truth (§2 of the spec),
* provenance (bounding boxes, candidates, applied corrections) travels with the
  data through every stage (§9), and
* the final JSON is produced deterministically via ``model_dump``.

The top-level result, :class:`ParseResult`, serializes to exactly the JSON shape
described in the spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (origin = top-left).

    Stored as (x, y, width, height) which is the most common convention for the
    OpenCV-based stages. Helper properties expose edge coordinates and area so
    callers never recompute them inconsistently.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        """Right edge (exclusive-ish) x coordinate."""
        return self.x + self.width

    @property
    def y1(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Box area in pixels."""
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        """Box center as (cx, cy)."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union with another box (0.0 when disjoint)."""
        ix0 = max(self.x, other.x)
        iy0 = max(self.y, other.y)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
class RegionType(str, Enum):
    """Coarse region categories produced by layout analysis (§4.2)."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    DRAWING = "drawing"
    ANNOTATION = "annotation"
    LABEL = "label"
    UNKNOWN = "unknown"


class LayoutRegion(BaseModel):
    """A detected page region with reading order and provenance."""

    id: str
    type: RegionType = RegionType.UNKNOWN
    bbox: BoundingBox
    reading_order: int = 0
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class TextCandidate(BaseModel):
    """An alternative transcription hypothesis for a line (§4.3 beam search)."""

    text: str
    score: float = 0.0


class TranscriptLine(BaseModel):
    """Line-level transcription with source region and alternatives."""

    id: str
    region_id: str
    text: str
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    candidates: list[TextCandidate] = Field(default_factory=list)
    # Human-readable record of correction steps applied to this line (§9).
    corrections: list[str] = Field(default_factory=list)
    needs_review: bool = False


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #
class SymbolKind(str, Enum):
    """What a recovered scientific token represents."""

    UNIT = "unit"
    SYMBOL = "symbol"
    NOTATION = "notation"  # e.g. scientific notation, super/subscripts


class SymbolToken(BaseModel):
    """A normalized scientific symbol/unit with provenance (§4.4)."""

    raw: str
    normalized: str
    kind: SymbolKind
    line_id: Optional[str] = None
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Chemistry
# --------------------------------------------------------------------------- #
class Quantity(BaseModel):
    """A numeric value paired with a unit (e.g. ``10 mL``)."""

    value: Optional[float] = None
    unit: Optional[str] = None
    raw: str


class ReagentRole(str, Enum):
    """Role a chemical plays in the experiment."""

    REAGENT = "reagent"
    SOLVENT = "solvent"
    CATALYST = "catalyst"
    PRODUCT = "product"
    UNKNOWN = "unknown"


class Reagent(BaseModel):
    """A reagent mention linked back to the transcript (§4.5, §8)."""

    name: str
    normalized_name: Optional[str] = None
    role: ReagentRole = ReagentRole.UNKNOWN
    quantity: Optional[Quantity] = None
    concentration: Optional["Concentration"] = None
    evidence: list[str] = Field(default_factory=list)  # transcript line ids
    confidence: float = 0.0


class Formula(BaseModel):
    """A chemical formula token (e.g. ``H2SO4``)."""

    raw: str
    normalized: str
    valid: Optional[bool] = None  # None when no validator (rdkit) available
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class Concentration(BaseModel):
    """A concentration mention (e.g. ``0.5 M HCl``)."""

    value: Optional[float] = None
    unit: Optional[str] = None
    species: Optional[str] = None
    raw: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class AtomNode(BaseModel):
    """A node in an inferred molecular graph."""

    id: int
    element: str = "C"


class BondEdge(BaseModel):
    """An edge in an inferred molecular graph."""

    source: int
    target: int
    order: int = 1


class ChemicalStructure(BaseModel):
    """A (possibly partial) machine-readable hand-drawn structure (§4.5).

    ``uncertain`` is set whenever full recovery was not possible so downstream
    consumers can tell inferred structures from confidently recovered ones.
    """

    region_id: Optional[str] = None
    name: Optional[str] = None
    smiles: Optional[str] = None
    atoms: list[AtomNode] = Field(default_factory=list)
    bonds: list[BondEdge] = Field(default_factory=list)
    confidence: float = 0.0
    uncertain: bool = True
    notes: Optional[str] = None


class ChemistrySection(BaseModel):
    """The ``chemistry`` block of the output schema."""

    reagents: list[Reagent] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    structures: list[ChemicalStructure] = Field(default_factory=list)
    concentrations: list[Concentration] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Experiment semantics
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    """A semantic statement linked to the transcript evidence it derives from.

    ``inferred`` distinguishes synthesized conclusions from directly observed
    text, which the spec requires be kept separable (§11).
    """

    text: str
    evidence: list[str] = Field(default_factory=list)  # transcript line ids
    inferred: bool = False
    confidence: float = 0.0


class ExperimentSection(BaseModel):
    """The ``experiment`` block of the output schema."""

    goal: Optional[str] = None
    conditions: list[EvidenceItem] = Field(default_factory=list)
    procedure: list[EvidenceItem] = Field(default_factory=list)
    observations: list[EvidenceItem] = Field(default_factory=list)
    results: list[EvidenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Confidence + top-level result
# --------------------------------------------------------------------------- #
class ConfidenceReport(BaseModel):
    """Aggregated confidence with per-field breakdown (§9)."""

    overall: float = 0.0
    fields: dict[str, float] = Field(default_factory=dict)


class ParseResult(BaseModel):
    """Top-level parser output; serializes to the schema in §2 of the spec."""

    page_id: str
    document_type: str = "chemistry_lab_notebook"
    layout: list[LayoutRegion] = Field(default_factory=list)
    transcript: list[TranscriptLine] = Field(default_factory=list)
    symbols: list[SymbolToken] = Field(default_factory=list)
    chemistry: ChemistrySection = Field(default_factory=ChemistrySection)
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize deterministically to a JSON string matching the spec."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict."""
        return self.model_dump(mode="json")


# Resolve the forward reference between Reagent and Concentration.
Reagent.model_rebuild()
