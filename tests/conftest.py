"""Shared test fixtures.

Provides a synthetic page image and a fully offline ``StubEngine`` pre-loaded with
canned structured responses (modeled on the sample Li-electrodeposition page), so
the entire pipeline is testable without network access or an API key.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from notebook_parser.llm.client import StubEngine
from notebook_parser.llm.schemas import (
    BoxOut,
    ChemistryResponse,
    ConcentrationOut,
    EvidenceOut,
    ExperimentResponse,
    FormulaOut,
    LineOut,
    QuantityOut,
    ReagentOut,
    RegionOut,
    RegionTypeOut,
    RoleOut,
    StructureOut,
    SymbolKindOut,
    SymbolOut,
    TranscriptionResponse,
)


@pytest.fixture
def synthetic_page() -> Image.Image:
    """A small synthetic page image (content irrelevant; the engine is stubbed)."""
    img = Image.new("RGB", (640, 360), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, t in enumerate(["Project: Li electrodeposition", "1M LiTFSI in diglyme"]):
        draw.text((20, 20 + i * 60), t, fill=(0, 0, 0))
    return img


@pytest.fixture
def transcription_response() -> TranscriptionResponse:
    """Canned transcription-pass output."""
    return TranscriptionResponse(
        regions=[
            RegionOut(id="r0", type=RegionTypeOut.HEADING, bbox=BoxOut(x=10, y=10, width=400, height=30), reading_order=0, confidence=0.95),
            RegionOut(id="r1", type=RegionTypeOut.TEXT, bbox=BoxOut(x=10, y=60, width=500, height=120), reading_order=1, confidence=0.9),
            RegionOut(id="r2", type=RegionTypeOut.DRAWING, bbox=BoxOut(x=20, y=200, width=200, height=120), reading_order=2, confidence=0.7),
        ],
        lines=[
            LineOut(id="r0_l0", region_id="r0", text="Project: Li electrodeposition - glyme electrolytes", confidence=0.94, alternatives=[]),
            LineOut(id="r1_l0", region_id="r1", text="Goal: Screen electrolyte 240604-B for stable Li plating at 30 °C", confidence=0.88, alternatives=[]),
            LineOut(id="r1_l1", region_id="r1", text="Electrolyte: 1M LiTFSI in diglyme : EtOH (4:1 v/v)", confidence=0.82, alternatives=["1M LiTFSI in diglycine : EtOH"]),
            LineOut(id="r1_l2", region_id="r1", text="Add 5 mol% 12-crown-4 as additive. Stir 20 min", confidence=0.8, alternatives=[]),
            LineOut(id="r1_l3", region_id="r1", text="Observed film looks grey + dull", confidence=0.5, alternatives=[]),
        ],
        symbols=[
            SymbolOut(raw="oC", normalized="°C", kind=SymbolKindOut.UNIT, line_id="r1_l0", confidence=0.9),
            SymbolOut(raw="v/v", normalized="v/v", kind=SymbolKindOut.NOTATION, line_id="r1_l1", confidence=0.85),
            SymbolOut(raw="mol%", normalized="mol%", kind=SymbolKindOut.UNIT, line_id="r1_l2", confidence=0.8),
        ],
    )


@pytest.fixture
def chemistry_response() -> ChemistryResponse:
    """Canned chemistry-pass output (note the deliberately non-canonical SMILES)."""
    return ChemistryResponse(
        reagents=[
            ReagentOut(name="LiTFSI", normalized_name="lithium bis(trifluoromethanesulfonyl)imide", role=RoleOut.REAGENT,
                       quantity=None, concentration=ConcentrationOut(value=1.0, unit="M", species="LiTFSI", raw="1M LiTFSI", evidence=["r1_l1"], confidence=0.85),
                       evidence=["r1_l1"], confidence=0.85),
            ReagentOut(name="EtOH", normalized_name="ethanol", role=RoleOut.SOLVENT, quantity=None, concentration=None, evidence=["r1_l1"], confidence=0.8),
            ReagentOut(name="12-crown-4", normalized_name="1,4,7,10-tetraoxacyclododecane", role=RoleOut.CATALYST,
                       quantity=QuantityOut(value=5.0, unit="mol%", raw="5 mol%"), concentration=None, evidence=["r1_l2"], confidence=0.75),
        ],
        formulas=[FormulaOut(raw="LiTFSI", normalized="LiTFSI", valid=None, evidence=["r1_l1"], confidence=0.7)],
        structures=[
            StructureOut(region_id="r2", name="12-crown-4", smiles="O1CCOCCOCCOCC1", uncertain=False, confidence=0.6, notes=None),
        ],
        concentrations=[ConcentrationOut(value=1.0, unit="M", species="LiTFSI", raw="1M LiTFSI", evidence=["r1_l1"], confidence=0.85)],
    )


@pytest.fixture
def experiment_response() -> ExperimentResponse:
    """Canned experiment-pass output."""
    return ExperimentResponse(
        goal="Screen electrolyte 240604-B for stable Li plating at 30 °C",
        conditions=[EvidenceOut(text="at 30 °C", evidence=["r1_l0"], inferred=False, confidence=0.85)],
        procedure=[EvidenceOut(text="Add 5 mol% 12-crown-4 as additive. Stir 20 min", evidence=["r1_l2"], inferred=False, confidence=0.8)],
        observations=[EvidenceOut(text="film looks grey + dull", evidence=["r1_l3"], inferred=False, confidence=0.5)],
        results=[],
    )


@pytest.fixture
def stub_engine(transcription_response, chemistry_response, experiment_response) -> StubEngine:
    """A StubEngine wired with all three canned pass responses."""
    engine = StubEngine()
    engine.register(TranscriptionResponse, transcription_response)
    engine.register(ChemistryResponse, chemistry_response)
    engine.register(ExperimentResponse, experiment_response)
    return engine
