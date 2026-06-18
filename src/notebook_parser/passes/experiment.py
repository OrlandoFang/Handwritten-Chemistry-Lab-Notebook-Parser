"""Experiment pass: evidence-constrained semantic synthesis (§4.5).

Text-only (no pixels) to minimize hallucination: the model receives the
transcript and extracted chemistry as its sole evidence and must cite line ids.
"""

from __future__ import annotations

from ..llm import prompts
from ..llm.client import LLMEngine
from ..llm.schemas import ExperimentResponse
from ..types import (
    ChemistrySection,
    EvidenceItem,
    ExperimentSection,
    TranscriptLine,
)
from .chemistry import build_transcript_block


def build_chemistry_summary(chemistry: ChemistrySection) -> str:
    """Render the chemistry section as a compact text block for the prompt."""
    parts: list[str] = []
    if chemistry.reagents:
        parts.append("reagents: " + ", ".join(r.name for r in chemistry.reagents))
    if chemistry.concentrations:
        parts.append("concentrations: " + ", ".join(c.raw for c in chemistry.concentrations))
    if chemistry.formulas:
        parts.append("formulas: " + ", ".join(f.normalized for f in chemistry.formulas))
    if chemistry.structures:
        named = [s.name or s.smiles or "structure" for s in chemistry.structures]
        parts.append("structures: " + ", ".join(named))
    return "\n".join(parts) or "(none extracted)"


def _map_items(items) -> list[EvidenceItem]:
    """Map structured evidence items to canonical evidence items."""
    return [
        EvidenceItem(
            text=i.text, evidence=list(i.evidence), inferred=i.inferred, confidence=i.confidence
        )
        for i in items
    ]


def run_experiment(
    engine: LLMEngine,
    lines: list[TranscriptLine],
    chemistry: ChemistrySection,
) -> ExperimentSection:
    """Run the experiment pass and return a canonical experiment section."""
    resp: ExperimentResponse = engine.extract(
        system=prompts.EXPERIMENT_SYSTEM,
        user=prompts.experiment_user(
            build_transcript_block(lines), build_chemistry_summary(chemistry)
        ),
        schema=ExperimentResponse,
        image_data_url=None,  # text-only, grounded in evidence
    )
    return ExperimentSection(
        goal=resp.goal,
        conditions=_map_items(resp.conditions),
        procedure=_map_items(resp.procedure),
        observations=_map_items(resp.observations),
        results=_map_items(resp.results),
    )
