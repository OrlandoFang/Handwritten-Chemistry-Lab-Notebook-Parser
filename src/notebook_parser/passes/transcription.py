"""Transcription pass: layout + transcript + symbols (§4.3).

Sends the conditioned page image to the model and maps the structured response
into canonical layout regions, transcript lines, and symbol tokens.
"""

from __future__ import annotations

from ..config import LLMConfig
from ..imaging import ConditionedImage
from ..llm import prompts
from ..llm.client import LLMEngine
from ..llm.schemas import TranscriptionResponse
from ..types import (
    BoundingBox,
    LayoutRegion,
    RegionType,
    SymbolKind,
    SymbolToken,
    TextCandidate,
    TranscriptLine,
)


def run_transcription(
    engine: LLMEngine,
    image: ConditionedImage,
    config: LLMConfig | None = None,
) -> tuple[list[LayoutRegion], list[TranscriptLine], list[SymbolToken]]:
    """Run the transcription pass and return (regions, lines, symbols)."""
    config = config or LLMConfig()
    resp: TranscriptionResponse = engine.extract(
        system=prompts.TRANSCRIPTION_SYSTEM,
        user=prompts.transcription_user(image.width, image.height, config.max_alternatives),
        schema=TranscriptionResponse,
        image_data_url=image.data_url,
    )

    regions = [
        LayoutRegion(
            id=r.id,
            type=RegionType(r.type.value),
            bbox=BoundingBox(x=r.bbox.x, y=r.bbox.y, width=r.bbox.width, height=r.bbox.height),
            reading_order=r.reading_order,
            confidence=r.confidence,
        )
        for r in resp.regions
    ]

    lines = [
        TranscriptLine(
            id=l.id,
            region_id=l.region_id,
            text=l.text,
            confidence=l.confidence,
            candidates=[TextCandidate(text=a, score=0.0) for a in l.alternatives],
        )
        for l in resp.lines
    ]

    symbols = [
        SymbolToken(
            raw=s.raw,
            normalized=s.normalized,
            kind=SymbolKind(s.kind.value),
            line_id=s.line_id,
            confidence=s.confidence,
        )
        for s in resp.symbols
    ]
    return regions, lines, symbols
