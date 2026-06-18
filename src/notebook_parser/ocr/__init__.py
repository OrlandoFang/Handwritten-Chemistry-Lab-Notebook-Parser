"""Handwriting transcription stage (§4.3).

Exposes :func:`transcribe`, which segments text-bearing regions into lines, runs
the (swappable) recognizer per line, applies LM rescoring and generic cleanup,
and emits :class:`~notebook_parser.types.TranscriptLine` objects carrying
confidence, alternatives, applied corrections, and a human-review flag.
"""

from __future__ import annotations

import numpy as np

from ..config import OCRConfig
from ..layout.detect import detect_text_lines
from ..types import BoundingBox, LayoutRegion, RegionType, TextCandidate, TranscriptLine
from .correct import clean_text
from .decode import rescore
from .recognize import (
    CallableRecognizer,
    FallbackRecognizer,
    Recognizer,
    RecognitionOutput,
    SequenceRecognizer,
    TesseractRecognizer,
    build_default_recognizer,
)

__all__ = [
    "transcribe",
    "Recognizer",
    "RecognitionOutput",
    "FallbackRecognizer",
    "SequenceRecognizer",
    "CallableRecognizer",
    "TesseractRecognizer",
    "build_default_recognizer",
]

# Region types whose pixels contain transcribable text.
_TEXT_LIKE = {
    RegionType.TEXT,
    RegionType.HEADING,
    RegionType.ANNOTATION,
    RegionType.LABEL,
    RegionType.TABLE,
}


def _crop(gray: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Return a clamped crop of ``gray`` for ``bbox`` (never out of bounds)."""
    h, w = gray.shape[:2]
    x0 = max(0, bbox.x)
    y0 = max(0, bbox.y)
    x1 = min(w, bbox.x1)
    y1 = min(h, bbox.y1)
    return gray[y0:y1, x0:x1]


def transcribe(
    gray: np.ndarray,
    binary: np.ndarray,
    regions: list[LayoutRegion],
    recognizer: Recognizer | None = None,
    config: OCRConfig | None = None,
    review_threshold: float = 0.6,
) -> list[TranscriptLine]:
    """Transcribe all text-like regions into ordered line records.

    Regions are visited in reading order so the transcript reads naturally.
    Lines below ``review_threshold`` confidence are flagged for human review but
    still emitted (§9). The recognizer defaults to the best available backend.
    """
    config = config or OCRConfig()
    recognizer = recognizer or build_default_recognizer(config)

    ordered = sorted(regions, key=lambda r: r.reading_order)
    lines: list[TranscriptLine] = []

    for region in ordered:
        if region.type not in _TEXT_LIKE:
            continue
        line_boxes = detect_text_lines(binary, region.bbox)
        for li, box in enumerate(line_boxes):
            crop = _crop(gray, box)
            if crop.size == 0:
                continue
            raw = recognizer.recognize(crop)
            scored = rescore(raw, config)
            text, corrections = clean_text(scored.text)
            candidates = [
                TextCandidate(text=t, score=float(s)) for t, s in scored.candidates
            ]
            lines.append(
                TranscriptLine(
                    id=f"{region.id}_l{li}",
                    region_id=region.id,
                    text=text,
                    bbox=box,
                    confidence=float(scored.confidence),
                    candidates=candidates,
                    corrections=corrections,
                    needs_review=scored.confidence < review_threshold,
                )
            )
    return lines
