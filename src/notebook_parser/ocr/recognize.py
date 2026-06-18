"""Handwriting recognition backends (§4.3).

Recognition is line-level and backend-agnostic. The pipeline programs against the
:class:`Recognizer` protocol so a real handwriting model can be dropped in. Three
backends ship by default:

* :class:`TesseractRecognizer` - uses pytesseract when the engine is installed;
* :class:`FallbackRecognizer` - emits empty, review-flagged lines when no model
  is available (honest degradation rather than fabricated text);
* :class:`SequenceRecognizer` - returns scripted outputs in call order, used to
  drive the downstream stages in tests and demos without real pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..config import OCRConfig


@dataclass
class RecognitionOutput:
    """One line's recognition result with alternatives for rescoring."""

    text: str
    confidence: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)


class Recognizer(Protocol):
    """Interface every recognition backend must implement."""

    def recognize(self, line_crop: np.ndarray) -> RecognitionOutput:
        """Transcribe a single cropped text line image."""
        ...


class FallbackRecognizer:
    """No-model backend: returns empty text flagged for human review.

    Used when no handwriting model/engine is available. It keeps the pipeline
    runnable and the output schema-valid while making the absence of a real
    transcription explicit (confidence 0) rather than hallucinating text.
    """

    def recognize(self, line_crop: np.ndarray) -> RecognitionOutput:
        """Return an empty, zero-confidence result."""
        return RecognitionOutput(text="", confidence=0.0, candidates=[])


class SequenceRecognizer:
    """Deterministic backend that replays predefined line outputs in order.

    Each call to :meth:`recognize` returns the next scripted string. This makes
    the symbol/chemistry/semantic stages fully testable independent of any OCR
    engine or image content.
    """

    def __init__(self, lines: list[str], confidence: float = 0.9) -> None:
        """Store the scripted lines and the confidence to report for each."""
        self._lines = list(lines)
        self._confidence = confidence
        self._i = 0

    def recognize(self, line_crop: np.ndarray) -> RecognitionOutput:
        """Pop and return the next scripted line (empty once exhausted)."""
        if self._i >= len(self._lines):
            return RecognitionOutput(text="", confidence=0.0)
        text = self._lines[self._i]
        self._i += 1
        return RecognitionOutput(
            text=text,
            confidence=self._confidence,
            candidates=[(text, self._confidence)],
        )


class CallableRecognizer:
    """Adapter wrapping any ``crop -> str`` function as a recognizer."""

    def __init__(self, fn: Callable[[np.ndarray], str], confidence: float = 0.8) -> None:
        """Store the transcription callable and a fixed reported confidence."""
        self._fn = fn
        self._confidence = confidence

    def recognize(self, line_crop: np.ndarray) -> RecognitionOutput:
        """Call the wrapped function and wrap its string output."""
        text = self._fn(line_crop) or ""
        return RecognitionOutput(text=text, confidence=self._confidence if text else 0.0)


class TesseractRecognizer:
    """OCR backend using the Tesseract engine via pytesseract (optional).

    Note: Tesseract is a *printed-text* engine, so this is a stand-in for a true
    handwriting model; it is wired here to demonstrate the swap-in path. Raises
    at construction if pytesseract/the binary are unavailable so callers can fall
    back deterministically.
    """

    def __init__(self, lang: str = "eng") -> None:
        """Import pytesseract eagerly so unavailability fails fast and clearly."""
        try:
            import pytesseract  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError("pytesseract is not installed") from exc
        # Probe the binary; image_to_string raises if tesseract is missing.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # pragma: no cover - depends on system binary
            raise RuntimeError("tesseract binary not found") from exc
        self._pt = pytesseract
        self._lang = lang

    def recognize(self, line_crop: np.ndarray) -> RecognitionOutput:  # pragma: no cover
        """Transcribe a line crop and average per-word confidences."""
        data = self._pt.image_to_data(
            line_crop, lang=self._lang, output_type=self._pt.Output.DICT, config="--psm 7"
        )
        words = [w for w in data.get("text", []) if w and w.strip()]
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
        confs = [c for c in confs if c >= 0]
        text = " ".join(words).strip()
        confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return RecognitionOutput(
            text=text,
            confidence=confidence,
            candidates=[(text, confidence)] if text else [],
        )


def build_default_recognizer(config: OCRConfig | None = None) -> Recognizer:
    """Select a recognizer per config, preferring real engines when present.

    ``auto`` tries Tesseract and silently falls back; ``tesseract``/``fallback``
    force a specific backend.
    """
    config = config or OCRConfig()
    backend = config.backend
    if backend in ("auto", "tesseract"):
        try:
            return TesseractRecognizer()
        except Exception:
            if backend == "tesseract":
                raise
    return FallbackRecognizer()
