"""Tests for the transcription stage (§4.3, §10)."""

from __future__ import annotations

import numpy as np

from notebook_parser.config import OCRConfig
from notebook_parser.layout import HeuristicLayoutDetector, assign_reading_order
from notebook_parser.ocr import SequenceRecognizer, transcribe
from notebook_parser.ocr.correct import clean_text
from notebook_parser.ocr.decode import language_model_score, rescore
from notebook_parser.ocr.recognize import FallbackRecognizer, RecognitionOutput
from notebook_parser.preprocessing import preprocess


def test_sequence_recognizer_replays_then_empties():
    """SequenceRecognizer returns scripted lines, then empty results."""
    rec = SequenceRecognizer(["first", "second"])
    crop = np.zeros((5, 5), np.uint8)
    assert rec.recognize(crop).text == "first"
    assert rec.recognize(crop).text == "second"
    assert rec.recognize(crop).text == ""


def test_fallback_recognizer_is_low_confidence():
    """The no-model backend emits empty, zero-confidence output."""
    out = FallbackRecognizer().recognize(np.zeros((4, 4), np.uint8))
    assert out.text == ""
    assert out.confidence == 0.0


def test_language_model_score_prefers_lab_text():
    """Lab vocabulary scores higher than random characters."""
    assert language_model_score("add solution and stir") > language_model_score("xqzv wkbp")


def test_rescore_reranks_candidates():
    """Rescoring promotes the more plausible candidate."""
    out = RecognitionOutput(
        text="qzxw",
        confidence=0.6,
        candidates=[("qzxw", 0.6), ("add the solution", 0.55)],
    )
    rescored = rescore(out, OCRConfig())
    assert rescored.text == "add the solution"


def test_clean_text_normalizes_whitespace_and_decimals():
    """Generic cleanup collapses spaces and rejoins split decimals."""
    text, corrections = clean_text("add   0 . 5  mL")
    assert text == "add 0.5 mL"
    assert "rejoin_decimal" in corrections
    assert "collapse_whitespace" in corrections


def test_transcribe_with_scripted_recognizer(synthetic_page, sample_lines):
    """End-to-end line transcription uses the injected recognizer output."""
    pre = preprocess(synthetic_page)
    regions = HeuristicLayoutDetector().detect(pre.binary)
    regions = assign_reading_order(regions, pre.binary.shape[1])
    lines = transcribe(
        pre.gray, pre.binary, regions, recognizer=SequenceRecognizer(sample_lines)
    )
    joined = " ".join(l.text for l in lines)
    assert "synthesize aspirin" in joined
    assert all(l.region_id for l in lines)
