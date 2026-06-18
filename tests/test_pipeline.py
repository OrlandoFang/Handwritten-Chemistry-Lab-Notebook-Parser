"""Integration tests for the full pipeline (§7, §10 integration)."""

from __future__ import annotations

from notebook_parser.config import PipelineConfig
from notebook_parser.ocr import SequenceRecognizer
from notebook_parser.pipeline import NotebookPipeline
from notebook_parser.validation import check_output_dict


def _run(page, lines):
    """Run the pipeline with a scripted recognizer for reproducibility."""
    pipeline = NotebookPipeline(PipelineConfig(page_id="test_page"))
    return pipeline.run(page, recognizer=SequenceRecognizer(lines))


def test_full_pipeline_produces_valid_schema(synthetic_page, sample_lines):
    """A full run yields a schema-valid result with populated sections."""
    result = _run(synthetic_page, sample_lines)
    check_output_dict(result.to_dict())
    assert result.page_id == "test_page"
    assert result.document_type == "chemistry_lab_notebook"
    assert result.transcript
    assert result.experiment.goal == "synthesize aspirin"
    assert any(c.unit == "M" for c in result.chemistry.concentrations)
    assert 0.0 <= result.confidence.overall <= 1.0


def test_full_pipeline_is_deterministic(synthetic_page, sample_lines):
    """Identical inputs yield identical JSON (§11 determinism)."""
    first = _run(synthetic_page, sample_lines).to_json()
    second = _run(synthetic_page, sample_lines).to_json()
    assert first == second


def test_symbols_normalized_in_transcript(synthetic_page, sample_lines):
    """Symbol recovery is reflected in the emitted transcript text."""
    result = _run(synthetic_page, sample_lines)
    joined = " ".join(l.text for l in result.transcript)
    assert "\u00b0C" in joined  # '80 oC' -> '80 °C'
