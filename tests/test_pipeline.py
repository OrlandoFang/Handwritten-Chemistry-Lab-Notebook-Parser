"""Integration tests for the full pipeline using the offline stub engine (§11)."""

from __future__ import annotations

from notebook_parser.config import PipelineConfig
from notebook_parser.pipeline import NotebookPipeline
from notebook_parser.validation import check_output_dict


def test_full_pipeline_produces_valid_schema(stub_engine, synthetic_page):
    """A full run yields a schema-valid result with populated sections."""
    pipeline = NotebookPipeline(PipelineConfig(page_id="page_57"), engine=stub_engine)
    result = pipeline.run(synthetic_page)

    check_output_dict(result.to_dict())
    assert result.page_id == "page_57"
    assert result.document_type == "chemistry_lab_notebook"
    assert result.transcript
    assert any(s.smiles for s in result.chemistry.structures)
    assert result.experiment.goal and "Li plating" in result.experiment.goal
    assert any(c.unit == "M" for c in result.chemistry.concentrations)
    assert 0.0 <= result.confidence.overall <= 1.0


def test_full_pipeline_output_keys_match_spec(stub_engine, synthetic_page):
    """Top-level JSON keys match the required output schema exactly."""
    pipeline = NotebookPipeline(engine=stub_engine)
    data = pipeline.run(synthetic_page, page_id="p").to_dict()
    assert set(data.keys()) == {
        "page_id", "document_type", "layout", "transcript", "symbols",
        "chemistry", "experiment", "confidence",
    }
    assert set(data["chemistry"].keys()) == {"reagents", "formulas", "structures", "concentrations"}
    assert set(data["experiment"].keys()) == {"goal", "conditions", "procedure", "observations", "results"}


def test_full_pipeline_deterministic(stub_engine, synthetic_page):
    """Identical inputs yield identical JSON with a fixed engine (§10)."""
    pipeline = NotebookPipeline(engine=stub_engine)
    first = pipeline.run(synthetic_page, page_id="p").to_json()
    second = pipeline.run(synthetic_page, page_id="p").to_json()
    assert first == second


def test_low_confidence_lines_flagged(stub_engine, synthetic_page):
    """The low-confidence observation line is flagged for review."""
    pipeline = NotebookPipeline(engine=stub_engine)
    result = pipeline.run(synthetic_page, page_id="p")
    low = [l for l in result.transcript if l.confidence < 0.6]
    assert low and all(l.needs_review for l in low)
