"""Tests for the Markdown report renderer."""

from __future__ import annotations

from notebook_parser.pipeline import NotebookPipeline
from notebook_parser.report import render_markdown, render_markdown_from_dict


def _result(stub_engine, page):
    """Run the pipeline with the stub engine to get a populated result."""
    return NotebookPipeline(engine=stub_engine).run(page, page_id="page_57")


def test_markdown_has_all_sections(stub_engine, synthetic_page):
    """The report contains the transcription, chemistry, and experiment sections."""
    md = render_markdown(_result(stub_engine, synthetic_page))
    assert "## Human-readable transcription" in md
    assert "## Chemistry" in md
    assert "### Hand-drawn structures" in md
    assert "### Reagents" in md
    assert "### Concentrations" in md
    assert "## What was happening in the experiment" in md
    assert "### Goal" in md and "### Procedure" in md


def test_markdown_renders_transcript_text(stub_engine, synthetic_page):
    """Transcribed lines appear verbatim in the transcription section."""
    md = render_markdown(_result(stub_engine, synthetic_page))
    assert "Project: Li electrodeposition" in md
    assert "1M LiTFSI in diglyme" in md


def test_markdown_shows_chemistry_details(stub_engine, synthetic_page):
    """Structures, reagents, and concentrations are shown with detail."""
    md = render_markdown(_result(stub_engine, synthetic_page))
    assert "12-crown-4" in md
    assert "SMILES `C1COCCOCCOCCO1`" in md  # canonicalized form
    assert "LiTFSI" in md
    assert "1 M" in md  # concentration formatted


def test_markdown_shows_experiment(stub_engine, synthetic_page):
    """Goal and procedure narrative are rendered."""
    md = render_markdown(_result(stub_engine, synthetic_page))
    assert "stable Li plating" in md
    assert "Stir 20 min" in md


def test_render_from_dict_matches(stub_engine, synthetic_page):
    """Rendering from a serialized dict matches rendering from the model."""
    result = _result(stub_engine, synthetic_page)
    assert render_markdown_from_dict(result.to_dict()) == render_markdown(result)
