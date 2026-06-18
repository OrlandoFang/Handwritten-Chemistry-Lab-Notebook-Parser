"""Tests for confidence scoring and schema validation (§9, §11)."""

from __future__ import annotations

import pytest

from notebook_parser.config import ConfidenceConfig
from notebook_parser.types import (
    BoundingBox,
    LayoutRegion,
    ParseResult,
    RegionType,
    TranscriptLine,
)
from notebook_parser.validation import (
    SchemaError,
    apply_review_flags,
    check_output_dict,
    compute_confidence,
    validate_result,
)


def _result() -> ParseResult:
    """A small but schema-complete result for validation tests."""
    return ParseResult(
        page_id="p1",
        layout=[
            LayoutRegion(id="r0", type=RegionType.TEXT, bbox=BoundingBox(x=0, y=0, width=10, height=10), confidence=0.8)
        ],
        transcript=[
            TranscriptLine(id="r0_l0", region_id="r0", text="hi", confidence=0.9),
            TranscriptLine(id="r0_l1", region_id="r0", text="lo", confidence=0.3),
        ],
    )


def test_compute_confidence_weighted_and_per_field():
    """Overall confidence is in range and excludes empty fields."""
    report = compute_confidence(_result())
    assert 0.0 <= report.overall <= 1.0
    assert "transcript" in report.fields and "layout" in report.fields
    assert "chemistry" not in report.fields


def test_apply_review_flags_marks_low_confidence():
    """Lines below the medium threshold are flagged for human review."""
    result = apply_review_flags(_result(), ConfidenceConfig())
    flags = {l.id: l.needs_review for l in result.transcript}
    assert flags["r0_l0"] is False
    assert flags["r0_l1"] is True


def test_validate_result_roundtrip():
    """A well-formed result validates and round-trips through the schema."""
    validated = validate_result(_result())
    data = validated.to_dict()
    assert set(data) >= {
        "page_id", "document_type", "layout", "transcript", "symbols",
        "chemistry", "experiment", "confidence",
    }


def test_check_output_dict_rejects_bad_payload():
    """A payload with a wrong-typed field raises SchemaError."""
    with pytest.raises(SchemaError):
        check_output_dict({"page_id": "p1", "layout": "not-a-list"})
