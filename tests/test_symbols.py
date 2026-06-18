"""Tests for scientific symbol/unit recovery (§4.4, §10)."""

from __future__ import annotations

import pytest

from notebook_parser.symbols import recover_symbols
from notebook_parser.symbols.normalize import normalize_text
from notebook_parser.symbols.patterns import to_superscript
from notebook_parser.types import TranscriptLine


@pytest.mark.parametrize(
    "raw, expected_fragment",
    [
        ("heat to 100 oC", "100 \u00b0C"),
        ("add 10 uL", "10 \u03bcL"),
        ("1.2 x10^-3 mol", "1.2 \u00d710\u207b\u00b3 mol"),
        ("lambda max 450 nm", "\u03bbmax 450 nm"),
        ("theta angle", "\u03b8 angle"),
    ],
)
def test_normalize_text_repairs(raw, expected_fragment):
    """Common OCR symbol confusions are repaired in the normalized text."""
    text, _tokens, _corr = normalize_text(raw, "L1")
    assert expected_fragment in text


def test_normalize_text_emits_unit_tokens():
    """Canonical units present in the text are catalogued as symbol tokens."""
    _text, tokens, _ = normalize_text("dissolve 5 mg in 2 mL water", "L1")
    units = {t.normalized for t in tokens}
    assert "mg" in units and "mL" in units


def test_to_superscript():
    """Exponent rendering uses Unicode superscript glyphs."""
    assert to_superscript("-3") == "\u207b\u00b3"
    assert to_superscript("12") == "\u00b9\u00b2"


def test_recover_symbols_mutates_lines_and_records_corrections():
    """recover_symbols normalizes line text and logs the corrections applied."""
    lines = [TranscriptLine(id="L1", region_id="r0", text="heat to 100 oC", confidence=0.9)]
    tokens = recover_symbols(lines)
    assert lines[0].text == "heat to 100 \u00b0C"
    assert any(c.startswith("symbol:") for c in lines[0].corrections)
    assert tokens  # at least the celsius repair token
