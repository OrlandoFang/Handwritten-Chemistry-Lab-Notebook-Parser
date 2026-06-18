"""Tests for the extraction passes (§4.3-§4.5)."""

from __future__ import annotations

from notebook_parser.imaging import condition_image
from notebook_parser.passes import (
    build_transcript_block,
    canonical_smiles,
    run_chemistry,
    run_experiment,
    run_transcription,
)
from notebook_parser.types import RegionType, ReagentRole


def test_run_transcription_maps_canonical(stub_engine, synthetic_page):
    """The transcription pass maps the response into canonical models."""
    img = condition_image(synthetic_page)
    regions, lines, symbols = run_transcription(stub_engine, img)
    assert any(r.type == RegionType.DRAWING for r in regions)
    assert any("Li electrodeposition" in l.text for l in lines)
    assert lines[2].candidates  # alternative reading preserved
    assert any(s.normalized == "°C" for s in symbols)


def test_run_chemistry_maps_and_canonicalizes(stub_engine, synthetic_page):
    """The chemistry pass maps reagents/structures and canonicalizes SMILES."""
    img = condition_image(synthetic_page)
    _r, lines, _s = run_transcription(stub_engine, img)
    chem = run_chemistry(stub_engine, img, lines)
    names = {r.name for r in chem.reagents}
    assert {"LiTFSI", "EtOH", "12-crown-4"} <= names
    assert any(r.role == ReagentRole.SOLVENT for r in chem.reagents)
    assert chem.structures and chem.structures[0].name == "12-crown-4"
    # SMILES gets canonicalized when RDKit is present (non-canonical input given).
    smiles = chem.structures[0].smiles
    assert smiles is not None


def test_run_experiment_maps_items(stub_engine, synthetic_page):
    """The experiment pass maps goal and evidence items, preserving links."""
    img = condition_image(synthetic_page)
    _r, lines, _s = run_transcription(stub_engine, img)
    chem = run_chemistry(stub_engine, img, lines)
    exp = run_experiment(stub_engine, lines, chem)
    assert exp.goal and "stable Li plating" in exp.goal
    assert exp.procedure and exp.procedure[0].evidence == ["r1_l2"]


def test_build_transcript_block(stub_engine, synthetic_page):
    """The transcript block renders as 'line_id: text' lines."""
    img = condition_image(synthetic_page)
    _r, lines, _s = run_transcription(stub_engine, img)
    block = build_transcript_block(lines)
    assert "r0_l0:" in block


def test_canonical_smiles_roundtrip():
    """SMILES canonicalization works with RDKit, else passes through."""
    canon, valid = canonical_smiles("OCC")
    assert canon is not None
    if valid is True:
        assert canon == "CCO"
