"""Tests for chemistry extraction and normalization (§4.5, §8, §10)."""

from __future__ import annotations

from notebook_parser.chemistry import canonical_smiles, extract_chemistry, normalize_chemistry
from notebook_parser.chemistry.extract import (
    extract_reagents,
    parse_concentrations,
    parse_formulas,
)
from notebook_parser.types import (
    ChemistrySection,
    Formula,
    Reagent,
    ReagentRole,
    TranscriptLine,
)


def test_parse_formulas_accepts_valid_and_rejects_words():
    """Valid formulas are extracted; English words/single elements are not."""
    found = {f.raw for f in parse_formulas("mix H2SO4 with NaCl in H2O", "L1")}
    assert {"H2SO4", "NaCl", "H2O"} <= found
    # "Add" and the element-symbol word "He" must not be treated as formulas.
    none = parse_formulas("Add the He balloon and stir", "L2")
    assert none == []


def test_parse_concentrations_extracts_value_unit_species():
    """A concentration like '0.5 M HCl' parses into structured fields."""
    concs = parse_concentrations("titrate with 0.5 M HCl", "L1")
    assert len(concs) == 1
    c = concs[0]
    assert c.value == 0.5 and c.unit == "M" and c.species == "HCl"


def test_parse_concentration_ignores_bare_percent_yield():
    """A bare percentage (yield) is not parsed as a concentration."""
    assert parse_concentrations("Yield: 75%", "L1") == []


def test_extract_reagents_associates_quantity():
    """A reagent is linked to a nearby quantity on the same line (§8)."""
    line = TranscriptLine(id="L1", region_id="r0", text="add 10 mL water", confidence=0.9)
    reagents = extract_reagents(line)
    names = {r.name.lower() for r in reagents}
    assert "water" in names
    water = next(r for r in reagents if r.name.lower() == "water")
    assert water.role == ReagentRole.SOLVENT
    assert water.quantity is not None and water.quantity.unit == "mL"


def test_normalize_chemistry_dedupes_formulas():
    """Duplicate formulas merge and union their evidence."""
    section = ChemistrySection(
        formulas=[
            Formula(raw="H2O", normalized="H2O", evidence=["L1"], confidence=0.5),
            Formula(raw="H2O", normalized="H2O", evidence=["L2"], confidence=0.7),
        ]
    )
    out = normalize_chemistry(section)
    assert len(out.formulas) == 1
    assert out.formulas[0].evidence == ["L1", "L2"]
    assert out.formulas[0].confidence == 0.7


def test_canonical_smiles_roundtrip():
    """SMILES canonicalization works when RDKit is available, else passes through."""
    canon, valid = canonical_smiles("OCC")
    assert canon is not None
    if valid is True:  # rdkit present
        assert canon == "CCO"


def test_extract_chemistry_text_only(sample_lines):
    """Text-only chemistry extraction returns a populated, normalized section."""
    lines = [
        TranscriptLine(id=f"L{i}", region_id="r0", text=t, confidence=0.9)
        for i, t in enumerate(sample_lines)
    ]
    section = extract_chemistry(lines)
    assert any(c.unit == "M" for c in section.concentrations)
