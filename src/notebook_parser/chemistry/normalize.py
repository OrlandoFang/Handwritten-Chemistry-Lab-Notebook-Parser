"""Chemistry normalization and validation (§4.5, tooling: rdkit).

Deduplicates extracted chemistry and, when RDKit is available, canonicalizes any
recovered SMILES and validates them. RDKit is imported lazily so the package
works without it; without RDKit, SMILES are passed through and validity is left
unknown (``None``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..types import (
    ChemicalStructure,
    ChemistrySection,
    Concentration,
    Formula,
    Reagent,
)


@lru_cache(maxsize=1)
def _get_rdkit() -> Any | None:
    """Return ``rdkit.Chem`` if importable, else ``None`` (cached)."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")  # silence parse warnings
        return Chem
    except Exception:  # pragma: no cover - exercised only without rdkit
        return None


def canonical_smiles(smiles: str) -> tuple[str | None, bool | None]:
    """Canonicalize a SMILES string with RDKit.

    Returns ``(canonical_smiles, is_valid)``. Without RDKit returns
    ``(smiles, None)`` (unknown validity); on a parse failure returns
    ``(smiles, False)``.
    """
    chem = _get_rdkit()
    if chem is None:
        return smiles, None
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False
    return chem.MolToSmiles(mol), True


def _normalize_structure(structure: ChemicalStructure) -> ChemicalStructure:
    """Canonicalize a structure's SMILES if present; flag invalid parses."""
    if not structure.smiles:
        return structure
    canon, valid = canonical_smiles(structure.smiles)
    structure.smiles = canon
    if valid is False:
        structure.uncertain = True
        structure.notes = (structure.notes or "") + "; invalid SMILES"
    return structure


def _dedupe_formulas(formulas: list[Formula]) -> list[Formula]:
    """Merge formulas sharing a normalized form, unioning their evidence."""
    by_key: dict[str, Formula] = {}
    for f in formulas:
        existing = by_key.get(f.normalized)
        if existing is None:
            by_key[f.normalized] = f.model_copy(deep=True)
        else:
            existing.evidence = sorted(set(existing.evidence) | set(f.evidence))
            existing.confidence = max(existing.confidence, f.confidence)
    return list(by_key.values())


def _dedupe_reagents(reagents: list[Reagent]) -> list[Reagent]:
    """Merge reagents with the same identity, keeping the richest record."""
    by_key: dict[str, Reagent] = {}
    for r in reagents:
        key = (r.normalized_name or r.name).lower()
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = r.model_copy(deep=True)
            continue
        existing.evidence = sorted(set(existing.evidence) | set(r.evidence))
        existing.confidence = max(existing.confidence, r.confidence)
        # Prefer a record that has quantity/concentration information.
        if existing.quantity is None and r.quantity is not None:
            existing.quantity = r.quantity
        if existing.concentration is None and r.concentration is not None:
            existing.concentration = r.concentration
    return list(by_key.values())


def _dedupe_concentrations(concs: list[Concentration]) -> list[Concentration]:
    """Drop duplicate concentrations (same value/unit/species), union evidence."""
    by_key: dict[tuple, Concentration] = {}
    for c in concs:
        key = (c.value, c.unit, (c.species or "").lower())
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = c.model_copy(deep=True)
        else:
            existing.evidence = sorted(set(existing.evidence) | set(c.evidence))
    return list(by_key.values())


def normalize_chemistry(section: ChemistrySection) -> ChemistrySection:
    """Canonicalize and deduplicate a whole chemistry section deterministically."""
    return ChemistrySection(
        reagents=_dedupe_reagents(section.reagents),
        formulas=_dedupe_formulas(section.formulas),
        structures=[_normalize_structure(s) for s in section.structures],
        concentrations=_dedupe_concentrations(section.concentrations),
    )
