"""Chemistry pass: reagents, formulas, structures, concentrations (§4.4).

Sends the image + transcript to the model, maps the structured response into the
canonical chemistry section, and (optionally) canonicalizes SMILES with RDKit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..imaging import ConditionedImage
from ..llm import prompts
from ..llm.client import LLMEngine
from ..llm.schemas import ChemistryResponse
from ..types import (
    ChemicalStructure,
    ChemistrySection,
    Concentration,
    Formula,
    Quantity,
    Reagent,
    ReagentRole,
    TranscriptLine,
)


def build_transcript_block(lines: list[TranscriptLine]) -> str:
    """Render transcript lines as a compact ``line_id: text`` evidence block."""
    return "\n".join(f"{l.id}: {l.text}" for l in lines if l.text) or "(no transcript)"


@lru_cache(maxsize=1)
def _get_rdkit() -> Any | None:
    """Return ``rdkit.Chem`` if importable, else ``None`` (cached)."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception:  # pragma: no cover - exercised only without rdkit
        return None


def canonical_smiles(smiles: str) -> tuple[str | None, bool | None]:
    """Canonicalize SMILES with RDKit; return (canonical, valid).

    Without RDKit returns ``(smiles, None)``; an unparsable SMILES yields
    ``(smiles, False)``.
    """
    chem = _get_rdkit()
    if chem is None:
        return smiles, None
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False
    return chem.MolToSmiles(mol), True


def _map_quantity(q) -> Quantity | None:
    """Map a structured quantity (or None) to the canonical model."""
    if q is None:
        return None
    return Quantity(value=q.value, unit=q.unit, raw=q.raw)


def _map_concentration(c) -> Concentration | None:
    """Map a structured concentration (or None) to the canonical model."""
    if c is None:
        return None
    return Concentration(
        value=c.value, unit=c.unit, species=c.species, raw=c.raw,
        evidence=list(c.evidence), confidence=c.confidence,
    )


def run_chemistry(
    engine: LLMEngine,
    image: ConditionedImage,
    lines: list[TranscriptLine],
    canonicalize: bool = True,
) -> ChemistrySection:
    """Run the chemistry pass and return a canonical chemistry section."""
    resp: ChemistryResponse = engine.extract(
        system=prompts.CHEMISTRY_SYSTEM,
        user=prompts.chemistry_user(build_transcript_block(lines)),
        schema=ChemistryResponse,
        image_data_url=image.data_url,
    )

    reagents = [
        Reagent(
            name=r.name,
            normalized_name=r.normalized_name,
            role=ReagentRole(r.role.value),
            quantity=_map_quantity(r.quantity),
            concentration=_map_concentration(r.concentration),
            evidence=list(r.evidence),
            confidence=r.confidence,
        )
        for r in resp.reagents
    ]

    formulas = [
        Formula(
            raw=f.raw, normalized=f.normalized, valid=f.valid,
            evidence=list(f.evidence), confidence=f.confidence,
        )
        for f in resp.formulas
    ]

    structures: list[ChemicalStructure] = []
    for s in resp.structures:
        smiles = s.smiles
        valid: bool | None = None
        if smiles and canonicalize:
            smiles, valid = canonical_smiles(smiles)
        structures.append(
            ChemicalStructure(
                region_id=s.region_id,
                name=s.name,
                smiles=smiles,
                confidence=s.confidence,
                uncertain=s.uncertain or valid is False,
                notes=(s.notes or "") + ("; invalid SMILES" if valid is False else "") or None,
            )
        )

    concentrations = [_map_concentration(c) for c in resp.concentrations]
    concentrations = [c for c in concentrations if c is not None]

    return ChemistrySection(
        reagents=reagents,
        formulas=formulas,
        structures=structures,
        concentrations=concentrations,
    )
