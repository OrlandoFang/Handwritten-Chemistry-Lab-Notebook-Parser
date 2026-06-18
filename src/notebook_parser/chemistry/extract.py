"""Chemistry text extraction: formulas, reagents, quantities, concentrations.

Operates on already symbol-normalized transcript lines (§4.5, §8). Everything is
rule-based and conservative: a candidate is only accepted as chemistry when it
decomposes into valid element symbols or matches the lab lexicon, so ordinary
English words are not mistyped as formulas. Each extraction links back to its
source line for provenance.
"""

from __future__ import annotations

import re

from ..types import (
    Concentration,
    Formula,
    Quantity,
    Reagent,
    ReagentRole,
    TranscriptLine,
)

# Periodic-table element symbols used to validate formula candidates.
_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn",
}

# Single/short element symbols that are also common English words; rejected as
# standalone formulas to avoid false positives.
_WORD_BLOCKLIST = {"I", "In", "He", "No", "At", "As", "Be", "Co", "Sn", "Po", "Os"}

# Common lab solvents and reagent abbreviations -> canonical names.
_SOLVENTS = {"water", "ethanol", "methanol", "acetone", "dcm", "thf", "dmso", "dmf", "etoac", "hexane", "toluene"}
_ABBREV = {
    "etoh": "ethanol", "meoh": "methanol", "dcm": "dichloromethane",
    "thf": "tetrahydrofuran", "dmso": "dimethyl sulfoxide",
    "dmf": "dimethylformamide", "etoac": "ethyl acetate", " accn": "acetonitrile",
    "mecn": "acetonitrile", "ac2o": "acetic anhydride",
}
# A few full reagent names recognized directly.
_REAGENT_NAMES = {
    "hcl", "naoh", "h2so4", "water", "ethanol", "methanol", "acetone",
    "acetic acid", "sodium chloride", "sodium hydroxide", "ammonia",
}

_NUMBER = r"\d+(?:\.\d+)?"
_QUANTITY_UNITS = [
    "\u00b0C", "mL", "L", "\u03bcL", "mg", "kg", "g", "\u03bcg", "ng",
    "mmol", "mol", "nm", "cm", "mm", "\u03bcm", "min", "rpm", "eq", "h", "hr", "s",
]
_QTY_ALT = "|".join(re.escape(u) for u in sorted(_QUANTITY_UNITS, key=len, reverse=True))
_QUANTITY_RE = re.compile(rf"(?P<val>{_NUMBER})\s*(?P<unit>{_QTY_ALT})(?![A-Za-z])")

# Bare "%" is intentionally excluded so yields like "75%" are not mistaken for
# concentrations; keep the explicit concentration-style percent units.
_CONC_UNITS = ["mM", "\u03bcM", "nM", "M", "N", "wt%", "w/v", "v/v", "ppm"]
_CONC_ALT = "|".join(re.escape(u) for u in sorted(_CONC_UNITS, key=len, reverse=True))
_CONC_RE = re.compile(
    rf"(?P<val>{_NUMBER})\s*(?P<unit>{_CONC_ALT})(?![A-Za-z])\s*(?P<species>[A-Z][A-Za-z0-9]*)?"
)

_FORMULA_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?\d*){1,}(?![a-z])")
_FORMULA_PART_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _decompose_formula(token: str) -> tuple[bool, int, bool]:
    """Validate a formula candidate.

    Returns ``(is_valid, n_distinct_elements, has_digit)``. Valid means the token
    is composed entirely of recognized element symbols with optional counts.
    """
    parts = _FORMULA_PART_RE.findall(token)
    if not parts or "".join(s + d for s, d in parts) != token:
        return False, 0, False
    elements = set()
    has_digit = False
    for sym, digits in parts:
        if sym not in _ELEMENTS:
            return False, 0, False
        elements.add(sym)
        if digits:
            has_digit = True
    return True, len(elements), has_digit


def parse_formulas(text: str, line_id: str) -> list[Formula]:
    """Extract well-formed chemical formulas from a line.

    Accepts a candidate only if it decomposes into valid elements and either
    contains a digit or names at least two elements, and is not an English word
    on the blocklist.
    """
    out: list[Formula] = []
    for match in _FORMULA_CANDIDATE_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2 or token in _WORD_BLOCKLIST:
            continue
        valid, n_elem, has_digit = _decompose_formula(token)
        if not valid or not (has_digit or n_elem >= 2):
            continue
        out.append(
            Formula(
                raw=token,
                normalized=token,
                valid=True,
                evidence=[line_id],
                confidence=0.7 if has_digit else 0.5,
            )
        )
    return out


def parse_quantities(text: str) -> list[tuple[Quantity, int]]:
    """Find ``value unit`` quantities; return each with its match start offset."""
    results: list[tuple[Quantity, int]] = []
    for m in _QUANTITY_RE.finditer(text):
        results.append(
            (
                Quantity(value=float(m.group("val")), unit=m.group("unit"), raw=m.group(0)),
                m.start(),
            )
        )
    return results


def parse_concentrations(text: str, line_id: str) -> list[Concentration]:
    """Extract concentration mentions like ``0.5 M HCl`` with optional species."""
    out: list[Concentration] = []
    for m in _CONC_RE.finditer(text):
        species = m.group("species") or None
        out.append(
            Concentration(
                value=float(m.group("val")),
                unit=m.group("unit"),
                species=species,
                raw=m.group(0).strip(),
                evidence=[line_id],
                confidence=0.75,
            )
        )
    return out


def _guess_role(name_lower: str) -> ReagentRole:
    """Heuristically assign a reagent role from its (lowercased) name."""
    if name_lower in _SOLVENTS:
        return ReagentRole.SOLVENT
    return ReagentRole.REAGENT


def extract_reagents(line: TranscriptLine) -> list[Reagent]:
    """Extract reagent mentions from a line, attaching nearby quantity/conc.

    Reagent candidates are formula tokens, recognized abbreviations, and known
    reagent names. The nearest preceding quantity (by character offset) on the
    same line is associated with the reagent, implementing the "associate reagent
    names with nearby quantities" rule (§8).
    """
    text = line.text
    if not text:
        return []
    quantities = parse_quantities(text)
    concentrations = parse_concentrations(text, line.id)
    reagents: list[Reagent] = []
    seen: set[str] = set()

    def _nearest_quantity(pos: int):
        """Return the quantity whose match is closest to ``pos`` on the line."""
        if not quantities:
            return None
        return min(quantities, key=lambda qp: abs(qp[1] - pos))[0]

    # Formula-based reagents.
    for f in parse_formulas(text, line.id):
        key = f.raw.lower()
        if key in seen:
            continue
        seen.add(key)
        pos = text.find(f.raw)
        reagents.append(
            Reagent(
                name=f.raw,
                normalized_name=_ABBREV.get(key, _REAGENT_NAMES_MAP.get(key)),
                role=_guess_role(key),
                quantity=_nearest_quantity(pos),
                concentration=concentrations[0] if concentrations else None,
                evidence=[line.id],
                confidence=0.6,
            )
        )

    # Lexicon / abbreviation based reagents.
    for word_match in re.finditer(r"[A-Za-z][A-Za-z0-9]*", text):
        word = word_match.group(0)
        key = word.lower()
        if key in seen:
            continue
        if key in _ABBREV or key in _REAGENT_NAMES or key in _SOLVENTS:
            seen.add(key)
            reagents.append(
                Reagent(
                    name=word,
                    normalized_name=_ABBREV.get(key, word.lower() if key in _REAGENT_NAMES else None),
                    role=_guess_role(key),
                    quantity=_nearest_quantity(word_match.start()),
                    concentration=concentrations[0] if concentrations else None,
                    evidence=[line.id],
                    confidence=0.65,
                )
            )
    return reagents


# Map known reagent names to a canonical lowercase form (identity for now).
_REAGENT_NAMES_MAP = {n.replace(" ", ""): n for n in _REAGENT_NAMES}


def extract_text_chemistry(
    lines: list[TranscriptLine],
) -> tuple[list[Reagent], list[Formula], list[Concentration]]:
    """Aggregate reagents, formulas, and concentrations across all lines."""
    reagents: list[Reagent] = []
    formulas: list[Formula] = []
    concentrations: list[Concentration] = []
    for line in lines:
        reagents.extend(extract_reagents(line))
        formulas.extend(parse_formulas(line.text, line.id))
        concentrations.extend(parse_concentrations(line.text, line.id))
    return reagents, formulas, concentrations
