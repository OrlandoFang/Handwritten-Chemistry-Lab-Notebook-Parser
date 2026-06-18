"""Pattern tables for scientific symbol/unit recovery (§4.4).

Centralizes the rule-based knowledge: OCR-confusion repairs, the canonical unit
vocabulary, and helpers for scientific-notation formatting. Rules are ordered and
applied sequentially by :mod:`notebook_parser.symbols.normalize`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import SymbolKind

# Unicode superscript/subscript digit maps for scientific-notation rendering.
_SUPERSCRIPT = str.maketrans("0123456789+-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b")


def to_superscript(exponent: str) -> str:
    """Render an exponent string (digits and sign) using Unicode superscripts."""
    return exponent.translate(_SUPERSCRIPT)


@dataclass(frozen=True)
class RepairRule:
    """A single regex-based symbol repair.

    ``pattern`` is matched against the line; ``repl`` may be a string or a
    callable (for dynamic replacements like superscripting an exponent).
    ``kind`` classifies the produced token and ``name`` documents the rule.
    """

    pattern: re.Pattern
    repl: object  # str | Callable[[re.Match], str]
    kind: SymbolKind
    name: str


def _scientific_repl(match: re.Match) -> str:
    """Normalize ``x10^-3`` style notation to ``×10⁻³`` with a real × sign."""
    exponent = match.group("exp")
    return "\u00d710" + to_superscript(exponent)


# Ordered repair rules. Earlier rules run first; later rules see their output.
REPAIR_RULES: list[RepairRule] = [
    # Scientific notation: 1.2 x 10^-3 / 1.2*10 -3 -> 1.2×10⁻³
    RepairRule(
        re.compile(r"(?:x|X|\*|\u00d7)\s*10\s*\^?\s*(?P<exp>[-+]?\d+)"),
        _scientific_repl,
        SymbolKind.NOTATION,
        "scientific_notation",
    ),
    # Degrees Celsius: 100 oC / 100 deg C / 100°C -> 100 °C
    RepairRule(
        re.compile(r"(?P<n>\d)\s*(?:deg(?:rees?)?|[oO0]|\u00b0)\s*C\b"),
        lambda m: f"{m.group('n')} \u00b0C",
        SymbolKind.UNIT,
        "celsius",
    ),
    # Bare degree word: 45 deg -> 45 °
    RepairRule(
        re.compile(r"(?P<n>\d)\s*deg(?:rees?)?\b"),
        lambda m: f"{m.group('n')}\u00b0",
        SymbolKind.SYMBOL,
        "degree",
    ),
    # Micro prefix confusions: uL/um/ug/uM/umol -> μL/μm/μg/μM/μmol
    RepairRule(
        re.compile(r"(?<![A-Za-z])u(?P<u>L|m|g|M|mol|l)\b"),
        lambda m: "\u03bc" + m.group("u"),
        SymbolKind.UNIT,
        "micro_prefix",
    ),
    # lambda max -> λmax
    RepairRule(
        re.compile(r"\blambda\s*max\b", re.IGNORECASE),
        "\u03bbmax",
        SymbolKind.SYMBOL,
        "lambda_max",
    ),
]

# Greek spelled-out -> letter, applied as whole words only (domain-appropriate).
GREEK_WORDS: dict[str, str] = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "theta": "\u03b8",
    "lambda": "\u03bb",
    "mu": "\u03bc",
    "omega": "\u03c9",
}
GREEK_WORD_RE = re.compile(r"\b(" + "|".join(GREEK_WORDS) + r")\b", re.IGNORECASE)

# Canonical scientific units recognized for the symbols output list.
CANONICAL_UNITS = [
    "\u00b0C", "K", "mL", "L", "\u03bcL", "\u03bcl", "mg", "kg", "g", "ng",
    "\u03bcg", "mmol", "mol", "mM", "\u03bcM", "nM", "M", "N", "nm", "cm",
    "mm", "\u03bcm", "min", "rpm", "Hz", "mbar", "atm", "ppm", "eq",
]
# Match the longest unit first so e.g. ``mL`` is not split into ``m`` + ``L``.
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(CANONICAL_UNITS, key=len, reverse=True))
UNIT_RE = re.compile(r"(?<![A-Za-z])(?:" + _UNIT_ALT + r")(?![A-Za-z])")
