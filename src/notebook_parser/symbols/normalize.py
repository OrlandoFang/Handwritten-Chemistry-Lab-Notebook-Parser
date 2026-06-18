"""Apply symbol/unit recovery to transcript lines (§4.4).

Runs the ordered repair rules, then catalogs canonical units and Greek symbols
present in the (repaired) text. Returns normalized text plus the
:class:`~notebook_parser.types.SymbolToken` list, with each repair recorded for
provenance. Rule-based repair runs first; a learned corrector could be slotted in
afterward for the ambiguous remainder.
"""

from __future__ import annotations

import re

from ..types import SymbolKind, SymbolToken
from .patterns import GREEK_WORD_RE, GREEK_WORDS, REPAIR_RULES, UNIT_RE


def normalize_text(text: str, line_id: str | None = None) -> tuple[str, list[SymbolToken], list[str]]:
    """Repair and catalog scientific symbols in a single line.

    Returns ``(normalized_text, tokens, corrections)``. ``tokens`` includes both
    repaired confusions (raw != normalized) and canonical symbols/units detected
    in the final text; duplicates are removed while preserving order.
    """
    corrections: list[str] = []
    tokens: list[SymbolToken] = []

    # 1) Sequential rule-based repairs; capture each match as a token.
    for rule in REPAIR_RULES:
        def _sub(match: re.Match, _rule=rule) -> str:
            replacement = _rule.repl(match) if callable(_rule.repl) else _rule.repl
            tokens.append(
                SymbolToken(
                    raw=match.group(0),
                    normalized=replacement,
                    kind=_rule.kind,
                    line_id=line_id,
                    confidence=0.8,
                )
            )
            return replacement

        new_text, n = rule.pattern.subn(_sub, text)
        if n:
            corrections.append(f"symbol:{rule.name}x{n}")
        text = new_text

    # 2) Greek spelled-out words -> letters.
    def _greek_sub(match: re.Match) -> str:
        word = match.group(0)
        letter = GREEK_WORDS[word.lower()]
        tokens.append(
            SymbolToken(
                raw=word, normalized=letter, kind=SymbolKind.SYMBOL,
                line_id=line_id, confidence=0.7,
            )
        )
        return letter

    new_text, n = GREEK_WORD_RE.subn(_greek_sub, text)
    if n:
        corrections.append(f"symbol:greek_wordx{n}")
    text = new_text

    # 3) Catalog canonical units already present (no text change).
    for match in UNIT_RE.finditer(text):
        unit = match.group(0)
        tokens.append(
            SymbolToken(
                raw=unit, normalized=unit, kind=SymbolKind.UNIT,
                line_id=line_id, confidence=0.9,
            )
        )

    return text, _dedupe(tokens), corrections


def _dedupe(tokens: list[SymbolToken]) -> list[SymbolToken]:
    """Drop duplicate tokens (same raw/normalized/kind/line) preserving order."""
    seen: set[tuple[str, str, str, str | None]] = set()
    unique: list[SymbolToken] = []
    for t in tokens:
        key = (t.raw, t.normalized, t.kind.value, t.line_id)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique
