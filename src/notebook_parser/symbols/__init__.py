"""Scientific symbol and unit recovery stage (§4.4).

Exposes :func:`recover_symbols`, which normalizes every transcript line in place
(repairing OCR-damaged scientific tokens and recording the corrections) and
returns the aggregated list of recovered symbol tokens for the output schema.
"""

from __future__ import annotations

from ..types import SymbolToken, TranscriptLine
from .normalize import normalize_text
from .patterns import to_superscript

__all__ = ["recover_symbols", "normalize_text", "to_superscript"]


def recover_symbols(lines: list[TranscriptLine]) -> list[SymbolToken]:
    """Normalize symbols across all lines and return recovered tokens.

    Mutates each line's ``text`` to its normalized form and appends the applied
    correction names to ``line.corrections`` so the repair trail is preserved.
    """
    all_tokens: list[SymbolToken] = []
    for line in lines:
        if not line.text:
            continue
        normalized, tokens, corrections = normalize_text(line.text, line.id)
        line.text = normalized
        if corrections:
            line.corrections.extend(corrections)
        all_tokens.extend(tokens)
    return all_tokens
