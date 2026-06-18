"""Generic post-OCR text cleanup (§4.3 fallback, §8).

Conservative, domain-agnostic normalization applied to every line before the
symbol/chemistry stages do their specialized repairs. Each transformation that
fires is recorded so the correction trail is preserved for provenance (§9).
Anything chemistry- or symbol-specific deliberately lives in those stages, not
here, to keep changes surgical.
"""

from __future__ import annotations

import re
import unicodedata

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# OCR frequently splits a decimal point from its digits: "0 . 5" -> "0.5".
_SPACED_DECIMAL_RE = re.compile(r"(\d)\s*\.\s*(\d)")


def clean_text(text: str) -> tuple[str, list[str]]:
    """Normalize one OCR line; return ``(clean_text, corrections_applied)``.

    Steps: Unicode NFC normalization, control-char stripping, decimal-point
    rejoining, and whitespace collapsing. Only steps that change the string are
    reported.
    """
    corrections: list[str] = []
    original = text

    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        corrections.append("unicode_nfc")
    text = normalized

    stripped = _CONTROL_RE.sub("", text)
    if stripped != text:
        corrections.append("strip_control_chars")
    text = stripped

    rejoined = _SPACED_DECIMAL_RE.sub(r"\1.\2", text)
    if rejoined != text:
        corrections.append("rejoin_decimal")
    text = rejoined

    collapsed = _MULTISPACE_RE.sub(" ", text).strip()
    if collapsed != text:
        corrections.append("collapse_whitespace")
    text = collapsed

    if text == original:
        return text, []
    return text, corrections
