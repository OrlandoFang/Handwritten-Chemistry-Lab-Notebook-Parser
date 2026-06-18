"""Candidate decoding and language-model rescoring (§4.3).

A full system would rescore beam-search hypotheses with a neural LM. Here we use
a lightweight, deterministic lexicon prior over lab/chemistry vocabulary: it
reorders candidates and nudges confidence toward hypotheses that look like real
lab language, without inventing tokens. This keeps the rescoring hook real and
swappable.
"""

from __future__ import annotations

import re

from ..config import OCRConfig
from .recognize import RecognitionOutput

# Small, domain-relevant lexicon used as a cheap language prior. Intentionally
# compact; a production system would load a learned LM here.
_LAB_LEXICON = {
    "add", "added", "stir", "stirred", "heat", "heated", "cool", "cooled",
    "mix", "mixture", "solution", "reflux", "filter", "filtered", "wash",
    "washed", "dry", "dried", "yield", "product", "reaction", "observed",
    "color", "colour", "precipitate", "ppt", "aqueous", "organic", "layer",
    "goal", "objective", "procedure", "result", "results", "observation",
    "conditions", "temperature", "weigh", "weighed", "dissolve", "dissolved",
    "acid", "base", "salt", "water", "ethanol", "methanol", "acetone",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_NUMBER_RE = re.compile(r"\d")


def language_model_score(text: str) -> float:
    """Return a [0,1] plausibility score for a line of lab text.

    Combines lexicon coverage with a small bonus for containing numeric data
    (measurements are pervasive in lab notes). Empty text scores 0.
    """
    if not text.strip():
        return 0.0
    words = [w.lower() for w in _TOKEN_RE.findall(text)]
    if not words:
        # No alphabetic words (e.g. pure measurements): mildly plausible.
        return 0.4 if _NUMBER_RE.search(text) else 0.2
    in_lex = sum(1 for w in words if w in _LAB_LEXICON)
    coverage = in_lex / len(words)
    number_bonus = 0.1 if _NUMBER_RE.search(text) else 0.0
    return min(1.0, 0.3 + 0.6 * coverage + number_bonus)


def rescore(output: RecognitionOutput, config: OCRConfig | None = None) -> RecognitionOutput:
    """Reorder candidates and blend LM prior into the confidence.

    The recognizer's acoustic/visual confidence is combined with the lexicon
    prior so a visually plausible but linguistically implausible hypothesis is
    down-weighted. Returns a new :class:`RecognitionOutput`; the input is left
    untouched.
    """
    config = config or OCRConfig()
    candidates = list(output.candidates) or ([(output.text, output.confidence)] if output.text else [])
    if not config.language_model_rescoring or not candidates:
        return output

    # Blend visual score with LM prior (equal weight) and sort descending.
    scored = sorted(
        ((t, 0.5 * s + 0.5 * language_model_score(t)) for t, s in candidates),
        key=lambda ts: ts[1],
        reverse=True,
    )
    best_text, best_score = scored[0]
    trimmed = scored[: max(1, config.max_alternatives)]
    return RecognitionOutput(text=best_text, confidence=best_score, candidates=trimmed)
