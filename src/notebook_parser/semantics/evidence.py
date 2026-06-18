"""Evidence-linked classification of transcript lines (§4.6, §11).

The semantic layer is *constrained*: it never invents content. It classifies
existing transcript lines into experiment categories using keyword cues and links
each resulting statement back to the line(s) it came from. Because statements
quote observed text, they are marked ``inferred=False``; only synthesized
conclusions (e.g. a fallback goal) are marked inferred.
"""

from __future__ import annotations

import re

from ..types import EvidenceItem, TranscriptLine

# Cue words per category. Word-boundary matched, case-insensitive. Kept compact
# and auditable; this is the "constrained" knowledge the summarizer may use.
_CUES: dict[str, list[str]] = {
    "conditions": [
        "reflux", "heat", "cool", "ice", "rt", "room temperature", "pressure",
        "ph", "catalyst", "inert", "nitrogen", "argon", "vacuum", "\u0394",
    ],
    "procedure": [
        "add", "added", "stir", "stirred", "mix", "mixed", "heat", "heated",
        "cool", "cooled", "filter", "filtered", "wash", "washed", "dry",
        "dried", "dissolve", "dissolved", "weigh", "weighed", "transfer",
        "distill", "extract", "extracted", "combine", "pour", "neutralize",
    ],
    "observations": [
        "observed", "observe", "color", "colour", "turned", "precipitate",
        "ppt", "bubbles", "gas", "evolved", "smell", "odor", "cloudy", "clear",
        "solid", "crystal",
    ],
    "results": [
        "yield", "result", "product", "obtained", "recovered", "purity",
        "mp", "melting point", "bp", "rf", "conversion", "%",
    ],
}
_GOAL_RE = re.compile(
    r"\b(?:goal|objective|aim|purpose|hypothesis)\b\s*[:\-]?\s*(?P<rest>.*)",
    re.IGNORECASE,
)

# Temperature/time patterns strengthen the "conditions" category.
_TEMP_RE = re.compile(r"\d+\s*\u00b0C")
_TIME_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:min|h|hr|hrs|s|sec)\b")

# Precompiled per-category cue matchers (longest cue first for stable matching).
_CUE_RES: dict[str, re.Pattern] = {
    cat: re.compile(
        r"(?<![A-Za-z])(?:"
        + "|".join(re.escape(c) for c in sorted(words, key=len, reverse=True))
        + r")(?![A-Za-z])",
        re.IGNORECASE,
    )
    for cat, words in _CUES.items()
}


def _item_confidence(line: TranscriptLine, base: float) -> float:
    """Blend a category base confidence with the line's recognition confidence."""
    # When recognition confidence is 0 (no model), keep a small floor so the item
    # is still emitted but clearly low-confidence for human review.
    visual = line.confidence if line.confidence > 0 else 0.2
    return round(min(1.0, 0.5 * base + 0.5 * visual), 3)


def find_goal(lines: list[TranscriptLine]) -> EvidenceItem | None:
    """Return the experiment goal as an evidence-linked item, if stated."""
    for line in lines:
        m = _GOAL_RE.search(line.text)
        if m:
            rest = m.group("rest").strip()
            text = rest if rest else line.text.strip()
            return EvidenceItem(
                text=text,
                evidence=[line.id],
                inferred=False,
                confidence=_item_confidence(line, 0.7),
            )
    return None


def classify_lines(lines: list[TranscriptLine]) -> dict[str, list[EvidenceItem]]:
    """Classify lines into condition/procedure/observation/result evidence items.

    A line may contribute to multiple categories (e.g. an action performed at a
    given temperature). Temperature/time patterns reinforce ``conditions``.
    """
    out: dict[str, list[EvidenceItem]] = {
        "conditions": [],
        "procedure": [],
        "observations": [],
        "results": [],
    }
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        for category, matcher in _CUE_RES.items():
            if matcher.search(text):
                out[category].append(
                    EvidenceItem(
                        text=text,
                        evidence=[line.id],
                        inferred=False,
                        confidence=_item_confidence(line, 0.6),
                    )
                )
        # Reinforce conditions from explicit temperature/time even without a cue.
        if (_TEMP_RE.search(text) or _TIME_RE.search(text)) and not any(
            it.evidence == [line.id] for it in out["conditions"]
        ):
            out["conditions"].append(
                EvidenceItem(
                    text=text,
                    evidence=[line.id],
                    inferred=False,
                    confidence=_item_confidence(line, 0.55),
                )
            )
    return out
