"""Experiment-level semantic synthesis (§4.6).

Assembles the :class:`~notebook_parser.types.ExperimentSection` from
evidence-linked classifications. This is the constrained summarizer: it only
emits statements backed by transcript evidence, satisfying the constraint that
the summary must not invent chemistry details (§11). A learned/LLM summarizer
could replace this while keeping the same evidence-only contract.
"""

from __future__ import annotations

from ..types import EvidenceItem, ExperimentSection, TranscriptLine
from .evidence import classify_lines, find_goal


def _dedupe_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Remove items with identical text, unioning evidence and keeping max conf."""
    by_text: dict[str, EvidenceItem] = {}
    for it in items:
        existing = by_text.get(it.text)
        if existing is None:
            by_text[it.text] = it.model_copy(deep=True)
        else:
            existing.evidence = sorted(set(existing.evidence) | set(it.evidence))
            existing.confidence = max(existing.confidence, it.confidence)
            existing.inferred = existing.inferred and it.inferred
    return list(by_text.values())


def synthesize_experiment(lines: list[TranscriptLine]) -> ExperimentSection:
    """Distill transcript lines into a structured, evidence-linked summary."""
    goal_item = find_goal(lines)
    classified = classify_lines(lines)
    return ExperimentSection(
        goal=goal_item.text if goal_item else None,
        conditions=_dedupe_items(classified["conditions"]),
        procedure=_dedupe_items(classified["procedure"]),
        observations=_dedupe_items(classified["observations"]),
        results=_dedupe_items(classified["results"]),
    )
