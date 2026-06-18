"""Tests for experiment-level semantic synthesis (§4.6, §10, §11)."""

from __future__ import annotations

from notebook_parser.semantics import classify_lines, find_goal, synthesize_experiment
from notebook_parser.types import TranscriptLine


def _lines(texts: list[str]) -> list[TranscriptLine]:
    """Build transcript lines (confidence set) from raw strings."""
    return [
        TranscriptLine(id=f"L{i}", region_id="r0", text=t, confidence=0.9)
        for i, t in enumerate(texts)
    ]


def test_find_goal_extracts_objective():
    """A 'Goal:' line yields the goal text linked to its line."""
    item = find_goal(_lines(["Goal: synthesize aspirin", "add water"]))
    assert item is not None
    assert item.text == "synthesize aspirin"
    assert item.evidence == ["L0"]
    assert item.inferred is False


def test_classify_lines_categories():
    """Lines route to procedure/observation/result/condition by cue."""
    classified = classify_lines(
        _lines(
            [
                "Add 10 mL water and stir",
                "Observed white precipitate",
                "Yield: 75%",
                "Heat to 80 \u00b0C for 30 min",
            ]
        )
    )
    assert any("Add" in i.text for i in classified["procedure"])
    assert any("precipitate" in i.text for i in classified["observations"])
    assert any("Yield" in i.text for i in classified["results"])
    assert any("80 \u00b0C" in i.text for i in classified["conditions"])


def test_synthesize_experiment_only_uses_evidence():
    """Every emitted statement is backed by a transcript line (no invention)."""
    lines = _lines(
        ["Goal: make salt", "Add HCl to NaOH", "Observed bubbles", "Yield: 90%"]
    )
    exp = synthesize_experiment(lines)
    valid_ids = {l.id for l in lines}
    assert exp.goal == "make salt"
    for group in (exp.conditions, exp.procedure, exp.observations, exp.results):
        for item in group:
            assert item.evidence
            assert set(item.evidence) <= valid_ids
