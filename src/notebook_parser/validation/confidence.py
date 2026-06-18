"""Confidence aggregation and human-review routing (§9).

Computes a per-field confidence breakdown and a weighted overall score, and flags
low-confidence transcript lines for review. Empty fields are excluded from the
overall average so absent content does not unfairly depress (or inflate) the
score.
"""

from __future__ import annotations

from statistics import mean

from ..config import ConfidenceConfig
from ..types import ConfidenceReport, ParseResult

# Relative weights for the overall confidence; reflect evaluation emphasis (§13).
_FIELD_WEIGHTS = {
    "layout": 0.15,
    "transcript": 0.35,
    "symbols": 0.1,
    "chemistry": 0.2,
    "experiment": 0.2,
}


def _safe_mean(values: list[float]) -> float | None:
    """Mean of ``values`` rounded to 3 dp, or ``None`` if empty."""
    return round(mean(values), 3) if values else None


def compute_confidence(result: ParseResult) -> ConfidenceReport:
    """Compute per-field and overall confidence for a parse result.

    The overall score is the weighted average over fields that actually have
    content, with weights renormalized to those present.
    """
    chem = result.chemistry
    chem_scores = (
        [r.confidence for r in chem.reagents]
        + [f.confidence for f in chem.formulas]
        + [c.confidence for c in chem.concentrations]
        + [s.confidence for s in chem.structures]
    )
    exp = result.experiment
    exp_scores = [
        it.confidence
        for group in (exp.conditions, exp.procedure, exp.observations, exp.results)
        for it in group
    ]

    field_means: dict[str, float | None] = {
        "layout": _safe_mean([r.confidence for r in result.layout]),
        "transcript": _safe_mean([l.confidence for l in result.transcript]),
        "symbols": _safe_mean([t.confidence for t in result.symbols]),
        "chemistry": _safe_mean(chem_scores),
        "experiment": _safe_mean(exp_scores),
    }

    present = {k: v for k, v in field_means.items() if v is not None}
    if present:
        total_weight = sum(_FIELD_WEIGHTS[k] for k in present)
        overall = sum(_FIELD_WEIGHTS[k] * v for k, v in present.items()) / total_weight
        overall = round(overall, 3)
    else:
        overall = 0.0

    return ConfidenceReport(overall=overall, fields=present)


def apply_review_flags(result: ParseResult, config: ConfidenceConfig | None = None) -> ParseResult:
    """Set ``needs_review`` on transcript lines below the medium threshold."""
    config = config or ConfidenceConfig()
    for line in result.transcript:
        line.needs_review = config.band(line.confidence) == "low"
    return result
