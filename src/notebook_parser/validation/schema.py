"""Output schema enforcement (§2, §7 validation/).

Provides a single entry point to validate that a result conforms to the typed
schema before it leaves the pipeline. Validation is via Pydantic round-trip,
which guarantees the emitted JSON matches the documented structure exactly.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..types import ParseResult

# The required top-level keys of the output JSON (§2).
REQUIRED_TOP_LEVEL_KEYS = {
    "page_id",
    "document_type",
    "layout",
    "transcript",
    "symbols",
    "chemistry",
    "experiment",
    "confidence",
}


def validate_result(result: ParseResult) -> ParseResult:
    """Round-trip a result through the schema, raising on any inconsistency.

    Returns a validated :class:`ParseResult`. A :class:`SchemaError` is raised if
    the data does not conform (e.g. a stage produced an out-of-contract value).
    """
    try:
        data = result.model_dump(mode="json")
        validated = ParseResult.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - defensive
        raise SchemaError(str(exc)) from exc

    missing = REQUIRED_TOP_LEVEL_KEYS - set(validated.model_dump().keys())
    if missing:  # pragma: no cover - schema guarantees presence
        raise SchemaError(f"missing top-level keys: {sorted(missing)}")
    return validated


def check_output_dict(data: dict[str, Any]) -> None:
    """Validate an already-serialized dict against the schema (raises on error)."""
    try:
        ParseResult.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(str(exc)) from exc


class SchemaError(ValueError):
    """Raised when a result violates the output schema."""
