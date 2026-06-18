"""Validation stage (§7, §9).

Exposes confidence scoring/human-review routing and schema enforcement used as
the final step of the pipeline.
"""

from .confidence import apply_review_flags, compute_confidence
from .schema import SchemaError, check_output_dict, validate_result

__all__ = [
    "compute_confidence",
    "apply_review_flags",
    "validate_result",
    "check_output_dict",
    "SchemaError",
]
