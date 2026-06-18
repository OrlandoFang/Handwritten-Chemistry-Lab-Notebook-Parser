"""Experiment-level semantic synthesis stage (§4.6).

Exposes :func:`synthesize_experiment`, the constrained summarizer that turns
transcript lines into an evidence-linked experiment summary without inventing
content.
"""

from .evidence import classify_lines, find_goal
from .summarize import synthesize_experiment

__all__ = ["synthesize_experiment", "classify_lines", "find_goal"]
