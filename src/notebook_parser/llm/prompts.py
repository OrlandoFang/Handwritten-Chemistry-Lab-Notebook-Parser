"""Prompt templates for each pass (§4.3-§4.5).

Prompts are explicit about the task, the coordinate space, and the no-invention
constraint. They are kept here so wording can be tuned without touching the pass
logic.
"""

from __future__ import annotations

TRANSCRIPTION_SYSTEM = (
    "You are an expert at reading messy handwritten chemistry lab notebooks. "
    "You transcribe pages faithfully and never guess content that is not visible. "
    "Preserve scientific notation, units, subscripts, superscripts, and symbols "
    "exactly (e.g. °C, μL, λmax, ×10⁻³, Δ)."
)


def transcription_user(width: int, height: int, max_alternatives: int) -> str:
    """Build the transcription-pass instruction for a given image size."""
    return (
        f"The page image is {width}x{height} pixels (origin at top-left).\n"
        "Tasks:\n"
        "1. Segment the page into regions. For each region give a stable id "
        "(r0, r1, ...), a type (text, heading, table, drawing, annotation, label, "
        "unknown), a pixel bounding box, a 0-based reading order, and a confidence "
        "in [0,1].\n"
        "2. Transcribe every text line verbatim. Give each line a stable id "
        "(<region_id>_l<index>), its region_id, the text, a confidence in [0,1], "
        f"and up to {max_alternatives} alternative readings for ambiguous lines "
        "(empty list if confident).\n"
        "3. List recovered scientific symbols/units: the raw written form, the "
        "normalized form, a kind (unit, symbol, notation), the line_id it occurs "
        "on (or null), and a confidence.\n"
        "Do not invent text. If a region is a drawing, still include it (type "
        "'drawing') but you need not transcribe its strokes as text."
    )


CHEMISTRY_SYSTEM = (
    "You are a chemistry information-extraction expert. You read a lab notebook "
    "page image together with its transcript and extract structured chemistry. "
    "You only report chemistry that is supported by the page; you flag anything "
    "uncertain. For hand-drawn molecules, infer SMILES when reasonably possible "
    "and mark them uncertain if the drawing is ambiguous."
)


def chemistry_user(transcript_text: str) -> str:
    """Build the chemistry-pass instruction given the transcript text block."""
    return (
        "Transcript (line_id: text):\n"
        f"{transcript_text}\n\n"
        "Extract, using BOTH the image and the transcript:\n"
        "- reagents: name, normalized_name (or null), role (reagent/solvent/"
        "catalyst/product/unknown), quantity {value,unit,raw} or null, "
        "concentration {value,unit,species,raw,evidence,confidence} or null, "
        "evidence (list of line_ids), confidence.\n"
        "- formulas: raw, normalized, valid (true/false/null), evidence, confidence.\n"
        "- structures (hand-drawn molecules): region_id (or null), name (or null), "
        "smiles (or null), uncertain (bool), confidence, notes (or null).\n"
        "- concentrations: value, unit, species (or null), raw, evidence, confidence.\n"
        "Associate each reagent with the nearest quantity/concentration on its "
        "line. Use line_ids from the transcript as evidence. Do not invent "
        "chemistry that is not on the page."
    )


EXPERIMENT_SYSTEM = (
    "You distill a chemistry experiment from extracted evidence ONLY. You must "
    "not introduce any chemistry detail that is absent from the provided "
    "transcript and chemistry data. Every statement must cite the transcript "
    "line ids it derives from. Mark a statement inferred=true when it is a "
    "conclusion rather than a direct quote."
)


def experiment_user(transcript_text: str, chemistry_summary: str) -> str:
    """Build the experiment-pass instruction from transcript + chemistry text."""
    return (
        "Transcript (line_id: text):\n"
        f"{transcript_text}\n\n"
        "Extracted chemistry:\n"
        f"{chemistry_summary}\n\n"
        "Produce the experiment summary:\n"
        "- goal: a single concise objective string (or null if none stated).\n"
        "- conditions, procedure, observations, results: each a list of items "
        "{text, evidence (line_ids), inferred (bool), confidence}.\n"
        "Use only the evidence above. Prefer quoting/paraphrasing observed text; "
        "set inferred=true only for synthesized conclusions."
    )
