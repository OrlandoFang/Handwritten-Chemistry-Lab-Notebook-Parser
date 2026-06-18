"""Human-readable Markdown report renderer.

Turns a :class:`~notebook_parser.types.ParseResult` (i.e. the contents of
``result.json``) into a readable ``result.md`` with three parts:

1. a clean reconstructed transcription (reading order),
2. the chemistry (drawn structures, formulas, reagents, concentrations) shown and
   briefly explained from the extracted fields,
3. the experiment narrative (goal, conditions, procedure, observations, results).

Only data present in the result is rendered; nothing is invented.
"""

from __future__ import annotations

from collections import OrderedDict

from .types import (
    ChemicalStructure,
    Concentration,
    EvidenceItem,
    Formula,
    ParseResult,
    Quantity,
    Reagent,
)


def _ordered_lines(result: ParseResult):
    """Yield transcript lines grouped by region in reading order.

    Regions are ordered by their ``reading_order``; lines keep their transcript
    order within a region. Lines whose region is unknown are emitted last.
    """
    region_order = {r.id: r.reading_order for r in result.layout}
    groups: "OrderedDict[str, list]" = OrderedDict()
    for line in result.transcript:
        groups.setdefault(line.region_id, []).append(line)
    for rid in sorted(groups, key=lambda r: region_order.get(r, 10**9)):
        for line in groups[rid]:
            yield line


def _render_transcription(result: ParseResult) -> str:
    """Render the transcript as blank-line-separated lines (one per notebook line)."""
    paragraphs = [line.text.strip() for line in _ordered_lines(result) if line.text.strip()]
    body = "\n\n".join(paragraphs) if paragraphs else "_No text was transcribed._"
    return "## Human-readable transcription\n\n" + body


def _fmt_quantity(q: Quantity | None) -> str | None:
    """Format a quantity as a short string (e.g. ``5 mol%``), or None."""
    if q is None:
        return None
    if q.value is not None and q.unit:
        return f"{_fmt_num(q.value)} {q.unit}"
    return q.raw or None


def _fmt_concentration(c: Concentration | None) -> str | None:
    """Format a concentration (e.g. ``1 M LiTFSI``), or None."""
    if c is None:
        return None
    parts = []
    if c.value is not None and c.unit:
        parts.append(f"{_fmt_num(c.value)} {c.unit}")
    elif c.raw:
        parts.append(c.raw)
    if c.species:
        parts.append(c.species)
    return " ".join(parts) or None


def _fmt_num(x: float) -> str:
    """Render a float without a trailing ``.0`` when it is integral."""
    return str(int(x)) if float(x).is_integer() else str(x)


def _evidence_suffix(evidence: list[str]) -> str:
    """Return a compact ' (from ...)' provenance suffix, or empty string."""
    return f" _(from {', '.join(evidence)})_" if evidence else ""


def _render_structure(s: ChemicalStructure) -> str:
    """Render a single hand-drawn structure with a brief explanation."""
    title = s.name or "Unnamed structure"
    bits = [f"**{title}**"]
    if s.smiles:
        bits.append(f"SMILES `{s.smiles}`")
    line = " — ".join(bits)
    extras = []
    if s.region_id:
        extras.append(f"drawn in region {s.region_id}")
    extras.append(f"confidence {s.confidence:.2f}")
    if s.uncertain:
        extras.append("**uncertain** (drawing was ambiguous)")
    if s.notes:
        extras.append(s.notes)
    return f"- {line} — " + "; ".join(extras)


def _render_chemistry(result: ParseResult) -> str:
    """Render structures, formulas, reagents, and concentrations with context."""
    chem = result.chemistry
    out: list[str] = ["## Chemistry"]

    out.append("\n### Hand-drawn structures")
    if chem.structures:
        out.append(
            "Molecules recognized from drawings on the page, given as SMILES "
            "(a text encoding of the molecular graph):\n"
        )
        out.extend(_render_structure(s) for s in chem.structures)
    else:
        out.append("_No hand-drawn structures detected._")

    out.append("\n### Formulas")
    if chem.formulas:
        out.extend(_render_formula(f) for f in chem.formulas)
    else:
        out.append("_No chemical formulas detected._")

    out.append("\n### Reagents")
    if chem.reagents:
        out.append("Chemicals used, with quantities/concentrations where stated:\n")
        out.extend(_render_reagent(r) for r in chem.reagents)
    else:
        out.append("_No reagents detected._")

    out.append("\n### Concentrations")
    if chem.concentrations:
        out.extend(
            f"- {_fmt_concentration(c) or c.raw}{_evidence_suffix(c.evidence)}"
            for c in chem.concentrations
        )
    else:
        out.append("_No concentrations detected._")

    return "\n".join(out)


def _render_formula(f: Formula) -> str:
    """Render a formula line with validity annotation."""
    label = f.normalized or f.raw
    note = ""
    if f.valid is True:
        note = " (valid formula)"
    elif f.valid is False:
        note = " (could not be validated)"
    return f"- `{label}`{note}{_evidence_suffix(f.evidence)}"


def _render_reagent(r: Reagent) -> str:
    """Render a reagent as a readable sentence with role/quantity/concentration."""
    name = f"**{r.name}**"
    if r.normalized_name and r.normalized_name.lower() != r.name.lower():
        name += f" ({r.normalized_name})"
    details = [f"role: {r.role.value}"]
    qty = _fmt_quantity(r.quantity)
    if qty:
        details.append(f"amount: {qty}")
    conc = _fmt_concentration(r.concentration)
    if conc:
        details.append(f"concentration: {conc}")
    return f"- {name} — " + "; ".join(details) + _evidence_suffix(r.evidence)


def _render_items(items: list[EvidenceItem], ordered: bool = False) -> list[str]:
    """Render a list of evidence items as bullets or a numbered list."""
    if not items:
        return ["_None recorded._"]
    rendered = []
    for i, it in enumerate(items, start=1):
        marker = f"{i}." if ordered else "-"
        inferred = " _(inferred)_" if it.inferred else ""
        rendered.append(f"{marker} {it.text}{inferred}{_evidence_suffix(it.evidence)}")
    return rendered


def _render_experiment(result: ParseResult) -> str:
    """Render the experiment narrative: goal, conditions, procedure, results."""
    exp = result.experiment
    out: list[str] = ["## What was happening in the experiment"]

    out.append("\n### Goal")
    out.append(exp.goal if exp.goal else "_No explicit goal stated._")

    out.append("\n### Conditions")
    out.extend(_render_items(exp.conditions))

    out.append("\n### Procedure")
    out.extend(_render_items(exp.procedure, ordered=True))

    out.append("\n### Observations")
    out.extend(_render_items(exp.observations))

    out.append("\n### Results")
    out.extend(_render_items(exp.results))

    return "\n".join(out)


def _render_header(result: ParseResult) -> str:
    """Render the document title and a one-line confidence summary."""
    flagged = sum(1 for l in result.transcript if l.needs_review)
    lines = [
        f"# Lab notebook — page {result.page_id}",
        "",
        f"_Overall confidence: {result.confidence.overall:.2f}. "
        f"{flagged} line(s) flagged for review._",
    ]
    return "\n".join(lines)


def render_markdown(result: ParseResult) -> str:
    """Render a full Markdown report for a parse result."""
    sections = [
        _render_header(result),
        _render_transcription(result),
        _render_chemistry(result),
        _render_experiment(result),
    ]
    return "\n\n".join(sections).strip() + "\n"


def render_markdown_from_dict(data: dict) -> str:
    """Render Markdown from a serialized result dict (e.g. parsed ``result.json``)."""
    return render_markdown(ParseResult.model_validate(data))
