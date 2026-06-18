#!/usr/bin/env python3
"""Evaluation harness for the notebook parser (§5 evaluation tools, §10).

Implements the metrics the spec calls for - character/word error rate, symbol
accuracy, and reagent/formula F1 - and a driver that runs the pipeline over a
folder of ``<name>.{png,jpg}`` images each paired with a ``<name>.gt.json``
ground-truth file.

Ground-truth schema (all fields optional):
    {
      "transcript": ["line 1", "line 2", ...],
      "reagents": ["HCl", "NaOH", ...],
      "formulas": ["H2SO4", ...],
      "symbols":  ["°C", "μL", ...]
    }

The metric functions are importable and unit-tested independently of any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from notebook_parser.pipeline import NotebookPipeline


def levenshtein(a: Iterable, b: Iterable) -> int:
    """Edit distance between two sequences using O(min(len)) memory.

    Works on any sequences (characters for CER, word lists for WER). Uses a
    rolling two-row DP for memory efficiency on long lines.
    """
    a = list(a)
    b = list(b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def error_rate(refs: list[str], hyps: list[str], by_word: bool) -> float:
    """Corpus CER/WER: total edits divided by total reference length.

    ``by_word`` selects word-level (WER) vs character-level (CER). Pairs are
    aligned positionally; missing hypotheses count as full deletions.
    """
    total_edits = 0
    total_len = 0
    for i, ref in enumerate(refs):
        hyp = hyps[i] if i < len(hyps) else ""
        ref_seq = ref.split() if by_word else list(ref)
        hyp_seq = hyp.split() if by_word else list(hyp)
        total_edits += levenshtein(ref_seq, hyp_seq)
        total_len += len(ref_seq)
    return (total_edits / total_len) if total_len else 0.0


def set_f1(reference: Iterable[str], predicted: Iterable[str]) -> dict[str, float]:
    """Precision/recall/F1 over two string sets (case-insensitive)."""
    ref = {s.lower() for s in reference}
    pred = {s.lower() for s in predicted}
    if not ref and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(ref & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate_page(result, gt: dict) -> dict:
    """Compute all metrics for one page given a parse result and ground truth."""
    pred_lines = [l.text for l in result.transcript]
    ref_lines = gt.get("transcript", [])
    pred_reagents = [r.name for r in result.chemistry.reagents]
    pred_formulas = [f.normalized for f in result.chemistry.formulas]
    pred_symbols = [s.normalized for s in result.symbols]
    return {
        "cer": round(error_rate(ref_lines, pred_lines, by_word=False), 3),
        "wer": round(error_rate(ref_lines, pred_lines, by_word=True), 3),
        "reagent_f1": set_f1(gt.get("reagents", []), pred_reagents),
        "formula_f1": set_f1(gt.get("formulas", []), pred_formulas),
        "symbol_f1": set_f1(gt.get("symbols", []), pred_symbols),
    }


def evaluate_dataset(data_dir: str | Path) -> dict:
    """Run the pipeline over a labeled dataset directory and aggregate metrics."""
    data_dir = Path(data_dir)
    pipeline = NotebookPipeline()
    per_page: dict[str, dict] = {}
    for gt_path in sorted(data_dir.glob("*.gt.json")):
        stem = gt_path.name[: -len(".gt.json")]
        image = next((p for ext in (".png", ".jpg", ".jpeg") if (p := data_dir / f"{stem}{ext}").exists()), None)
        if image is None:
            continue
        gt = json.loads(gt_path.read_text())
        result = pipeline.run(str(image))
        per_page[stem] = evaluate_page(result, gt)
    return per_page


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate a dataset directory and print a JSON metrics report."""
    parser = argparse.ArgumentParser(description="Evaluate the notebook parser.")
    parser.add_argument("data_dir", help="Directory of <name>.{png,jpg} + <name>.gt.json")
    args = parser.parse_args(argv)
    report = evaluate_dataset(args.data_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(sys.argv[1:]))
