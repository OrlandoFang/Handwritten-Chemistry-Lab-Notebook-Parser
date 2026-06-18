#!/usr/bin/env python3
"""Training entry point for a handwriting OCR model (§4.3, §5 optional models).

Like ``train_layout.py``, this is an honest scaffold: no handwriting model or
transcription dataset is bundled, so it validates prerequisites and documents the
training contract rather than faking a run. A trained recognizer plugs into the
pipeline behind the ``Recognizer`` protocol with no other changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _check_prerequisites(data_dir: Path) -> list[str]:
    """Return a list of missing prerequisites for training (empty if ready)."""
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("PyTorch (install with: pip install '.[ml]')")
    if not data_dir.exists():
        missing.append(f"line-image/transcription dataset at {data_dir}")
    return missing


def main(argv: list[str] | None = None) -> int:
    """Validate prerequisites and describe the training contract."""
    parser = argparse.ArgumentParser(description="Train a handwriting recognizer.")
    parser.add_argument("--data", default="models/ocr/dataset", help="Line-image dataset dir.")
    parser.add_argument("--out", default="models/ocr/model.pt", help="Output weights path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args(argv)

    missing = _check_prerequisites(Path(args.data))
    if missing:
        print("Cannot train: missing prerequisites:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nProvide a dataset of cropped text-line images paired with their "
            "transcriptions (CTC- or seq2seq-ready). Implement the encoder, "
            "decoder with beam search, and CTC/LM rescoring here. The resulting "
            "model loads behind the Recognizer protocol.",
            file=sys.stderr,
        )
        return 1

    raise NotImplementedError(
        "Model architecture/training loop not bundled; implement against your dataset."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
