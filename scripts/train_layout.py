#!/usr/bin/env python3
"""Training entry point for a layout-detection model (§4.2, §5 optional models).

This is an intentionally minimal, honest scaffold. No layout-detection weights or
annotated notebook-layout dataset ship with this repository, so this script does
not fabricate a training run. It defines the expected interface (data directory,
hyperparameters, output path) and validates prerequisites, then explains exactly
what is required to train a real detector that can be swapped in via the
``LayoutDetector`` protocol.
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
        missing.append(f"annotated layout dataset at {data_dir}")
    return missing


def main(argv: list[str] | None = None) -> int:
    """Validate prerequisites and describe the training contract."""
    parser = argparse.ArgumentParser(description="Train a notebook layout detector.")
    parser.add_argument("--data", default="models/layout/dataset", help="Annotated dataset dir.")
    parser.add_argument("--out", default="models/layout/model.pt", help="Output weights path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    missing = _check_prerequisites(Path(args.data))
    if missing:
        print("Cannot train: missing prerequisites:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nProvide a dataset of page images with region annotations "
            "(boxes + RegionType labels), then implement the model and training "
            "loop here. Trained weights load behind the LayoutDetector protocol "
            "so they drop into the pipeline without other changes.",
            file=sys.stderr,
        )
        return 1

    raise NotImplementedError(
        "Model architecture/training loop not bundled; implement against your dataset."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
