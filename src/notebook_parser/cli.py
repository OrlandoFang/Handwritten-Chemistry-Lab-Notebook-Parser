"""Command-line entry point: parse a page image to JSON.

Thin wrapper around :class:`~notebook_parser.pipeline.NotebookPipeline` so the
parser is usable as ``parse-notebook-page <image> [-o out.json]``.
"""

from __future__ import annotations

import argparse
import sys

from .config import PipelineConfig
from .pipeline import NotebookPipeline


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the pipeline, and write/print the JSON result."""
    parser = argparse.ArgumentParser(description="Parse a handwritten chemistry notebook page.")
    parser.add_argument("image", help="Path to the page image.")
    parser.add_argument("-o", "--output", help="Output JSON path (defaults to stdout).")
    parser.add_argument("--page-id", help="Override the page id.")
    parser.add_argument(
        "--ocr-backend",
        default="auto",
        choices=["auto", "tesseract", "fallback"],
        help="OCR backend selection.",
    )
    args = parser.parse_args(argv)

    config = PipelineConfig(page_id=args.page_id)
    config.ocr.backend = args.ocr_backend
    pipeline = NotebookPipeline(config)
    result = pipeline.run(args.image)

    payload = result.to_json()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
