"""Command-line entry point: parse a page image to JSON.

Thin wrapper around :class:`~notebook_parser.pipeline.NotebookPipeline`. Requires
``OPENAI_API_KEY`` in the environment (the model does the recognition).
"""

from __future__ import annotations

import argparse
import sys

from .config import PipelineConfig
from .llm.client import LLMError
from .pipeline import NotebookPipeline


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the pipeline, and write/print the JSON result."""
    parser = argparse.ArgumentParser(description="Parse a handwritten chemistry notebook page.")
    parser.add_argument("image", help="Path to the page image.")
    parser.add_argument("-o", "--output", help="Output JSON path (defaults to stdout).")
    parser.add_argument("--page-id", help="Override the page id.")
    parser.add_argument("--model", help="OpenAI model to use (overrides OPENAI_MODEL).")
    parser.add_argument("--no-deskew", action="store_true", help="Disable local deskew.")
    args = parser.parse_args(argv)

    config = PipelineConfig(page_id=args.page_id)
    if args.model:
        config.llm.model = args.model
    if args.no_deskew:
        config.imaging.deskew = False

    try:
        result = NotebookPipeline(config).run(args.image)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_json()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
