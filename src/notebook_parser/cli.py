"""Command-line entry point: parse a page image to JSON.

Thin wrapper around :class:`~notebook_parser.pipeline.NotebookPipeline`. Requires
``OPENAI_API_KEY`` in the environment (the model does the recognition).
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from .config import PipelineConfig
from .llm.client import LLMError
from .pipeline import NotebookPipeline
from .report import render_markdown


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the pipeline, and write/print the JSON + Markdown."""
    parser = argparse.ArgumentParser(description="Parse a handwritten chemistry notebook page.")
    parser.add_argument("image", help="Path to the page image.")
    parser.add_argument("-o", "--output", help="Output JSON path (defaults to stdout).")
    parser.add_argument(
        "-m", "--markdown",
        help="Output Markdown path. Defaults to the JSON path with a .md suffix.",
    )
    parser.add_argument("--no-markdown", action="store_true", help="Do not write a Markdown report.")
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

    # Resolve where (if anywhere) to write the Markdown report.
    md_path = _resolve_markdown_path(args)
    if md_path is not None:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(result))
        print(f"wrote {md_path}", file=sys.stderr)
    return 0


def _resolve_markdown_path(args) -> str | None:
    """Decide the Markdown output path from the CLI flags (None = skip)."""
    if args.no_markdown:
        return None
    if args.markdown:
        return args.markdown
    if args.output:
        return str(Path(args.output).with_suffix(".md"))
    return None  # JSON went to stdout and no explicit path was given


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
