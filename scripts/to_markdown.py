#!/usr/bin/env python3
"""Convert an existing ``result.json`` into a human-readable ``result.md``.

Usage:
    python scripts/to_markdown.py result.json [-o result.md]

Works fully offline (no API key needed) since it only re-renders saved output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from notebook_parser.report import render_markdown_from_dict


def main(argv: list[str] | None = None) -> int:
    """Read a result JSON file and write/print its Markdown rendering."""
    parser = argparse.ArgumentParser(description="Render result.json as Markdown.")
    parser.add_argument("json_path", help="Path to a result.json produced by the parser.")
    parser.add_argument("-o", "--output", help="Output .md path (defaults to stdout).")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    markdown = render_markdown_from_dict(data)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
