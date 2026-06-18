#!/usr/bin/env python3
"""Run the parser on a single page image and emit JSON.

Usage:
    python scripts/run_page.py path/to/page.png [-o out.json] [--ocr-backend auto]

This is a thin wrapper around :func:`notebook_parser.cli.main` so the package's
console-script and the repo script behave identically.
"""

from __future__ import annotations

import sys

from notebook_parser.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
