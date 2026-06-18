"""Handwritten chemistry lab notebook parser.

A multi-pass, multimodal pipeline that converts a scanned handwritten chemistry
lab notebook page into structured, machine-readable JSON. See ``spec.md`` for the
full design.

The public surface is intentionally small: construct/configure a
:class:`~notebook_parser.pipeline.NotebookPipeline` and call
:meth:`~notebook_parser.pipeline.NotebookPipeline.run`.
"""

from .config import PipelineConfig
from .llm.client import LLMEngine, OpenAIEngine, StubEngine
from .pipeline import NotebookPipeline
from .report import render_markdown, render_markdown_from_dict
from .types import ParseResult

__all__ = [
    "PipelineConfig",
    "NotebookPipeline",
    "ParseResult",
    "LLMEngine",
    "OpenAIEngine",
    "StubEngine",
    "render_markdown",
    "render_markdown_from_dict",
    "__version__",
]

__version__ = "0.3.0"
