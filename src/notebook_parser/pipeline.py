"""End-to-end multi-pass pipeline orchestration (§7).

Wires the passes together:

    condition image -> transcription -> chemistry -> experiment
    -> assemble -> confidence -> validate

The LLM engine is injectable, so the entire pipeline runs offline in tests with a
stub and against the OpenAI API in production with no code change.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .imaging import condition_image
from .llm.client import LLMEngine, OpenAIEngine
from .passes import run_chemistry, run_experiment, run_transcription
from .types import ChemistrySection, ExperimentSection, ParseResult
from .validation import apply_review_flags, compute_confidence, validate_result


class NotebookPipeline:
    """Multimodal parser for handwritten chemistry notebook pages."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        engine: LLMEngine | None = None,
    ) -> None:
        """Store config and (optionally) an injected LLM engine."""
        self.config = config or PipelineConfig()
        self.engine = engine or OpenAIEngine(self.config.llm)

    def _resolve_page_id(self, source) -> str:
        """Derive a stable page id from config, the source filename, or default."""
        if self.config.page_id:
            return self.config.page_id
        if isinstance(source, (str, os.PathLike)):
            return Path(source).stem
        return "page"

    def run(
        self,
        image: np.ndarray | Image.Image | str | os.PathLike,
        page_id: str | None = None,
    ) -> ParseResult:
        """Parse a page (array, Pillow image, or path) into a validated result."""
        source = image if isinstance(image, (str, os.PathLike)) else None
        resolved_page_id = page_id or self._resolve_page_id(source)

        # 1) Condition the image (load/deskew/resize/encode).
        conditioned = condition_image(image, self.config.imaging)

        # 2) Transcription pass (vision).
        regions, lines, symbols = run_transcription(self.engine, conditioned, self.config.llm)

        # 3) Chemistry pass (vision).
        chemistry = (
            run_chemistry(self.engine, conditioned, lines, self.config.canonicalize_smiles)
            if self.config.do_chemistry
            else ChemistrySection()
        )

        # 4) Experiment pass (text-only, evidence-constrained).
        experiment = (
            run_experiment(self.engine, lines, chemistry)
            if self.config.do_experiment
            else ExperimentSection()
        )

        # 5) Assemble, score confidence, route review, validate schema.
        result = ParseResult(
            page_id=resolved_page_id,
            layout=regions,
            transcript=lines,
            symbols=symbols,
            chemistry=chemistry,
            experiment=experiment,
        )
        result.confidence = compute_confidence(result)
        result = apply_review_flags(result, self.config.confidence)
        return validate_result(result)
