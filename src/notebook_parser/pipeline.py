"""End-to-end pipeline orchestration (§7).

Wires the staged pipeline together:

    load -> preprocess -> layout -> transcribe -> symbols -> chemistry
    -> semantics -> confidence -> validate

Each stage is independently swappable (recognizer, layout detector, structure
extractor) and gated by :class:`~notebook_parser.config.PipelineConfig`. The run
is deterministic for a fixed config and input.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .chemistry import extract_chemistry
from .config import PipelineConfig
from .layout import HeuristicLayoutDetector, LayoutDetector, assign_reading_order
from .ocr import Recognizer, transcribe
from .preprocessing import preprocess
from .semantics import synthesize_experiment
from .symbols import recover_symbols
from .types import ChemistrySection, ExperimentSection, ParseResult
from .validation import apply_review_flags, compute_confidence, validate_result


def load_image(path: str | os.PathLike) -> np.ndarray:
    """Load an image file into a NumPy array (BGR or grayscale).

    Prefers OpenCV; falls back to Pillow. Raises ``FileNotFoundError`` if the
    path does not exist and ``RuntimeError`` if no image backend is available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    from .preprocessing._cv import get_cv2

    cv2 = get_cv2()
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return np.asarray(im.convert("L"))
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"could not load image {path}: {exc}") from exc


class NotebookPipeline:
    """Configurable multimodal parser for handwritten chemistry notebook pages."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        recognizer: Recognizer | None = None,
        layout_detector: LayoutDetector | None = None,
    ) -> None:
        """Store config and (optionally) injected backends for OCR and layout."""
        self.config = config or PipelineConfig()
        self.recognizer = recognizer
        self.layout_detector = layout_detector or HeuristicLayoutDetector(self.config.layout)

    def _resolve_page_id(self, source: str | os.PathLike | None) -> str:
        """Derive a stable page id from config, the source filename, or default."""
        if self.config.page_id:
            return self.config.page_id
        if isinstance(source, (str, os.PathLike)):
            return Path(source).stem
        return "page"

    def run(
        self,
        image: np.ndarray | str | os.PathLike,
        page_id: str | None = None,
        recognizer: Recognizer | None = None,
    ) -> ParseResult:
        """Parse a page (array or path) into a validated :class:`ParseResult`.

        ``recognizer`` overrides the instance/default backend for this call,
        which is the seam used by tests and demos to inject transcriptions.
        """
        source = image if isinstance(image, (str, os.PathLike)) else None
        resolved_page_id = page_id or self._resolve_page_id(source)
        if source is not None:
            image = load_image(source)

        # 1) Preprocess.
        pre = preprocess(image, self.config.preprocessing)

        # 2) Layout + reading order.
        regions = self.layout_detector.detect(pre.binary)
        regions = assign_reading_order(regions, pre.binary.shape[1], self.config.layout)

        # 3) Transcription.
        lines = transcribe(
            pre.gray,
            pre.binary,
            regions,
            recognizer=recognizer or self.recognizer,
            config=self.config.ocr,
            review_threshold=self.config.confidence.medium,
        )

        # 4) Symbol recovery (mutates line text in place).
        symbols = recover_symbols(lines) if self.config.do_symbols else []

        # 5) Chemistry extraction (text + drawings).
        chemistry = (
            extract_chemistry(lines, gray=pre.gray, regions=regions, binary=pre.binary)
            if self.config.do_chemistry
            else ChemistrySection()
        )

        # 6) Experiment semantics.
        experiment = (
            synthesize_experiment(lines)
            if self.config.do_semantics
            else ExperimentSection()
        )

        # 7) Assemble, score confidence, route review, validate schema.
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
