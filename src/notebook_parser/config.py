"""Pipeline configuration.

All tunables live here so each stage stays free of magic numbers and the whole
pipeline can be reconfigured (or made deterministic for a given model version)
from a single object. The config is a Pydantic model so values are validated and
the object is easy to serialize alongside results for provenance.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Knobs for the image-normalization stage (§4.1)."""

    enabled: bool = True
    deskew: bool = True
    # Skew search is bounded; pages are rarely rotated more than this.
    max_skew_deg: float = Field(default=15.0, ge=0.0, le=45.0)
    denoise: bool = True
    normalize_illumination: bool = True
    # Longest image side is capped to keep downstream stages fast and memory
    # bounded; 0 disables resizing.
    max_long_side: int = Field(default=2200, ge=0)


class LayoutConfig(BaseModel):
    """Knobs for layout segmentation + reading order (§4.2)."""

    # Regions smaller than this fraction of the page area are discarded as noise.
    min_region_area_frac: float = Field(default=0.0008, ge=0.0, le=1.0)
    # Vertical gap (as a fraction of median line height) above which two text
    # rows are considered separate lines.
    line_gap_frac: float = Field(default=0.6, ge=0.0)
    # Columns whose centers differ by more than this fraction of page width are
    # treated as separate reading-order columns.
    column_split_frac: float = Field(default=0.45, ge=0.0, le=1.0)


class OCRConfig(BaseModel):
    """Knobs for handwriting transcription (§4.3)."""

    # Preferred backend; the pipeline falls back gracefully if unavailable.
    backend: str = Field(default="auto")  # auto | tesseract | fallback
    language_model_rescoring: bool = True
    max_alternatives: int = Field(default=3, ge=0)


class ConfidenceConfig(BaseModel):
    """Thresholds that drive the human-review routing (§9)."""

    high: float = Field(default=0.85, ge=0.0, le=1.0)
    medium: float = Field(default=0.6, ge=0.0, le=1.0)

    def band(self, score: float) -> str:
        """Map a numeric confidence into a coarse review band."""
        if score >= self.high:
            return "high"
        if score >= self.medium:
            return "medium"
        return "low"


class PipelineConfig(BaseModel):
    """Top-level configuration for the whole pipeline."""

    page_id: Optional[str] = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)

    # Stage toggles let callers run partial pipelines (e.g. layout-only).
    do_symbols: bool = True
    do_chemistry: bool = True
    do_semantics: bool = True
