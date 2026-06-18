"""Pipeline configuration.

All tunables live here so the pipeline can be reconfigured from a single object
and the config can be serialized alongside results for provenance. It is a
Pydantic model so values are validated.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class ImagingConfig(BaseModel):
    """Knobs for local image conditioning before sending to the model (§4.2)."""

    deskew: bool = True
    max_skew_deg: float = Field(default=15.0, ge=0.0, le=45.0)
    # Longest image side sent to the API; bounds token cost while staying legible.
    max_long_side: int = Field(default=1600, ge=256)
    # JPEG quality for the encoded data URL (PNG used when lossless preferred).
    jpeg_quality: int = Field(default=90, ge=50, le=100)
    encode_format: str = Field(default="jpeg")  # jpeg | png


class LLMConfig(BaseModel):
    """OpenAI engine configuration (§4.1).

    The API key is read from the environment by default and never stored in the
    serialized config.
    """

    model: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    # Newer GPT-5 family models only accept their default temperature, so we omit
    # temperature by default (None) and rely on a fixed seed for best-effort
    # determinism (§10). Set a value for older models (e.g. gpt-4o) that support it.
    # The engine also drops any parameter a given model rejects.
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    seed: Optional[int] = 7
    max_retries: int = Field(default=4, ge=0)
    timeout_s: float = Field(default=120.0, gt=0.0)
    # Max alternative readings the transcription pass may return per line.
    max_alternatives: int = Field(default=2, ge=0)

    def api_key(self) -> Optional[str]:
        """Return the API key from the environment (not persisted in config)."""
        return os.environ.get("OPENAI_API_KEY")


class ConfidenceConfig(BaseModel):
    """Thresholds that drive human-review routing (§9)."""

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
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)

    # Stage toggles let callers run partial pipelines.
    do_chemistry: bool = True
    do_experiment: bool = True
    # Canonicalize/validate SMILES with RDKit when available.
    canonicalize_smiles: bool = True
