"""Tests for the LLM engine seam (§4.1)."""

from __future__ import annotations

import pytest

from notebook_parser.config import LLMConfig
from notebook_parser.llm.client import LLMError, OpenAIEngine, StubEngine
from notebook_parser.llm.schemas import ChemistryResponse, TranscriptionResponse


def test_stub_engine_returns_registered(transcription_response):
    """StubEngine returns the registered response for a schema."""
    engine = StubEngine({TranscriptionResponse: transcription_response})
    out = engine.extract("sys", "user", TranscriptionResponse)
    assert out is transcription_response


def test_stub_engine_supports_callables(transcription_response):
    """A registered zero-arg factory is invoked per call."""
    engine = StubEngine()
    engine.register(TranscriptionResponse, lambda: transcription_response)
    assert engine.extract("s", "u", TranscriptionResponse) is transcription_response


def test_stub_engine_unregistered_raises():
    """Requesting an unregistered schema raises LLMError."""
    engine = StubEngine()
    with pytest.raises(LLMError):
        engine.extract("s", "u", ChemistryResponse)


def test_openai_engine_requires_api_key(monkeypatch):
    """Without OPENAI_API_KEY the engine raises a clear error (no network)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = OpenAIEngine(LLMConfig())
    with pytest.raises(LLMError):
        engine.extract("s", "u", TranscriptionResponse)
