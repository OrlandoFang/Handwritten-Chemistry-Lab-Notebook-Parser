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


def test_openai_engine_drops_unsupported_params(monkeypatch, transcription_response):
    """When a model rejects a sampling param, the engine drops it and retries.

    Simulates newer GPT-5 family behavior (custom temperature unsupported) with a
    fake client injected in place of the real OpenAI client.
    """
    from types import SimpleNamespace

    calls: list[dict] = []

    class _FakeCompletions:
        def parse(self, **kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise RuntimeError(
                    "Unsupported value: 'temperature' does not support 0.0 with this model"
                )
            message = SimpleNamespace(parsed=transcription_response, refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    engine = OpenAIEngine(LLMConfig(temperature=0.0, seed=7))
    engine._client = fake_client  # bypass real client construction

    out = engine.extract("s", "u", TranscriptionResponse)
    assert out is transcription_response
    assert len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    assert "seed" in calls[1]  # supported param retained on retry
