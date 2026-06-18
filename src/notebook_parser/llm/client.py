"""LLM engine: the single seam through which all model calls flow (§4.1).

``LLMEngine`` is the interface the passes program against. ``OpenAIEngine`` is the
production backend (vision + Structured Outputs). ``StubEngine`` returns canned
responses so the whole pipeline runs offline and deterministically in tests, and
so an alternative model can be dropped in without touching pass logic.
"""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

from ..config import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model call fails or is refused."""


class LLMEngine(Protocol):
    """Interface for a schema-constrained, optionally multimodal model call."""

    def extract(
        self,
        system: str,
        user: str,
        schema: type[T],
        image_data_url: str | None = None,
    ) -> T:
        """Return a validated ``schema`` instance for the given prompts/image."""
        ...


class OpenAIEngine:
    """OpenAI-backed engine using ``chat.completions.parse`` (vision + schema)."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Create the engine; the OpenAI client is built lazily on first use."""
        self.config = config or LLMConfig()
        self._client = None

    def _get_client(self):
        """Lazily construct and cache the OpenAI client, validating the key."""
        if self._client is not None:
            return self._client
        api_key = self.config.api_key()
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Export it (or add it as a Cursor "
                "Cloud Agent secret) before running the pipeline."
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency guaranteed by deps
            raise LLMError("the 'openai' package is not installed") from exc
        self._client = OpenAI(
            api_key=api_key,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout_s,
        )
        return self._client

    def extract(
        self,
        system: str,
        user: str,
        schema: type[T],
        image_data_url: str | None = None,
    ) -> T:
        """Call the model with optional image and parse into ``schema``.

        Uses temperature 0 and a fixed seed for best-effort determinism. Raises
        :class:`LLMError` on refusal or empty parse.
        """
        client = self._get_client()
        content: list[dict] = [{"type": "text", "text": user}]
        if image_data_url:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        params: dict = {
            "model": self.config.model,
            "messages": messages,
            "response_format": schema,
        }
        # Only send optional sampling params when configured; some newer models
        # reject non-default temperature/seed entirely.
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.seed is not None:
            params["seed"] = self.config.seed

        completion = self._parse_with_fallback(client, params)
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError(f"model refused: {message.refusal}")
        if message.parsed is None:
            raise LLMError("model returned no parsable structured output")
        return message.parsed

    @staticmethod
    def _parse_with_fallback(client, params: dict):
        """Call ``chat.completions.parse``, dropping params the model rejects.

        Some models reject optional sampling params (e.g. a custom ``temperature``
        or ``seed``). On such an error we remove the offending parameter(s) named
        in the error message and retry once, so newer models work out of the box
        without sacrificing determinism on models that do support seeding.
        """
        try:
            return client.chat.completions.parse(**params)
        except Exception as exc:
            message = str(exc).lower()
            dropped = [
                p for p in ("temperature", "seed")
                if p in params and p in message
            ]
            if not dropped:
                raise LLMError(f"model call failed: {exc}") from exc
            for p in dropped:
                params.pop(p, None)
            try:
                return client.chat.completions.parse(**params)
            except Exception as exc2:
                raise LLMError(f"model call failed: {exc2}") from exc2


class StubEngine:
    """Offline engine returning pre-registered responses keyed by schema type.

    Register either a model instance or a zero-arg callable per schema class. Used
    by tests and offline runs; deterministic and network-free.
    """

    def __init__(self, responses: dict[type[BaseModel], object] | None = None) -> None:
        """Store the mapping from response schema class to instance/factory."""
        self._responses: dict[type[BaseModel], object] = dict(responses or {})

    def register(self, schema: type[T], response: T | Callable[[], T]) -> None:
        """Register a response (or factory) for a schema class."""
        self._responses[schema] = response

    def extract(
        self,
        system: str,
        user: str,
        schema: type[T],
        image_data_url: str | None = None,
    ) -> T:
        """Return the registered response for ``schema`` (calling it if callable)."""
        if schema not in self._responses:
            raise LLMError(f"StubEngine has no response registered for {schema.__name__}")
        value = self._responses[schema]
        result = value() if callable(value) and not isinstance(value, BaseModel) else value
        if not isinstance(result, schema):
            raise LLMError(f"StubEngine response for {schema.__name__} has wrong type")
        return result
