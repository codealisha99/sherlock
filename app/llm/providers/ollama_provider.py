"""Ollama provider adapter for the Sherlock LLM Gateway.

Wraps the existing OllamaClient (app/llm/ollama_client.py) so it conforms to
the LLMProvider protocol. No changes are made to OllamaClient itself.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from app.llm.models import GenerationRequest, GenerationResult, LLMProvider, TokenUsage
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 0.75) + text.count(".") + text.count(","))


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class OllamaProvider:
    """Ollama provider adapter implementing LLMProvider."""

    name = "ollama"

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self._client = ollama_client or OllamaClient()
        self.model = getattr(self._client, "model", None) or ""

    # ------------------------------------------------------------------ #
    # Provider interface
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = _now_ms()
        try:
            text = self._client.generate(
                prompt=request.to_prompt(),
                system_prompt=request.messages[0].get("content")
                if request.messages
                and request.messages[0].get("role") == "system"
                else None,
                temperature=request.temperature if request.temperature is not None else 0.2,
            )
            total_ms = _now_ms() - t0
            usage = TokenUsage(
                estimated_input=_estimate_tokens(request.to_prompt()),
                estimated_output=_estimate_tokens(text),
            )
            return GenerationResult(
                request_id=request.request_id,
                provider=self.name,
                model=request.model or self.model,
                text=text,
                usage=usage,
                timing={"ttft_ms": total_ms, "total_latency_ms": total_ms},
                finish_reason="stop",
            )
        except Exception as exc:  # noqa: BLE001
            total_ms = _now_ms() - t0
            return GenerationResult(
                request_id=request.request_id,
                provider=self.name,
                model=request.model or self.model,
                text="",
                usage=TokenUsage(),
                timing={"ttft_ms": None, "total_latency_ms": total_ms},
                error=str(exc),
                error_category=self._error_category(exc),
            )

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        t0 = _now_ms()
        ttft_emitted = False
        accumulated = ""

        try:
            system_prompt = None
            if request.messages and request.messages[0].get("role") == "system":
                system_prompt = request.messages[0].get("content")

            async for chunk in self._client.generate_stream(
                prompt=request.to_prompt(),
                system_prompt=system_prompt,
                temperature=request.temperature if request.temperature is not None else 0.2,
            ):
                if not ttft_emitted:
                    ttft_ms = _now_ms() - t0
                    ttft_emitted = True
                    yield f"__ttft__:{ttft_ms:.2f}__\n"
                accumulated += chunk
                yield chunk

        except Exception as exc:  # noqa: BLE001
            yield f"__error__:{json_dumps({'message': str(exc), 'error_category': self._error_category(exc)})}__\n"

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _error_category(self, exc: Exception) -> str:
        import httpx

        if isinstance(exc, (httpx.ConnectError,)):
            return "network"
        if isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)):
            return "network"
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 429:
                return "rate_limit"
            return "provider"
        return "provider"

    def check_health(self) -> Dict[str, Any]:
        try:
            health = self._client.check_health()
            return {
                "provider": self.name,
                "model": self.model,
                "available": health.get("available", False),
                "model_ready": health.get("model_ready", False),
                "configured_model": health.get("configured_model", self.model),
                "error": health.get("error"),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "provider": self.name,
                "model": self.model,
                "available": False,
                "error": str(exc),
            }

    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)
