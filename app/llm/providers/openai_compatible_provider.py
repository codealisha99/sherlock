"""Generic OpenAI-compatible chat-completions provider (httpx only, no SDK).

Backs bring-your-own-key (BYOK) calls to OpenAI, Anthropic's OpenAI-compatible
endpoint, Groq, OpenRouter, or any user-supplied compatible base URL. Emits the
same stream markers as the gateway contract (`__ttft__:` / `__error__:`).
Never logs or exposes the API key.
"""

from __future__ import annotations

import json
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, AsyncGenerator, Dict, Mapping, Optional, Tuple

import httpx

from app.llm.models import GenerationRequest, GenerationResult, TokenUsage
from app.llm.providers.groq_provider import _estimate_tokens

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 45.0
_DONE = "[DONE]"


class ProviderError(Exception):
    """Malformed response, or an error reported inside a 200 SSE stream."""


class RateLimitError(Exception):
    """HTTP 429 with optional Retry-After (seconds)."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def parse_sse_line(line: str) -> Tuple[bool, str]:
    """Return (done, text_delta) for one SSE line. Non-data lines yield ''."""
    line = line.strip()
    if not line.startswith("data:"):
        return False, ""
    payload = line[len("data:"):].strip()
    if payload == _DONE:
        return True, ""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False, ""
    if not isinstance(data, dict):
        return False, ""
    if data.get("error"):
        err = data["error"]
        message = err.get("message") if isinstance(err, dict) else None
        raise ProviderError(str(message or "Provider stream error"))
    delta = _first_choice(data).get("delta")
    content = delta.get("content") if isinstance(delta, dict) else None
    return False, content or ""


def _first_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _message_content(data: Dict[str, Any]) -> str:
    message = _first_choice(data).get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content or ""


def _provider_usage(data: Dict[str, Any]) -> Optional[TokenUsage]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    return TokenUsage(
        provider_input=usage.get("prompt_tokens"),
        provider_output=usage.get("completion_tokens"),
    )


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise RateLimitError(
            "Provider rate limit exceeded",
            _parse_retry_after(response.headers.get("retry-after")),
        )
    response.raise_for_status()


def _error_category(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return "rate_limit"
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return "network"
    return "provider"


def _describe(exc: Exception) -> str:
    """User-safe message. Avoids echoing URLs, headers, or response bodies."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Provider returned HTTP {exc.response.status_code}."
    if isinstance(exc, httpx.TimeoutException):
        return "Provider request timed out."
    if isinstance(exc, httpx.TransportError):
        return "Could not reach the provider."
    return str(exc) or exc.__class__.__name__


def _error_event(exc: Exception) -> str:
    payload: Dict[str, Any] = {
        "type": "error",
        "message": _describe(exc),
        "error_category": _error_category(exc),
    }
    if isinstance(exc, RateLimitError) and exc.retry_after_seconds is not None:
        payload["retry_after_seconds"] = exc.retry_after_seconds
    return f"__error__:{json.dumps(payload)}__\n"


class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible base URL.

    `transport` / `async_transport` exist so tests can inject httpx mock transports.
    The request's `model` is ignored in favour of the provider's own model, so a
    platform model name can never be sent to a BYOK endpoint.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        if not api_key:
            raise ValueError("API key is required.")
        if not base_url:
            raise ValueError("Base URL is required.")
        if not model:
            raise ValueError("Model is required.")
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._extra_headers = dict(extra_headers or {})
        self._transport = transport
        self._async_transport = async_transport

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self.name!r}, model={self.model!r})"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        # Authorization last so extra_headers can never override it.
        return {
            **self._extra_headers,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _payload(self, request: GenerationRequest, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in request.messages
            ],
            "temperature": request.temperature if request.temperature is not None else 0.2,
            "stream": stream,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        return payload

    def _estimate(self, request: GenerationRequest, text: str) -> TokenUsage:
        return TokenUsage(
            estimated_input=_estimate_tokens(request.to_prompt()),
            estimated_output=_estimate_tokens(text),
        )

    def _error_result(self, request: GenerationRequest, exc: Exception, total_ms: float) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            provider=self.name,
            model=self.model,
            text="",
            usage=TokenUsage(),
            timing={"ttft_ms": None, "total_latency_ms": total_ms},
            error=_describe(exc),
            error_category=_error_category(exc),
        )

    # ------------------------------------------------------------------ #
    # Provider interface
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = _now_ms()
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(self._url, json=self._payload(request, False), headers=self._headers())
                _raise_for_status(response)
                data = response.json()
            if not isinstance(data, dict):
                raise ProviderError("Malformed provider response.")
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            return self._error_result(request, exc, _now_ms() - t0)

        total_ms = _now_ms() - t0
        text = _message_content(data)
        return GenerationResult(
            request_id=request.request_id,
            provider=self.name,
            model=self.model,
            text=text,
            usage=_provider_usage(data) or self._estimate(request, text),
            timing={"ttft_ms": total_ms, "total_latency_ms": total_ms},
            finish_reason=_first_choice(data).get("finish_reason"),
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        t0 = _now_ms()
        ttft_emitted = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._async_transport) as client:
                async with client.stream(
                    "POST", self._url, json=self._payload(request, True), headers=self._headers()
                ) as response:
                    _raise_for_status(response)
                    async for line in response.aiter_lines():
                        done, text = parse_sse_line(line)
                        if done:
                            break
                        if not text:
                            continue
                        if not ttft_emitted:
                            ttft_emitted = True
                            yield f"__ttft__:{_now_ms() - t0:.2f}__\n"
                        yield text
        except Exception as exc:  # noqa: BLE001
            yield _error_event(exc)

    def check_health(self) -> Dict[str, Any]:
        # No network call: BYOK health must not spend the user's tokens.
        return {"provider": self.name, "model": self.model, "available": bool(self._api_key)}

    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def close(self) -> None:
        """No-op: httpx clients are created per call."""
