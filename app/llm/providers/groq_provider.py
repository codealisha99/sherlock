"""LLM provider implementations for the Sherlock LLM Gateway."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from app.llm.models import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    TokenUsage,
)

logger = logging.getLogger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_API_BASE}/chat/completions"


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~0.75 words per token + punctuation)."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 0.75) + text.count(".") + text.count(","))


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class GroqProvider:
    """GroqCloud chat-completions provider.

    Uses the Groq Python SDK when available, falling back to a direct
    httpx call to the OpenAI-compatible chat/completions endpoint.
    Never logs or exposes API keys.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        timeout_seconds: int = 45,
        max_retries: int = 0,  # no automatic retries for partially streamed calls
    ):
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sdk = None
        self._sdk_client = None
        self._try_init_sdk()

    # ------------------------------------------------------------------ #
    # SDK init
    # ------------------------------------------------------------------ #
    def _try_init_sdk(self) -> None:
        try:
            from groq import Groq  # type: ignore

            self._sdk = Groq
            self._sdk_client = Groq(api_key=self.api_key)
        except Exception:
            # SDK unavailable; fall back to raw httpx.
            self._sdk = None
            self._sdk_client = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _messages_to_groq(self, request: GenerationRequest) -> list[Dict[str, str]]:
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in request.messages]

    def _estimate_usage(self, request: GenerationRequest, text: str) -> TokenUsage:
        prompt = request.to_prompt()
        return TokenUsage(
            estimated_input=_estimate_tokens(prompt),
            estimated_output=_estimate_tokens(text),
        )

    # ------------------------------------------------------------------ #
    # Provider interface
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = _now_ms()
        ttft_ms: Optional[float] = None

        try:
            if self._sdk_client is not None:
                return self._generate_sdk(request, t0, ttft_ms)
            return self._generate_http(request, t0, ttft_ms)
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            total_ms = _now_ms() - t0
            return self._error_result(request, exc, t0, total_ms, ttft_ms)

    def _generate_sdk(
        self, request: GenerationRequest, t0: float, ttft_ms: Optional[float]
    ) -> GenerationResult:
        assert self._sdk_client is not None
        completion = self._sdk_client.chat.completions.create(
            model=request.model or self.model,
            messages=self._messages_to_groq(request),
            temperature=request.temperature if request.temperature is not None else 0.2,
            max_tokens=request.max_output_tokens,
            stream=False,
            timeout=self.timeout_seconds,
        )
        total_ms = _now_ms() - t0
        text = completion.choices[0].message.content or ""
        ttft_ms = total_ms  # non-streaming: first token = completion

        usage = self._extract_sdk_usage(completion) or self._estimate_usage(request, text)
        return GenerationResult(
            request_id=request.request_id,
            provider=self.name,
            model=request.model or self.model,
            text=text,
            usage=usage,
            timing={"ttft_ms": ttft_ms, "total_latency_ms": total_ms},
            finish_reason=completion.choices[0].finish_reason,
        )

    def _generate_http(
        self, request: GenerationRequest, t0: float, ttft_ms: Optional[float]
    ) -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": request.model or self.model,
            "messages": self._messages_to_groq(request),
            "temperature": request.temperature if request.temperature is not None else 0.2,
            "stream": False,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(GROQ_CHAT_ENDPOINT, json=payload, headers=self._headers())
            if response.status_code == 429:
                retry_after = self._parse_retry_after(response)
                raise GroqRateLimitError(
                    "Groq rate limit exceeded",
                    status_code=429,
                    retry_after_seconds=retry_after,
                )
            response.raise_for_status()
            data = response.json()

        total_ms = _now_ms() - t0
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        ttft_ms = total_ms

        usage = self._extract_http_usage(data) or self._estimate_usage(request, text)
        return GenerationResult(
            request_id=request.request_id,
            provider=self.name,
            model=request.model or self.model,
            text=text,
            usage=usage,
            timing={"ttft_ms": ttft_ms, "total_latency_ms": total_ms},
            finish_reason=data.get("choices", [{}])[0].get("finish_reason"),
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        t0 = _now_ms()
        ttft_emitted = False
        accumulated = ""

        try:
            if self._sdk_client is not None:
                async for chunk in self._stream_sdk(request, t0, ttft_emitted, accumulated):
                    yield chunk
            else:
                async for chunk in self._stream_http(request, t0, ttft_emitted, accumulated):
                    yield chunk
        except Exception as exc:  # noqa: BLE001
            yield self._error_event(str(exc), exc)

    async def _stream_sdk(
        self, request: GenerationRequest, t0: float, ttft_emitted: bool, accumulated: str
    ):
        assert self._sdk_client is not None
        stream = self._sdk_client.chat.completions.create(
            model=request.model or self.model,
            messages=self._messages_to_groq(request),
            temperature=request.temperature if request.temperature is not None else 0.2,
            max_tokens=request.max_output_tokens,
            stream=True,
            timeout=self.timeout_seconds,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                if not ttft_emitted:
                    ttft_ms = _now_ms() - t0
                    ttft_emitted = True
                    # Yield a timing event as a special prefix the gateway can use.
                    yield f"__ttft__:{ttft_ms:.2f}__\n"
                accumulated += content
                yield content

    async def _stream_http(
        self, request: GenerationRequest, t0: float, ttft_emitted: bool, accumulated: str
    ):
        payload: Dict[str, Any] = {
            "model": request.model or self.model,
            "messages": self._messages_to_groq(request),
            "temperature": request.temperature if request.temperature is not None else 0.2,
            "stream": True,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream(
                "POST", GROQ_CHAT_ENDPOINT, json=payload, headers=self._headers()
            ) as response:
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    raise GroqRateLimitError(
                        "Groq rate limit exceeded",
                        status_code=429,
                        retry_after_seconds=retry_after,
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            if not ttft_emitted:
                                ttft_ms = _now_ms() - t0
                                ttft_emitted = True
                                yield f"__ttft__:{ttft_ms:.2f}__\n"
                            accumulated += content
                            yield content

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _extract_sdk_usage(self, completion: Any) -> Optional[TokenUsage]:
        try:
            u = completion.usage
            if u is None:
                return None
            return TokenUsage(
                estimated_input=0,
                estimated_output=0,
                provider_input=getattr(u, "prompt_tokens", None),
                provider_output=getattr(u, "completion_tokens", None),
            )
        except Exception:
            return None

    def _extract_http_usage(self, data: Dict[str, Any]) -> Optional[TokenUsage]:
        u = data.get("usage")
        if not u:
            return None
        return TokenUsage(
            estimated_input=0,
            estimated_output=0,
            provider_input=u.get("prompt_tokens"),
            provider_output=u.get("completion_tokens"),
        )

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        try:
            ra = response.headers.get("retry-after")
            if ra is not None:
                return float(ra)
        except (ValueError, TypeError):
            pass
        return None

    def _error_result(
        self,
        request: GenerationRequest,
        exc: Exception,
        t0: float,
        total_ms: float,
        ttft_ms: Optional[float],
    ) -> GenerationResult:
        category = self._error_category(exc)
        return GenerationResult(
            request_id=request.request_id,
            provider=self.name,
            model=request.model or self.model,
            text="",
            usage=TokenUsage(),
            timing={"ttft_ms": ttft_ms, "total_latency_ms": total_ms},
            finish_reason=None,
            error=str(exc),
            error_category=category,
        )

    def _error_event(self, message: str, exc: Exception) -> str:
        category = self._error_category(exc)
        payload = {"type": "error", "message": message, "error_category": category}
        return f"__error__:{json.dumps(payload)}__\n"

    @staticmethod
    def _error_category(exc: Exception) -> str:
        from httpx import HTTPStatusError, ConnectError, ReadError, RemoteProtocolError, TimeoutException

        if isinstance(exc, GroqRateLimitError):
            return "rate_limit"
        if isinstance(exc, ConnectError):
            return "network"
        if isinstance(exc, (ReadError, RemoteProtocolError)):
            return "network"
        if isinstance(exc, TimeoutException):
            return "timeout"
        if isinstance(exc, HTTPStatusError):
            if exc.response.status_code == 429:
                return "rate_limit"
            if exc.response.status_code >= 500:
                return "provider"
            return "provider"
        return "provider"

    def check_health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "available": bool(self.api_key),
            "sdk_available": self._sdk is not None,
        }

    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def close(self) -> None:
        self._sdk_client = None
        self._sdk = None


class GroqRateLimitError(Exception):
    """Groq HTTP 429 rate-limit error with optional Retry-After."""

    def __init__(self, message: str, status_code: int = 429, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
