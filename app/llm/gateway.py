"""LLM Gateway for Sherlock.

Routes inference calls through provider adapters (GroqCloud, local Ollama)
while enforcing application-level budgets, tracking usage, normalizing
errors, and preserving streaming semantics.

Usage example:

    from app.llm.gateway import LLMGateway
    from app.llm.models import GenerationRequest

    gateway = LLMGateway(config)
    result = gateway.generate(
        GenerationRequest(
            messages=[{"role": "user", "content": "Explain this code"}],
            inference_operation="rag_answer",
            repo_id="repo-123",
        )
    )
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from app.llm.models import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    TokenUsage,
)
from app.llm.providers.groq_provider import GroqProvider, GroqRateLimitError
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.quota_manager import QuotaManager
from app.llm.usage_tracker import UsageTracker
from config import config, SherlockConfig

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """Structured gateway error."""

    def __init__(
        self,
        message: str,
        error_category: str = "provider",
        status_code: int = 500,
        retry_after_seconds: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_category = error_category
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class LLMGateway:
    """Central inference gateway for Sherlock.

    Responsibilities:
      - Provider selection (Groq or Ollama) based on configuration.
      - Quota enforcement before any inference call.
      - Usage tracking and telemetry.
      - Error normalization (no API keys leaked).
      - Streaming with accurate TTFT measurement.
    """

    def __init__(self, sherlock_cfg: Optional[SherlockConfig] = None):
        self.cfg = sherlock_cfg or config
        self._quota = QuotaManager(
            db_path=self.cfg.llm_quota_db_path,
            daily_request_budget=int(self.cfg.llm_daily_request_budget),
            daily_token_budget=int(self.cfg.llm_daily_token_budget),
            per_minute_request_limit=int(self.cfg.llm_per_minute_request_limit),
            per_minute_token_limit=int(self.cfg.llm_per_minute_token_limit),
            max_concurrent_requests=int(self.cfg.llm_max_concurrent_requests),
        )
        self._tracker = UsageTracker(self._quota)
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider: Optional[str] = None
        self._initialize_providers()
        self._lock = __import__("threading").Lock()

    # ------------------------------------------------------------------ #
    # Provider initialization
    # ------------------------------------------------------------------ #
    def _initialize_providers(self) -> None:
        provider_name = self.cfg.llm_provider.lower()

        # Ollama adapter (always available, wraps existing OllamaClient).
        ollama_client = OllamaClient(self.cfg)
        self._providers["ollama"] = OllamaProvider(ollama_client)

        # Groq provider (only if API key is configured).
        groq_key = self.cfg.llm_groq_api_key
        if groq_key:
            self._providers["groq"] = GroqProvider(
                api_key=groq_key,
                model=self.cfg.llm_groq_model,
                timeout_seconds=int(self.cfg.llm_groq_timeout_seconds),
                max_retries=0,
            )

        # Determine default provider.
        if provider_name == "groq" and "groq" in self._providers:
            self._default_provider = "groq"
        elif provider_name == "ollama":
            self._default_provider = "ollama"
        elif "groq" in self._providers:
            self._default_provider = "groq"
        else:
            self._default_provider = "ollama"

        logger.info(
            "LLM Gateway initialized. default_provider=%s, providers=%s",
            self._default_provider,
            list(self._providers.keys()),
        )

    def _get_provider(self, name: Optional[str] = None) -> LLMProvider:
        name = name or self._default_provider or "ollama"
        if name not in self._providers:
            raise GatewayError(
                f"Provider '{name}' is not configured. Available: {list(self._providers.keys())}",
                error_category="provider",
                status_code=400,
            )
        return self._providers[name]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Non-streaming generation with quota enforcement.

        Checks budget before calling the provider. Reserves max_output_tokens
        (plus estimated input) and reconciles actual usage afterward.
        """
        provider = self._get_provider(request.model and self._provider_for_model(request.model))
        return self._generate_with_quota(provider, request, stream=False)

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming generation with quota enforcement.

        Yields text chunks as they arrive. TTFT is measured against the first
        non-empty generated chunk, not the first SSE status event.
        """
        provider = self._get_provider(request.model and self._provider_for_model(request.model))
        agen = self._generate_with_quota(provider, request, stream=True)
        async for chunk in agen:
            yield chunk

    def _provider_for_model(self, model: Optional[str]) -> Optional[str]:
        """If a specific model is requested, pick a provider that supports it."""
        if not model:
            return None
        # Groq models typically contain '/', Ollama models use ':tag'.
        if "/" in model:
            return "groq" if "groq" in self._providers else None
        return "ollama" if "ollama" in self._providers else None

    # ------------------------------------------------------------------ #
    # Core generation with quota
    # ------------------------------------------------------------------ #
    def _generate_with_quota(
        self,
        provider: LLMProvider,
        request: GenerationRequest,
        stream: bool,
    ) -> Any:
        # Estimate input tokens for admission.
        estimated_input = provider.estimate_tokens(request.to_prompt())
        max_output = request.max_output_tokens or int(self.cfg.llm_max_output_tokens)

        # Check and reserve budget BEFORE calling the provider.
        decision = self._quota.check_and_reserve(
            request_id=request.request_id,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_output,
            operation=request.inference_operation,
        )
        if not decision.allowed:
            return self._budget_exceeded_result(request, decision)

        reservation_id = decision.reservation_id
        start_wall = time.time()
        t0 = time.monotonic()

        try:
            if stream:
                return self._stream_with_reconcile(
                    provider, request, reservation_id, estimated_input, max_output, t0
                )
            result = provider.generate(request)
            self._reconcile_and_record(
                provider,
                request,
                reservation_id,
                estimated_input,
                max_output,
                result,
                start_wall,
                t0,
            )
            return result
        except GatewayError:
            self._quota.reconcile(
                reservation_id,
                actual_input_tokens=estimated_input,
                actual_output_tokens=0,
                success=False,
                error_category="budget",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self._quota.reconcile(
                reservation_id,
                actual_input_tokens=estimated_input,
                actual_output_tokens=0,
                success=False,
                error_category=self._error_category(exc),
            )
            raise GatewayError(
                f"Generation failed: {exc}",
                error_category=self._error_category(exc),
                status_code=502,
            ) from exc

    # ------------------------------------------------------------------ #
    # Streaming with reconcile
    # ------------------------------------------------------------------ #
    async def _stream_with_reconcile(
        self,
        provider: LLMProvider,
        request: GenerationRequest,
        reservation_id: str,
        estimated_input: int,
        max_output: int,
        t0: float,
    ):
        """Async generator that streams chunks and reconciles at the end."""
        accumulated = ""
        ttft_ms: Optional[float] = None
        error_category: Optional[str] = None
        error_message: Optional[str] = None
        success = True

        try:
            async for chunk in provider.generate_stream(request):
                # Gateway-internal timing metadata (not sent to frontend as-is;
                # the gateway consumer extracts it). We prefix with a marker.
                if chunk.startswith("__ttft__:"):
                    ttft_ms = float(chunk.split(":", 1)[1].split("__", 1)[0])
                    continue
                if chunk.startswith("__error__:"):
                    import json

                    payload = json.loads(chunk[len("__error__:") : -2])
                    error_message = payload.get("message", "Stream error")
                    error_category = payload.get("error_category", "provider")
                    success = False
                    break
                accumulated += chunk
                yield chunk

            total_ms = (time.monotonic() - t0) * 1000
            actual_output = provider.estimate_tokens(accumulated)

            result = GenerationResult(
                request_id=request.request_id,
                provider=provider.name,
                model=request.model or getattr(provider, "model", ""),
                text=accumulated,
                usage=TokenUsage(
                    estimated_input=estimated_input,
                    estimated_output=actual_output,
                ),
                timing={"ttft_ms": ttft_ms, "total_latency_ms": total_ms},
                finish_reason="stop" if success else None,
                error=error_message,
                error_category=error_category,
            )

            self._reconcile_and_record(
                provider,
                request,
                reservation_id,
                estimated_input,
                max_output,
                result,
                start_wall=start_wall,
                t0=t0,
                streamed=True,
            )

        except Exception as exc:  # noqa: BLE001
            error_category = self._error_category(exc)
            success = False
            total_ms = (time.monotonic() - t0) * 1000
            yield f"__error__:{self._json_dumps({'message': str(exc), 'error_category': error_category})}__\n"

            result = GenerationResult(
                request_id=request.request_id,
                provider=provider.name,
                model=request.model or getattr(provider, "model", ""),
                text=accumulated,
                usage=TokenUsage(estimated_input=estimated_input, estimated_output=0),
                timing={"ttft_ms": ttft_ms, "total_latency_ms": total_ms},
                error=str(exc),
                error_category=error_category,
            )
            self._quota.reconcile(
                reservation_id,
                actual_input_tokens=estimated_input,
                actual_output_tokens=0,
                success=False,
                error_category=error_category,
            )
            self._tracker.record(
                request_id=request.request_id,
                provider=provider.name,
                model=request.model or getattr(provider, "model", ""),
                operation=request.inference_operation,
                repo_id=request.repo_id,
                result=result,
                success=False,
                error_category=error_category,
            )

    # ------------------------------------------------------------------ #
    # Reconcile + track
    # ------------------------------------------------------------------ #
    def _reconcile_and_record(
        self,
        provider: LLMProvider,
        request: GenerationRequest,
        reservation_id: str,
        estimated_input: int,
        max_output: int,
        result: GenerationResult,
        start_wall: float,
        t0: float,
        streamed: bool = False,
    ) -> None:
        actual_input = estimated_input
        actual_output = result.usage.output
        success = result.error is None

        self._quota.reconcile(
            reservation_id,
            actual_input_tokens=actual_input,
            actual_output_tokens=actual_output,
            success=success,
            error_category=result.error_category,
        )

        self._tracker.record(
            request_id=request.request_id,
            provider=provider.name,
            model=request.model or getattr(provider, "model", ""),
            operation=request.inference_operation,
            repo_id=request.repo_id,
            result=result,
            success=success,
            error_category=result.error_category,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _budget_exceeded_result(
        self, request: GenerationRequest, decision: Any
    ) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            provider="gateway",
            model=request.model or "",
            text="",
            usage=TokenUsage(),
            timing={"ttft_ms": None, "total_latency_ms": 0.0},
            error=(
                "Sherlock's daily inference budget has been reached. "
                "Please try again later or reduce context size."
            ),
            error_category="budget",
        )

    def _error_category(self, exc: Exception) -> str:
        if isinstance(exc, GroqRateLimitError):
            return "rate_limit"
        import httpx

        if isinstance(exc, httpx.ConnectError):
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

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        import json

        return json.dumps(obj)

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def get_usage_summary(self) -> Dict[str, Any]:
        return self._tracker.get_summary()

    def get_recent_events(self, limit: int = 50) -> list[Dict[str, Any]]:
        return self._tracker.get_recent_events(limit)

    def get_provider_health(self) -> Dict[str, Any]:
        return {
            name: p.check_health() for name, p in self._providers.items()
        }

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def close(self) -> None:
        self._quota.close()
        for p in self._providers.values():
            try:
                p.close()
            except Exception:
                pass
