"""LangSmith tracing instrumentation for Sherlock RAG pipeline.

Instruments the RAG execution path with LangSmith traces. Tracing is opt-in
and disabled by default - Sherlock works entirely without LangSmith.

Traced stages:
  1. Request validation
  2. Retrieval
  3. Context assembly
  4. Ollama generation
  5. Citation validation
  6. Grounding and abstention
  7. Final response

Only safe metadata is recorded in traces: request IDs, repository identifiers,
retrieval strategy, counts, ranks, scores, timing, and validation outcomes.
Source code, full chunks, secrets, and environment variables are never sent
to LangSmith by default.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.langsmith_config import (
    LangSmithConfig,
    get_langsmith_config,
    safe_repo_id,
    sanitize_for_trace,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# LangSmith SDK availability
# --------------------------------------------------------------------------- #

try:
    from langsmith import Client, traceable  # type: ignore

    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    Client = None  # type: ignore[assignment]
    traceable = None  # type: ignore[assignment]
    _LANGSMITH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Trace metadata structures
# --------------------------------------------------------------------------- #


@dataclass
class RAGTraceMetadata:
    """Safe, non-sensitive metadata for a RAG trace."""

    request_id: str
    repo_id: str  # Pseudonymous (truncated) repository identifier
    retrieval_method: str  # "lexical", "semantic", or "hybrid"
    limit: int
    num_chunks_retrieved: int = 0
    num_chunks_used: int = 0
    context_tokens: int = 0
    retrieval_latency_ms: Optional[float] = None
    assembly_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    citation_validation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    citations_valid: int = 0
    citations_total: int = 0
    abstained: bool = False
    abstention_reason: Optional[str] = None
    retrieval_error: Optional[str] = None
    generation_error: Optional[str] = None
    failure_stage: Optional[str] = None
    code_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for LangSmith trace metadata."""
        return {
            "request_id": self.request_id,
            "repo_id": self.repo_id,
            "retrieval_method": self.retrieval_method,
            "limit": self.limit,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "num_chunks_used": self.num_chunks_used,
            "context_tokens": self.context_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "assembly_latency_ms": self.assembly_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "citation_validation_latency_ms": self.citation_validation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model_name": self.model_name,
            "citations_valid": self.citations_valid,
            "citations_total": self.citations_total,
            "abstained": self.abstained,
            "abstention_reason": sanitize_for_trace(self.abstention_reason),
            "retrieval_error": sanitize_for_trace(self.retrieval_error),
            "generation_error": sanitize_for_trace(self.generation_error),
            "failure_stage": self.failure_stage,
            "code_revision": self.code_revision,
        }


# --------------------------------------------------------------------------- #
# RAG Tracer
# --------------------------------------------------------------------------- #


class RAGTracer:
    """Traces RAG pipeline execution using LangSmith.

    When LangSmith is unavailable or tracing is disabled, all methods are
    no-ops and RAG execution proceeds unaffected.
    """

    def __init__(self, config: Optional[LangSmithConfig] = None):
        self.config = config or get_langsmith_config()
        self._client: Optional[Any] = None
        self._trace_id: Optional[str] = None
        self._start_time: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """Return True if tracing is enabled and the SDK is available."""
        return self.config.tracing_enabled and _LANGSMITH_AVAILABLE

    @property
    def client(self) -> Optional[Any]:
        """Lazily create a LangSmith client."""
        if not self.enabled or self._client is not None:
            return self._client
        try:
            client_kwargs: Dict[str, Any] = {}
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            if self.config.endpoint:
                client_kwargs["api_url"] = self.config.endpoint
            self._client = Client(**client_kwargs)
        except Exception as exc:  # noqa: BLE001 - tracing must not break RAG
            logger.warning("Failed to create LangSmith client: %s", exc)
            self._client = None
        return self._client

    # ------------------------------------------------------------------ #
    # Trace lifecycle
    # ------------------------------------------------------------------ #

    def start_trace(
        self,
        repo_id: str,
        retrieval_method: str,
        limit: int,
        code_revision: Optional[str] = None,
    ) -> str:
        """Start a new RAG trace. Returns a request ID."""
        request_id = f"rag-{uuid.uuid4().hex[:12]}"
        self._trace_id = request_id
        self._start_time = time.monotonic()

        if self.enabled and self.client:
            try:
                self.client.create_run(
                    name="sherlock-rag-request",
                    run_type="chain",
                    inputs={
                        "repo_id": safe_repo_id(repo_id, self.config.redact_paths),
                        "retrieval_method": retrieval_method,
                        "limit": limit,
                    },
                    project_name=self.config.project,
                    extra={
                        "metadata": {
                            "request_id": request_id,
                            "repo_id": safe_repo_id(repo_id, self.config.redact_paths),
                            "retrieval_method": retrieval_method,
                            "code_revision": code_revision,
                        }
                    },
                )
            except Exception as exc:  # noqa: BLE001 - tracing must not break RAG
                logger.debug("LangSmith start_trace failed: %s", exc)

        return request_id

    def end_trace(self, metadata: RAGTraceMetadata) -> None:
        """End the current RAG trace with final metadata."""
        if not self.enabled or not self.client or not self._trace_id:
            return

        total_ms = None
        if self._start_time is not None:
            total_ms = (time.monotonic() - self._start_time) * 1000
            metadata.total_latency_ms = round(total_ms, 2)

        try:
            self.client.update_run(
                run_id=self._trace_id,
                outputs={
                    "abstained": metadata.abstained,
                    "abstention_reason": sanitize_for_trace(metadata.abstention_reason),
                    "citations_valid": metadata.citations_valid,
                    "citations_total": metadata.citations_total,
                    "num_chunks_used": metadata.num_chunks_used,
                    "total_latency_ms": metadata.total_latency_ms,
                    "failure_stage": metadata.failure_stage,
                },
                extra={
                    "metadata": metadata.to_dict(),
                },
                project_name=self.config.project,
            )
        except Exception as exc:  # noqa: BLE001 - tracing must not break RAG
            logger.debug("LangSmith end_trace failed: %s", exc)

        self._trace_id = None
        self._start_time = None

    # ------------------------------------------------------------------ #
    # Stage instrumentation
    # ------------------------------------------------------------------ #

    @contextmanager
    def trace_stage(
        self,
        stage_name: str,
        **extra_metadata: Any,
    ):
        """Context manager for tracing a pipeline stage.

        Records stage start/end timing. Failures are logged but do not
        propagate to the RAG pipeline.
        """
        start = time.monotonic()
        stage_failed = False
        stage_error: Optional[str] = None

        try:
            if self.enabled and self.client:
                try:
                    self.client.create_run(
                        name=f"sherlock-rag-{stage_name}",
                        run_type="tool",
                        inputs={k: v for k, v in extra_metadata.items() if k not in ("question", "answer")},
                        project_name=self.config.project,
                        extra={
                            "metadata": {
                                "request_id": self._trace_id,
                                "stage": stage_name,
                            }
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LangSmith stage start failed (%s): %s", stage_name, exc)

            yield

        except Exception as exc:  # noqa: BLE001 - tracing must not break RAG
            stage_failed = True
            stage_error = str(exc)
            logger.debug("LangSmith stage error (%s): %s", stage_name, exc)
            raise

        finally:
            duration_ms = (time.monotonic() - start) * 1000
            if self.enabled and self.client:
                try:
                    self.client.update_run(
                        run_id=self._trace_id,
                        outputs={
                            "stage": stage_name,
                            "duration_ms": round(duration_ms, 2),
                            "failed": stage_failed,
                            "error": sanitize_for_trace(stage_error),
                        },
                        extra={
                            "metadata": {
                                "request_id": self._trace_id,
                                "stage": stage_name,
                                "duration_ms": round(duration_ms, 2),
                            }
                        },
                        project_name=self.config.project,
                    )
                except Exception as exc2:  # noqa: BLE001
                    logger.debug("LangSmith stage end failed (%s): %s", stage_name, exc2)

    def record_metadata(self, metadata: RAGTraceMetadata) -> None:
        """Record metadata to the current trace without ending it."""
        if not self.enabled or not self.client or not self._trace_id:
            return
        try:
            self.client.update_run(
                run_id=self._trace_id,
                extra={
                    "metadata": metadata.to_dict(),
                },
                project_name=self.config.project,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith record_metadata failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Convenience: trace a function call
    # ------------------------------------------------------------------ #

    def trace_function(self, func: Callable) -> Callable:
        """Decorator wrapper that traces a function call with LangSmith.

        If LangSmith tracing is disabled or unavailable, returns the
        original function unchanged.
        """
        if not self.enabled or traceable is None:
            return func
        return traceable(func)


# --------------------------------------------------------------------------- #
# Global tracer instance
# --------------------------------------------------------------------------- #

_tracer: Optional[RAGTracer] = None


def get_tracer() -> RAGTracer:
    """Get the global RAG tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer (mainly for testing)."""
    global _tracer
    _tracer = None


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def tracing_diagnostics() -> Dict[str, Any]:
    """Return diagnostic information about LangSmith tracing configuration.

    Used by the CLI to report tracing status without exposing credentials.
    """
    config = get_langsmith_config()
    sdk_available = _LANGSMITH_AVAILABLE

    if not config.tracing_enabled:
        status = "disabled"
    elif config.tracing_enabled and not sdk_available:
        status = "sdk_missing"
    elif config.tracing_enabled and not config.api_key:
        status = "credentials_missing"
    elif config.tracing_enabled and config.api_key and not config.is_ready():
        status = "not_ready"
    else:
        status = "enabled"

    return {
        "langsmith_installed": sdk_available,
        "tracing_enabled": config.tracing_enabled,
        "tracing_status": status,
        "project": config.project,
        "api_key_configured": config.api_key is not None,
        "capture_io": config.should_capture_io(),
        "redact_paths": config.redact_paths,
    }
