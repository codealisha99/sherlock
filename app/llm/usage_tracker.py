"""Lightweight usage tracker for LLM Gateway telemetry.

Provides a bounded, append-only record of inference events and a local
usage summary. No external observability stack is required.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.models import GenerationResult
from app.llm.quota_manager import QuotaManager, UsageRecord


@dataclass
class InferenceEvent:
    """A single inference event for telemetry."""

    timestamp_utc: str
    request_id: str
    provider: str
    model: str
    operation: str
    repo_id: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]
    success: bool
    error_category: Optional[str] = None


class UsageTracker:
    """In-memory + SQLite-backed usage tracker.

    Writes each inference event through the QuotaManager's usage_events table
    and keeps a rolling in-memory window for fast summary computation.
    """

    def __init__(self, quota_manager: QuotaManager, max_events: int = 10_000):
        self.quota_manager = quota_manager
        self.max_events = max_events
        self._events: List[InferenceEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        request_id: str,
        provider: str,
        model: str,
        operation: str,
        repo_id: Optional[str],
        result: GenerationResult,
        success: bool = True,
        error_category: Optional[str] = None,
    ) -> None:
        """Record a completed inference event."""
        event = InferenceEvent(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            provider=provider,
            model=model,
            operation=operation,
            repo_id=repo_id,
            input_tokens=result.usage.input,
            output_tokens=result.usage.output,
            total_tokens=result.usage.total,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,
            success=success,
            error_category=error_category,
        )

        record = UsageRecord(
            timestamp_utc=event.timestamp_utc,
            request_id=event.request_id,
            provider=event.provider,
            model=event.model,
            operation=event.operation,
            repo_id=event.repo_id,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            total_tokens=event.total_tokens,
            ttft_ms=event.ttft_ms,
            total_latency_ms=event.total_latency_ms,
            success=event.success,
            error_category=event.error_category,
        )
        self.quota_manager.record_usage(record)

        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]

    def get_summary(self) -> Dict[str, Any]:
        """Return a local usage summary."""
        quota = self.quota_manager.get_usage_summary()
        with self._lock:
            events = list(self._events)

        total = len(events)
        successes = sum(1 for e in events if e.success)
        failures = total - successes

        latencies = [e.total_latency_ms for e in events if e.total_latency_ms is not None]
        ttfts = [e.ttft_ms for e in events if e.ttft_ms is not None]
        tps = [
            (e.output_tokens / (e.total_latency_ms / 1000.0))
            for e in events
            if e.success and e.total_latency_ms and e.total_latency_ms > 0 and e.output_tokens > 0
        ]

        def pct(values: List[float], p: int) -> Optional[float]:
            if not values:
                return None
            s = sorted(values)
            k = (len(s) - 1) * (p / 100)
            f = int(k)
            c = k - f
            if f + 1 < len(s):
                return s[f] + (s[f + 1] - s[f]) * c
            return s[f]

        total_input = sum(e.input_tokens for e in events)
        total_output = sum(e.output_tokens for e in events)
        total_tokens = sum(e.total_tokens for e in events)

        by_operation: Dict[str, Dict[str, Any]] = {}
        for e in events:
            b = by_operation.setdefault(
                e.operation,
                {"requests": 0, "tokens": 0, "failures": 0},
            )
            b["requests"] += 1
            b["tokens"] += e.total_tokens
            if not e.success:
                b["failures"] += 1

        error_categories: Dict[str, int] = {}
        for e in events:
            if e.error_category:
                error_categories[e.error_category] = error_categories.get(e.error_category, 0) + 1

        return {
            "quota": quota,
            "total_requests": total,
            "successes": successes,
            "failures": failures,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "ttft_p50_ms": pct(ttfts, 50),
            "ttft_p95_ms": pct(ttfts, 95),
            "total_latency_p50_ms": pct(latencies, 50),
            "total_latency_p95_ms": pct(latencies, 95),
            "avg_tokens_per_second": (sum(tps) / len(tps)) if tps else None,
            "by_operation": by_operation,
            "error_categories": error_categories,
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._events[-limit:])
        return [
            {
                "timestamp_utc": e.timestamp_utc,
                "request_id": e.request_id,
                "provider": e.provider,
                "model": e.model,
                "operation": e.operation,
                "repo_id": e.repo_id,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "total_tokens": e.total_tokens,
                "ttft_ms": e.ttft_ms,
                "total_latency_ms": e.total_latency_ms,
                "success": e.success,
                "error_category": e.error_category,
            }
            for e in events
        ]
