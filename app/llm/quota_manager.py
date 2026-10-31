"""Quota manager for LLM Gateway budget enforcement.

Enforces daily request/token budgets, per-minute rate limiting, and maximum
concurrent inference requests using durable SQLite records. All accounting is
UTC-based.

The quota manager checks budget availability **before** a request is sent and
reconciles actual usage after the request completes. Stale reservations from
crashed or cancelled requests are reclaimed on a bounded TTL.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass
class QuotaDecision:
    """Outcome of a quota admission check."""

    allowed: bool
    reason: Optional[str] = None  # e.g. "daily_request_budget_exceeded"
    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reservation_id: Optional[str] = None  # id of the held reservation, if any
    retry_after_seconds: Optional[float] = None  # for rate-limit scenarios


@dataclass
class UsageRecord:
    """A single recorded inference usage event."""

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


class QuotaManager:
    """Thread-safe quota enforcement backed by SQLite.

    Tables:
      quota_state: daily counters (requests, tokens) keyed by UTC date.
      reservations: in-flight reservations keyed by reservation_id.
      rate_limits: per-minute request/token counters keyed by minute bucket.
    """

    def __init__(
        self,
        db_path: Path,
        daily_request_budget: int = 100,
        daily_token_budget: int = 50_000,
        per_minute_request_limit: int = 20,
        per_minute_token_limit: int = 10_000,
        max_concurrent_requests: int = 2,
        reservation_ttl_seconds: int = 300,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_request_budget = daily_request_budget
        self.daily_token_budget = daily_token_budget
        self.per_minute_request_limit = per_minute_request_limit
        self.per_minute_token_limit = per_minute_token_limit
        self.max_concurrent_requests = max_concurrent_requests
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quota_state (
                utc_date TEXT PRIMARY KEY,
                requests INTEGER NOT NULL DEFAULT 0,
                tokens INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                estimated_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rate_limits (
                minute_bucket TEXT PRIMARY KEY,
                requests INTEGER NOT NULL DEFAULT 0,
                tokens INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _utc_date() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _utc_minute() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _reclaim_stale_reservations(self, conn: sqlite3.Connection) -> None:
        """Delete reservations past their TTL and return estimated tokens to the day."""
        now = self._utc_now_iso()
        stale = conn.execute(
            "SELECT reservation_id, estimated_tokens FROM reservations WHERE expires_at < ?",
            (now,),
        ).fetchall()
        for res_id, est_tokens in stale:
            # Return tokens to the daily counter (they were never consumed).
            conn.execute(
                "UPDATE quota_state SET tokens = MAX(0, tokens - ?) WHERE utc_date = ?",
                (est_tokens, self._utc_date()),
            )
            conn.execute("DELETE FROM reservations WHERE reservation_id = ?", (res_id,))
        if stale:
            conn.commit()

    # ------------------------------------------------------------------ #
    # Admission
    # ------------------------------------------------------------------ #
    def check_and_reserve(
        self,
        request_id: str,
        estimated_input_tokens: int,
        max_output_tokens: int,
        operation: str,
    ) -> QuotaDecision:
        """Check budget availability and reserve estimated max tokens.

        Returns a QuotaDecision. If allowed, a reservation is held until
        reconcile() is called with the actual usage. The reservation holds
        max_output_tokens (conservative estimate) plus estimated input tokens.
        """
        total_estimated = estimated_input_tokens + max_output_tokens
        now = self._utc_now_iso()
        today = self._utc_date()
        minute = self._utc_minute()

        with self._lock:
            conn = self._connect()
            try:
                # Reclaim stale reservations first.
                self._reclaim_stale_reservations(conn)

                # Current daily usage.
                row = conn.execute(
                    "SELECT requests, tokens FROM quota_state WHERE utc_date = ?",
                    (today,),
                ).fetchone()
                current_requests = row["requests"] if row else 0
                current_tokens = row["tokens"] if row else 0

                # Active concurrent reservations.
                active = conn.execute(
                    "SELECT COUNT(*) AS c, COALESCE(SUM(estimated_tokens), 0) AS t FROM reservations"
                ).fetchone()
                active_count = active["c"] if active else 0
                active_tokens = active["t"] if active else 0

                # Per-minute usage.
                mrow = conn.execute(
                    "SELECT requests, tokens FROM rate_limits WHERE minute_bucket = ?",
                    (minute,),
                ).fetchone()
                minute_requests = mrow["requests"] if mrow else 0
                minute_tokens = mrow["tokens"] if mrow else 0

                # --- Budget checks ---
                if current_requests + active_count >= self.max_concurrent_requests:
                    return QuotaDecision(
                        allowed=False,
                        reason="max_concurrent_requests_exceeded",
                        remaining_requests=max(0, self.max_concurrent_requests - active_count),
                        remaining_tokens=max(0, self.daily_token_budget - current_tokens - active_tokens),
                        retry_after_seconds=60.0,
                    )

                if current_requests >= self.daily_request_budget:
                    return QuotaDecision(
                        allowed=False,
                        reason="daily_request_budget_exceeded",
                        remaining_requests=0,
                        remaining_tokens=max(0, self.daily_token_budget - current_tokens),
                        retry_after_seconds=86400.0,
                    )

                if current_tokens + active_tokens + total_estimated > self.daily_token_budget:
                    return QuotaDecision(
                        allowed=False,
                        reason="daily_token_budget_exceeded",
                        remaining_requests=max(0, self.daily_request_budget - current_requests),
                        remaining_tokens=max(0, self.daily_token_budget - current_tokens - active_tokens),
                        retry_after_seconds=86400.0,
                    )

                if minute_requests >= self.per_minute_request_limit:
                    return QuotaDecision(
                        allowed=False,
                        reason="per_minute_request_limit_exceeded",
                        remaining_requests=0,
                        remaining_tokens=max(0, self.daily_token_budget - current_tokens - active_tokens),
                        retry_after_seconds=60.0,
                    )

                if minute_tokens + total_estimated > self.per_minute_token_limit:
                    return QuotaDecision(
                        allowed=False,
                        reason="per_minute_token_limit_exceeded",
                        remaining_requests=max(0, self.daily_request_budget - current_requests),
                        remaining_tokens=max(0, self.per_minute_token_limit - minute_tokens),
                        retry_after_seconds=60.0,
                    )

                # --- Reserve ---
                reservation_id = str(uuid.uuid4())
                expires_at = datetime.now(timezone.utc).fromisoformat(now.replace("Z", "+00:00")) if False else None
                # Compute expiry simply:
                from datetime import timedelta

                expiry = datetime.now(timezone.utc) + timedelta(seconds=self.reservation_ttl_seconds)

                conn.execute(
                    """
                    INSERT INTO reservations
                    (reservation_id, request_id, estimated_tokens, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        reservation_id,
                        request_id,
                        total_estimated,
                        now,
                        expiry.isoformat(),
                    ),
                )
                # Tentatively increment daily counters so concurrent checks see the hold.
                conn.execute(
                    """
                    INSERT INTO quota_state (utc_date, requests, tokens)
                    VALUES (?, 1, ?)
                    ON CONFLICT(utc_date) DO UPDATE SET
                        requests = requests + 1,
                        tokens = tokens + ?
                    """,
                    (today, total_estimated, total_estimated),
                )
                conn.execute(
                    """
                    INSERT INTO rate_limits (minute_bucket, requests, tokens)
                    VALUES (?, 1, ?)
                    ON CONFLICT(minute_bucket) DO UPDATE SET
                        requests = requests + 1,
                        tokens = tokens + ?
                    """,
                    (minute, total_estimated, total_estimated),
                )
                conn.commit()

                return QuotaDecision(
                    allowed=True,
                    reservation_id=reservation_id,
                    remaining_requests=max(0, self.daily_request_budget - current_requests - 1),
                    remaining_tokens=max(0, self.daily_token_budget - current_tokens - total_estimated),
                )
            finally:
                # Keep connection open for reconcile; do not close here.
                pass

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #
    def reconcile(
        self,
        reservation_id: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
        success: bool = True,
        error_category: Optional[str] = None,
    ) -> None:
        """Replace a reservation's estimated hold with actual usage.

        Called after a generation completes (or fails). If the reservation
        does not exist (already reconciled or expired), this is a no-op.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT estimated_tokens FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    # Already reconciled or expired; nothing to do.
                    return

                estimated = row["estimated_tokens"]
                actual = actual_input_tokens + actual_output_tokens
                delta = actual - estimated  # can be negative (over-reserved)

                today = self._utc_date()
                minute = self._utc_minute()

                # Adjust daily tokens: remove estimated, add actual.
                conn.execute(
                    """
                    UPDATE quota_state
                    SET tokens = MAX(0, tokens - ? + ?)
                    WHERE utc_date = ?
                    """,
                    (estimated, actual, today),
                )
                # Ensure requests counter reflects at least the reserved request.
                conn.execute(
                    """
                    INSERT INTO quota_state (utc_date, requests, tokens)
                    VALUES (?, 1, 0)
                    ON CONFLICT(utc_date) DO UPDATE SET requests = MAX(requests, 1)
                    """,
                    (today,),
                )

                # Adjust rate-limit counters similarly.
                conn.execute(
                    """
                    UPDATE rate_limits
                    SET tokens = MAX(0, tokens - ? + ?)
                    WHERE minute_bucket = ?
                    """,
                    (estimated, actual, minute),
                )
                conn.execute(
                    """
                    INSERT INTO rate_limits (minute_bucket, requests, tokens)
                    VALUES (?, 1, 0)
                    ON CONFLICT(minute_bucket) DO UPDATE SET requests = MAX(requests, 1)
                    """,
                    (minute,),
                )

                # Delete the reservation.
                conn.execute(
                    "DELETE FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                )
                conn.commit()
            finally:
                pass

    # ------------------------------------------------------------------ #
    # Usage summary
    # ------------------------------------------------------------------ #
    def get_usage_summary(self) -> Dict[str, object]:
        """Return current budget status and today's usage."""
        with self._lock:
            conn = self._connect()
            try:
                today = self._utc_date()
                row = conn.execute(
                    "SELECT requests, tokens FROM quota_state WHERE utc_date = ?",
                    (today,),
                ).fetchone()
                current_requests = row["requests"] if row else 0
                current_tokens = row["tokens"] if row else 0

                active = conn.execute(
                    "SELECT COUNT(*) AS c, COALESCE(SUM(estimated_tokens), 0) AS t FROM reservations"
                ).fetchone()
                active_count = active["c"] if active else 0
                active_tokens = active["t"] if active else 0

                return {
                    "utc_date": today,
                    "daily_request_budget": self.daily_request_budget,
                    "daily_token_budget": self.daily_token_budget,
                    "requests_today": current_requests,
                    "tokens_today": current_tokens,
                    "active_reservations": active_count,
                    "active_reserved_tokens": active_tokens,
                    "remaining_requests": max(0, self.daily_request_budget - current_requests - active_count),
                    "remaining_tokens": max(0, self.daily_token_budget - current_tokens - active_tokens),
                    "max_concurrent_requests": self.max_concurrent_requests,
                    "per_minute_request_limit": self.per_minute_request_limit,
                    "per_minute_token_limit": self.per_minute_token_limit,
                }
            finally:
                pass

    def record_usage(self, record: UsageRecord) -> None:
        """Persist a usage record for telemetry (append-only)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        repo_id TEXT,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        ttft_ms REAL,
                        total_latency_ms REAL,
                        success INTEGER NOT NULL DEFAULT 1,
                        error_category TEXT
                    )
                    """,
                )
                conn.execute(
                    """
                    INSERT INTO usage_events
                    (timestamp_utc, request_id, provider, model, operation, repo_id,
                     input_tokens, output_tokens, total_tokens, ttft_ms,
                     total_latency_ms, success, error_category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.timestamp_utc,
                        record.request_id,
                        record.provider,
                        record.model,
                        record.operation,
                        record.repo_id,
                        record.input_tokens,
                        record.output_tokens,
                        record.total_tokens,
                        record.ttft_ms,
                        record.total_latency_ms,
                        1 if record.success else 0,
                        record.error_category,
                    ),
                )
                conn.commit()
            finally:
                pass

    def get_recent_usage(self, limit: int = 100) -> list[Dict[str, object]]:
        """Return the most recent usage events."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        repo_id TEXT,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        ttft_ms REAL,
                        total_latency_ms REAL,
                        success INTEGER NOT NULL DEFAULT 1,
                        error_category TEXT
                    )
                    """,
                )
                rows = conn.execute(
                    """
                    SELECT * FROM usage_events ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                pass
