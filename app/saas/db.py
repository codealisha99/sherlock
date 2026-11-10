"""Persistence for the SaaS layer (users, sessions, keys, jobs, billing).

Two engines behind one API:
  * Postgres (e.g. Neon) when DATABASE_URL is a postgres:// URL -- production.
  * SQLite file otherwise -- local dev and tests.

SQL is written once in a portable subset (`?` placeholders are translated for
Postgres). Account data is kept apart from the code index so it can be backed
up and migrated independently. All methods are thread-safe.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from config import config

try:  # optional: only needed when DATABASE_URL points at Postgres
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    _UNIQUE_ERRORS: Tuple[Type[BaseException], ...] = (sqlite3.IntegrityError, psycopg.errors.UniqueViolation)
except ImportError:  # pragma: no cover - exercised only without psycopg installed
    psycopg = None  # type: ignore[assignment]
    _UNIQUE_ERRORS = (sqlite3.IntegrityError,)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hint TEXT NOT NULL,
    base_url TEXT,
    model TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, provider)
);
CREATE TABLE IF NOT EXISTS repo_owners (
    repo_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_url TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (repo_id, user_id)
);
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    github_url TEXT NOT NULL,
    repo_id TEXT,
    analysis_id TEXT,
    status TEXT NOT NULL,
    stage TEXT,
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user_kind ON usage_events(user_id, kind, created_at);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    stripe_session_id TEXT UNIQUE,
    status TEXT NOT NULL,
    plan TEXT NOT NULL,
    amount_cents INTEGER,
    currency TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY,
    received_at TEXT NOT NULL
);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def _default_db_path() -> Path:
    env = os.getenv("SHERLOCK_SAAS_DB")
    return Path(env) if env else config.index_dir / "saas.db"


def _is_postgres_url(url: Optional[str]) -> bool:
    return bool(url) and url.split("://", 1)[0] in ("postgres", "postgresql")  # type: ignore[union-attr]


class _SqliteEngine:
    dialect = "sqlite"

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connection() as c:
            c.executescript(SCHEMA)

    @staticmethod
    def sql(statement: str) -> str:
        return statement


class _PostgresEngine:
    dialect = "postgres"

    def __init__(self, url: str):
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is Postgres but psycopg is not installed")
        # ponytail: one pool per process; size via SHERLOCK_DB_POOL_MAX if Neon limits bite.
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=int(os.getenv("SHERLOCK_DB_POOL_MAX", "10")),
            kwargs={"row_factory": dict_row},
            check=ConnectionPool.check_connection,  # Neon drops idle connections
            open=True,
        )

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self.pool.connection() as conn:  # commits on success, rolls back on error
            yield conn

    def init_schema(self) -> None:
        ddl = SCHEMA.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        with self.connection() as c:
            c.execute(ddl)

    @staticmethod
    def sql(statement: str) -> str:
        # Our SQL never contains literal '?' or '%', so this translation is exact.
        return statement.replace("?", "%s")


class SaasStore:
    """Thin data-access layer. Returns plain dicts; never mutates arguments."""

    def __init__(self, db_path: Optional[Path] = None, database_url: Optional[str] = None):
        url = database_url if database_url is not None else (None if db_path else os.getenv("DATABASE_URL"))
        if _is_postgres_url(url):
            self._engine: Any = _PostgresEngine(url)  # type: ignore[arg-type]
            self.db_path: Optional[Path] = None
        else:
            self.db_path = Path(db_path) if db_path else _default_db_path()
            self._engine = _SqliteEngine(self.db_path)
        self._engine.init_schema()

    @property
    def dialect(self) -> str:
        return self._engine.dialect

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        with self._engine.connection() as conn:
            yield conn

    def _one(self, sql: str, args: tuple = ()) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(self._engine.sql(sql), args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(self._engine.sql(sql), args).fetchall()]

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._conn() as c:
            return c.execute(self._engine.sql(sql), args).rowcount

    # ------------------------------------------------------------------ users
    def create_user(self, email: str, password_hash: str) -> Dict[str, Any]:
        user_id = uuid.uuid4().hex
        try:
            self._exec(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, password_hash, iso(utc_now())),
            )
        except _UNIQUE_ERRORS:
            raise ValueError("email_taken")
        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM users WHERE id = ?", (user_id,))

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM users WHERE email = ?", (email,))

    def get_user_by_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,))

    def update_user(self, user_id: str, **fields: Any) -> None:
        allowed = {"password_hash", "plan", "stripe_customer_id", "failed_logins", "locked_until"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {sorted(bad)}")
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE users SET {cols} WHERE id = ?", (*fields.values(), user_id))

    # --------------------------------------------------------------- sessions
    def create_session(self, token_hash: str, user_id: str, ttl: timedelta) -> None:
        now = utc_now()
        self._exec(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token_hash, user_id, iso(now), iso(now + ttl)),
        )

    def get_session_user(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._one(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash, iso(utc_now())),
        )

    def delete_session(self, token_hash: str) -> None:
        self._exec("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def delete_user_sessions(self, user_id: str) -> None:
        self._exec("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    # --------------------------------------------------------------- api keys
    def upsert_api_key(
        self, user_id: str, provider: str, encrypted_key: str, key_hint: str,
        base_url: Optional[str], model: Optional[str],
    ) -> None:
        self._exec(
            "INSERT INTO api_keys (id, user_id, provider, encrypted_key, key_hint, base_url, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, provider) DO UPDATE SET "
            "encrypted_key = excluded.encrypted_key, key_hint = excluded.key_hint, "
            "base_url = excluded.base_url, model = excluded.model, created_at = excluded.created_at",
            (uuid.uuid4().hex, user_id, provider, encrypted_key, key_hint, base_url, model, iso(utc_now())),
        )

    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """Metadata only -- never returns the encrypted secret."""
        return self._all(
            "SELECT id, provider, key_hint, base_url, model, created_at FROM api_keys "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )

    def get_api_key(self, user_id: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if provider:
            return self._one(
                "SELECT * FROM api_keys WHERE user_id = ? AND provider = ?", (user_id, provider)
            )
        return self._one(
            "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)
        )

    def delete_api_key(self, user_id: str, provider: str) -> bool:
        return self._exec(
            "DELETE FROM api_keys WHERE user_id = ? AND provider = ?", (user_id, provider)
        ) > 0

    # ------------------------------------------------------------ repo owners
    def add_repo_owner(self, repo_id: str, user_id: str, source_url: Optional[str] = None) -> None:
        self._exec(
            "INSERT INTO repo_owners (repo_id, user_id, source_url, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (repo_id, user_id) DO NOTHING",
            (repo_id, user_id, source_url, iso(utc_now())),
        )

    def user_owns_repo(self, user_id: str, repo_id: str) -> bool:
        return self._one(
            "SELECT 1 AS ok FROM repo_owners WHERE user_id = ? AND repo_id = ?", (user_id, repo_id)
        ) is not None

    def list_user_repo_ids(self, user_id: str) -> List[str]:
        return [r["repo_id"] for r in self._all(
            "SELECT repo_id FROM repo_owners WHERE user_id = ?", (user_id,)
        )]

    def remove_repo_owner(self, repo_id: str, user_id: str) -> None:
        self._exec("DELETE FROM repo_owners WHERE repo_id = ? AND user_id = ?", (repo_id, user_id))

    # ------------------------------------------------------------ ingest jobs
    def create_job(self, user_id: str, github_url: str) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = iso(utc_now())
        self._exec(
            "INSERT INTO ingest_jobs (id, user_id, github_url, status, stage, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, 'queued', 'queued', 0, ?, ?)",
            (job_id, user_id, github_url, now, now),
        )
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"repo_id", "analysis_id", "status", "stage", "progress", "error", "result"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {sorted(bad)}")
        values = dict(fields)
        if "result" in values:
            values["result_json"] = json.dumps(values.pop("result"))
        values["updated_at"] = iso(utc_now())
        cols = ", ".join(f"{k} = ?" for k in values)
        self._exec(f"UPDATE ingest_jobs SET {cols} WHERE id = ?", (*values.values(), job_id))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self._one("SELECT * FROM ingest_jobs WHERE id = ?", (job_id,))
        if row is None:
            return None
        raw = row.pop("result_json")
        return {**row, "result": json.loads(raw) if raw else None}

    def list_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self._all(
            "SELECT id, github_url, repo_id, analysis_id, status, stage, progress, error, created_at, updated_at "
            "FROM ingest_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )

    def user_owns_analysis(self, user_id: str, analysis_id: str) -> bool:
        return self._one(
            "SELECT 1 AS ok FROM ingest_jobs WHERE user_id = ? AND analysis_id = ?", (user_id, analysis_id)
        ) is not None

    def count_active_jobs(self, user_id: str) -> int:
        row = self._one(
            "SELECT COUNT(*) AS n FROM ingest_jobs WHERE user_id = ? AND status IN ('queued', 'running')",
            (user_id,),
        )
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ usage
    def record_usage(self, user_id: str, kind: str) -> None:
        self._exec(
            "INSERT INTO usage_events (user_id, kind, created_at) VALUES (?, ?, ?)",
            (user_id, kind, iso(utc_now())),
        )

    def count_usage_since(self, user_id: str, kind: str, since: datetime) -> int:
        row = self._one(
            "SELECT COUNT(*) AS n FROM usage_events WHERE user_id = ? AND kind = ? AND created_at >= ?",
            (user_id, kind, iso(since)),
        )
        return int(row["n"]) if row else 0

    # --------------------------------------------------------------- payments
    def create_payment(
        self, user_id: str, stripe_session_id: str, plan: str,
        amount_cents: Optional[int], currency: Optional[str],
    ) -> None:
        now = iso(utc_now())
        self._exec(
            "INSERT INTO payments (id, user_id, stripe_session_id, status, plan, amount_cents, currency, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, user_id, stripe_session_id, plan, amount_cents, currency, now, now),
        )

    def get_payment_by_session(self, stripe_session_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM payments WHERE stripe_session_id = ?", (stripe_session_id,))

    def update_payment_status(self, stripe_session_id: str, status: str) -> None:
        self._exec(
            "UPDATE payments SET status = ?, updated_at = ? WHERE stripe_session_id = ?",
            (status, iso(utc_now()), stripe_session_id),
        )

    def list_payments(self, user_id: str) -> List[Dict[str, Any]]:
        return self._all(
            "SELECT stripe_session_id, status, plan, amount_cents, currency, created_at FROM payments "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )

    def mark_webhook_processed(self, event_id: str) -> bool:
        """Return True if newly recorded, False if already seen (idempotency)."""
        try:
            self._exec(
                "INSERT INTO webhook_events (event_id, received_at) VALUES (?, ?)",
                (event_id, iso(utc_now())),
            )
            return True
        except _UNIQUE_ERRORS:
            return False

    def unmark_webhook(self, event_id: str) -> None:
        """Release a claimed event id so a failed handler can be retried."""
        self._exec("DELETE FROM webhook_events WHERE event_id = ?", (event_id,))


_store: Optional[SaasStore] = None


def get_store() -> SaasStore:
    """Process-wide store; tests can reset via set_store()."""
    global _store
    if _store is None:
        _store = SaasStore()
    return _store


def set_store(store: Optional[SaasStore]) -> None:
    global _store
    _store = store
