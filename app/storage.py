"""SQLite storage layer for Sherlock's model-free codebase index.

Provides persistent, transactional storage of repository indexes with
full-text search (FTS5) over code chunks. All operations are local and
require no LLM, embedding model, or network service.

Tables:
  repositories        - registered repositories and indexing status
  files               - scanned files with metadata and content hashes
  chunks              - searchable code chunks with line ranges
  chunk_fts           - FTS5 virtual table over chunk content
  definitions         - extracted classes, functions, methods
  imports             - extracted import statements
  symbol_references   - symbol usages/references with provenance and resolution status
  index_log           - audit log of indexing actions
"""

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import config, SherlockConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS repositories (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        indexed_at TEXT,
        status TEXT NOT NULL DEFAULT 'indexing',
        error TEXT,
        file_count INTEGER DEFAULT 0,
        total_lines INTEGER DEFAULT 0,
        schema_version INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        language TEXT,
        size_bytes INTEGER,
        line_count INTEGER,
        is_binary INTEGER DEFAULT 0,
        is_secret INTEGER DEFAULT 0,
        mtime REAL,
        content_hash TEXT,
        UNIQUE(repo_id, path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        chunk_type TEXT NOT NULL DEFAULT 'code'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        content,
        content='chunks',
        content_rowid='id',
        tokenize='unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS definitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        line INTEGER NOT NULL,
        signature TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS imports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        module TEXT NOT NULL,
        line INTEGER NOT NULL,
        kind TEXT NOT NULL DEFAULT 'import'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        action TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        details TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbol_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
        file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        symbol_name TEXT NOT NULL,
        symbol_kind TEXT NOT NULL,
        line INTEGER NOT NULL,
        column INTEGER,
        resolution_status TEXT NOT NULL DEFAULT 'unresolved',
        resolved_to_file_id TEXT,
        resolved_to_line INTEGER,
        evidence_type TEXT NOT NULL,
        evidence_detail TEXT,
        UNIQUE(repo_id, file_id, symbol_name, line)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
    CREATE INDEX IF NOT EXISTS idx_definitions_repo ON definitions(repo_id);
    CREATE INDEX IF NOT EXISTS idx_definitions_file ON definitions(file_id);
    CREATE INDEX IF NOT EXISTS idx_imports_repo ON imports(repo_id);
    CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);
    CREATE INDEX IF NOT EXISTS idx_index_log_repo ON index_log(repo_id);
    CREATE INDEX IF NOT EXISTS idx_symbol_references_repo ON symbol_references(repo_id);
    CREATE INDEX IF NOT EXISTS idx_symbol_references_symbol ON symbol_references(symbol_name);
    CREATE INDEX IF NOT EXISTS idx_symbol_references_file ON symbol_references(file_id);
    """,
]


class IndexStore:
    """SQLite-backed persistent index for one or more repositories."""

    def __init__(self, db_path: Optional[Path] = None, cfg: Optional[SherlockConfig] = None):
        self.cfg = cfg or config
        self.db_path = Path(db_path) if db_path else self.cfg.index_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self.connect()
        for statement in _SCHEMA:
            conn.executescript(statement)
        conn.commit()
        logger.info("Index schema initialized at %s", self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "IndexStore":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Repository registration
    # ------------------------------------------------------------------ #
    def register_repository(
        self,
        repo_id: str,
        path: str,
        name: str,
    ) -> str:
        """Insert a repository record (or reset its indexing state).

        Returns the actual repository id stored in the database (the upsert
        targets ``path``, so re-indexing the same path reuses the existing row).
        """
        conn = self.connect()
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO repositories (id, path, name, indexed_at, status, error, file_count, total_lines, schema_version)
            VALUES (?, ?, ?, ?, 'indexing', NULL, 0, 0, ?)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                indexed_at=excluded.indexed_at,
                status='indexing',
                error=NULL,
                file_count=0,
                total_lines=0,
                schema_version=excluded.schema_version
            """,
            (repo_id, path, name, now, SCHEMA_VERSION),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM repositories WHERE path = ?", (path,)
        ).fetchone()
        return str(row["id"]) if row else repo_id

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM repositories WHERE id = ?", (repo_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_repositories(self) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM repositories ORDER BY indexed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def finalize_repository(
        self,
        repo_id: str,
        file_count: int,
        total_lines: int,
        status: str = "complete",
        error: Optional[str] = None,
    ) -> None:
        conn = self.connect()
        conn.execute(
            """
            UPDATE repositories
            SET indexed_at = ?, status = ?, error = ?, file_count = ?, total_lines = ?
            WHERE id = ?
            """,
            (_utc_now(), status, error, file_count, total_lines, repo_id),
        )
        conn.commit()

    def remove_repository(self, repo_id: str) -> bool:
        """Delete a repository and all derived index data (cascades)."""
        conn = self.connect()
        cur = conn.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))
        conn.commit()
        removed = cur.rowcount > 0
        if removed:
            logger.info("Removed repository %s and all derived index data", repo_id)
        return removed

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #
    def upsert_file(
        self,
        repo_id: str,
        file_id: str,
        path: str,
        language: str,
        size_bytes: int,
        line_count: int,
        is_binary: bool,
        is_secret: bool,
        mtime: float,
        content_hash: str,
    ) -> None:
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO files (id, repo_id, path, language, size_bytes, line_count,
                               is_binary, is_secret, mtime, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_id, path) DO UPDATE SET
                language=excluded.language,
                size_bytes=excluded.size_bytes,
                line_count=excluded.line_count,
                is_binary=excluded.is_binary,
                is_secret=excluded.is_secret,
                mtime=excluded.mtime,
                content_hash=excluded.content_hash
            """,
            (
                file_id, repo_id, path, language, size_bytes, line_count,
                int(is_binary), int(is_secret), mtime, content_hash,
            ),
        )
        conn.commit()

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def get_files_by_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM files WHERE repo_id = ? ORDER BY path", (repo_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_file(self, file_id: str) -> None:
        conn = self.connect()
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()

    # ------------------------------------------------------------------ #
    # Chunks + FTS
    # ------------------------------------------------------------------ #
    def add_chunk(
        self,
        repo_id: str,
        file_id: str,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str = "code",
    ) -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO chunks (repo_id, file_id, content, start_line, end_line, chunk_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (repo_id, file_id, content, start_line, end_line, chunk_type),
        )
        chunk_id = cur.lastrowid
        # Mirror into FTS5 external-content table
        conn.execute(
            "INSERT INTO chunk_fts(rowid, content) VALUES (?, ?)",
            (chunk_id, content),
        )
        conn.commit()
        return chunk_id

    def delete_chunks_for_file(self, file_id: str) -> None:
        conn = self.connect()
        rows = conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        for r in rows:
            conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (r["id"],))
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.commit()

    def search_chunks(
        self,
        repo_id: str,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Lexical FTS search over chunks within a repository, ranked by FTS5 bm25."""
        conn = self.connect()
        # FTS5 query: prefix terms for partial matching, AND across terms
        fts_query = " OR ".join(f'"{term}*"' for term in query.split() if term)
        if not fts_query:
            return []
        rows = conn.execute(
            """
            SELECT c.id, c.file_id, c.content, c.start_line, c.end_line,
                   c.chunk_type, f.path AS file_path,
                   bm25(chunk_fts) AS rank
            FROM chunk_fts
            JOIN chunks c ON c.id = chunk_fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunk_fts MATCH ? AND c.repo_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, repo_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Definitions & imports
    # ------------------------------------------------------------------ #
    def add_definition(
        self,
        repo_id: str,
        file_id: str,
        name: str,
        kind: str,
        line: int,
        signature: Optional[str] = None,
    ) -> None:
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO definitions (repo_id, file_id, name, kind, line, signature)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (repo_id, file_id, name, kind, line, signature),
        )
        conn.commit()

    def add_import(
        self,
        repo_id: str,
        file_id: str,
        module: str,
        line: int,
        kind: str = "import",
    ) -> None:
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO imports (repo_id, file_id, module, line, kind)
            VALUES (?, ?, ?, ?, ?)
            """,
            (repo_id, file_id, module, line, kind),
        )
        conn.commit()

    def get_definitions_by_file(self, file_id: str) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM definitions WHERE file_id = ? ORDER BY line", (file_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_definitions_by_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM definitions WHERE repo_id = ? ORDER BY file_id, line", (repo_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def search_definitions(self, repo_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self.connect()
        like = f"%{query}%"
        rows = conn.execute(
            """
            SELECT d.*, f.path AS file_path
            FROM definitions d
            JOIN files f ON f.id = d.file_id
            WHERE d.repo_id = ? AND d.name LIKE ?
            ORDER BY d.name
            LIMIT ?
            """,
            (repo_id, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_imports_by_file(self, file_id: str) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM imports WHERE file_id = ? ORDER BY line", (file_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_imports_by_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT i.*, f.path AS file_path
            FROM imports i
            JOIN files f ON f.id = i.file_id
            WHERE i.repo_id = ?
            ORDER BY f.path, i.line
            """,
            (repo_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Symbol references (provenance-tracked symbol usages)
    # ------------------------------------------------------------------ #
    def add_symbol_reference(
        self,
        repo_id: str,
        file_id: str,
        symbol_name: str,
        symbol_kind: str,
        line: int,
        column: Optional[int] = None,
        resolution_status: str = "unresolved",
        resolved_to_file_id: Optional[str] = None,
        resolved_to_line: Optional[int] = None,
        evidence_type: str = "text_match",
        evidence_detail: Optional[str] = None,
    ) -> int:
        """Record a symbol reference with provenance.

        Returns the new row id. Uses INSERT OR IGNORE so re-indexing the
        same file does not create duplicate reference rows.
        """
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO symbol_references
            (repo_id, file_id, symbol_name, symbol_kind, line, column,
             resolution_status, resolved_to_file_id, resolved_to_line,
             evidence_type, evidence_detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                repo_id,
                file_id,
                symbol_name,
                symbol_kind,
                line,
                column,
                resolution_status,
                resolved_to_file_id,
                resolved_to_line,
                evidence_type,
                evidence_detail,
            ),
        )
        conn.commit()
        # If the row already existed (UNIQUE constraint), lastrowid may be 0;
        # fetch the existing id instead.
        row_id = cur.lastrowid
        if row_id == 0:
            existing = conn.execute(
                """
                SELECT id FROM symbol_references
                WHERE repo_id = ? AND file_id = ? AND symbol_name = ? AND line = ?
                """,
                (repo_id, file_id, symbol_name, line),
            ).fetchone()
            row_id = existing["id"] if existing else 0
        return int(row_id)

    def get_symbol_references(
        self,
        repo_id: str,
        symbol_name: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return symbol references for a repository.

        Optionally filter by symbol name or file id. Each result includes
        provenance fields so callers can trace back to source evidence.
        """
        conn = self.connect()
        query = """
            SELECT sr.*, f.path AS file_path
            FROM symbol_references sr
            JOIN files f ON f.id = sr.file_id
            WHERE sr.repo_id = ?
        """
        params: List[Any] = [repo_id]
        if symbol_name is not None:
            query += " AND sr.symbol_name = ?"
            params.append(symbol_name)
        if file_id is not None:
            query += " AND sr.file_id = ?"
            params.append(file_id)
        query += " ORDER BY sr.file_id, sr.line"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_symbol_definitions_with_references(
        self, repo_id: str
    ) -> List[Dict[str, Any]]:
        """Return definitions joined with their references.

        Each definition row includes a ``references`` list of reference
        dicts. References with resolution_status 'unresolved' are
        explicitly marked so consumers know the relationship is inferred
        or ambiguous rather than proven.
        """
        conn = self.connect()
        defs = conn.execute(
            """
            SELECT d.*, f.path AS file_path
            FROM definitions d
            JOIN files f ON f.id = d.file_id
            WHERE d.repo_id = ?
            ORDER BY f.path, d.line
            """,
            (repo_id,),
        ).fetchall()
        result: List[Dict[str, Any]] = []
        for d in defs:
            d_dict = dict(d)
            refs = conn.execute(
                """
                SELECT sr.*, f.path AS file_path
                FROM symbol_references sr
                JOIN files f ON f.id = sr.file_id
                WHERE sr.repo_id = ? AND sr.symbol_name = ?
                ORDER BY sr.file_id, sr.line
                """,
                (repo_id, d_dict["name"]),
            ).fetchall()
            d_dict["references"] = [dict(r) for r in refs]
            result.append(d_dict)
        return result

    def delete_references_for_file(self, file_id: str) -> None:
        """Delete all symbol references for a given file (idempotent re-indexing)."""
        conn = self.connect()
        conn.execute("DELETE FROM symbol_references WHERE file_id = ?", (file_id,))
        conn.commit()

    # ------------------------------------------------------------------ #
    # Index log
    # ------------------------------------------------------------------ #
    def log_action(self, repo_id: str, action: str, details: Optional[str] = None) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT INTO index_log (repo_id, action, timestamp, details) VALUES (?, ?, ?, ?)",
            (repo_id, action, _utc_now(), details),
        )
        conn.commit()

    def get_index_log(self, repo_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM index_log WHERE repo_id = ? ORDER BY id DESC LIMIT ?",
            (repo_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Consistency
    # ------------------------------------------------------------------ #
    def consistency_check(self, repo_id: str) -> Dict[str, Any]:
        """Verify index consistency: orphaned chunks/definitions/imports."""
        conn = self.connect()
        orphan_chunks = conn.execute(
            """
            SELECT COUNT(*) AS c FROM chunks c
            LEFT JOIN files f ON f.id = c.file_id
            WHERE c.repo_id = ? AND f.id IS NULL
            """,
            (repo_id,),
        ).fetchone()["c"]
        orphan_defs = conn.execute(
            """
            SELECT COUNT(*) AS c FROM definitions d
            LEFT JOIN files f ON f.id = d.file_id
            WHERE d.repo_id = ? AND f.id IS NULL
            """,
            (repo_id,),
        ).fetchone()["c"]
        orphan_imports = conn.execute(
            """
            SELECT COUNT(*) AS c FROM imports i
            LEFT JOIN files f ON f.id = i.file_id
            WHERE i.repo_id = ? AND f.id IS NULL
            """,
            (repo_id,),
        ).fetchone()["c"]
        file_count = conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE repo_id = ?", (repo_id,)
        ).fetchone()["c"]
        chunk_count = conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone()["c"]
        return {
            "repo_id": repo_id,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "orphan_chunks": orphan_chunks,
            "orphan_definitions": orphan_defs,
            "orphan_imports": orphan_imports,
            "consistent": orphan_chunks == 0 and orphan_defs == 0 and orphan_imports == 0,
        }

    # ------------------------------------------------------------------ #
    # Storage inventory
    # ------------------------------------------------------------------ #
    def get_storage_inventory(self) -> Dict[str, Any]:
        """Return a summary of what is persisted and where.

        Used for documentation and security transparency: callers can
        inspect exactly which tables exist, how many rows per repository,
        and whether the database file is encrypted (it is not; protection
        relies on OS disk encryption and filesystem permissions).
        """
        conn = self.connect()
        tables = [
            "repositories",
            "files",
            "chunks",
            "chunk_fts",
            "definitions",
            "imports",
            "symbol_references",
            "index_log",
        ]
        table_info: Dict[str, Any] = {}
        for table in tables:
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {table}"
                ).fetchone()
                table_info[table] = row["c"] if row else 0
            except Exception:
                table_info[table] = "error"

        repos = conn.execute("SELECT id, path, name, file_count FROM repositories").fetchall()
        repo_list = [dict(r) for r in repos]

        db_path = str(self.db_path)
        db_exists = self.db_path.exists()
        db_size_bytes = self.db_path.stat().st_size if db_exists else 0

        return {
            "database_path": db_path,
            "database_exists": db_exists,
            "database_size_bytes": db_size_bytes,
            "database_encrypted": False,
            "encryption_mechanism": None,
            "protection_baseline": "OS disk encryption + filesystem permissions",
            "tables": table_info,
            "repositories": repo_list,
            "stored_data_summary": {
                "repositories": "metadata + indexing status",
                "files": "path, language, size, line count, content hash",
                "chunks": "60-line source snippets with start/end line ranges (raw source text)",
                "chunk_fts": "FTS5 full-text index over chunk content",
                "definitions": "symbol name, kind (function/class/method), line, signature",
                "imports": "imported module, line, kind (import/from_import)",
                "symbol_references": "symbol usage with provenance, resolution status, evidence",
                "index_log": "indexing action audit trail",
            },
        }


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
