"""SQLite storage layer for Sherlock's model-free codebase index.

Provides persistent, transactional storage of repository indexes with
full-text search (FTS5) over code chunks. All operations are local and
require no LLM, embedding model, or network service.

Tables:
  repositories        - registered repositories and indexing status
  files               - scanned files with metadata and content hashes
  chunks              - searchable code chunks with line ranges
  chunk_fts           - FTS5 virtual table over chunk content
  chunk_embeddings    - sqlite-vec virtual table over chunk embedding vectors
  definitions         - extracted classes, functions, methods
  imports             - extracted import statements
  symbol_references   - symbol usages/references with provenance and resolution status
  index_log           - audit log of indexing actions
"""

import hashlib
import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import config, SherlockConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Optional sqlite-vec import; vector storage is unavailable without it.
try:
    import sqlite_vec

    _SQLITE_VEC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    sqlite_vec = None  # type: ignore[assignment]
    _SQLITE_VEC_AVAILABLE = False

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
            self._init_vec_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self.connect()
        for statement in _SCHEMA:
            conn.executescript(statement)
        conn.commit()
        logger.info("Index schema initialized at %s", self.db_path)

    def _init_vec_schema(self) -> None:
        """Create the sqlite-vec virtual table for chunk embeddings.

        Uses ``CREATE VIRTUAL TABLE IF NOT EXISTS`` so re-initialization on
        an existing database is safe. Vectors are stored L2-normalized so
        that L2 distance ranking is equivalent to cosine similarity ranking.
        """
        if not _SQLITE_VEC_AVAILABLE:
            logger.debug("sqlite-vec not available; skipping vector schema init")
            return
        conn = self.connect()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        dimension = int(self.cfg.embedding_dimension)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                embedding float[{dimension}]
            )
            """
        )
        conn.commit()
        logger.info(
            "Vector schema initialized at %s (dimension=%d)", self.db_path, dimension
        )

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
        """Delete a repository and all derived index data (cascades).

        Because sqlite-vec virtual tables do not support foreign keys or
        cascading deletes, chunk embeddings for this repository are removed
        explicitly before the repository row is deleted.
        """
        conn = self.connect()
        # Remove embeddings for this repo's chunks first (vec0 has no FK
        # cascade support).
        if _SQLITE_VEC_AVAILABLE:
            conn.execute(
                """
                DELETE FROM chunk_embeddings
                WHERE rowid IN (
                    SELECT c.id FROM chunks c WHERE c.repo_id = ?
                )
                """,
                (repo_id,),
            )
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
        # Delete embeddings for this file's chunks first. vec0 has no FK
        # cascade support, so deleting the file row alone would orphan vectors.
        chunk_rows = conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        if _SQLITE_VEC_AVAILABLE and chunk_rows:
            for r in chunk_rows:
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE rowid = ?",
                    (r["id"],),
                )
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
            # vec0 has no FK cascade support; remove embeddings explicitly.
            if _SQLITE_VEC_AVAILABLE:
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE rowid = ?",
                    (r["id"],),
                )
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
    # Vector storage (sqlite-vec)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_vector(embedding: List[float]) -> List[float]:
        """L2-normalize a vector so that L2 distance ranking equals cosine
        similarity ranking. Returns a new list."""
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            return list(embedding)
        return [v / norm for v in embedding]

    @staticmethod
    def _validate_embedding(embedding: List[float], dimension: int) -> List[float]:
        """Validate that an embedding is nonempty, one-dimensional, finite,
        and matches the expected dimension. Returns a normalized copy."""
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError("Embedding must be a nonempty list of floats.")
        if len(embedding) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}."
            )
        if not all(math.isfinite(v) for v in embedding):
            raise ValueError("Embedding contains non-finite values.")
        return IndexStore._normalize_vector(embedding)

    def upsert_chunk_embedding(
        self,
        chunk_id: int,
        embedding: List[float],
    ) -> None:
        """Insert or replace a chunk embedding vector.

        sqlite-vec virtual tables do not support ``ON CONFLICT`` or foreign
        keys, so an existing row with the same ``rowid`` is deleted first,
        then the new vector is inserted. The embedding is L2-normalized
        before storage so that L2 distance ranking is equivalent to cosine
        similarity ranking.
        """
        if not _SQLITE_VEC_AVAILABLE:
            raise RuntimeError("sqlite-vec is not available; vector storage is disabled.")
        dimension = int(self.cfg.embedding_dimension)
        normalized = self._validate_embedding(embedding, dimension)
        conn = self.connect()
        # Delete any existing row for this chunk_id (no ON CONFLICT on vec0).
        conn.execute("DELETE FROM chunk_embeddings WHERE rowid = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(normalized)),
        )
        conn.commit()

    def get_chunk_embedding(self, chunk_id: int) -> Optional[List[float]]:
        """Retrieve the stored embedding vector for a chunk, if any."""
        if not _SQLITE_VEC_AVAILABLE:
            return None
        conn = self.connect()
        row = conn.execute(
            "SELECT embedding FROM chunk_embeddings WHERE rowid = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        import numpy as np

        return np.frombuffer(row["embedding"], dtype=np.float32).tolist()

    def delete_chunk_embedding(self, chunk_id: int) -> None:
        """Delete the embedding vector for a chunk, if present."""
        if not _SQLITE_VEC_AVAILABLE:
            return
        conn = self.connect()
        conn.execute("DELETE FROM chunk_embeddings WHERE rowid = ?", (chunk_id,))
        conn.commit()

    def sync_chunk_embeddings(self, repo_id: str) -> Dict[str, int]:
        """Synchronize the ``chunk_embeddings`` table with the ``chunks`` table.

        Inserts missing embeddings (as zero vectors placeholder) and removes
        embeddings for chunks that no longer exist. Returns a summary of
        actions taken. Callers should replace placeholder vectors with real
        embeddings via :meth:`upsert_chunk_embedding`.

        This is useful after re-indexing when chunks may have changed.
        """
        if not _SQLITE_VEC_AVAILABLE:
            return {"inserted": 0, "deleted": 0}
        dimension = int(self.cfg.embedding_dimension)
        conn = self.connect()
        # Chunk IDs that exist in the chunks table but not in vec0.
        missing = conn.execute(
            """
            SELECT c.id FROM chunks c
            LEFT JOIN chunk_embeddings e ON e.rowid = c.id
            WHERE c.repo_id = ? AND e.rowid IS NULL
            """,
            (repo_id,),
        ).fetchall()
        # Embedding rows whose chunk no longer exists.
        orphaned = conn.execute(
            """
            SELECT e.rowid FROM chunk_embeddings e
            LEFT JOIN chunks c ON c.id = e.rowid
            WHERE c.id IS NULL
            """,
        ).fetchall()
        inserted = 0
        zero_vec = [0.0] * dimension
        for r in missing:
            conn.execute(
                "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
                (r["id"], sqlite_vec.serialize_float32(zero_vec)),
            )
            inserted += 1
        deleted = 0
        for r in orphaned:
            conn.execute("DELETE FROM chunk_embeddings WHERE rowid = ?", (r["rowid"],))
            deleted += 1
        conn.commit()
        return {"inserted": inserted, "deleted": deleted}

    def search_chunk_embeddings(
        self,
        repo_id: str,
        query_embedding: List[float],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbor search over chunk embeddings using cosine similarity.

        The query embedding is L2-normalized before search. Stored vectors are
        also L2-normalized, so L2 distance ranking is equivalent to cosine
        similarity ranking. Distance is converted to a similarity score via
        ``similarity = 1 - distance^2 / 2`` (valid for unit vectors).

        Returns a list of dicts with chunk metadata and a ``similarity`` key.
        """
        if not _SQLITE_VEC_AVAILABLE:
            raise RuntimeError("sqlite-vec is not available; vector search is disabled.")
        dimension = int(self.cfg.embedding_dimension)
        normalized = self._validate_embedding(query_embedding, dimension)
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT c.id, c.file_id, c.content, c.start_line, c.end_line,
                   c.chunk_type, f.path AS file_path,
                   e.distance
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.rowid
            JOIN files f ON f.id = c.file_id
            WHERE e.embedding MATCH ? AND k = ? AND c.repo_id = ?
            ORDER BY e.distance
            """,
            (sqlite_vec.serialize_float32(normalized), limit, repo_id),
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            distance = float(r["distance"])
            # For unit vectors: cosine_similarity = 1 - distance^2 / 2
            similarity = max(0.0, min(1.0, 1.0 - (distance * distance) / 2.0))
            results.append(
                {
                    "id": r["id"],
                    "file_id": r["file_id"],
                    "content": r["content"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "chunk_type": r["chunk_type"],
                    "file_path": r["file_path"],
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return results

    def get_chunk_embedding_count(self, repo_id: str) -> int:
        """Return the number of embedding vectors for a repository."""
        if not _SQLITE_VEC_AVAILABLE:
            return 0
        conn = self.connect()
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.rowid
            WHERE c.repo_id = ?
            """,
            (repo_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

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
            "chunk_embeddings",
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
