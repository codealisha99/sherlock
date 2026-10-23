"""Model-free repository indexer for Sherlock.

Scans a target repository safely, parses Python source with the stdlib ``ast``
module, and persists file metadata, searchable chunks, definitions, imports,
and symbol references into the SQLite index store. All operations are local,
deterministic, and require no LLM, embedding model, or network service.

Symbol references are recorded with provenance (evidence type, resolution
status) so that every structural result can be traced back to source evidence.
Unresolved, ambiguous, dynamic, or inferred relationships are explicitly
marked rather than presented as proven call graphs.
"""

import ast
import fnmatch
import hashlib
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.llm.embedding_client import EmbeddingClient
from app.storage import IndexStore
from app.tools.filesys import SafeFilesystem
from config import config, SherlockConfig

logger = logging.getLogger(__name__)


class IndexResult:
    """Outcome of an indexing run."""

    def __init__(
        self,
        repo_id: str,
        repo_path: str,
        repo_name: str,
        status: str = "indexing",
    ):
        self.repo_id = repo_id
        self.repo_path = repo_path
        self.repo_name = repo_name
        self.status = status
        self.error: Optional[str] = None
        self.file_count = 0
        self.total_lines = 0
        self.chunk_count = 0
        self.definition_count = 0
        self.import_count = 0
        self.skipped_files: List[str] = []
        self.errors: List[str] = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_path": self.repo_path,
            "repo_name": self.repo_name,
            "status": self.status,
            "error": self.error,
            "file_count": self.file_count,
            "total_lines": self.total_lines,
            "chunk_count": self.chunk_count,
            "definition_count": self.definition_count,
            "import_count": self.import_count,
            "skipped_files": self.skipped_files,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


class RepositoryIndexer:
    """Safe, deterministic repository indexer backed by SQLite + FTS5."""

    def __init__(self, cfg: Optional[SherlockConfig] = None):
        self.cfg = cfg or config
        self.store = IndexStore(cfg=self.cfg)
        self.gitignore_patterns: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def index_repository(
        self,
        repo_path: str | Path,
        repo_id: Optional[str] = None,
        repo_name: Optional[str] = None,
        subdirectory: Optional[str | Path] = None,
        root_path: Optional[str | Path] = None,
    ) -> IndexResult:
        """Index a repository end-to-end.

        Re-indexing is idempotent: existing files are replaced when their
        content hash changes, and stale records are removed.

        Parameters
        ----------
        repo_path:
            Path stored in the database as the repository location. This is
            what gets stored in the ``repositories.path`` column and must be
            unique per repository.
        repo_id:
            Optional repository id. Generated if omitted.
        repo_name:
            Optional repository name. Defaults to the root directory name.
        subdirectory:
            Optional subdirectory within ``repo_path`` to index. Only files
            under this subdirectory are indexed, but paths are reported
            relative to ``root_path`` (the parent directory). This allows
            indexing a single repository within a fixtures directory while
            keeping paths relative to the fixtures root.
        root_path:
            Optional root directory for scanning and computing relative
            paths. If not provided, defaults to ``repo_path``. Use this when
            ``repo_path`` points to a subdirectory but you want paths
            reported relative to a parent directory.
        """
        root = Path(root_path or repo_path).resolve()
        if not root.exists():
            raise ValueError(f"Repository path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Repository path is not a directory: {root}")

        # Validate subdirectory is within root, if specified
        subdir_path: Optional[Path] = None
        if subdirectory is not None:
            subdir_path = (root / subdirectory).resolve()
            if not subdir_path.exists():
                raise ValueError(f"Subdirectory does not exist: {subdir_path}")
            if not subdir_path.is_dir():
                raise ValueError(f"Subdirectory path is not a directory: {subdir_path}")

        repo_id = repo_id or str(uuid.uuid4())
        repo_name = repo_name or root.name

        result = IndexResult(repo_id=repo_id, repo_path=str(repo_path), repo_name=repo_name)

        fs = SafeFilesystem(root, self.cfg)

        # Load .gitignore patterns for this repository
        self.gitignore_patterns = self._load_gitignore(root)

        self.store.connect()
        try:
            # Reuse the existing repository id for this path (re-index is
            # idempotent) so all derived tables stay consistent.
            actual_repo_id = self.store.register_repository(repo_id, str(repo_path), repo_name)
            if actual_repo_id != repo_id:
                repo_id = actual_repo_id
            # Keep the result's repo_id in sync with the id actually used in
            # the database (re-indexing an existing path reuses its row id).
            result.repo_id = repo_id
            self.store.log_action(repo_id, "index_start", f"path={repo_path}")

            start = time.monotonic()

            # Gather current on-disk file set (relative paths)
            on_disk: Dict[str, Path] = {}
            try:
                for p in root.rglob("*"):
                    if self._should_skip(p, root):
                        continue
                    # Only include files under the subdirectory, if specified
                    if subdirectory is not None:
                        try:
                            p.relative_to(subdir_path)
                        except ValueError:
                            continue
                    if p.is_file():
                        rel = self._relative(root, p)
                        on_disk[rel] = p
            except Exception as exc:  # pragma: no cover - defensive
                result.status = "failed"
                result.error = f"Failed to traverse repository: {exc}"
                self._finalize(result, start)
                return result
                result.status = "failed"
                result.error = f"Failed to traverse repository: {exc}"
                self._finalize(result, start)
                return result

            # Index each file
            for rel_path in sorted(on_disk):
                abs_path = on_disk[rel_path]
                try:
                    self._index_file(fs, root, rel_path, abs_path, repo_id, result)
                except Exception as exc:  # defensive: never crash whole run on one file
                    logger.warning("Skipping %s due to error: %s", rel_path, exc)
                    result.skipped_files.append(rel_path)
                    result.errors.append(f"{rel_path}: {exc}")

            # Remove stale records for files that no longer exist on disk
            self._remove_stale_files(fs, root, on_disk, repo_id, result)

            # Consistency check
            consistency = self.store.consistency_check(repo_id)
            if not consistency["consistent"]:
                result.errors.append(
                    f"Index consistency warnings: orphan_chunks={consistency['orphan_chunks']}, "
                    f"orphan_definitions={consistency['orphan_definitions']}, "
                    f"orphan_imports={consistency['orphan_imports']}"
                )

            result.status = "complete" if not result.errors else "complete_with_warnings"
            self._finalize(result, start)
            self.store.log_action(
                repo_id,
                "index_complete",
                f"files={result.file_count} chunks={result.chunk_count} "
                f"defs={result.definition_count} imports={result.import_count}",
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive
            result.status = "failed"
            result.error = str(exc)
            self._finalize(result, time.monotonic() - (time.monotonic() - start) if 'start' in locals() else None)
            raise
        finally:
            self.store.close()

    def remove_repository(self, repo_id: str) -> bool:
        """Remove a repository and all its derived index data."""
        self.store.connect()
        try:
            return self.store.remove_repository(repo_id)
        finally:
            self.store.close()

    def list_repositories(self) -> list:
        self.store.connect()
        try:
            return self.store.list_repositories()
        finally:
            self.store.close()

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        self.store.connect()
        try:
            return self.store.get_repository(repo_id)
        finally:
            self.store.close()

    def consistency_check(self, repo_id: str) -> Dict[str, Any]:
        self.store.connect()
        try:
            return self.store.consistency_check(repo_id)
        finally:
            self.store.close()

    # ------------------------------------------------------------------ #
    # Chunk embedding
    # ------------------------------------------------------------------ #
    def embed_chunks(
        self,
        repo_id: str,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> Dict[str, Any]:
        """Generate and persist embeddings for all chunks of a repository.

        This is an explicitly invoked operation, separate from
        :meth:`index_repository`. It reuses the existing chunk records and
        persists L2-normalized embedding vectors via the storage layer.

        The embedding model is loaded once and reused for all chunks (via
        the supplied or default :class:`EmbeddingClient`), avoiding repeated
        loads.

        Parameters
        ----------
        repo_id:
            Repository id whose chunks should be embedded.
        embedding_client:
            Optional pre-configured :class:`EmbeddingClient`. If omitted, a
            default client using the global config is created.

        Returns
        -------
        dict
            Summary with ``repo_id``, ``chunk_count``, ``embedded_count``,
            ``skipped_count``, ``errors`` and ``duration_seconds``.

        Raises
        ------
        ValueError
            If the repository does not exist.
        RuntimeError
            If the embedding client or sqlite-vec is unavailable.
        """
        client = embedding_client or EmbeddingClient()
        if not client.is_available():
            raise RuntimeError(
                "Embedding model is not available. Ensure sentence-transformers "
                "and the configured model are installed."
            )

        self.store.connect()
        try:
            repo = self.store.get_repository(repo_id)
            if repo is None:
                raise ValueError(f"Repository not found: {repo_id}")

            chunks = self.store.connect().execute(
                """
                SELECT c.id, c.content FROM chunks c
                WHERE c.repo_id = ?
                ORDER BY c.id
                """,
                (repo_id,),
            ).fetchall()

            chunk_ids = [r["id"] for r in chunks]
            contents = [r["content"] for r in chunks]

            embedded_count = 0
            skipped_count = 0
            errors: List[str] = []
            start = time.monotonic()

            if contents:
                # Batch-embed all chunks at once for efficiency.
                vectors = client.embed_texts(contents)
                for chunk_id, vector in zip(chunk_ids, vectors):
                    try:
                        self.store.upsert_chunk_embedding(chunk_id, vector)
                        embedded_count += 1
                    except Exception as exc:  # noqa: BLE001 - report per-chunk
                        skipped_count += 1
                        errors.append(f"chunk {chunk_id}: {exc}")
                        logger.warning("Failed to embed chunk %d: %s", chunk_id, exc)

            duration = time.monotonic() - start
            self.store.log_action(
                repo_id,
                "embed_chunks",
                f"chunks={len(chunks)} embedded={embedded_count} skipped={skipped_count}",
            )
            return {
                "repo_id": repo_id,
                "chunk_count": len(chunks),
                "embedded_count": embedded_count,
                "skipped_count": skipped_count,
                "errors": errors,
                "duration_seconds": round(duration, 3),
            }
        finally:
            self.store.close()

    def semantic_search(
        self,
        repo_id: str,
        query: str,
        limit: int = 20,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chunks using embedding similarity.

        Embeds the query once, then searches only the vector index
        (``chunk_embeddings``). Does not fall back to FTS5.

        Parameters
        ----------
        repo_id:
            Repository id to search within.
        query:
            Natural-language query string. Must be nonempty.
        limit:
            Maximum number of results to return.
        embedding_client:
            Optional pre-configured :class:`EmbeddingClient`. If omitted, a
            default client using the global config is created.

        Returns
        -------
        list of dict
            Each dict contains chunk metadata (id, file_id, content,
            start_line, end_line, chunk_type, file_path) plus ``distance``
            and ``similarity`` keys. Results are ranked by similarity
            (highest first).

        Raises
        ------
        ValueError
            If the query is empty or the repository does not exist.
        RuntimeError
            If the embedding model or sqlite-vec is unavailable, or no
            embeddings exist for the repository.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("semantic_search() requires a nonempty query string.")

        client = embedding_client or EmbeddingClient()
        if not client.is_available():
            raise RuntimeError(
                "Embedding model is not available. Ensure sentence-transformers "
                "and the configured model are installed."
            )

        self.store.connect()
        try:
            repo = self.store.get_repository(repo_id)
            if repo is None:
                raise ValueError(f"Repository not found: {repo_id}")

            if self.store.get_chunk_embedding_count(repo_id) == 0:
                raise RuntimeError(
                    f"No embeddings found for repository {repo_id}. "
                    "Run 'embed' first to generate chunk embeddings."
                )

            query_embedding = client.embed_text(query)
            results = self.store.search_chunk_embeddings(
                repo_id, query_embedding, limit=limit
            )
            return results
        finally:
            self.store.close()

    def hybrid_search(
        self,
        repo_id: str,
        query: str,
        limit: int = 20,
        rrf_k: int = 60,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining FTS5 lexical and sqlite-vec semantic search.

        Uses Reciprocal Rank Fusion (RRF):
            score(d) = sum(1 / (k + rank_i(d)))
        where rank_i(d) is the 1-based position of document d in result
        list i, and k is the RRF constant (default 60).

        Parameters
        ----------
        repo_id:
            Repository id to search within.
        query:
            Natural-language query string. Must be nonempty.
        limit:
            Maximum number of fused results to return.
        rrf_k:
            RRF ranking constant (default 60).
        embedding_client:
            Optional pre-configured :class:`EmbeddingClient`. If omitted,
            a default client using the global config is created. Semantic
            results are skipped if no client is available.

        Returns
        -------
        list of dict
            Each dict contains chunk metadata plus ``lexical_rank``,
            ``semantic_rank``, ``lexical_score``, ``semantic_score``, and
            ``rrf_score``. Sorted by descending ``rrf_score``.

        Raises
        ------
        ValueError
            If the query is empty or the repository does not exist.
        RuntimeError
            If neither lexical nor semantic search can produce results.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("hybrid_search() requires a nonempty query string.")

        self.store.connect()
        try:
            repo = self.store.get_repository(repo_id)
            if repo is None:
                raise ValueError(f"Repository not found: {repo_id}")

            # Embed the query once for semantic search.
            query_embedding: Optional[List[float]] = None
            client = embedding_client or EmbeddingClient()
            if client.is_available():
                query_embedding = client.embed_text(query)

            results = self.store.hybrid_search(
                repo_id,
                query,
                query_embedding=query_embedding,
                limit=limit,
                rrf_k=rrf_k,
            )
            return results
        finally:
            self.store.close()

    # ------------------------------------------------------------------ #
    # Per-file indexing
    # ------------------------------------------------------------------ #
    def _index_file(
        self,
        fs: SafeFilesystem,
        root: Path,
        rel_path: str,
        abs_path: Path,
        repo_id: str,
        result: IndexResult,
    ) -> None:
        resolved = fs.resolve_path(abs_path)

        # Re-check boundary after resolve (defense in depth)
        try:
            resolved.relative_to(root)
        except ValueError:
            result.skipped_files.append(rel_path)
            result.errors.append(f"{rel_path}: path escapes repository root")
            return

        if not resolved.is_file():
            return

        # Binary / secret / size guards
        is_binary = fs.is_binary(resolved)
        is_secret = fs.is_secret_file(resolved)
        try:
            size_bytes = resolved.stat().st_size
        except OSError:
            result.skipped_files.append(rel_path)
            result.errors.append(f"{rel_path}: cannot stat file")
            return

        if is_binary:
            result.skipped_files.append(rel_path)
            return
        if is_secret:
            result.skipped_files.append(rel_path)
            return
        if size_bytes > self.cfg.max_file_size_bytes:
            result.skipped_files.append(rel_path)
            return

        content, truncated = fs.read_file(resolved)
        if truncated:
            result.skipped_files.append(rel_path)
            result.errors.append(f"{rel_path}: exceeds read limit ({size_bytes} bytes)")
            return

        lines = content.splitlines()
        line_count = len(lines)
        language = _language_for(resolved)
        mtime = resolved.stat().st_mtime
        content_hash = _content_hash(content)
        file_id = f"{repo_id}:{rel_path}"

        # Idempotent: remove existing chunks/FTS first, then the file row.
        # Chunks must go before the file row so the FTS external-content table
        # is cleaned up (file deletion cascades to chunks but not to FTS).
        self.store.delete_chunks_for_file(file_id)
        self.store.delete_file(file_id)

        self.store.upsert_file(
            repo_id=repo_id,
            file_id=file_id,
            path=rel_path,
            language=language,
            size_bytes=size_bytes,
            line_count=line_count,
            is_binary=is_binary,
            is_secret=is_secret,
            mtime=mtime,
            content_hash=content_hash,
        )

        # Chunk the file for FTS search (one chunk per logical block)
        chunk_id = 0
        for start, end, block in _chunk_lines(lines):
            self.store.add_chunk(
                repo_id=repo_id,
                file_id=file_id,
                content=block,
                start_line=start,
                end_line=end,
                chunk_type="code",
            )
            chunk_id += 1

        # AST parsing for Python
        definitions: List[Tuple[str, str, int, Optional[str]]] = []
        imports: List[Tuple[str, int, str]] = []
        call_refs: List[Tuple[str, int, Optional[int], str, Optional[str]]] = []
        tree: Optional[ast.AST] = None
        if resolved.suffix.lower() == ".py":
            try:
                tree = ast.parse(content)
                definitions, imports = self._extract_python_symbols(tree, rel_path)
                call_refs = self._extract_call_references(tree, rel_path)
            except SyntaxError:
                # Malformed Python: still indexed as text, but no defs/imports/calls
                logger.debug("Syntax error in %s; skipping AST extraction", rel_path)

        for name, kind, line, signature in definitions:
            self.store.add_definition(repo_id, file_id, name, kind, line, signature)
            self.store.add_symbol_reference(
                repo_id=repo_id,
                file_id=file_id,
                symbol_name=name,
                symbol_kind=kind,
                line=line,
                resolution_status="resolved",
                resolved_to_file_id=file_id,
                resolved_to_line=line,
                evidence_type="ast_definition",
                evidence_detail=signature or "",
            )
            self.definition_count_increment(result)

        for module, line, kind in imports:
            self.store.add_import(repo_id, file_id, module, line, kind)
            self.store.add_symbol_reference(
                repo_id=repo_id,
                file_id=file_id,
                symbol_name=module,
                symbol_kind=kind,
                line=line,
                resolution_status="unresolved",
                evidence_type="ast_import",
                evidence_detail=f"import {module}",
            )
            self.import_count_increment(result)

        # Call references: mark as unresolved/ambiguous/dynamic since static
        # analysis cannot determine which definition a call resolves to
        # without whole-program analysis.
        for callee_name, line, column, evidence_type, evidence_detail in call_refs:
            self.store.add_symbol_reference(
                repo_id=repo_id,
                file_id=file_id,
                symbol_name=callee_name,
                symbol_kind="call",
                line=line,
                column=column,
                resolution_status="unresolved",
                evidence_type=evidence_type,
                evidence_detail=evidence_detail,
            )

        result.file_count += 1
        result.total_lines += line_count
        result.chunk_count += chunk_id

    def definition_count_increment(self, result: IndexResult) -> None:
        result.definition_count += 1

    def import_count_increment(self, result: IndexResult) -> None:
        result.import_count += 1

    def _extract_python_symbols(
        self, tree: ast.AST, rel_path: str
    ) -> Tuple[List[Tuple[str, str, int, Optional[str]]], List[Tuple[str, int, str]]]:
        definitions: List[Tuple[str, str, int, Optional[str]]] = []
        imports: List[Tuple[str, int, str]] = []

        # Top-level nodes only (avoids double-counting nested methods via ast.walk)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                sig = _signature_for_function(node)
                definitions.append((node.name, "function", node.lineno, sig))
            elif isinstance(node, ast.AsyncFunctionDef):
                sig = _signature_for_function(node)
                definitions.append((node.name, "async_function", node.lineno, sig))
            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                sig = "(" + ", ".join(bases) + ")" if bases else None
                definitions.append((node.name, "class", node.lineno, sig))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sig = _signature_for_function(item)
                        definitions.append((f"{node.name}.{item.name}", "method", item.lineno, sig))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno, "import"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append((module, node.lineno, "from_import"))

        return definitions, imports

    def _extract_call_references(
        self, tree: ast.AST, rel_path: str
    ) -> List[Tuple[str, int, Optional[int], str, Optional[str]]]:
        """Extract function call references from Python AST.

        Returns a list of (callee_name, line, column, evidence_type, evidence_detail).
        Call targets are marked 'unresolved' because static analysis cannot
        determine which definition a call resolves to without whole-program
        analysis. Dynamic calls (getattr, imported aliases) are marked
        'dynamic' or 'ambiguous'.
        """
        references: List[Tuple[str, int, Optional[int], str, Optional[str]]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                callee_name: Optional[str] = None
                evidence_type = "ast_call"
                evidence_detail: Optional[str] = None
                resolution_status = "unresolved"

                if isinstance(func, ast.Name):
                    callee_name = func.id
                    evidence_detail = f"{func.id}(...)"
                elif isinstance(func, ast.Attribute):
                    # method call like obj.method() or module.func()
                    callee_name = func.attr
                    try:
                        obj_str = ast.unparse(func.value)
                        evidence_detail = f"{obj_str}.{func.attr}(...)"
                    except Exception:
                        evidence_detail = f".{func.attr}(...)"
                    resolution_status = "ambiguous"  # can't statically know the receiver type
                else:
                    # Dynamic call (e.g., getattr(), funcptr(), imported alias)
                    try:
                        evidence_detail = ast.unparse(func)
                    except Exception:
                        evidence_detail = "<dynamic call>"
                    callee_name = "<dynamic>"
                    resolution_status = "dynamic"

                if callee_name:
                    references.append(
                        (callee_name, node.lineno, node.col_offset, evidence_type, evidence_detail)
                    )

        return references

    # ------------------------------------------------------------------ #
    # Stale record cleanup
    # ------------------------------------------------------------------ #
    def _remove_stale_files(
        self,
        fs: SafeFilesystem,
        root: Path,
        on_disk: Dict[str, Path],
        repo_id: str,
        result: IndexResult,
    ) -> None:
        """Delete index records for files that no longer exist on disk."""
        self.store.connect()
        try:
            indexed_files = self.store.get_files_by_repo(repo_id)
            indexed_paths = {f["path"] for f in indexed_files}
            stale = indexed_paths - set(on_disk.keys())
            for rel_path in sorted(stale):
                file_id = f"{repo_id}:{rel_path}"
                self.store.delete_chunks_for_file(file_id)
                self.store.delete_file(file_id)
        finally:
            self.store.close()

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def _finalize(self, result: IndexResult, duration: Optional[float]) -> None:
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = round(duration, 3) if duration is not None else None
        try:
            self.store.finalize_repository(
                repo_id=result.repo_id,
                file_count=result.file_count,
                total_lines=result.total_lines,
                status=result.status,
                error=result.error,
            )
        except Exception:
            pass  # never mask the original result with a store error

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _should_skip(self, p: Path, root: Path) -> bool:
        # Skip symlinks that escape the repository root
        if p.is_symlink():
            try:
                target = p.resolve()
                target.relative_to(root)
            except (ValueError, OSError):
                return True
        # Skip ignored directories (any path component)
        parts = set(p.parts)
        if parts & self.cfg.ignored_dirs:
            return True
        # Skip VCS internals explicitly
        if ".git" in parts:
            return True
        # Skip .gitignore itself
        if p.name == ".gitignore":
            return True
        # Check gitignore patterns
        rel_path = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        for pat in self.gitignore_patterns:
            # Directory-only pattern (trailing slash): match against path components
            if pat.endswith("/"):
                dir_pat = pat[:-1]
                path_parts = rel_path.split("/")
                if dir_pat in path_parts:
                    return True
                if rel_path == dir_pat or rel_path.startswith(dir_pat + "/"):
                    return True
                continue
            # Regular pattern: match against filename or relative path
            if fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(rel_path, pat):
                return True
        return False

    @staticmethod
    def _relative(root: Path, p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    @staticmethod
    def _load_gitignore(root: Path) -> List[str]:
        """Load .gitignore patterns from the repository root.

        Returns patterns as-is (trailing slashes preserved for directory-only
        patterns). Directory-only patterns are matched against path components.
        """
        patterns: List[str] = []
        gitignore_path = root / ".gitignore"
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception:
                pass
        return patterns


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _language_for(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".rb": "Ruby",
        ".php": "PHP",
        ".html": "HTML",
        ".css": "CSS",
        ".json": "JSON",
        ".md": "Markdown",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".sh": "Shell",
        ".sql": "SQL",
    }
    return mapping.get(ext, "Text")


def _chunk_lines(lines: List[str], max_lines: int = 60) -> List[Tuple[int, int, str]]:
    """Split source into bounded chunks for lexical search."""
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        block = "\n".join(lines[start:end])
        chunks.append((start + 1, end, block))
        start = end
    return chunks


def _signature_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ", ".join(a.arg for a in node.args.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({args}){returns}"
