"""API Routes and HTTP Request Handler for Sherlock.

Provides RESTful endpoints for starting codebase analysis, checking progress,
retrieving the 10-page report, viewing source files, and serving the frontend SPA.
"""

import json
import logging
import mimetypes
from pathlib import Path
import threading
import asyncio
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional

from config import config
from app.orchestrator.orchestrator import Orchestrator
from app.analysis.context import AnalysisContext
from app.indexer import RepositoryIndexer

logger = logging.getLogger(__name__)


class SherlockRequestHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for Sherlock API and Frontend."""

    orchestrator: Optional[Orchestrator] = None
    indexer: Optional[RepositoryIndexer] = None
    enable_ai: bool = False
    frontend_dir: Path = Path(__file__).resolve().parent.parent.parent / "frontend"

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Restrict CORS to same-origin by default; do not reflect arbitrary origins.
        self.send_header("Access-Control-Allow-Origin", "null")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status_code: int = 400):
        self._send_json({"error": message, "status": "error"}, status_code=status_code)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Health check
        if path == "/api/health":
            if self.enable_ai and self.orchestrator:
                health = self.orchestrator.llm.check_health()
            else:
                health = {"available": False, "ai_enabled": False}
            self._send_json({"status": "ok", "ollama": health})
            return

        # 1b. Model-free: list indexed repositories
        if path == "/api/index/repos":
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                repos = indexer.store.list_repositories()
            finally:
                indexer.store.close()
            self._send_json({"repositories": repos, "count": len(repos)})
            return

        # 1c. Model-free: repository status/detail
        if path.startswith("/api/index/repos/") and "/search" not in path and "/file" not in path:
            repo_id = path.replace("/api/index/repos/", "").strip("/")
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            repo = indexer.get_repository(repo_id)
            if not repo:
                self._send_error_json("Repository not found", 404)
                return
            c = indexer.consistency_check(repo_id)
            self._send_json({**repo, "consistency": c})
            return

        # 1f. Model-free: symbol definitions with references
        if path == "/api/index/symbols":
            repo_id = query.get("repo_id", [None])[0]
            symbol_name = query.get("symbol", [None])[0]
            if not repo_id:
                self._send_error_json("'repo_id' is required", 400)
                return
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                if symbol_name:
                    refs = indexer.store.get_symbol_references(repo_id, symbol_name=symbol_name)
                    self._send_json({"symbol": symbol_name, "references": refs, "count": len(refs)})
                else:
                    defs = indexer.store.get_symbol_definitions_with_references(repo_id)
                    self._send_json({"symbols": defs, "count": len(defs)})
            finally:
                indexer.store.close()
            return

        # 1g. Model-free: symbol references lookup
        if path == "/api/index/references":
            repo_id = query.get("repo_id", [None])[0]
            symbol_name = query.get("symbol", [None])[0]
            if not repo_id or not symbol_name:
                self._send_error_json("'repo_id' and 'symbol' are required", 400)
                return
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                refs = indexer.store.get_symbol_references(repo_id, symbol_name=symbol_name)
            finally:
                indexer.store.close()
            self._send_json({"symbol": symbol_name, "references": refs, "count": len(refs)})
            return

        # 1d. Model-free: search indexed chunks
        if path == "/api/index/search":
            repo_id = query.get("repo_id", [None])[0]
            query_term = query.get("query", [None])[0]
            limit = int(query.get("limit", ["20"])[0])
            if not repo_id or not query_term:
                self._send_error_json("'repo_id' and 'query' are required", 400)
                return
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                results = indexer.store.search_chunks(repo_id, query_term, limit=limit)
            finally:
                indexer.store.close()
            self._send_json({"results": results, "count": len(results)})
            return

        # 1e. Model-free: inspect a source file with line numbers
        if path.startswith("/api/index/repos/") and path.endswith("/file"):
            parts = path.strip("/").split("/")
            # /api/index/repos/{id}/file
            repo_id = parts[3]
            file_rel_path = query.get("path", [""])[0]
            start_line = int(query.get("start_line", ["1"])[0])
            end_line = int(query.get("end_line", ["50"])[0])
            if not file_rel_path:
                self._send_error_json("Missing 'path' query parameter", 400)
                return
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                repo_row = indexer.store.get_repository(repo_id)
                if repo_row is None:
                    self._send_error_json("Repository not found", 404)
                    return
                file_row = None
                for f in indexer.store.get_files_by_repo(repo_id):
                    if f["path"] == file_rel_path:
                        file_row = f
                        break
                if file_row is None:
                    self._send_error_json(f"File not in index: {file_rel_path}", 404)
                    return
                root = Path(repo_row["path"])
            finally:
                indexer.store.close()
            from app.tools.filesys import SafeFilesystem
            fs = SafeFilesystem(root, config)
            content, truncated = fs.read_file(file_rel_path, max_bytes=config.max_file_size_bytes)
            lines = content.splitlines()
            out = []
            for i, line in enumerate(lines, start=start_line):
                if end_line is not None and i > end_line:
                    break
                out.append({"line": i, "content": line})
            self._send_json({
                "path": file_rel_path,
                "language": file_row["language"],
                "line_count": file_row["line_count"],
                "truncated": truncated,
                "lines": out,
            })
            return

        # 2. Get Analysis Status
        if path.startswith("/api/analysis/") and "/page/" not in path and not path.endswith("/file") and not path.endswith("/report"):
            analysis_id = path.replace("/api/analysis/", "").strip("/")
            ctx = self.orchestrator.get_analysis(analysis_id) if self.orchestrator else None
            if not ctx:
                # Check disk
                report = self.orchestrator.artifact_mgr.load_ten_page_report(analysis_id) if self.orchestrator else None
                if report:
                    self._send_json({
                        "analysis_id": analysis_id,
                        "status": "completed",
                        "current_agent": "Completed",
                        "agent_statuses": {k: "completed" for k in ["Repository Mapper", "Flow Tracer", "Component Analyst", "Data Analyst", "Gap Analyst", "Teacher"]},
                        "has_report": True,
                    })
                    return
                self._send_error_json("Analysis ID not found", 404)
                return
            self._send_json(ctx.to_progress_dict())
            return

        # 3. Get Full Report
        if path.startswith("/api/analysis/") and path.endswith("/report"):
            parts = path.strip("/").split("/")
            analysis_id = parts[2]
            ctx = self.orchestrator.get_analysis(analysis_id) if self.orchestrator else None
            if ctx and ctx.ten_page_report:
                self._send_json(ctx.ten_page_report.model_dump())
                return
            # Check disk
            report = self.orchestrator.artifact_mgr.load_ten_page_report(analysis_id) if self.orchestrator else None
            if report:
                self._send_json(report.model_dump())
                return
            self._send_error_json("Report not ready or not found", 404)
            return

        # 4. Get Specific Report Page (1 to 10)
        if "/page/" in path:
            parts = path.strip("/").split("/")
            # e.g. api / analysis / {id} / page / {num}
            try:
                analysis_id = parts[2]
                page_num = int(parts[4])
            except (IndexError, ValueError):
                self._send_error_json("Invalid page URL format", 400)
                return

            ctx = self.orchestrator.get_analysis(analysis_id) if self.orchestrator else None
            report = ctx.ten_page_report if ctx else (self.orchestrator.artifact_mgr.load_ten_page_report(analysis_id) if self.orchestrator else None)

            if not report:
                self._send_error_json("Report not found", 404)
                return

            page = next((p for p in report.pages if p.page_number == page_num), None)
            if not page:
                self._send_error_json(f"Page {page_num} not found", 404)
                return

            self._send_json(page.model_dump())
            return

        # 5. Safe File Content Viewer
        if path.startswith("/api/analysis/") and path.endswith("/file"):
            parts = path.strip("/").split("/")
            analysis_id = parts[2]
            file_rel_path = query.get("path", [""])[0]

            if not file_rel_path:
                self._send_error_json("Missing 'path' query parameter", 400)
                return

            ctx = self.orchestrator.get_analysis(analysis_id) if self.orchestrator else None
            if not ctx:
                self._send_error_json("Analysis session not active", 404)
                return

            try:
                content, truncated = ctx.fs.read_file(file_rel_path)
                self._send_json({
                    "path": file_rel_path,
                    "content": content,
                    "truncated": truncated,
                })
            except Exception as e:
                self._send_error_json(str(e), 400)
            return

        # 6. Static files / Frontend Single Page Application
        self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            project_path_str = data.get("project_path", "").strip()
            if not project_path_str:
                self._send_error_json("Missing 'project_path'", 400)
                return

            project_path = Path(project_path_str).resolve()
            if not project_path.exists() or not project_path.is_dir():
                self._send_error_json(f"Directory not found: {project_path_str}", 400)
                return

            # AI analysis requires explicit opt-in
            if not self.enable_ai:
                self._send_error_json(
                    "AI analysis is disabled. Start Sherlock with --enable-ai to use Ollama-powered analysis, "
                    "or use the model-free endpoints (/api/index/*) instead.",
                    status_code=403,
                )
                return

            # Initialize orchestrator if needed
            if not self.orchestrator:
                self.orchestrator = Orchestrator()

            # Create context
            ctx = AnalysisContext(project_path=project_path)
            self.orchestrator.active_analyses[ctx.analysis_id] = ctx

            # Run in background thread
            def worker():
                try:
                    self.orchestrator.run_analysis(project_path=project_path, analysis_id=ctx.analysis_id)
                except Exception as e:
                    logger.error(f"Background analysis worker failed: {e}")

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            self._send_json({
                "status": "started",
                "analysis_id": ctx.analysis_id,
                "project_name": project_path.name,
                "project_path": str(project_path),
            }, status_code=202)
            return

        # Model-free: start indexing a repository (no LLM)
        if path == "/api/index":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            project_path_str = data.get("project_path", "").strip()
            if not project_path_str:
                self._send_error_json("Missing 'project_path'", 400)
                return

            project_path = Path(project_path_str).resolve()
            if not project_path.exists() or not project_path.is_dir():
                self._send_error_json(f"Directory not found: {project_path_str}", 400)
                return

            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            repo_id = data.get("repo_id")
            repo_name = data.get("repo_name")

            def worker():
                try:
                    indexer.index_repository(project_path, repo_id=repo_id, repo_name=repo_name)
                except Exception as e:
                    logger.error(f"Background model-free indexing failed: {e}")

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            self._send_json({
                "status": "started",
                "project_name": project_path.name,
                "project_path": str(project_path),
                "note": "Indexing runs in the background. Poll GET /api/index/repos for status.",
            }, status_code=202)
            return

        # RAG: ask a natural-language question (grounded Q&A)
        if path == "/api/ask":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            repo_id = data.get("repo_id", "").strip()
            question = data.get("question", "").strip()
            retrieval_method = data.get("retrieval_method", "hybrid")
            limit = int(data.get("limit", 10))

            if not repo_id:
                self._send_error_json("'repo_id' is required", 400)
                return
            if not question:
                self._send_error_json("'question' is required", 400)
                return
            if retrieval_method not in ("hybrid", "lexical", "semantic"):
                self._send_error_json("'retrieval_method' must be 'hybrid', 'lexical', or 'semantic'", 400)
                return
            if limit < 1 or limit > 50:
                self._send_error_json("'limit' must be between 1 and 50", 400)
                return

            from app.rag.pipeline import RAGPipeline

            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                repo = indexer.store.get_repository(repo_id)
                if not repo:
                    self._send_error_json(f"Repository '{repo_id}' not found", 404)
                    return
            finally:
                indexer.store.close()

            try:
                pipeline = RAGPipeline(indexer=indexer)
                result = pipeline.ask(
                    repo_id=repo_id,
                    question=question,
                    retrieval_method=retrieval_method,
                    limit=limit,
                )
            except Exception as exc:  # noqa: BLE001 - report pipeline failure
                logger.error(f"RAG pipeline error: {exc}")
                self._send_error_json(f"RAG pipeline error: {exc}", 500)
                return

            citations = [
                {
                    "source_id": c.source_id,
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "chunk_type": c.chunk_type,
                }
                for c in result.citations
            ]

            self._send_json({
                "query": result.query,
                "repo_id": result.repo_id,
                "answer": result.answer,
                "abstained": result.abstained,
                "abstention_reason": result.abstention_reason,
                "retrieval_method": result.retrieval_method,
                "num_chunks_retrieved": result.num_chunks_retrieved,
                "num_chunks_used": result.num_chunks_used,
                "citations": citations,
                "generation_error": result.generation_error,
                "retrieval_error": result.retrieval_error,
            })
            return

        # RAG: ask a natural-language question with streaming (SSE)
        if path == "/api/ask/stream":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            repo_id = data.get("repo_id", "").strip()
            question = data.get("question", "").strip()
            retrieval_method = data.get("retrieval_method", "hybrid")
            limit = int(data.get("limit", 10))

            if not repo_id:
                self._send_error_json("'repo_id' is required", 400)
                return
            if not question:
                self._send_error_json("'question' is required", 400)
                return
            if retrieval_method not in ("hybrid", "lexical", "semantic"):
                self._send_error_json("'retrieval_method' must be 'hybrid', 'lexical', or 'semantic'", 400)
                return
            if limit < 1 or limit > 50:
                self._send_error_json("'limit' must be between 1 and 50", 400)
                return

            from app.rag.pipeline import RAGPipeline

            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                repo = indexer.store.get_repository(repo_id)
                if not repo:
                    self._send_error_json(f"Repository '{repo_id}' not found", 404)
                    return
            finally:
                indexer.store.close()

            # Set SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

            pipeline = RAGPipeline(indexer=indexer)

            def _send_event(event: Dict[str, Any]) -> None:
                """Send a single SSE event."""
                try:
                    data_line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(data_line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected - stop streaming
                    raise

            def _run_stream() -> None:
                """Run the async streaming generator in a new event loop."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    gen = pipeline.ask_stream(
                        repo_id=repo_id,
                        question=question,
                        retrieval_method=retrieval_method,
                        limit=limit,
                    )
                    while True:
                        try:
                            event = loop.run_until_complete(gen.__anext__())
                            _send_event(event)
                            if event.get("type") in ("done", "error"):
                                break
                        except StopAsyncIteration:
                            break
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected
                    pass
                finally:
                    try:
                        loop.run_until_complete(gen.aclose())
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass

            try:
                _run_stream()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client disconnected during streaming - not an error
                logger.debug("Client disconnected during streaming")
            return

        self._send_error_json("Endpoint not found", 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Model-free: remove a repository and its derived index data
        if path.startswith("/api/index/repos/"):
            repo_id = path.replace("/api/index/repos/", "").strip("/")
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            removed = indexer.remove_repository(repo_id)
            if removed:
                self._send_json({"status": "removed", "repo_id": repo_id})
            else:
                self._send_error_json("Repository not found", 404)
            return

        self._send_error_json("Endpoint not found", 404)

    def _serve_static(self, request_path: str):
        """Serve frontend static files (HTML, CSS, JS)."""
        clean_path = request_path.lstrip("/")
        if not clean_path or clean_path == "/":
            clean_path = "index.html"

        file_path = (self.frontend_dir / clean_path).resolve()

        # Prevent directory traversal
        try:
            file_path.relative_to(self.frontend_dir)
        except ValueError:
            self._send_error_json("Forbidden", 403)
            return

        if not file_path.is_file():
            # SPA fallback to index.html
            file_path = self.frontend_dir / "index.html"

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type else mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_error_json(f"Error reading asset: {e}", 500)

    def log_message(self, format, *args):
        # Override to use standard logging
        logger.debug(f"{self.address_string()} - {format % args}")
