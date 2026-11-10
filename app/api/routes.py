"""API Routes and HTTP Request Handler for Sherlock.

Provides RESTful endpoints for starting codebase analysis, checking progress,
retrieving the 10-page report, viewing source files, and serving the frontend SPA.
"""

import json
import logging
import mimetypes
import re
from pathlib import Path
import threading
import asyncio
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

from config import config
from app.saas.llm_platform import resolve_gateway
from app.saas.router import auth_required, consume_chat_quota
from app.orchestrator.orchestrator import Orchestrator
from app.analysis.context import AnalysisContext
from app.indexer import RepositoryIndexer

logger = logging.getLogger(__name__)


class SherlockRequestHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for Sherlock API and Frontend."""

    orchestrator: Optional[Orchestrator] = None
    indexer: Optional[RepositoryIndexer] = None
    gateway: Optional[Any] = None
    enable_ai: bool = False
    frontend_dir: Path = Path(__file__).resolve().parent.parent.parent / "frontend"

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        from app.saas.security import SECURITY_HEADERS
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status_code: int = 400):
        self._send_json({"error": message, "status": "error"}, status_code=status_code)

    def do_OPTIONS(self):
        # Same-origin app: no CORS preflight grants.
        self.send_response(204)
        self.end_headers()

    # ------------------------------------------------------------------ #
    # Entry points: SaaS middleware first, then legacy routes.
    # ------------------------------------------------------------------ #
    def do_GET(self):
        self._dispatch("GET", self._legacy_get)

    def do_POST(self):
        self._dispatch("POST", self._legacy_post)

    def do_DELETE(self):
        self._dispatch("DELETE", self._legacy_delete)

    def _dispatch(self, method: str, legacy) -> None:
        from app.saas.router import process

        if process(self, method):
            return
        try:
            legacy()
        except ValueError:
            # Malformed numeric params etc. -> client error, not a dropped socket.
            self._send_error_json("Invalid request parameter", 400)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected")

    def _legacy_get(self):
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
            start_line = max(1, start_line)
            end_line = max(start_line, min(end_line, start_line + 1000))
            lines = content.splitlines()[start_line - 1:end_line]
            out = [{"line": i, "content": line} for i, line in enumerate(lines, start=start_line)]
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

        # 6. Chat: list sessions for a repo
        if path == "/api/chat/sessions":
            repo_id = query.get("repo_id", [None])[0]
            if not repo_id:
                self._send_error_json("'repo_id' is required", 400)
                return
            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                sessions = indexer.store.list_chat_sessions(repo_id)
            finally:
                indexer.store.close()
            self._send_json({"sessions": sessions, "count": len(sessions)})
            return

        # 6. Static files / Frontend Single Page Application
        self._serve_static(path)

    def _legacy_post(self):
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

        # Chat: create a new session
        if path == "/api/chat/sessions":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            repo_id = data.get("repo_id", "").strip()
            title = data.get("title", "").strip() or None
            if not repo_id:
                self._send_error_json("'repo_id' is required", 400)
                return

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

            session_id = indexer.store.create_chat_session(repo_id, title=title)
            self._send_json({
                "session_id": session_id,
                "repo_id": repo_id,
                "title": title or "New Chat",
            }, status_code=201)
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

            gateway, _ = _chat_gateway(self)
            if gateway is None and auth_required():
                self._send_json(NO_MODEL_BODY, 409)
                return

            try:
                pipeline = RAGPipeline(indexer=indexer, gateway=gateway)
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

            gateway, _ = _chat_gateway(self)
            if gateway is None and auth_required():
                self._send_json(NO_MODEL_BODY, 409)
                return

            # Set SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            from app.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline(indexer=indexer, gateway=gateway)

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

        # Chat: send a message with streaming (SSE)
        if path.startswith("/api/chat/sessions/") and path.endswith("/messages"):
            parts = path.strip("/").split("/")
            # /api/chat/sessions/{id}/messages
            session_id = parts[3]
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                self._send_error_json("Invalid JSON payload", 400)
                return

            question = data.get("question", "").strip()
            retrieval_method = data.get("retrieval_method", "hybrid")
            limit = int(data.get("limit", 10))

            if not question:
                self._send_error_json("'question' is required", 400)
                return
            if retrieval_method not in ("hybrid", "lexical", "semantic"):
                self._send_error_json("'retrieval_method' must be 'hybrid', 'lexical', or 'semantic'", 400)
                return
            if limit < 1 or limit > 50:
                self._send_error_json("'limit' must be between 1 and 50", 400)
                return

            indexer = self.indexer or RepositoryIndexer()
            self.indexer = indexer
            indexer.store.connect()
            try:
                session = indexer.store.get_chat_session(session_id)
                if not session:
                    self._send_error_json(f"Session '{session_id}' not found", 404)
                    return
                repo_id = session["repo_id"]
                repo = indexer.store.get_repository(repo_id)
                if not repo:
                    self._send_error_json(f"Repository '{repo_id}' not found", 404)
                    return
            finally:
                indexer.store.close()

            try:
                focus = _parse_focus(data.get("focus"))
            except ValueError as exc:
                self._send_error_json(str(exc), 400)
                return

            # Model check and focus context come before quota and SSE headers, so a
            # request that cannot be answered costs nothing and gets a clean 409.
            gateway, _ = _chat_gateway(self)
            if gateway is None and auth_required():
                self._send_json(NO_MODEL_BODY, 409)
                return
            from app.saas.focus_context import build_focus_context

            focus_chunks = build_focus_context(repo_id, focus) if focus else []

            user = getattr(self, "saas_user", None)
            if user is not None:
                limit_error = consume_chat_quota(user)
                if limit_error:
                    self._send_error_json(limit_error, 402)
                    return

            # Save user message
            indexer.store.add_chat_message(session_id, "user", question)

            # Set SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            from app.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline(indexer=indexer, gateway=gateway)
            pipeline_question = _focused_question(question, focus)

            def _send_event(event: Dict[str, Any]) -> None:
                """Send a single SSE event."""
                try:
                    data_line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(data_line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    raise

            def _run_stream() -> None:
                """Run the async streaming generator in a new event loop."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    gen = pipeline.ask_stream(
                        repo_id=repo_id,
                        question=pipeline_question,
                        retrieval_method=retrieval_method,
                        limit=limit,
                        extra_context=focus_chunks or None,
                    )
                    full_answer = ""
                    citation_list = []
                    while True:
                        try:
                            event = loop.run_until_complete(gen.__anext__())
                            _send_event(event)
                            if event.get("type") == "token":
                                full_answer += event.get("text", "")
                            elif event.get("type") == "done":
                                citation_list = event.get("citations", [])
                                break
                        except StopAsyncIteration:
                            break
                    # Save assistant message
                    if full_answer:
                        indexer.store.add_chat_message(
                            session_id, "assistant", full_answer,
                            citations=citation_list,
                        )
                except (BrokenPipeError, ConnectionResetError, OSError):
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
                logger.debug("Client disconnected during streaming")
            return

        self._send_error_json("Endpoint not found", 404)

    def _legacy_delete(self):
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

    # Pretty URLs for the HTML pages.
    PAGE_ROUTES = {
        "": "landing.html",
        "pricing": "landing.html",
        "login": "auth.html",
        "signup": "auth.html",
        "settings": "settings.html",
        "app": "index.html",
    }

    def _serve_static(self, request_path: str):
        """Serve frontend static files (HTML, CSS, JS)."""
        clean_path = request_path.strip("/")
        clean_path = self.PAGE_ROUTES.get(clean_path, clean_path)

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
            from app.saas.security import STATIC_SECURITY_HEADERS
            for name, value in STATIC_SECURITY_HEADERS:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            logger.exception("Error reading static asset %s", file_path)
            self._send_error_json("Error reading asset", 500)

    def log_message(self, format, *args):
        # Override to use standard logging
        logger.debug(f"{self.address_string()} - {format % args}")


# --------------------------------------------------------------------------- #
# Chat helpers (SaaS): node focus, per-plan quota, BYOK gateway selection.
# --------------------------------------------------------------------------- #
# No whitespace/quotes/brackets: these values reach the prompt. `$?!` cover JS ($el) and
# Ruby (valid?, save!) identifiers that tree-sitter emits.
_FOCUS_PATH_RE = re.compile(r"^[\w./@+$-]{1,300}$")
_FOCUS_SYMBOL_RE = re.compile(r"^[A-Za-z_$][\w.$?!]{0,200}$")
_FOCUS_NODE_RE = re.compile(r"^(file|sym):[\w./@+:$?!-]{1,300}$")


def _parse_focus(raw: Any) -> Optional[Dict[str, str]]:
    """Validate the optional node focus. Values reach the prompt, so be strict."""
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("'focus' must be an object")
    file_path = raw.get("file_path") or ""
    symbol = raw.get("symbol") or ""
    node_id = raw.get("node_id") or ""
    if not all(isinstance(v, str) for v in (file_path, symbol, node_id)):
        raise ValueError("'focus' fields must be strings")
    if node_id and not _FOCUS_NODE_RE.match(node_id):
        raise ValueError("Invalid focus node_id")
    kind, _, rest = node_id.partition(":")
    if kind == "file":
        file_path = file_path or rest
    elif kind == "sym":
        path_part, _, sym_part = rest.partition("::")
        file_path = file_path or path_part
        symbol = symbol or sym_part
    if file_path and (not _FOCUS_PATH_RE.match(file_path) or ".." in file_path):
        raise ValueError("Invalid focus file_path")
    if symbol and not _FOCUS_SYMBOL_RE.match(symbol):
        raise ValueError("Invalid focus symbol")
    if not file_path and not symbol:
        return None
    return {"file_path": file_path, "symbol": symbol, "node_id": node_id}


def _focused_question(question: str, focus: Optional[Dict[str, str]]) -> str:
    if not focus:
        return question
    parts = [focus["symbol"], f"in {focus['file_path']}" if focus["file_path"] else ""]
    target = " ".join(p for p in parts if p)
    return f"{question}\n\n(Focus on: {target})"


NO_MODEL_BODY = {
    "error": "No AI model is configured. Add your own API key in Settings.",
    "code": "no_model",
}


def _chat_gateway(handler: Any) -> Tuple[Any, Optional[str]]:
    """BYOK key, then platform model, then legacy handler gateway (see llm_platform).

    Single-user local dev (auth off, no user) keeps the legacy gateway unchanged.
    """
    user = getattr(handler, "saas_user", None)
    legacy = getattr(handler, "gateway", None)
    if user is None and not auth_required():
        return legacy, "legacy"
    return resolve_gateway(user, legacy, allow_ollama=bool(getattr(handler, "enable_ai", False)))
