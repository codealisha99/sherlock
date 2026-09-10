"""API Routes and HTTP Request Handler for Sherlock.

Provides RESTful endpoints for starting codebase analysis, checking progress,
retrieving the 10-page report, viewing source files, and serving the frontend SPA.
"""

import json
import logging
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler
from typing import Optional

from config import config
from app.orchestrator.orchestrator import Orchestrator
from app.analysis.context import AnalysisContext

logger = logging.getLogger(__name__)


class SherlockRequestHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for Sherlock API and Frontend."""

    orchestrator: Optional[Orchestrator] = None
    frontend_dir: Path = Path(__file__).resolve().parent.parent.parent / "frontend"

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
            health = self.orchestrator.llm.check_health() if self.orchestrator else {"available": False}
            self._send_json({"status": "ok", "ollama": health})
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
