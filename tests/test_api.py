"""Unit tests for SherlockRequestHandler without network socket binding.

Covers:
- Health endpoint (Ollama online/offline)
- Analyze endpoint with AI enabled/disabled
- Model-free indexing endpoint
- Missing workspace validation
- Frontend state transitions (via backend API contract)
"""

import unittest
import io
from pathlib import Path
import json
import tempfile
import shutil

from config import SherlockConfig
from app.orchestrator.orchestrator import Orchestrator
from app.api.routes import SherlockRequestHandler


class DummySocket:
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, data):
        self.wfile.write(data)


def simulate_http_request(method: str, path: str, body: bytes = b"", headers: dict = None) -> tuple[int, dict, bytes]:
    """Simulate an HTTP request against SherlockRequestHandler."""
    headers = headers or {}
    header_lines = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    if body:
        header_lines.append(f"Content-Length: {len(body)}")
    for k, v in headers.items():
        header_lines.append(f"{k}: {v}")
    header_lines.append("\r\n")

    raw_req = "\r\n".join(header_lines).encode("utf-8") + body
    sock = DummySocket(raw_req)

    client_address = ("127.0.0.1", 12345)
    server = None
    handler = SherlockRequestHandler(sock, client_address, server)

    sock.wfile.seek(0)
    raw_response = sock.wfile.read()
    status_line = raw_response.split(b"\r\n")[0].decode("utf-8")
    status_code = int(status_line.split(" ")[1])

    parts = raw_response.split(b"\r\n\r\n", 1)
    res_body = parts[1] if len(parts) > 1 else b""
    return status_code, {}, res_body


class TestAPIHandler(unittest.TestCase):
    """Tests for Sherlock API handler with AI enabled (default test config)."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = True

    def test_health_endpoint_ollama_online(self):
        """Health endpoint returns 200 with ollama status."""
        status, _, body = simulate_http_request("GET", "/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "ok")
        self.assertIn("ollama", data)

    def test_frontend_static_index(self):
        """Frontend SPA is served at root."""
        status, _, body = simulate_http_request("GET", "/")
        self.assertEqual(status, 200)
        content = body.decode("utf-8")
        self.assertIn("SHERLOCK", content)
        self.assertIn("10-PAGE BRIEFING", content)

    def test_analyze_trigger_with_ai(self):
        """Analyze endpoint starts analysis when AI is enabled."""
        project_dir = str(Path(__file__).resolve().parent.parent)
        payload = json.dumps({"project_path": project_dir}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 202)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "started")
        self.assertIn("analysis_id", data)

        # Poll status
        analysis_id = data["analysis_id"]
        status_code, _, poll_body = simulate_http_request("GET", f"/api/analysis/{analysis_id}")
        self.assertEqual(status_code, 200)
        poll_data = json.loads(poll_body.decode("utf-8"))
        self.assertEqual(poll_data["analysis_id"], analysis_id)

    def test_analyze_missing_project_path(self):
        """Analyze endpoint returns 400 when project_path is missing."""
        SherlockRequestHandler.enable_ai = True
        payload = json.dumps({}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_analyze_nonexistent_directory(self):
        """Analyze endpoint returns 400 for nonexistent directory."""
        SherlockRequestHandler.enable_ai = True
        payload = json.dumps({"project_path": "/nonexistent/path/xyz"}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)


class TestAPIHandlerNoAI(unittest.TestCase):
    """Tests for Sherlock API handler with AI disabled (offline mode)."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = False
        self.created_repo_ids = []

    def tearDown(self):
        # Clean up any repositories created during indexing tests
        from app.indexer import RepositoryIndexer
        import os
        indexer = RepositoryIndexer()
        indexer.store.connect()
        try:
            repos = indexer.store.list_repositories()
            for repo in repos:
                indexer.remove_repository(repo["id"])
        finally:
            indexer.store.close()
        # Remove the default database file to prevent cross-test pollution
        db_path = Path(self.cfg.index_db_path)
        if db_path.exists():
            db_path.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    p.unlink(missing_ok=True)

    def test_health_endpoint_ollama_offline(self):
        """Health endpoint returns ollama.available=False when AI disabled."""
        status, _, body = simulate_http_request("GET", "/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "ok")
        self.assertIn("ollama", data)
        self.assertFalse(data["ollama"]["available"])

    def test_analyze_requires_ai_403(self):
        """Analyze endpoint returns 403 when AI is disabled."""
        project_dir = str(Path(__file__).resolve().parent.parent)
        payload = json.dumps({"project_path": project_dir}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("AI analysis is disabled", data.get("error", ""))

    def test_index_trigger_without_ai(self):
        """Model-free indexing endpoint works without AI (returns 202)."""
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="sherlock_api_test_")
        Path(tmpdir, "test.py").write_text("x = 1\n")
        try:
            payload = json.dumps({"project_path": tmpdir, "repo_name": "test-repo"}).encode("utf-8")
            status, _, body = simulate_http_request(
                "POST",
                "/api/index",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(status, 202)
            data = json.loads(body.decode("utf-8"))
            self.assertEqual(data["status"], "started")
            self.assertIn("project_name", data)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_index_missing_project_path(self):
        """Index endpoint returns 400 when project_path is missing."""
        payload = json.dumps({}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/index",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_index_nonexistent_directory(self):
        """Index endpoint returns 400 for nonexistent directory."""
        payload = json.dumps({"project_path": "/nonexistent/path/xyz"}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/index",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_index_repos_list(self):
        """Index repos list endpoint works without AI."""
        status, _, body = simulate_http_request("GET", "/api/index/repos")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("repositories", data)
        self.assertIn("count", data)

    def test_no_unintended_model_startup(self):
        """Verify that no Ollama model is pulled or started when AI is disabled."""
        # The OllamaClient is only instantiated when enable_ai is True
        # or when /api/analyze is called. With enable_ai=False, the handler
        # returns 403 before touching Ollama.
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=json.dumps({"project_path": str(Path(__file__).resolve().parent.parent)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)


class TestFrontendContract(unittest.TestCase):
    """Tests that verify the backend API contract matches frontend expectations.

    These tests verify that the backend responses contain all fields the
    frontend JavaScript expects, ensuring frontend state transitions work
    correctly.
    """

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = False

    def test_health_response_has_ollama_available_field(self):
        """Health response includes ollama.available for frontend status display."""
        status, _, body = simulate_http_request("GET", "/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("ollama", data)
        self.assertIn("available", data["ollama"])
        # Frontend checks ollama.available to determine connected vs offline
        self.assertIsInstance(data["ollama"]["available"], bool)

    def test_index_response_has_project_name_for_frontend_tracking(self):
        """Index response includes project_name for frontend repo tracking."""
        project_dir = str(Path(__file__).resolve().parent.parent)
        payload = json.dumps({"project_path": project_dir, "repo_name": "test-repo"}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/index",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 202)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("project_name", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "started")

    def test_index_repos_response_has_repositories_array(self):
        """Index repos response includes repositories array for frontend polling."""
        status, _, body = simulate_http_request("GET", "/api/index/repos")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data["repositories"], list)
        self.assertIsInstance(data["count"], int)

    def test_analyze_error_response_has_error_field(self):
        """Analyze error response includes error field for frontend alert display."""
        project_dir = str(Path(__file__).resolve().parent.parent)
        payload = json.dumps({"project_path": project_dir}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
