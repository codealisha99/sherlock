"""Unit tests for SherlockRequestHandler without network socket binding."""

import unittest
import io
from pathlib import Path
import json

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
    headers = headers or {}
    header_lines = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    if body:
        header_lines.append(f"Content-Length: {len(body)}")
    for k, v in headers.items():
        header_lines.append(f"{k}: {v}")
    header_lines.append("\r\n")

    raw_req = "\r\n".join(header_lines).encode("utf-8") + body
    sock = DummySocket(raw_req)

    # Initialize handler
    client_address = ("127.0.0.1", 12345)
    server = None  # not used by handler directly
    handler = SherlockRequestHandler(sock, client_address, server)

    # Parse response from wfile
    sock.wfile.seek(0)
    raw_response = sock.wfile.read()
    status_line = raw_response.split(b"\r\n")[0].decode("utf-8")
    status_code = int(status_line.split(" ")[1])

    # Extract body
    parts = raw_response.split(b"\r\n\r\n", 1)
    res_body = parts[1] if len(parts) > 1 else b""
    return status_code, {}, res_body


class TestAPIHandler(unittest.TestCase):
    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator

    def test_health_endpoint(self):
        status, _, body = simulate_http_request("GET", "/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "ok")
        self.assertIn("ollama", data)

    def test_frontend_static_index(self):
        status, _, body = simulate_http_request("GET", "/")
        self.assertEqual(status, 200)
        content = body.decode("utf-8")
        self.assertIn("SHERLOCK", content)
        self.assertIn("10-PAGE BRIEFING", content)

    def test_analyze_trigger(self):
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


if __name__ == "__main__":
    unittest.main()
