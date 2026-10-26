"""API tests for the /api/ask RAG endpoint.

Uses deterministic mocks for the RAG pipeline. No model downloads,
no Ollama server, no network access.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from config import SherlockConfig
from app.api.routes import SherlockRequestHandler
from app.orchestrator.orchestrator import Orchestrator


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


def simulate_http_request(method: str, path: str, body: bytes = b"", headers: dict = None) -> tuple:
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


class _FakeRepo:
    def __init__(self, repo_id="repo-test"):
        self.id = repo_id
        self.name = "test-repo"
        self.path = "/tmp/test-repo"
        self.status = "complete"
        self.file_count = 2
        self.total_lines = 100


class _FakeStore:
    """Fake store that returns a repo."""

    def __init__(self, repo_id="repo-test"):
        self.repo_id = repo_id

    def connect(self):
        pass

    def close(self):
        pass

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {
                "id": self.repo_id,
                "name": "test-repo",
                "path": "/tmp/test-repo",
                "status": "complete",
                "file_count": 2,
                "total_lines": 100,
            }
        return None


class _FakeIndexer:
    """Fake indexer with a fake store and search methods."""

    def __init__(self, repo_id="repo-test"):
        self.store = _FakeStore(repo_id)

    def hybrid_search(self, repo_id, query, query_embedding=None, limit=10, rrf_k=60, embedding_client=None):
        return []

    def semantic_search(self, repo_id, query, limit=10, embedding_client=None):
        return []


class _FakeCitation:
    def __init__(self, source_id="S1", file_path="src/calculator.py", start_line=1, end_line=2, chunk_type="code"):
        self.source_id = source_id
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type


class _FakeRAGResult:
    def __init__(
        self,
        abstained=False,
        abstention_reason=None,
        answer="The add function is defined in calculator.py [S1].",
        citations=None,
        retrieval_method="hybrid",
        num_chunks_retrieved=2,
        num_chunks_used=2,
        generation_error=None,
        retrieval_error=None,
    ):
        self.query = "How do I add?"
        self.repo_id = "repo-test"
        self.answer = answer
        self.abstained = abstained
        self.abstention_reason = abstention_reason
        self.retrieval_method = retrieval_method
        self.num_chunks_retrieved = num_chunks_retrieved
        self.num_chunks_used = num_chunks_used
        self.citations = citations or []
        self.generation_error = generation_error
        self.retrieval_error = retrieval_error


class TestAPIAsk(unittest.TestCase):
    """Tests for the /api/ask RAG endpoint."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = True

    def test_ask_grounded_answer(self):
        """A grounded answer returns 200 with answer and citations."""
        fake_result = _FakeRAGResult(
            abstained=False,
            answer="Use the add function [S1].",
            citations=[_FakeCitation()],
            retrieval_method="hybrid",
            num_chunks_retrieved=2,
            num_chunks_used=2,
        )

        with patch("app.api.routes.RepositoryIndexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                payload = json.dumps({
                    "repo_id": "repo-test",
                    "question": "How do I add?",
                }).encode("utf-8")

                status, _, body = simulate_http_request(
                    "POST",
                    "/api/ask",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["query"], "How do I add?")
        self.assertEqual(data["repo_id"], "repo-test")
        self.assertEqual(data["answer"], "Use the add function [S1].")
        self.assertFalse(data["abstained"])
        self.assertEqual(data["retrieval_method"], "hybrid")
        self.assertEqual(data["num_chunks_retrieved"], 2)
        self.assertEqual(data["num_chunks_used"], 2)
        self.assertEqual(len(data["citations"]), 1)
        self.assertEqual(data["citations"][0]["source_id"], "S1")
        self.assertEqual(data["citations"][0]["file_path"], "src/calculator.py")

    def test_ask_abstention(self):
        """Abstention returns 200 with abstained=true and a reason."""
        fake_result = _FakeRAGResult(
            abstained=True,
            abstention_reason="No relevant content found in the indexed repository.",
            answer="",
            citations=[],
        )

        with patch("app.api.routes.RepositoryIndexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                payload = json.dumps({
                    "repo_id": "repo-test",
                    "question": "Anything?",
                }).encode("utf-8")

                status, _, body = simulate_http_request(
                    "POST",
                    "/api/ask",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data["abstained"])
        self.assertIn("No relevant content found", data["abstention_reason"])
        self.assertEqual(data["answer"], "")
        self.assertEqual(len(data["citations"]), 0)

    def test_ask_missing_repo_id(self):
        """Missing repo_id returns 400."""
        payload = json.dumps({"question": "test"}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/ask",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("repo_id", data["error"].lower())

    def test_ask_missing_question(self):
        """Missing question returns 400."""
        payload = json.dumps({"repo_id": "repo-test"}).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/ask",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("question", data["error"].lower())

    def test_ask_invalid_retrieval_method(self):
        """Invalid retrieval_method returns 400."""
        payload = json.dumps({
            "repo_id": "repo-test",
            "question": "test",
            "retrieval_method": "invalid",
        }).encode("utf-8")
        status, _, body = simulate_http_request(
            "POST",
            "/api/ask",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("retrieval_method", data["error"].lower())

    def test_ask_invalid_json(self):
        """Invalid JSON body returns 400."""
        status, _, body = simulate_http_request(
            "POST",
            "/api/ask",
            body=b"not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("invalid json", data["error"].lower())

    def test_ask_pipeline_error(self):
        """Pipeline error returns 500 when EmbeddingClient fails to initialize."""
        with patch("app.api.routes.RepositoryIndexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.EmbeddingClient") as mock_embed_class:
                mock_embed_class.side_effect = RuntimeError("Model not found")

                payload = json.dumps({
                    "repo_id": "repo-test",
                    "question": "test",
                }).encode("utf-8")

                status, _, body = simulate_http_request(
                    "POST",
                    "/api/ask",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 500)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("Model not found", data["error"])


if __name__ == "__main__":
    unittest.main()
