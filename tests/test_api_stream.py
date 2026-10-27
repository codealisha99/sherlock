"""API tests for the /api/ask/stream SSE endpoint.

Uses deterministic mocks for the RAG pipeline. No Ollama required.
"""

from __future__ import annotations

import asyncio
import io
import json
import unittest
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


def simulate_sse_request(method: str, path: str, body: bytes = b"", headers: dict = None):
    """Simulate an HTTP request and return (status_code, headers_dict, sse_events_list)."""
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

    # Parse SSE events from body
    parts = raw_response.split(b"\r\n\r\n", 1)
    res_body = parts[1] if len(parts) > 1 else b""
    sse_text = res_body.decode("utf-8", errors="replace")

    # Parse SSE events: each event is "data: {json}\n\n"
    events = []
    for block in sse_text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            try:
                events.append(json.loads(block[6:]))
            except json.JSONDecodeError:
                pass

    return status_code, {}, events


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
        self.generation_error = None
        self.retrieval_error = None


class _FakeAsyncGen:
    """Fake async generator for ask_stream."""

    def __init__(self, events):
        self.events = events
        self._index = 0

    async def __aiter__(self):
        for event in self.events:
            yield event


class _FakeRAGPipeline:
    """Fake RAG pipeline with controllable ask_stream output."""

    def __init__(self, events=None):
        self.events = events or []
        self.ask_stream_called = False
        self.ask_stream_kwargs = None

    async def ask_stream(self, repo_id, question, retrieval_method="hybrid", limit=10):
        self.ask_stream_called = True
        self.ask_stream_kwargs = {
            "repo_id": repo_id,
            "question": question,
            "retrieval_method": retrieval_method,
            "limit": limit,
        }
        for event in self.events:
            yield event


class TestAPIAskStream(unittest.TestCase):
    """Tests for the /api/ask/stream SSE endpoint."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = True

    def _make_streaming_events(self):
        """Create a sequence of SSE events simulating a streaming RAG response."""
        return [
            {"type": "progress", "stage": "validation", "message": "Request validated", "elapsed_ms": 1.0},
            {"type": "progress", "stage": "retrieval", "message": "Retrieving relevant chunks", "elapsed_ms": 2.0},
            {"type": "progress", "stage": "retrieval_complete", "num_chunks": 2, "elapsed_ms": 5.0},
            {"type": "progress", "stage": "context_assembly", "message": "Assembling context", "elapsed_ms": 6.0},
            {"type": "progress", "stage": "context_assembly_complete", "num_chunks": 2, "tokens": 50, "elapsed_ms": 7.0},
            {"type": "progress", "stage": "generation", "message": "Generating answer", "elapsed_ms": 8.0},
            {"type": "token", "text": "The ", "elapsed_ms": 50.0},
            {"type": "token", "text": "add ", "elapsed_ms": 100.0},
            {"type": "token", "text": "function", "elapsed_ms": 150.0},
            {"type": "progress", "stage": "generation_complete", "tokens": 3, "elapsed_ms": 200.0},
            {"type": "progress", "stage": "citation_validation", "message": "Validating citations", "elapsed_ms": 201.0},
            {"type": "progress", "stage": "citation_validation_complete", "citations_valid": 1, "citations_total": 1, "elapsed_ms": 202.0},
            {
                "type": "done",
                "answer": "The add function",
                "abstained": False,
                "abstention_reason": None,
                "citations": [
                    {
                        "source_id": "S1",
                        "file_path": "src/calculator.py",
                        "start_line": 1,
                        "end_line": 2,
                        "chunk_type": "code",
                    }
                ],
                "num_chunks_retrieved": 2,
                "num_chunks_used": 2,
                "total_ms": 203.0,
            },
        ]

    def test_sse_streaming_returns_events(self):
        """SSE endpoint returns a sequence of events including tokens and done."""
        fake_events = self._make_streaming_events()
        fake_pipeline = _FakeRAGPipeline(events=fake_events)

        with patch("app.api.routes.RepositoryIndexer") as mock_indexer_class:
            mock_indexer = mock_indexer_class.return_value
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test", "name": "test", "path": "/tmp",
                "status": "complete", "file_count": 1, "total_lines": 10,
            }
            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                payload = json.dumps({
                    "repo_id": "repo-test",
                    "question": "How do I add?",
                }).encode("utf-8")

                status, _, events = simulate_sse_request(
                    "POST",
                    "/api/ask/stream",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 200)
        self.assertGreater(len(events), 0)
        # Should have progress events
        progress_events = [e for e in events if e.get("type") == "progress"]
        self.assertGreater(len(progress_events), 0)
        # Should have token events
        token_events = [e for e in events if e.get("type") == "token"]
        self.assertEqual(len(token_events), 3)
        # Should have done event
        done_events = [e for e in events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertFalse(done_events[0].get("abstained"))

    def test_sse_streaming_missing_repo_id(self):
        """Missing repo_id returns 400."""
        payload = json.dumps({"question": "test"}).encode("utf-8")
        status, _, _ = simulate_sse_request(
            "POST", "/api/ask/stream", body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_sse_streaming_missing_question(self):
        """Missing question returns 400."""
        payload = json.dumps({"repo_id": "repo-test"}).encode("utf-8")
        status, _, _ = simulate_sse_request(
            "POST", "/api/ask/stream", body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_sse_streaming_invalid_json(self):
        """Invalid JSON body returns 400."""
        status, _, _ = simulate_sse_request(
            "POST", "/api/ask/stream", body=b"not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_sse_streaming_abstention(self):
        """Abstention done event includes abstention_reason."""
        events = [
            {
                "type": "done",
                "answer": "",
                "abstained": True,
                "abstention_reason": "No relevant content found",
                "citations": [],
                "num_chunks_retrieved": 0,
                "num_chunks_used": 0,
                "total_ms": 100.0,
            },
        ]
        fake_pipeline = _FakeRAGPipeline(events=events)

        with patch("app.api.routes.RepositoryIndexer") as mock_indexer_class:
            mock_indexer = mock_indexer_class.return_value
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test", "name": "test", "path": "/tmp",
                "status": "complete", "file_count": 1, "total_lines": 10,
            }
            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                payload = json.dumps({
                    "repo_id": "repo-test",
                    "question": "quantum physics",
                }).encode("utf-8")

                status, _, sse_events = simulate_sse_request(
                    "POST",
                    "/api/ask/stream",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 200)
        done_events = [e for e in sse_events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertTrue(done_events[0].get("abstained"))
        self.assertIn("No relevant content", done_events[0].get("abstention_reason", ""))


if __name__ == "__main__":
    unittest.main()
