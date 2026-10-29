"""API tests for /api/chat endpoints (sessions and messages).

Tests cover:
- POST /api/chat/sessions - create session
- GET /api/chat/sessions - list sessions
- POST /api/chat/sessions/{id}/messages - send message with streaming SSE

Uses deterministic mocks for the RAG pipeline. No Ollama required.
"""

from __future__ import annotations

import io
import json
import unittest
import datetime
from unittest.mock import patch


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


def simulate_http_request(method: str, path: str, body: bytes = b"", headers: dict = None):
    """Simulate an HTTP request and return (status_code, headers, body_bytes)."""
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


from config import SherlockConfig
from app.api.routes import SherlockRequestHandler
from app.orchestrator.orchestrator import Orchestrator


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


class _FakeIndexer:
    """Fake indexer with a fake store."""

    def __init__(self, repo_id="repo-test"):
        self.repo_id = repo_id
        self.store = _FakeStore(repo_id)

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "test-repo", "path": "/tmp/test", "status": "complete"}
        return None


class _FakeStore:
    """Fake store for chat tests."""

    def __init__(self, repo_id="repo-test"):
        self.repo_id = repo_id
        self.sessions = {}
        self.messages = {}

    def connect(self):
        pass

    def close(self):
        pass

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "test-repo", "path": "/tmp/test", "status": "complete"}
        return None

    def create_chat_session(self, repo_id, session_id=None, title=None):
        import uuid
        if session_id is None:
            session_id = str(uuid.uuid4())
        if title is None:
            title = "New Chat"
        import datetime
        now = datetime.datetime.now(timezone.utc).isoformat()
        self.sessions[session_id] = {
            "id": session_id,
            "repo_id": repo_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        return session_id

    def get_chat_session(self, session_id):
        return self.sessions.get(session_id)

    def list_chat_sessions(self, repo_id):
        return [s for s in self.sessions.values() if s["repo_id"] == repo_id]

    def update_chat_session_title(self, session_id, title):
        if session_id in self.sessions:
            self.sessions[session_id]["title"] = title
            import datetime
            self.sessions[session_id]["updated_at"] = datetime.datetime.now(timezone.utc).isoformat()
            return True
        return False

    def delete_chat_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def add_chat_message(self, session_id, role, content, citations=None, context_files=None, message_id=None):
        import uuid
        import datetime
        if message_id is None:
            message_id = str(uuid.uuid4())
        now = datetime.datetime.now(timezone.utc).isoformat()
        self.messages[message_id] = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "context_files": context_files or [],
            "created_at": now,
        }
        return message_id

    def get_chat_messages(self, session_id):
        import json
        messages = []
        for m in self.messages.values():
            if m["session_id"] == session_id:
                msg = dict(m)
                try:
                    msg["citations"] = json.loads(msg["citations"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    msg["citations"] = []
                try:
                    msg["context_files"] = json.loads(msg["context_files"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    msg["context_files"] = []
                messages.append(msg)
        messages.sort(key=lambda m: m["created_at"])
        return messages


class TestAPIChatSession(unittest.TestCase):
    """Tests for /api/chat/sessions endpoints."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = True

    def test_create_session(self):
        """POST /api/chat/sessions creates a new session."""
        with patch("app.api.routes.RepositoryIndexer") as mock_indexer_class:
            mock_indexer = mock_indexer_class.return_value
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test", "name": "test", "path": "/tmp", "status": "complete"
            }
            mock_indexer.store.create_chat_session.return_value = "session-123"

            payload = json.dumps({"repo_id": "repo-test"}).encode("utf-8")
            status, _, body = simulate_http_request(
                "POST", "/api/chat/sessions",
                body=payload, headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 201)
        data = json.loads(body)
        self.assertEqual(data["session_id"], "session-123")
        self.assertEqual(data["repo_id"], "repo-test")

    def test_create_session_missing_repo_id(self):
        """POST /api/chat/sessions without repo_id returns 400."""
        payload = json.dumps({"title": "Test"}).encode("utf-8")
        status, _, _ = simulate_http_request(
            "POST", "/api/chat/sessions",
            body=payload, headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_list_sessions(self):
        """GET /api/chat/sessions lists sessions for a repo."""
        with patch("app.api.routes.RepositoryIndexer") as mock_indexer_class:
            mock_indexer = mock_indexer_class.return_value
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test", "name": "test", "path": "/tmp", "status": "complete"
            }
            mock_indexer.store.list_chat_sessions.return_value = [
                {"id": "s1", "repo_id": "repo-test", "title": "Session 1", "created_at": "t1", "updated_at": "t1"},
                {"id": "s2", "repo_id": "repo-test", "title": "Session 2", "created_at": "t2", "updated_at": "t2"},
            ]

            status, _, body = simulate_http_request(
                "GET", "/api/chat/sessions?repo_id=repo-test",
            )

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(len(data["sessions"]), 2)
        self.assertEqual(data["sessions"][0]["id"], "s1")
        self.assertEqual(data["sessions"][1]["id"], "s2")


class TestAPIChatMessage(unittest.TestCase):
    """Tests for POST /api/chat/sessions/{session_id}/messages."""

    def setUp(self):
        self.cfg = SherlockConfig()
        self.orchestrator = Orchestrator(sherlock_cfg=self.cfg)
        SherlockRequestHandler.orchestrator = self.orchestrator
        SherlockRequestHandler.enable_ai = True

    def test_send_message_streaming(self):
        """POST /api/chat/sessions/{id}/messages streams tokens and saves answer."""
        events = [
            {"type": "progress", "stage": "validation", "message": "OK", "elapsed_ms": 1.0},
            {"type": "progress", "stage": "retrieval_complete", "num_chunks": 2, "elapsed_ms": 5.0},
            {"type": "progress", "stage": "generation", "message": "Generating", "elapsed_ms": 8.0},
            {"type": "token", "text": "Hello ", "elapsed_ms": 50.0},
            {"type": "token", "text": "World", "elapsed_ms": 100.0},
            {
                "type": "done",
                "answer": "Hello World",
                "abstained": False,
                "citations": [
                    {"source_id": "S1", "file_path": "src/calc.py", "start_line": 1, "end_line": 5, "chunk_type": "code"}
                ],
                "num_chunks_retrieved": 2,
                "num_chunks_used": 2,
                "total_ms": 200.0,
            },
        ]
        fake_pipeline = _FakeRAGPipeline(events=events)

        with patch("app.api.routes.RepositoryIndexer") as mock_indexer_class:
            mock_indexer = mock_indexer_class.return_value
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test", "name": "test", "path": "/tmp", "status": "complete"
            }
            mock_indexer.store.get_chat_session.return_value = {
                "id": "session-1", "repo_id": "repo-test", "title": "Test", "created_at": "t1", "updated_at": "t1"
            }
            mock_indexer.store.add_chat_message.return_value = "msg-1"
            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                payload = json.dumps({"question": "How does add work?"}).encode("utf-8")
                status, _, body = simulate_http_request(
                    "POST", "/api/chat/sessions/session-1/messages",
                    body=payload, headers={"Content-Type": "application/json"},
                )

        self.assertEqual(status, 200)
        # Parse SSE events
        sse_text = body.decode("utf-8", errors="replace")
        sse_events = []
        for block in sse_text.split("\n\n"):
            block = block.strip()
            if block.startswith("data: "):
                try:
                    sse_events.append(json.loads(block[6:]))
                except json.JSONDecodeError:
                    pass

        token_events = [e for e in sse_events if e.get("type") == "token"]
        self.assertEqual(len(token_events), 2)
        self.assertEqual(token_events[0]["text"], "Hello ")
        self.assertEqual(token_events[1]["text"], "World")
        done_events = [e for e in sse_events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0]["answer"], "Hello World")


if __name__ == "__main__":
    unittest.main()
