"""CLI tests for the `ask --stream` RAG command.

Uses deterministic mocks for the RAG pipeline. No Ollama required.
"""

from __future__ import annotations

import asyncio
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.main import cmd_ask


class _FakeAsyncGen:
    """Fake async generator for ask_stream."""

    def __init__(self, events):
        self.events = events

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


class TestCLIAskStream(unittest.TestCase):
    """Tests for the `ask --stream` CLI command."""

    def _make_streaming_events(self):
        """Create a sequence of streaming events."""
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

    def test_stream_prints_tokens_to_stdout(self):
        """Streaming prints token chunks to stdout immediately."""
        fake_events = self._make_streaming_events()
        fake_pipeline = _FakeRAGPipeline(events=fake_events)

        with patch("app.main._get_indexer") as mock_get_indexer:
            mock_indexer = MagicMock()
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test",
                "name": "test",
                "path": "/tmp",
                "status": "complete",
                "file_count": 1,
                "total_lines": 10,
            }
            mock_get_indexer.return_value = mock_indexer

            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(SimpleNamespace(
                        repo_id="repo-test",
                        question="How to add?",
                        retrieval_method="hybrid",
                        limit=10,
                        stream=True,
                    ))

        self.assertEqual(exit_code, 0)
        stdout_output = stdout.getvalue()
        self.assertIn("The ", stdout_output)
        self.assertIn("add ", stdout_output)
        self.assertIn("function", stdout_output)
        self.assertIn("The add function", stdout_output)

    def test_stream_prints_progress_to_stderr(self):
        """Streaming prints progress events to stderr, not stdout."""
        fake_events = self._make_streaming_events()
        fake_pipeline = _FakeRAGPipeline(events=fake_events)

        with patch("app.main._get_indexer") as mock_get_indexer:
            mock_indexer = MagicMock()
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test",
                "name": "test",
                "path": "/tmp",
                "status": "complete",
                "file_count": 1,
                "total_lines": 10,
            }
            mock_get_indexer.return_value = mock_indexer

            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(SimpleNamespace(
                        repo_id="repo-test",
                        question="How to add?",
                        retrieval_method="hybrid",
                        limit=10,
                        stream=True,
                    ))

        self.assertEqual(exit_code, 0)
        stderr_output = stderr.getvalue()
        self.assertIn("[validation]", stderr_output)
        self.assertIn("[retrieval]", stderr_output)
        self.assertIn("[generation]", stderr_output)
        self.assertIn("[citation_validation]", stderr_output)

    def test_stream_prints_citations_to_stderr(self):
        """Streaming prints citations to stderr after the answer."""
        fake_events = self._make_streaming_events()
        fake_pipeline = _FakeRAGPipeline(events=fake_events)

        with patch("app.main._get_indexer") as mock_get_indexer:
            mock_indexer = MagicMock()
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test",
                "name": "test",
                "path": "/tmp",
                "status": "complete",
                "file_count": 1,
                "total_lines": 10,
            }
            mock_get_indexer.return_value = mock_indexer

            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(SimpleNamespace(
                        repo_id="repo-test",
                        question="How to add?",
                        retrieval_method="hybrid",
                        limit=10,
                        stream=True,
                    ))

        self.assertEqual(exit_code, 0)
        stderr_output = stderr.getvalue()
        self.assertIn("Citations", stderr_output)
        self.assertIn("[S1]", stderr_output)
        self.assertIn("src/calculator.py", stderr_output)

    def test_stream_abstention(self):
        """Streaming abstention produces exit code 2 and error message."""
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

        with patch("app.main._get_indexer") as mock_get_indexer:
            mock_indexer = MagicMock()
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test",
                "name": "test",
                "path": "/tmp",
                "status": "complete",
                "file_count": 1,
                "total_lines": 10,
            }
            mock_get_indexer.return_value = mock_indexer

            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(SimpleNamespace(
                        repo_id="repo-test",
                        question="quantum physics",
                        retrieval_method="hybrid",
                        limit=10,
                        stream=True,
                    ))

        self.assertEqual(exit_code, 2)
        stderr_output = stderr.getvalue()
        self.assertIn("Abstained", stderr_output)
        self.assertIn("No relevant content", stderr_output)

    def test_stream_pipeline_uses_stream_method(self):
        """Pipeline's ask_stream is called (not ask)."""
        fake_events = self._make_streaming_events()
        fake_pipeline = _FakeRAGPipeline(events=fake_events)

        with patch("app.main._get_indexer") as mock_get_indexer:
            mock_indexer = MagicMock()
            mock_indexer.store.connect = lambda: None
            mock_indexer.store.close = lambda: None
            mock_indexer.store.get_repository.return_value = {
                "id": "repo-test",
                "name": "test",
                "path": "/tmp",
                "status": "complete",
                "file_count": 1,
                "total_lines": 10,
            }
            mock_get_indexer.return_value = mock_indexer

            with patch("app.rag.pipeline.RAGPipeline", return_value=fake_pipeline):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(SimpleNamespace(
                        repo_id="repo-test",
                        question="How to add?",
                        retrieval_method="hybrid",
                        limit=10,
                        stream=True,
                    ))

        self.assertEqual(exit_code, 0)
        self.assertTrue(fake_pipeline.ask_stream_called)
        self.assertEqual(fake_pipeline.ask_stream_kwargs["repo_id"], "repo-test")
        self.assertEqual(fake_pipeline.ask_stream_kwargs["question"], "How to add?")
        self.assertEqual(fake_pipeline.ask_stream_kwargs["limit"], 10)


if __name__ == "__main__":
    unittest.main()
