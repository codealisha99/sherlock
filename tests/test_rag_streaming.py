"""Unit tests for RAG pipeline streaming (app/rag/pipeline.py ask_stream).

Uses deterministic mocks for the Ollama client. No network access required.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.indexer import RepositoryIndexer
from app.rag.pipeline import RAGPipeline


class _FakeAsyncGen:
    """Wrapper to make a regular iterable into an async generator."""

    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class _FakeOllamaClient:
    """Deterministic fake Ollama client with streaming support."""

    def __init__(self, chunks=None):
        self.chunks = chunks or ["Hello", " world", "!"]
        self.generate_stream_called = False
        self.generate_stream_kwargs = None

    async def generate_stream(self, prompt=None, system_prompt=None, temperature=0.2):
        self.generate_stream_called = True
        self.generate_stream_kwargs = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        for chunk in self.chunks:
            yield chunk

    def check_health(self):
        return {"available": True, "model_ready": True, "configured_model": "test"}


class _FakeIndexer:
    """Fake indexer that returns controlled search results."""

    def __init__(self, results=None, repo_id="repo-test"):
        self.results = results or []
        self.repo_id = repo_id
        self.store = _FakeStore(results, repo_id)

    def hybrid_search(self, repo_id, query, limit=10, rrf_k=60, embedding_client=None):
        return self.results

    def semantic_search(self, repo_id, query, limit=10, embedding_client=None):
        return self.results

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "test-repo", "path": "/tmp/test"}
        return None


class _FakeStore:
    """Fake store that mirrors indexer results."""

    def __init__(self, results, repo_id):
        self.results = results
        self.repo_id = repo_id

    def connect(self):
        pass

    def close(self):
        pass

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "test-repo", "path": "/tmp/test"}
        return None


def _make_chunk(
    chunk_id: str,
    content: str,
    file_path: str = "src/calculator.py",
    start_line: int = 1,
    end_line: int = 5,
    rank: int = 1,
    score: float = 0.9,
) -> dict:
    return {
        "id": chunk_id,
        "repo_id": "repo-test",
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        "chunk_type": "code",
        "lexical_rank": rank,
        "semantic_rank": rank,
        "lexical_score": score,
        "semantic_score": score,
        "rrf_score": score,
    }


class TestRAGStreaming(unittest.TestCase):
    """Tests for RAGPipeline.ask_stream() generator."""

    def setUp(self):
        self.indexer = _FakeIndexer(
            results=[
                _make_chunk(
                    "c1",
                    "def add(a, b): return a + b",
                    file_path="src/calculator.py",
                    start_line=1,
                    end_line=2,
                    rank=1,
                    score=0.9,
                ),
                _make_chunk(
                    "c2",
                    "def multiply(a, b): return a * b",
                    file_path="src/operations.py",
                    start_line=5,
                    end_line=6,
                    rank=2,
                    score=0.8,
                ),
            ]
        )
        self.ollama_client = _FakeOllamaClient(
            chunks=["The ", "add ", "function ", "is ", "in ", "calculator.py ", "[S1]."]
        )
        self.pipeline = RAGPipeline(
            indexer=self.indexer,
            ollama_client=self.ollama_client,
        )

    def _run_stream(self):
        """Run the async stream and collect all events."""
        events = []

        async def _collect():
            async for event in self.pipeline.ask_stream(
                repo_id="repo-test",
                question="How do I add?",
                retrieval_method="hybrid",
                limit=10,
            ):
                events.append(event)

        asyncio.run(_collect())
        return events

    def test_stream_yields_progress_events(self):
        """Stream yields progress events for each stage."""
        events = self._run_stream()
        stages = [e.get("stage") for e in events if e.get("type") == "progress"]
        self.assertIn("validation", stages)
        self.assertIn("retrieval", stages)
        self.assertIn("retrieval_complete", stages)
        self.assertIn("context_assembly", stages)
        self.assertIn("context_assembly_complete", stages)
        self.assertIn("generation", stages)
        self.assertIn("citation_validation", stages)
        self.assertIn("citation_validation_complete", stages)

    def test_stream_yields_token_events(self):
        """Stream yields token events with incremental text."""
        events = self._run_stream()
        token_events = [e for e in events if e.get("type") == "token"]
        self.assertGreater(len(token_events), 0)
        # Verify tokens are incremental chunks of the answer
        full_text = "".join(e.get("text", "") for e in token_events)
        self.assertIn("add", full_text)
        self.assertIn("[S1]", full_text)

    def test_stream_yields_done_event(self):
        """Stream yields a done event with final answer and citations."""
        events = self._run_stream()
        done_events = [e for e in events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        done = done_events[0]
        self.assertFalse(done.get("abstained"))
        self.assertIsNotNone(done.get("answer"))
        self.assertIn("[S1]", done.get("answer", ""))
        self.assertEqual(len(done.get("citations", [])), 1)
        self.assertEqual(done["citations"][0]["source_id"], "S1")

    def test_stream_timestamps_present(self):
        """Events include elapsed_ms timestamps."""
        events = self._run_stream()
        for event in events:
            if event.get("type") in ("progress", "token", "done"):
                self.assertIn("elapsed_ms", event)
                self.assertIsInstance(event["elapsed_ms"], (int, float))

    def test_stream_tokens_before_done(self):
        """Token events arrive before the done event (proves streaming)."""
        events = self._run_stream()
        token_indices = [
            i for i, e in enumerate(events) if e.get("type") == "token"
        ]
        done_indices = [
            i for i, e in enumerate(events) if e.get("type") == "done"
        ]
        self.assertGreater(len(token_indices), 0)
        self.assertGreater(len(done_indices), 0)
        # All tokens should come before done
        self.assertLess(max(token_indices), min(done_indices))

    def test_empty_question_abstains(self):
        """Empty question produces error event."""
        events = []

        async def _collect():
            async for event in self.pipeline.ask_stream(
                repo_id="repo-test",
                question="",
                retrieval_method="hybrid",
                limit=10,
            ):
                events.append(event)

        asyncio.run(_collect())
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertIn("empty", error_events[0].get("message", "").lower())

    def test_nonexistent_repo_abstains(self):
        """Nonexistent repository produces error event."""
        events = []

        async def _collect():
            async for event in self.pipeline.ask_stream(
                repo_id="nonexistent",
                question="test",
                retrieval_method="hybrid",
                limit=10,
            ):
                events.append(event)

        asyncio.run(_collect())
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertIn("not found", error_events[0].get("message", "").lower())

    def test_ollama_client_stream_called(self):
        """Pipeline calls the Ollama client's generate_stream method."""
        self._run_stream()
        self.assertTrue(self.ollama_client.generate_stream_called)
        self.assertIsNotNone(self.ollama_client.generate_stream_kwargs)
        self.assertIn("How do I add?", self.ollama_client.generate_stream_kwargs["prompt"])


class TestRAGStreamingNoResults(unittest.TestCase):
    """Tests streaming with no retrieval results."""

    def setUp(self):
        self.indexer = _FakeIndexer(results=[], repo_id="repo-empty")
        self.ollama_client = _FakeOllamaClient(chunks=["unused"])
        self.pipeline = RAGPipeline(
            indexer=self.indexer,
            ollama_client=self.ollama_client,
        )

    def test_no_results_done_with_abstention(self):
        """No retrieval results yields done with abstention."""
        events = []

        async def _collect():
            async for event in self.pipeline.ask_stream(
                repo_id="repo-empty",
                question="quantum physics",
                retrieval_method="hybrid",
                limit=10,
            ):
                events.append(event)

        asyncio.run(_collect())
        done_events = [e for e in events if e.get("type") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertTrue(done_events[0].get("abstained"))
        self.assertIn("No relevant content", done_events[0].get("abstention_reason", ""))


class TestRAGStreamingOllamaError(unittest.TestCase):
    """Tests streaming when Ollama fails."""

    def setUp(self):
        self.indexer = _FakeIndexer(
            results=[
                _make_chunk(
                    "c1",
                    "def add(a, b): return a + b",
                    file_path="src/calculator.py",
                    start_line=1,
                    end_line=2,
                )
            ]
        )

    def test_ollama_error_yields_error_event(self):
        """Ollama error yields error event, doesn't hang."""

        class _ErrorOllamaClient:
            async def generate_stream(self, prompt=None, system_prompt=None, temperature=0.2):
                if False:  # pragma: no cover - makes this an async generator function
                    yield
                raise ConnectionError("Ollama unreachable")

        pipeline = RAGPipeline(
            indexer=self.indexer,
            ollama_client=_ErrorOllamaClient(),
        )

        events = []

        async def _collect():
            async for event in pipeline.ask_stream(
                repo_id="repo-test",
                question="test",
                retrieval_method="hybrid",
                limit=10,
            ):
                events.append(event)

        asyncio.run(_collect())
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertIn("Ollama unreachable", error_events[0].get("message", ""))


if __name__ == "__main__":
    unittest.main()
