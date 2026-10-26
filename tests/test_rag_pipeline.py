"""Unit tests for RAG pipeline (app/rag/pipeline.py).

Uses deterministic mocks for the embedding client and Ollama client.
No model downloads, no Ollama server, no network access.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.indexer import RepositoryIndexer
from app.llm.embedding_client import EmbeddingClient
from app.llm.ollama_client import OllamaClient
from app.rag.pipeline import RAGPipeline, RAGResult


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


class _FakeEmbeddingClient:
    """Deterministic fake embedding client for tests."""

    def __init__(self):
        self.embed_text_called = False
        self.embed_texts_called = False

    def is_available(self):
        return True

    def embed_text(self, text):
        self.embed_text_called = True
        return [0.1] + [0.0] * 383

    def embed_texts(self, texts):
        self.embed_texts_called = True
        return [[0.1] + [0.0] * 383 for _ in texts]

    def close(self):
        pass


class _FakeOllamaClient:
    """Deterministic fake Ollama client for tests."""

    def __init__(self, answer: str = "The answer is based on [S1]."):
        self.answer = answer
        self.generate_called = False
        self.last_prompt = None
        self.last_system = None

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.generate_called = True
        self.last_prompt = prompt
        self.last_system = system_prompt
        return self.answer

    def check_health(self):
        return {"available": True, "model_ready": True, "configured_model": "test"}

    def close(self):
        pass


class _FakeIndexer:
    """Fake indexer that returns controlled search results."""

    def __init__(self, results=None, repo_id="repo-test"):
        self.results = results or []
        self.repo_id = repo_id
        self.store = _FakeStore(results, repo_id)

    def hybrid_search(self, repo_id, query, query_embedding=None, limit=10, rrf_k=60):
        return self.results

    def semantic_search(self, repo_id, query, limit=10, embedding_client=None):
        return self.results

    def store_search_chunks(self, repo_id, query, limit=10):
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

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "test-repo", "path": "/tmp/test"}
        return None


class TestRAGPipeline(unittest.TestCase):
    """Tests for RAGPipeline with mocked clients."""

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
        self.embed_client = _FakeEmbeddingClient()
        self.ollama_client = _FakeOllamaClient(answer="To add numbers, use the add function [S1].")
        self.pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=self.embed_client,
            ollama_client=self.ollama_client,
        )

    def test_grounded_answer_with_citation(self):
        """A relevant question produces a grounded answer with valid citation."""
        result = self.pipeline.ask("repo-test", "How do I add two numbers?")
        self.assertFalse(result.abstained)
        self.assertIsNone(result.abstention_reason)
        self.assertIsNone(result.generation_error)
        self.assertIsNone(result.retrieval_error)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].source_id, "S1")
        self.assertEqual(result.citations[0].file_path, "src/calculator.py")

    def test_source_metadata_from_trusted_store(self):
        """Citation metadata comes from trusted store, not model output."""
        result = self.pipeline.ask("repo-test", "How do I add?")
        self.assertEqual(len(result.citations), 1)
        c = result.citations[0]
        self.assertEqual(c.file_path, "src/calculator.py")
        self.assertEqual(c.start_line, 1)
        self.assertEqual(c.end_line, 2)
        self.assertEqual(c.source_id, "S1")

    def test_no_relevant_evidence_abstains(self):
        """No retrieval results → abstention with clear reason."""
        self.indexer.results = []
        result = self.pipeline.ask("repo-test", "Anything?")
        self.assertTrue(result.abstained)
        self.assertIsNotNone(result.abstention_reason)
        self.assertEqual(len(result.citations), 0)
        self.assertEqual(result.answer, "")

    def test_nonexistent_repo_abstains(self):
        """Nonexistent repository → abstention."""
        result = self.pipeline.ask("nonexistent-repo", "anything")
        self.assertTrue(result.abstained)
        self.assertIn("not found", result.abstention_reason)

    def test_empty_question_abstains(self):
        """Empty question → abstention."""
        result = self.pipeline.ask("repo-test", "")
        self.assertTrue(result.abstained)
        self.assertIn("empty", result.abstention_reason.lower())

    def test_whitespace_question_abstains(self):
        """Whitespace-only question → abstention."""
        result = self.pipeline.ask("repo-test", "   ")
        self.assertTrue(result.abstained)

    def test_missing_embeddings_handled(self):
        """Missing embedding client → graceful degradation (lexical only)."""
        no_embed = _FakeEmbeddingClient()
        no_embed.is_available = lambda: False
        pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=no_embed,
            ollama_client=self.ollama_client,
        )
        result = pipeline.ask("repo-test", "How do I add?")
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.citations), 1)

    def test_unavailable_llm_abstains(self):
        """Unavailable Ollama → clear error, abstention."""
        no_ollama = _FakeOllamaClient(answer="x")
        no_ollama.generate = lambda *a, **k: (_ for _ in ()).throw(
            ConnectionError("Ollama unreachable")
        )
        pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=self.embed_client,
            ollama_client=no_ollama,
        )
        result = pipeline.ask("repo-test", "How do I add?")
        self.assertTrue(result.abstained)
        self.assertIn("Ollama unreachable", result.generation_error or "")

    def test_empty_llm_response_abstains(self):
        """Empty LLM response → abstention."""
        empty_client = _FakeOllamaClient(answer="")
        pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=self.embed_client,
            ollama_client=empty_client,
        )
        result = pipeline.ask("repo-test", "How do I add?")
        self.assertTrue(result.abstained)
        self.assertIn("empty", result.abstention_reason.lower())

    def test_invalid_citation_rejected(self):
        """Citation referencing non-existent source ID is rejected."""
        # Model cites [S99] which doesn't exist in evidence
        bad_client = _FakeOllamaClient(answer="The answer is [S99].")
        pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=self.embed_client,
            ollama_client=bad_client,
        )
        result = pipeline.ask("repo-test", "How do I add?")
        self.assertTrue(result.abstained)
        self.assertIn("could not be grounded", result.abstention_reason)

    def test_repository_isolation(self):
        """Querying a different repo returns its own results, not leaked data."""
        other_indexer = _FakeIndexer(
            results=[
                _make_chunk(
                    "c9",
                    "def secret(): return 42",
                    file_path="src/secret.py",
                    start_line=1,
                    end_line=1,
                    rank=1,
                    score=0.9,
                )
            ],
            repo_id="other-repo",
        )
        pipeline = RAGPipeline(
            indexer=other_indexer,
            embedding_client=self.embed_client,
            ollama_client=self.ollama_client,
        )
        result = pipeline.ask("other-repo", "secret")
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].file_path, "src/secret.py")
        self.assertEqual(result.citations[0].source_id, "S1")

    def test_context_budget_respected(self):
        """Very small budget limits number of chunks used."""
        pipeline = RAGPipeline(
            indexer=self.indexer,
            embedding_client=self.embed_client,
            ollama_client=self.ollama_client,
            max_context_tokens=10,  # Very small - might fit only 1 chunk
        )
        result = pipeline.ask("repo-test", "add")
        self.assertLessEqual(result.num_chunks_used, 2)

    def test_pipeline_uses_embedding_client(self):
        """Pipeline calls embedding client for semantic retrieval."""
        self.pipeline.ask("repo-test", "How do I add?")
        self.assertTrue(self.embed_client.embed_text_called)

    def test_pipeline_uses_ollama_client(self):
        """Pipeline calls Ollama client for generation."""
        self.pipeline.ask("repo-test", "How do I add?")
        self.assertTrue(self.ollama_client.generate_called)
        self.assertIsNotNone(self.ollama_client.last_prompt)
        self.assertIn("add", self.ollama_client.last_prompt)
        self.assertIsNotNone(self.ollama_client.last_system)
        self.assertIn("evidence", self.ollama_client.last_system.lower())


if __name__ == "__main__":
    unittest.main()
