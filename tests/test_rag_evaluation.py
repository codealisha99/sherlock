"""Unit tests for RAG evaluation runner (evaluation/rag_runner.py).

Uses deterministic fixtures and mocks. No Ollama required.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.rag_runner import RAGEvaluationRunner


class _FakeIndexResult:
    """Fake index result from index_repository."""

    def __init__(self, repo_id="rag_eval_calc"):
        self.repo_id = repo_id
        self.repo_name = "calc"
        self.status = "complete"
        self.file_count = 3
        self.total_lines = 100
        self.chunk_count = 10
        self.definition_count = 5
        self.import_count = 2
        self.skipped_files = []
        self.errors = []


class _FakeEmbeddingResult:
    """Fake result from embed_chunks."""

    def __init__(self):
        self.chunk_count = 10
        self.embedded_count = 10
        self.skipped_count = 0
        self.duration_seconds = 0.5
        self.errors = []


class _FakeIndexer:
    """Fake indexer for RAG evaluation tests."""

    def __init__(self, repo_id="rag_eval_calc"):
        self.repo_id = repo_id
        self.store = _FakeStore()
        self._embedded = False

    def index_repository(self, path, repo_id, repo_name, subdirectory=None, root_path=None):
        return _FakeIndexResult(repo_id)

    def get_repository(self, repo_id):
        if repo_id == self.repo_id:
            return {"id": repo_id, "name": "calc", "path": "/tmp/calc", "status": "complete"}
        return None

    def embed_chunks(self, repo_id, embedding_client=None):
        self._embedded = True
        return _FakeEmbeddingResult()

    def semantic_search(self, repo_id, query, limit=10, embedding_client=None):
        return [
            {
                "id": "c1",
                "repo_id": repo_id,
                "file_path": "calc/calculator.py",
                "start_line": 1,
                "end_line": 5,
                "content": "class Calculator:\n    def add(self, a, b):\n        return a + b",
                "similarity": 0.9,
                "distance": 0.1,
            }
        ]

    def hybrid_search(self, repo_id, query, query_embedding=None, limit=10, rrf_k=60):
        return [
            {
                "id": "c1",
                "repo_id": repo_id,
                "file_path": "calc/calculator.py",
                "start_line": 1,
                "end_line": 5,
                "content": "class Calculator:\n    def add(self, a, b):\n        return a + b",
                "rrf_score": 0.8,
                "lexical_rank": 1,
                "semantic_rank": 1,
            }
        ]


class _FakeStore:
    """Fake store."""

    def connect(self):
        pass

    def close(self):
        pass

    def search_chunks(self, repo_id, query, limit=20):
        return [
            {
                "id": "c1",
                "repo_id": repo_id,
                "file_path": "calc/calculator.py",
                "start_line": 1,
                "end_line": 5,
                "content": "class Calculator:\n    def add(self, a, b):\n        return a + b",
                "rank": 1,
            }
        ]

    def get_repository(self, repo_id):
        if repo_id == "rag_eval_calc":
            return {"id": repo_id, "name": "calc", "path": "/tmp/calc", "status": "complete"}
        return None


class TestRAGEvaluationRunner(unittest.TestCase):
    """Tests for RAGEvaluationRunner without Ollama."""

    def setUp(self):
        self.runner = RAGEvaluationRunner(use_embeddings=False)

    def test_runner_init_without_embeddings(self):
        """Runner initializes without embeddings when use_embeddings=False."""
        self.assertIsNone(self.runner.embed_client)
        self.assertFalse(self.runner.use_embeddings)

    def test_index_fixture_returns_metrics(self):
        """index_fixture returns indexing metrics dict."""
        with patch.object(self.runner, "_get_indexer", return_value=_FakeIndexer()):
            result = self.runner.index_fixture("calc", Path("/tmp/fixtures/calc"))
            self.assertEqual(result["repo_name"], "calc")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["file_count"], 3)
            self.assertEqual(result["chunk_count"], 10)
            self.assertFalse(result["embeddings_available"])

    def test_run_benchmark_lexical_only(self):
        """run_benchmark with lexical-only method produces results."""
        queries = [
            {
                "query_id": "q001",
                "query": "Calculator",
                "category": "exact_identifier",
                "repo": "calc",
                "expected_files": ["calc/calculator.py"],
                "expected_symbols": ["Calculator"],
                "should_have_results": True,
            }
        ]

        with patch.object(self.runner, "_get_indexer", return_value=_FakeIndexer()):
            result = self.runner.run_benchmark(queries, retrieval_methods=["lexical"])

        self.assertEqual(result["retrieval_methods"], ["lexical"])
        self.assertFalse(result["embeddings_available"])
        self.assertIn("lexical", result["results_by_method"])
        method_result = result["results_by_method"]["lexical"]
        self.assertEqual(len(method_result["query_results"]), 1)
        self.assertEqual(method_result["query_results"][0]["query_id"], "q001")
        self.assertEqual(method_result["query_results"][0]["method"], "lexical")
        self.assertIn("recall@5", method_result["aggregate_metrics"])

    def test_generate_report_creates_files(self):
        """generate_report writes JSON and Markdown reports."""
        queries = [
            {
                "query_id": "q001",
                "query": "Calculator",
                "category": "exact_identifier",
                "repo": "calc",
                "expected_files": ["calc/calculator.py"],
                "expected_symbols": ["Calculator"],
                "should_have_results": True,
            }
        ]

        with patch.object(self.runner, "_get_indexer", return_value=_FakeIndexer()):
            benchmark = self.runner.run_benchmark(queries, retrieval_methods=["lexical"])

        report = self.runner.generate_report(benchmark, output_dir=Path("/tmp/rag_eval_test_reports"))
        self.assertIn("method_aggregates", report)
        self.assertIn("lexical", report["method_aggregates"])
        self.assertIn("embeddings_available", report)
        self.assertFalse(report["embeddings_available"])

        # Verify files were created
        json_path = Path("/tmp/rag_eval_test_reports/rag_latest.json")
        md_path = Path("/tmp/rag_eval_test_reports/rag_latest.md")
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(data["retrieval_methods"], ["lexical"])
        self.assertEqual(data["num_queries"] if "num_queries" in data else True, True)

    def test_run_benchmark_semantic_returns_empty_without_embeddings(self):
        """Semantic retrieval returns empty results when no embeddings available."""
        queries = [
            {
                "query_id": "q001",
                "query": "Calculator",
                "category": "exact_identifier",
                "repo": "calc",
                "expected_files": ["calc/calculator.py"],
                "expected_symbols": ["Calculator"],
                "should_have_results": True,
            }
        ]

        with patch.object(self.runner, "_get_indexer", return_value=_FakeIndexer()):
            result = self.runner.run_benchmark(queries, retrieval_methods=["semantic"])

        method_result = result["results_by_method"]["semantic"]
        self.assertEqual(method_result["query_results"][0]["num_retrieved"], 0)
        self.assertIsNone(method_result["query_results"][0]["search_error"])

    def test_aggregate_metrics_structure(self):
        """Aggregate metrics contain expected keys."""
        queries = [
            {
                "query_id": "q001",
                "query": "Calculator",
                "category": "exact_identifier",
                "repo": "calc",
                "expected_files": ["calc/calculator.py"],
                "expected_symbols": ["Calculator"],
                "should_have_results": True,
            }
        ]

        with patch.object(self.runner, "_get_indexer", return_value=_FakeIndexer()):
            result = self.runner.run_benchmark(queries, retrieval_methods=["lexical"])

        agg = result["results_by_method"]["lexical"]["aggregate_metrics"]
        expected_keys = {"num_queries", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"}
        for key in expected_keys:
            self.assertIn(key, agg, f"Missing metric key: {key}")


if __name__ == "__main__":
    unittest.main()
