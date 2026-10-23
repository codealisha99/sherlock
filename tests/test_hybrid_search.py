"""Unit tests for hybrid retrieval (RRF fusion) in IndexStore.

Uses isolated temporary SQLite databases. Never touches index/sherlock.db.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

import sqlite_vec  # noqa: E402
from app.storage import IndexStore  # noqa: E402


def _make_vector(values):
    """Serialize a float list for sqlite-vec insertion."""
    return sqlite_vec.serialize_float32(list(values))


class TestHybridSearch(unittest.TestCase):
    """Tests for hybrid_search RRF fusion."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_hybrid_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self.store = IndexStore(db_path=self.db_path)
        self.store.connect()
        self.repo_id = "repo-hybrid-test"
        self.store.register_repository(self.repo_id, str(self.test_dir / "repo"), "repo")

        # Create files and chunks
        self.store.upsert_file(
            self.repo_id, "f1", "src/calculator.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.store.upsert_file(
            self.repo_id, "f2", "src/operations.py", "Python", 100, 10, False, False, 1.0, "def"
        )

        self.chunk_ids = []
        chunks = [
            ("def add(a, b): return a + b", 1, 1),
            ("def multiply(a, b): return a * b", 3, 3),
            ("class Calculator: pass", 5, 5),
            ("def subtract(a, b): return a - b", 1, 1),
            ("def divide(a, b): return a / b", 3, 3),
        ]
        for content, start, end in chunks:
            cid = self.store.add_chunk(
                self.repo_id, "f1", content, start, end
            )
            self.chunk_ids.append(cid)

        # Add embeddings for all chunks
        import numpy as np

        self.vectors = {}
        for i, cid in enumerate(self.chunk_ids):
            # Create distinct vectors: chunk 0 and 1 are similar, others differ
            if cid == self.chunk_ids[0]:
                vec = [1.0] + [0.0] * 383  # "add" vector
            elif cid == self.chunk_ids[1]:
                vec = [1.0] + [0.0] * 383  # "multiply" - similar to add
            elif cid == self.chunk_ids[2]:
                vec = [0.0] + [1.0] + [0.0] * 382  # "Calculator" - orthogonal
            elif cid == self.chunk_ids[3]:
                vec = [0.0] + [0.0] + [1.0] + [0.0] * 381  # "subtract"
            else:
                vec = [0.0] * 383 + [1.0]  # "divide"
            self.store.upsert_chunk_embedding(cid, vec)
            self.vectors[cid] = vec

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Basic hybrid search
    # ------------------------------------------------------------------ #
    def test_hybrid_search_lexical_only(self):
        """Hybrid search with no embedding returns lexical results."""
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=None, limit=10
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("rrf_score", r)
            self.assertIn("lexical_rank", r)
            self.assertIn("semantic_rank", r)
            self.assertIn("lexical_score", r)
            self.assertIn("semantic_score", r)

    def test_hybrid_search_with_semantic(self):
        """Hybrid search with embedding returns fused results."""
        query_vec = [1.0] + [0.0] * 383  # matches "add" and "multiply"
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=10
        )
        self.assertGreater(len(results), 0)
        # Results should be sorted by descending RRF score
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["rrf_score"], results[i + 1]["rrf_score"])

    def test_hybrid_search_repository_scoping(self):
        """Hybrid search only returns results from the specified repo."""
        # Create a second repo with a matching chunk
        repo_id2 = "repo-hybrid-test-2"
        self.store.register_repository(repo_id2, str(self.test_dir / "repo2"), "repo2")
        self.store.upsert_file(
            repo_id2, "f1_repo2", "src/calculator.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        cid2 = self.store.add_chunk(repo_id2, "f1_repo2", "def add(a, b): return a + b", 1, 1)
        self.store.upsert_chunk_embedding(cid2, [1.0] + [0.0] * 383)

        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=[1.0] + [0.0] * 383, limit=10
        )
        for r in results:
            self.assertIn(r["id"], self.chunk_ids)

    def test_hybrid_search_limit(self):
        """The limit parameter caps results."""
        query_vec = [1.0] + [0.0] * 383
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=2
        )
        self.assertEqual(len(results), 2)

    def test_hybrid_search_empty_query(self):
        """Empty query raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.hybrid_search(self.repo_id, "", query_embedding=None)

    def test_hybrid_search_nonexistent_repo(self):
        """Nonexistent repo raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.hybrid_search("nonexistent", "add", query_embedding=None)

    # ------------------------------------------------------------------ #
    # RRF scoring correctness
    # ------------------------------------------------------------------ #
    def test_rrf_score_formula(self):
        """RRF score = sum(1 / (k + rank)) for each list the doc appears in."""
        query_vec = [1.0] + [0.0] * 383
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=10, rrf_k=60
        )
        # Find a result that appears in both lists
        both_lists = [r for r in results if r["lexical_rank"] is not None and r["semantic_rank"] is not None]
        if both_lists:
            r = both_lists[0]
            expected = 1.0 / (60 + r["lexical_rank"]) + 1.0 / (60 + r["semantic_rank"])
            self.assertAlmostEqual(r["rrf_score"], expected, places=6)

    def test_rrf_dedup(self):
        """Same chunk appears only once in results."""
        query_vec = [1.0] + [0.0] * 383
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=10
        )
        ids = [r["id"] for r in results]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate chunk IDs in hybrid results")

    def test_rrf_score_range(self):
        """RRF scores are positive and bounded."""
        query_vec = [1.0] + [0.0] * 383
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=10
        )
        for r in results:
            self.assertGreater(r["rrf_score"], 0.0)
            self.assertLessEqual(r["rrf_score"], 1.0)  # Max when rank=1 in both lists with k=0... but k=60

    def test_hybrid_search_result_metadata(self):
        """Each result contains complete chunk metadata."""
        query_vec = [1.0] + [0.0] * 383
        results = self.store.hybrid_search(
            self.repo_id, "add", query_embedding=query_vec, limit=5
        )
        for r in results:
            self.assertIn("id", r)
            self.assertIn("file_id", r)
            self.assertIn("content", r)
            self.assertIn("start_line", r)
            self.assertIn("end_line", r)
            self.assertIn("chunk_type", r)
            self.assertIn("file_path", r)
            self.assertIsInstance(r["start_line"], int)
            self.assertIsInstance(r["end_line"], int)

    # ------------------------------------------------------------------ #
    # Validation at storage layer
    # ------------------------------------------------------------------ #
    def test_storage_invalid_limit(self):
        """storage.hybrid_search rejects limit < 1."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search(self.repo_id, "add", limit=0)
        self.assertIn("limit >= 1", str(ctx.exception))

    def test_storage_negative_limit(self):
        """storage.hybrid_search rejects negative limit."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search(self.repo_id, "add", limit=-1)
        self.assertIn("limit >= 1", str(ctx.exception))

    def test_storage_invalid_rrf_k(self):
        """storage.hybrid_search rejects rrf_k < 1."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search(self.repo_id, "add", rrf_k=0)
        self.assertIn("rrf_k >= 1", str(ctx.exception))

    def test_storage_negative_rrf_k(self):
        """storage.hybrid_search rejects negative rrf_k."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search(self.repo_id, "add", rrf_k=-5)
        self.assertIn("rrf_k >= 1", str(ctx.exception))

    def test_storage_empty_query(self):
        """storage.hybrid_search rejects empty query."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search(self.repo_id, "")
        self.assertIn("nonempty", str(ctx.exception))

    def test_storage_nonexistent_repo(self):
        """storage.hybrid_search rejects nonexistent repo."""
        with self.assertRaises(ValueError) as ctx:
            self.store.hybrid_search("nonexistent", "add")
        self.assertIn("not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
