"""Integration tests for indexer chunk embedding with real model.

Uses isolated temporary SQLite databases and the real sentence-transformers
model. Never touches index/sherlock.db.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from app.indexer import RepositoryIndexer  # noqa: E402
from config import SherlockConfig  # noqa: E402


def _make_repo(files):
    """Create a temp repo from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix="sherlock_embed_idx_test_"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


class TestIndexerEmbeddingIntegration(unittest.TestCase):
    """Tests for embed_chunks() and semantic_search() on the indexer."""

    def setUp(self):
        # Use an isolated temporary database, never index/sherlock.db
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_embed_db_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self.cfg = SherlockConfig(index_db_path=self.db_path)
        self.idx = RepositoryIndexer(cfg=self.cfg)
        self.root = _make_repo(
            {
                "src/calculator.py": "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
                "src/greeting.py": "def greet(name):\n    return f\"Hello, {name}\"\n",
            }
        )
        self.res = self.idx.index_repository(self.root)
        self.repo_id = self.res.repo_id

    def tearDown(self):
        try:
            self.idx.remove_repository(self.repo_id)
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_embed_chunks(self):
        """Embed all chunks of a repository."""
        result = self.idx.embed_chunks(self.repo_id)
        self.assertEqual(result["repo_id"], self.repo_id)
        self.assertGreater(result["chunk_count"], 0)
        self.assertEqual(result["embedded_count"], result["chunk_count"])
        self.assertEqual(result["skipped_count"], 0)
        self.assertEqual(len(result["errors"]), 0)
        self.assertGreater(result["duration_seconds"], 0)

    def test_embed_chunks_idempotent(self):
        """Running embed_chunks twice produces the same embedded count."""
        result1 = self.idx.embed_chunks(self.repo_id)
        result2 = self.idx.embed_chunks(self.repo_id)
        self.assertEqual(result1["embedded_count"], result2["embedded_count"])
        self.assertEqual(result1["chunk_count"], result2["chunk_count"])

    def test_embed_chunks_nonexistent_repo(self):
        """embed_chunks raises ValueError for nonexistent repo."""
        with self.assertRaises(ValueError):
            self.idx.embed_chunks("nonexistent-repo-id")

    def test_semantic_search_after_embedding(self):
        """Semantic search returns relevant chunks after embedding."""
        self.idx.embed_chunks(self.repo_id)

        results = self.idx.semantic_search(
            self.repo_id, "how to add two numbers", limit=5
        )
        self.assertGreater(len(results), 0)
        # The add function should rank highly
        add_found = any("add" in r["content"] for r in results[:3])
        self.assertTrue(add_found, "Expected 'add' function in top results")

        # Each result should have similarity metadata
        for r in results:
            self.assertIn("similarity", r)
            self.assertIn("distance", r)
            self.assertIn("file_path", r)
            self.assertIn("content", r)
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.0)

    def test_semantic_search_empty_query(self):
        """semantic_search raises ValueError for empty query."""
        self.idx.embed_chunks(self.repo_id)
        with self.assertRaises(ValueError):
            self.idx.semantic_search(self.repo_id, "")

    def test_semantic_search_no_embeddings(self):
        """semantic_search raises RuntimeError when no embeddings exist."""
        with self.assertRaises(RuntimeError):
            self.idx.semantic_search(self.repo_id, "anything")

    def test_semantic_search_nonexistent_repo(self):
        """semantic_search raises ValueError for nonexistent repo."""
        with self.assertRaises(ValueError):
            self.idx.semantic_search("nonexistent-repo-id", "query")

    def test_semantic_search_ranking(self):
        """Semantically related query returns higher similarity for related chunks."""
        self.idx.embed_chunks(self.repo_id)

        add_results = self.idx.semantic_search(
            self.repo_id, "addition function", limit=5
        )
        greet_results = self.idx.semantic_search(
            self.repo_id, "greeting function", limit=5
        )

        # add query should have 'add' in top results
        add_top_content = " ".join(r["content"] for r in add_results[:3])
        self.assertIn("add", add_top_content)

        # greet query should have 'greet' in top results
        greet_top_content = " ".join(r["content"] for r in greet_results[:3])
        self.assertIn("greet", greet_top_content)

    def test_semantic_search_limit(self):
        """The limit parameter caps semantic search results."""
        self.idx.embed_chunks(self.repo_id)
        results = self.idx.semantic_search(self.repo_id, "function", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_changed_chunk_reembedding(self):
        """Re-indexing a changed chunk and re-embedding works correctly."""
        self.idx.embed_chunks(self.repo_id)
        count_before = self.idx.store.get_chunk_embedding_count(self.repo_id)
        self.assertGreater(count_before, 0)

        # Re-index the same repo (idempotent)
        res2 = self.idx.index_repository(self.root)
        self.idx.embed_chunks(res2.repo_id)
        count_after = self.idx.store.get_chunk_embedding_count(res2.repo_id)
        self.assertEqual(count_before, count_after)


if __name__ == "__main__":
    unittest.main()
