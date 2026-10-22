"""Unit tests for vector storage in IndexStore (sqlite-vec).

Uses isolated temporary SQLite databases. Never touches index/sherlock.db.
sqlite-vec is exercised directly (real library, no mocks) since it is a
lightweight dependency already installed in the environment.
"""

from __future__ import annotations

import math
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

import sqlite_vec  # noqa: E402
from app.storage import IndexStore  # noqa: E402


class TestVectorStorage(unittest.TestCase):
    """Tests for the chunk_embeddings vec0 table and related methods."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_vec_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self.store = IndexStore(db_path=self.db_path)
        self.store.connect()
        self.repo_id = "repo-test-vec"
        self.store.register_repository(self.repo_id, str(self.test_dir / "repo"), "repo")

        # Create a file and some chunks
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.chunk_ids = []
        for i, content in enumerate(
            ["def add(a, b): return a + b", "def multiply(a, b): return a * b", "class Calculator: pass"]
        ):
            cid = self.store.add_chunk(
                self.repo_id, "f1", content, i * 2 + 1, i * 2 + 2
            )
            self.chunk_ids.append(cid)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Schema initialization
    # ------------------------------------------------------------------ #
    def test_vec_table_created(self):
        """The chunk_embeddings vec0 table exists after initialization."""
        conn = self.store.connect()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
        ).fetchone()
        self.assertIsNotNone(row)

    def test_vec_table_has_correct_dimension(self):
        """The vec0 table uses the configured embedding dimension (384)."""
        from config import config

        dimension = int(config.embedding_dimension)
        conn = self.store.connect()
        # Inserting a wrong-dimension vector should fail.
        wrong = [0.1] * (dimension + 1)
        with self.assertRaises(Exception):
            self.store.upsert_chunk_embedding(self.chunk_ids[0], wrong)

    # ------------------------------------------------------------------ #
    # Insert / retrieve / persistence
    # ------------------------------------------------------------------ #
    def test_upsert_and_get_embedding(self):
        """Insert an embedding and retrieve it by chunk ID."""
        vector = [0.1] * 384
        self.store.upsert_chunk_embedding(self.chunk_ids[0], vector)
        retrieved = self.store.get_chunk_embedding(self.chunk_ids[0])
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 384)
        # Stored vectors are L2-normalized; check the direction matches.
        norm = math.sqrt(sum(v * v for v in vector))
        expected = [v / norm for v in vector]
        for got, exp in zip(retrieved, expected):
            self.assertAlmostEqual(got, exp, places=5)

    def test_persistence_across_reopen(self):
        """Embeddings survive closing and reopening the database."""
        vector = [0.5] * 384
        self.store.upsert_chunk_embedding(self.chunk_ids[1], vector)
        self.store.close()

        store2 = IndexStore(db_path=self.db_path)
        store2.connect()
        try:
            retrieved = store2.get_chunk_embedding(self.chunk_ids[1])
            self.assertIsNotNone(retrieved)
            self.assertEqual(len(retrieved), 384)
        finally:
            store2.close()

    def test_get_nonexistent_embedding(self):
        """Retrieving an embedding for a chunk with no vector returns None."""
        self.assertIsNone(self.store.get_chunk_embedding(99999))

    def test_delete_embedding(self):
        """Deleting an embedding removes it from the vec0 table."""
        vector = [0.3] * 384
        self.store.upsert_chunk_embedding(self.chunk_ids[2], vector)
        self.assertIsNotNone(self.store.get_chunk_embedding(self.chunk_ids[2]))
        self.store.delete_chunk_embedding(self.chunk_ids[2])
        self.assertIsNone(self.store.get_chunk_embedding(self.chunk_ids[2]))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def test_invalid_dimension_rejected(self):
        """Upserting a vector with wrong dimension raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.store.upsert_chunk_embedding(self.chunk_ids[0], [0.1] * 128)
        self.assertIn("dimension", str(ctx.exception).lower())

    def test_empty_vector_rejected(self):
        """Upserting an empty vector raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.upsert_chunk_embedding(self.chunk_ids[0], [])

    def test_non_finite_vector_rejected(self):
        """Upserting a vector with NaN or Inf raises ValueError."""
        import math

        bad = [0.1] * 383 + [float("nan")]
        with self.assertRaises(ValueError):
            self.store.upsert_chunk_embedding(self.chunk_ids[0], bad)

        bad2 = [0.1] * 383 + [float("inf")]
        with self.assertRaises(ValueError):
            self.store.upsert_chunk_embedding(self.chunk_ids[0], bad2)

    def test_upsert_is_idempotent(self):
        """Re-upserting the same chunk replaces rather than duplicates."""
        v1 = [1.0] + [0.0] * 383
        v2 = [0.0] + [1.0] + [0.0] * 382
        self.store.upsert_chunk_embedding(self.chunk_ids[0], v1)
        self.store.upsert_chunk_embedding(self.chunk_ids[0], v2)
        retrieved = self.store.get_chunk_embedding(self.chunk_ids[0])
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(retrieved[0], 0.0, places=5)
        self.assertAlmostEqual(retrieved[1], 1.0, places=5)

    # ------------------------------------------------------------------ #
    # Nearest-neighbor search
    # ------------------------------------------------------------------ #
    def test_nearest_neighbor_ranking(self):
        """Similar vectors rank closer than dissimilar vectors."""
        import numpy as np

        # v1 and v2 are identical; v3 is orthogonal.
        v1 = [1.0] + [0.0] * 383
        v2 = [1.0] + [0.0] * 383
        v3 = [0.0] + [1.0] + [0.0] * 382

        self.store.upsert_chunk_embedding(self.chunk_ids[0], v1)
        self.store.upsert_chunk_embedding(self.chunk_ids[1], v2)
        self.store.upsert_chunk_embedding(self.chunk_ids[2], v3)

        results = self.store.search_chunk_embeddings(
            self.repo_id, [1.0] + [0.0] * 383, limit=3
        )
        self.assertEqual(len(results), 3)
        # v1 and v2 are identical to the query; both should be in top 2.
        top_ids = {r["id"] for r in results[:2]}
        self.assertEqual(top_ids, {self.chunk_ids[0], self.chunk_ids[1]})
        # v3 is orthogonal; should be last.
        self.assertEqual(results[2]["id"], self.chunk_ids[2])
        # Similarity should be in [0, 1].
        for r in results:
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.0)

    def test_search_empty_index(self):
        """Searching a repo with no embeddings returns an empty list."""
        results = self.store.search_chunk_embeddings(
            self.repo_id, [0.1] * 384, limit=10
        )
        self.assertEqual(results, [])

    def test_search_invalid_query_dimension(self):
        """Searching with wrong-dimension query raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.search_chunk_embeddings(
                self.repo_id, [0.1] * 128, limit=10
            )

    def test_search_invalid_query_nonfinite(self):
        """Searching with non-finite query raises ValueError."""
        import math

        with self.assertRaises(ValueError):
            self.store.search_chunk_embeddings(
                self.repo_id, [float("nan")] * 384, limit=10
            )

    def test_search_limit(self):
        """The limit parameter caps the number of results."""
        for cid in self.chunk_ids:
            self.store.upsert_chunk_embedding(cid, [0.1] * 384)
        results = self.store.search_chunk_embeddings(
            self.repo_id, [0.1] * 384, limit=2
        )
        self.assertEqual(len(results), 2)

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #
    def test_sync_inserts_missing(self):
        """sync_chunk_embeddings inserts placeholder vectors for chunks without vectors."""
        summary = self.store.sync_chunk_embeddings(self.repo_id)
        self.assertEqual(summary["inserted"], len(self.chunk_ids))
        self.assertEqual(summary["deleted"], 0)
        for cid in self.chunk_ids:
            self.assertIsNotNone(self.store.get_chunk_embedding(cid))

    def test_sync_deletes_orphans(self):
        """sync_chunk_embeddings deletes vectors for chunks that no longer exist."""
        # Insert an embedding directly into vec0 for a chunk id, then delete
        # the chunk (bypassing delete_chunks_for_file which also cleans
        # embeddings) to create a true orphan vector.
        self.store.upsert_chunk_embedding(self.chunk_ids[0], [0.1] * 384)
        self.assertIsNotNone(self.store.get_chunk_embedding(self.chunk_ids[0]))

        # Manually delete the chunk row without going through
        # delete_chunks_for_file (which would also clean the embedding).
        conn = self.store.connect()
        conn.execute("DELETE FROM chunks WHERE id = ?", (self.chunk_ids[0],))
        conn.commit()

        summary = self.store.sync_chunk_embeddings(self.repo_id)
        self.assertEqual(summary["deleted"], 1)
        self.assertIsNone(self.store.get_chunk_embedding(self.chunk_ids[0]))

    def test_get_embedding_count(self):
        """get_chunk_embedding_count returns the correct count."""
        self.assertEqual(self.store.get_chunk_embedding_count(self.repo_id), 0)
        self.store.upsert_chunk_embedding(self.chunk_ids[0], [0.1] * 384)
        self.store.upsert_chunk_embedding(self.chunk_ids[1], [0.2] * 384)
        self.assertEqual(self.store.get_chunk_embedding_count(self.repo_id), 2)


if __name__ == "__main__":
    unittest.main()
