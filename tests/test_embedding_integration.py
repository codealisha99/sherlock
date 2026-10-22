"""Integration tests for embedding client with real model and sqlite-vec.

These tests require the sentence-transformers/all-MiniLM-L6-v2 model to be
available (downloaded and cached). They are run separately from unit tests
and report results distinctly.
"""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

import sqlite_vec  # noqa: E402
from app.llm.embedding_client import EmbeddingClient, EmbeddingError  # noqa: E402


class TestEmbeddingIntegration(unittest.TestCase):
    """Integration tests with the real sentence-transformers model."""

    @classmethod
    def setUpClass(cls):
        cls.client = EmbeddingClient(dimension=384)
        cls.snippets = [
            "def add(a, b): return a + b",
            "def multiply(a, b): return a * b",
            "class Calculator: pass",
            "def greet(name): return f\"Hello, {name}\"",
            "import os\nimport sys",
            "def calculate_total(items): return sum(items)",
            "def subtract(a, b): return a - b",
            "def hello_world(): print(\"Hello\")",
        ]

    def test_single_embedding_real_model(self):
        """Generate an embedding for a single code snippet."""
        vector = self.client.embed_text(self.snippets[0])
        self.assertEqual(len(vector), 384)
        self.assertTrue(all(math.isfinite(v) for v in vector))
        self.assertTrue(any(v != 0.0 for v in vector))  # nonempty/nonzero

    def test_batch_embedding_real_model(self):
        """Generate embeddings for multiple code snippets."""
        vectors = self.client.embed_texts(self.snippets)
        self.assertEqual(len(vectors), len(self.snippets))
        for v in vectors:
            self.assertEqual(len(v), 384)
            self.assertTrue(all(math.isfinite(x) for x in v))

    def test_embedding_dimension_consistency(self):
        """All embeddings have the same dimension (384)."""
        v1 = self.client.embed_text("def add(a, b): return a + b")
        v2 = self.client.embed_text("class Calculator: pass")
        self.assertEqual(len(v1), len(v2))
        self.assertEqual(len(v1), 384)

    def test_similar_texts_have_similar_embeddings(self):
        """Semantically similar snippets should have higher cosine similarity."""
        import numpy as np

        v_add = self.client.embed_text("def add(a, b): return a + b")
        v_sub = self.client.embed_text("def subtract(a, b): return a - b")
        v_import = self.client.embed_text("import os\nimport sys")

        def cosine(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_add_sub = cosine(v_add, v_sub)
        sim_add_import = cosine(v_add, v_import)

        # add and subtract are more similar than add and import
        self.assertGreater(sim_add_sub, sim_add_import)


class TestSqliteVecIntegration(unittest.TestCase):
    """Integration tests for sqlite-vec with real embeddings."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="sherlock_emb_test_")
        self.db_path = Path(self.tmpdir) / "test_embeddings.db"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_vec_db(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[384])")
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, content TEXT)")
        return conn

    def test_insert_and_nearest_neighbor(self):
        """Insert embeddings and retrieve nearest neighbors."""
        client = EmbeddingClient(dimension=384)
        conn = self._create_vec_db(self.db_path)

        try:
            snippets = [
                "def add(a, b): return a + b",
                "def multiply(a, b): return a * b",
                "class Calculator: pass",
                "import os\nimport sys",
            ]
            vectors = client.embed_texts(snippets)

            for i, (snippet, vec) in enumerate(zip(snippets, vectors)):
                conn.execute(
                    "INSERT INTO items(id, content) VALUES (?, ?)", (i + 1, snippet)
                )
                conn.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (i + 1, sqlite_vec.serialize_float32(vec)),
                )
            conn.commit()

            # Query for "how to add two numbers"
            query_vec = client.embed_text("how to add two numbers")
            rows = conn.execute(
                "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT 4",
                [sqlite_vec.serialize_float32(query_vec)],
            ).fetchall()

            # The add function should be the nearest neighbor
            self.assertEqual(rows[0][0], 1)  # rowid 1 = "def add(...)"
            print(f"\nNearest neighbors for 'how to add two numbers':")
            for rowid, dist in rows:
                content = conn.execute(
                    "SELECT content FROM items WHERE id = ?", (rowid,)
                ).fetchone()[0]
                print(f"  rowid={rowid} distance={dist:.4f} content={content!r}")
        finally:
            conn.close()

    def test_persistence_after_reopen(self):
        """Vectors persist after closing and reopening the database."""
        client = EmbeddingClient(dimension=384)
        conn = self._create_vec_db(self.db_path)

        try:
            snippet = "def add(a, b): return a + b"
            vec = client.embed_text(snippet)
            conn.execute(
                "INSERT INTO items(id, content) VALUES (?, ?)", (1, snippet)
            )
            conn.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (1, sqlite_vec.serialize_float32(vec)),
            )
            conn.commit()
        finally:
            conn.close()

        # Reopen and verify
        conn2 = sqlite3.connect(str(self.db_path))
        conn2.enable_load_extension(True)
        sqlite_vec.load(conn2)
        try:
            rows = conn2.execute(
                "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? LIMIT 1",
                [sqlite_vec.serialize_float32(vec)],
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 1)
            content = conn2.execute(
                "SELECT content FROM items WHERE id = ?", (1,)
            ).fetchone()[0]
            self.assertEqual(content, snippet)
        finally:
            conn2.close()

    def test_dimension_mismatch_rejected(self):
        """sqlite-vec rejects vectors with wrong dimension."""
        conn = self._create_vec_db(self.db_path)
        try:
            # Try to insert a 128-dim vector into a 384-dim table
            wrong_vec = [0.1] * 128
            with self.assertRaises(Exception):
                conn.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (1, sqlite_vec.serialize_float32(wrong_vec)),
                )
        finally:
            conn.close()

    def test_database_isolation(self):
        """Test database uses isolated temp path, not index/sherlock.db."""
        from config import config

        # The temp DB path should not be the production DB
        self.assertNotEqual(str(self.db_path), str(config.index_db_path))
        self.assertTrue("sherlock_emb_test_" in self.tmpdir)


if __name__ == "__main__":
    unittest.main()
