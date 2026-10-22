"""Unit tests for the embedding client using deterministic mocks.

These tests do not require the sentence-transformers model to be downloaded
or loaded. They validate the client's input validation, dimension checking,
finiteness checking, and error handling in isolation.
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure the project root is on the path
sys.path.insert(0, ".")

from app.llm.embedding_client import EmbeddingClient, EmbeddingError  # noqa: E402


class TestEmbeddingClientValidation(unittest.TestCase):
    """Tests for input validation and vector validation logic."""

    def setUp(self):
        self.client = EmbeddingClient(dimension=384)

    def test_embed_text_empty_string_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_text("")

    def test_embed_text_whitespace_only_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_text("   ")

    def test_embed_text_non_string_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_text(123)
        with self.assertRaises(EmbeddingError):
            self.client.embed_text(None)
        with self.assertRaises(EmbeddingError):
            self.client.embed_text(["a", "b"])

    def test_embed_texts_empty_list_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts([])

    def test_embed_texts_non_list_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts("not a list")
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts(None)

    def test_embed_texts_empty_string_item_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts(["hello", ""])
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts(["hello", "   "])

    def test_embed_texts_non_string_item_raises(self):
        with self.assertRaises(EmbeddingError):
            self.client.embed_texts(["hello", 123])


class TestEmbeddingClientDimensionMismatch(unittest.TestCase):
    """Tests that dimension mismatch is caught clearly."""

    def test_wrong_dimension_raises(self):
        client = EmbeddingClient(dimension=128)
        # Mock the model to return a 384-dim vector
        fake_model = MagicMock()
        fake_model.encode.return_value = __import__("numpy").zeros(384, dtype="float32")
        client._model = fake_model
        client._model_loaded = True

        with self.assertRaises(EmbeddingError) as ctx:
            client.embed_text("hello")
        self.assertIn("expected dimension 128, got 384", str(ctx.exception))

    def test_correct_dimension_passes(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones(384, dtype="float32")
        client._model = fake_model
        client._model_loaded = True

        result = client.embed_text("hello")
        self.assertEqual(len(result), 384)
        self.assertTrue(all(v == 1.0 for v in result))


class TestEmbeddingClientFiniteness(unittest.TestCase):
    """Tests that non-finite vectors are rejected."""

    def test_nan_vector_raises(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        vec = np.ones(384, dtype="float32")
        vec[0] = float("nan")
        fake_model.encode.return_value = vec
        client._model = fake_model
        client._model_loaded = True

        with self.assertRaises(EmbeddingError) as ctx:
            client.embed_text("hello")
        self.assertIn("non-finite", str(ctx.exception))

    def test_inf_vector_raises(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        vec = np.ones(384, dtype="float32")
        vec[0] = float("inf")
        fake_model.encode.return_value = vec
        client._model = fake_model
        client._model_loaded = True

        with self.assertRaises(EmbeddingError) as ctx:
            client.embed_text("hello")
        self.assertIn("non-finite", str(ctx.exception))

    def test_empty_vector_raises(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([], dtype="float32")
        client._model = fake_model
        client._model_loaded = True

        with self.assertRaises(EmbeddingError) as ctx:
            client.embed_text("hello")
        self.assertIn("empty", str(ctx.exception).lower())


class TestEmbeddingClientBatchValidation(unittest.TestCase):
    """Tests batch embedding validation with mocks."""

    def test_batch_wrong_dimension_raises(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        # Return 2D array with wrong second dimension
        fake_model.encode.return_value = np.zeros((2, 128), dtype="float32")
        client._model = fake_model
        client._model_loaded = True

        with self.assertRaises(EmbeddingError) as ctx:
            client.embed_texts(["a", "b"])
        self.assertIn("expected dimension 384, got 128", str(ctx.exception))

    def test_batch_correct_dimension_passes(self):
        client = EmbeddingClient(dimension=384)
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((3, 384), dtype="float32")
        client._model = fake_model
        client._model_loaded = True

        results = client.embed_texts(["a", "b", "c"])
        self.assertEqual(len(results), 3)
        for v in results:
            self.assertEqual(len(v), 384)


class TestEmbeddingClientMissingDependency(unittest.TestCase):
    """Tests behavior when sentence-transformers is not installed."""

    def test_missing_dependency_raises_clear_error(self):
        client = EmbeddingClient(dimension=384)

        # Simulate missing sentence-transformers
        with patch("app.llm.embedding_client._SENTENCE_TRANSFORMERS_AVAILABLE", False):
            with patch("app.llm.embedding_client.SentenceTransformer", None):
                with self.assertRaises(EmbeddingError) as ctx:
                    client.embed_text("hello")
                self.assertIn("sentence-transformers is not installed", str(ctx.exception))


class TestEmbeddingClientLoadFailure(unittest.TestCase):
    """Tests behavior when model loading fails."""

    def test_model_load_failure_raises_clear_error(self):
        client = EmbeddingClient(dimension=384, model_name="nonexistent-model")

        # Force load failure
        with patch("app.llm.embedding_client.SentenceTransformer", side_effect=Exception("model not found")):
            with self.assertRaises(EmbeddingError) as ctx:
                client.embed_text("hello")
            self.assertIn("Failed to load embedding model", str(ctx.exception))

    def test_is_available_returns_false_on_load_failure(self):
        client = EmbeddingClient(dimension=384, model_name="nonexistent-model")

        with patch("app.llm.embedding_client.SentenceTransformer", side_effect=Exception("model not found")):
            self.assertFalse(client.is_available())

    def test_close_resets_state(self):
        client = EmbeddingClient(dimension=384)
        client.close()
        self.assertIsNone(client._model)
        self.assertFalse(client._model_loaded)
        self.assertIsNone(client._load_error)


if __name__ == "__main__":
    unittest.main()
