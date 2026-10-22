"""Local embedding client for Sherlock using sentence-transformers.

This module provides a self-contained embedding client that is entirely
separate from the Ollama generation client. It uses a local
sentence-transformers model (default: all-MiniLM-L6-v2, 384-dimensional)
to produce embeddings for code snippets and natural-language text.

All operations are local and deterministic. No network calls are made
after the model has been downloaded and cached.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from config import config, SherlockConfig

logger = logging.getLogger(__name__)

# Optional import so the module can be imported without sentence-transformers
# installed; a clear error is raised only when embedding is actually attempted.
try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails for any reason."""


class EmbeddingClient:
    """Local embedding client backed by sentence-transformers.

    The model is loaded lazily on first use and cached for the lifetime of
    the client, avoiding repeated downloads or loads.
    """

    def __init__(
        self,
        sherlock_cfg: Optional[SherlockConfig] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.cfg = sherlock_cfg or config
        self.model_name = model_name or self.cfg.embedding_model_name
        self.dimension = dimension or self.cfg.embedding_dimension
        self.batch_size = batch_size or self.cfg.embedding_batch_size
        self.device = device or self.cfg.embedding_device
        self._model: Optional[SentenceTransformer] = None
        self._model_loaded = False
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> SentenceTransformer:
        """Load the sentence-transformers model if not already loaded.

        Raises EmbeddingError if the dependency is missing or the model
        cannot be loaded.
        """
        if self._model_loaded:
            if self._load_error is not None:
                raise EmbeddingError(self._load_error)
            assert self._model is not None
            return self._model

        self._model_loaded = True

        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_error = (
                "sentence-transformers is not installed. Install it with: "
                "pip install sentence-transformers"
            )
            raise EmbeddingError(self._load_error)

        try:
            logger.info(
                "Loading embedding model %s (device=%s)", self.model_name, self.device
            )
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
            logger.info("Embedding model %s loaded successfully", self.model_name)
        except Exception as exc:  # noqa: BLE001 - clear error for any load failure
            self._load_error = f"Failed to load embedding model '{self.model_name}': {exc}"
            logger.error(self._load_error)
            raise EmbeddingError(self._load_error) from exc

        assert self._model is not None
        return self._model

    def close(self) -> None:
        """Release the loaded model (if any)."""
        self._model = None
        self._model_loaded = False
        self._load_error = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text or code snippet.

        Parameters
        ----------
        text:
            The input text to embed. Must be a nonempty string.

        Returns
        -------
        List[float]
            A dense vector of ``self.dimension`` floats.

        Raises
        ------
        EmbeddingError
            If the input is invalid, the model is unavailable, or the
            returned vector fails validation.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("embed_text() requires a nonempty string input.")

        model = self._ensure_loaded()
        vector = model.encode(text, convert_to_numpy=True)
        return self._validate_vector(vector, source="embed_text()")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts or code snippets.

        Parameters
        ----------
        texts:
            A list of nonempty strings to embed.

        Returns
        -------
        List[List[float]]
            A list of dense vectors, each of ``self.dimension`` floats.

        Raises
        ------
        EmbeddingError
            If any input is invalid, the model is unavailable, or any
            returned vector fails validation.
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise EmbeddingError("embed_texts() requires a nonempty list of strings.")

        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise EmbeddingError(
                    f"embed_texts() item at index {i} is not a nonempty string."
                )

        model = self._ensure_loaded()
        vectors = model.encode(texts, convert_to_numpy=True, batch_size=self.batch_size)

        # model.encode returns a 2D numpy array for batch input
        import numpy as np

        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise EmbeddingError(
                "embed_texts() received an unexpected response shape from the model."
            )

        return [
            self._validate_vector(vectors[i], source=f"embed_texts()[{i}]")
            for i in range(vectors.shape[0])
        ]

    def is_available(self) -> bool:
        """Return True if the embedding model can be loaded."""
        try:
            self._ensure_loaded()
            return True
        except EmbeddingError:
            return False

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_vector(self, vector, source: str = "unknown") -> List[float]:
        """Validate a single embedding vector.

        Checks that the vector is nonempty, one-dimensional, finite, and
        has the expected dimension.
        """
        import numpy as np

        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)

        if vector.ndim != 1:
            raise EmbeddingError(
                f"{source}: expected a 1-D vector, got shape {vector.shape}."
            )

        if vector.size == 0:
            raise EmbeddingError(f"{source}: returned an empty vector.")

        if vector.size != self.dimension:
            raise EmbeddingError(
                f"{source}: expected dimension {self.dimension}, got {vector.size}."
            )

        if not bool(np.isfinite(vector).all()):
            raise EmbeddingError(f"{source}: vector contains non-finite values.")

        return [float(v) for v in vector.tolist()]


# ---------------------------------------------------------------------- #
# Module-level convenience instance (uses global config)
# ---------------------------------------------------------------------- #
_default_client: Optional[EmbeddingClient] = None


def get_default_client() -> EmbeddingClient:
    """Return a module-level EmbeddingClient using the global config.

    The client is created once and reused; the underlying sentence-transformers
    model is loaded lazily on first embed call.
    """
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


def embed_text(text: str) -> List[float]:
    """Module-level convenience: embed a single text using the default client."""
    return get_default_client().embed_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Module-level convenience: embed a batch of texts using the default client."""
    return get_default_client().embed_texts(texts)
