"""Sherlock RAG (Retrieval-Augmented Generation) package.

Provides context assembly, grounding, citation validation, abstention,
and RAG orchestration on top of the existing retrieval pipeline.
"""

from app.rag.context_builder import (
    ChunkSource,
    ContextBundle,
    assemble_context,
)

__all__ = [
    "ChunkSource",
    "ContextBundle",
    "assemble_context",
]
