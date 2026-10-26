"""Sherlock RAG (Retrieval-Augmented Generation) package.

Provides context assembly, grounding, citation validation, abstention,
and RAG orchestration on top of the existing retrieval pipeline.
"""

from app.rag.context_builder import (
    ChunkSource,
    ContextBundle,
    assemble_context,
)
from app.rag.citation_validator import Citation, extract_citation_ids, validate_citations
from app.rag.pipeline import RAGPipeline, RAGResult

__all__ = [
    "ChunkSource",
    "ContextBundle",
    "assemble_context",
    "Citation",
    "extract_citation_ids",
    "validate_citations",
    "RAGPipeline",
    "RAGResult",
]
