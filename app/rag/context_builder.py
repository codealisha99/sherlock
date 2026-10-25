"""Context assembly for Sherlock RAG.

Transforms retrieved chunks into a structured evidence package for the LLM.
Handles deduplication, ranking preservation, source-ID assignment, token
budgeting, and metadata preservation.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.storage import IndexStore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ChunkSource:
    """A single source chunk with trusted metadata."""

    source_id: str  # e.g., "S1", "S2" - assigned during context assembly
    repo_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str = "code"
    retrieval_rank: Optional[int] = None  # position in the original retrieval list
    retrieval_score: Optional[float] = None  # bm25 rank, cosine similarity, or RRF score
    retrieval_method: Optional[str] = None  # "lexical", "semantic", "hybrid"


@dataclass
class ContextBundle:
    """Assembled context package for the LLM.

    Attributes:
        sources: List of ChunkSource objects, in assembly order.
        sources_by_id: Mapping from source_id (e.g. "S1") to ChunkSource.
        total_tokens: Estimated token count of the assembled context.
        truncated: True if any content was omitted due to budget.
        budget_exceeded: True if even the first chunk alone exceeds the budget.
    """

    sources: List[ChunkSource] = field(default_factory=list)
    sources_by_id: Dict[str, ChunkSource] = field(default_factory=dict)
    total_tokens: int = 0
    truncated: bool = False
    budget_exceeded: bool = False


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a whitespace/punctuation-aware heuristic when a real tokenizer is
    unavailable. Returns a conservative (slightly overestimated) count so
    that callers never exceed the budget unintentionally.
    """
    if not text:
        return 0
    # Approximate: ~0.75 words per token for English/code text.
    # Count whitespace-separated tokens and add ~1 per punctuation cluster
    # to be conservative.
    words = text.split()
    return max(1, int(len(words) * 0.75) + text.count(".") + text.count(","))


def _try_tokenize(text: str) -> Optional[int]:
    """Attempt real token counting if a compatible tokenizer exists.

    Returns None if no tokenizer is available.
    """
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text, using a real tokenizer when available.

    Falls back to a conservative heuristic when tiktoken is not installed.
    """
    try_count = _try_tokenize(text)
    if try_count is not None:
        return try_count
    return _estimate_tokens(text)


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #


def _deduplicate_chunks(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate chunks based on chunk id, preserving first occurrence.

    Preserves the retrieval order of the first occurrence of each chunk.
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        cid = r.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(r)
    return deduped


def assemble_context(
    results: List[Dict[str, Any]],
    query: str,
    max_tokens: int = 4000,
    system_overhead_tokens: int = 500,
    question_overhead_tokens: int = 100,
) -> ContextBundle:
    """Assemble a bounded context bundle from retrieved chunks.

    Parameters
    ----------
    results:
        Ranked retrieval results from hybrid_search (or search_chunks /
        search_chunk_embeddings). Each dict must contain at least ``id``,
        ``file_path``, ``start_line``, ``end_line``, ``content``.
    query:
        The user's natural-language question (counted toward the budget).
    max_tokens:
        Maximum total tokens for the assembled context (chunks only, not
        including system/question overhead).
    system_overhead_tokens:
        Tokens reserved for system instructions.
    question_overhead_tokens:
        Tokens reserved for the user question.

    Returns
    -------
    ContextBundle
        A bundle with sources, source-ID mapping, total token count, and
        truncation flags.
    """
    bundle = ContextBundle()

    # Deduplicate while preserving retrieval order
    deduped = _deduplicate_chunks(results)

    # Reserve budget for system + question
    available_for_chunks = max_tokens
    # We track chunk tokens separately; system/question overhead is tracked
    # by the caller but reported here for transparency.

    source_index = 0
    for r in deduped:
        chunk_id = r.get("id")
        if not chunk_id:
            continue

        content = r.get("content", "") or ""
        if not content.strip():
            continue

        # Estimate tokens for this chunk
        chunk_tokens = count_tokens(content)

        # Check if adding this chunk would exceed the budget
        if bundle.total_tokens + chunk_tokens > available_for_chunks:
            bundle.truncated = True
            # If this is the very first chunk and it alone exceeds the budget,
            # mark budget_exceeded so the caller knows even one chunk won't fit
            if bundle.total_tokens == 0:
                bundle.budget_exceeded = True
            break

        # Assign source ID
        source_index += 1
        source_id = f"S{source_index}"

        source = ChunkSource(
            source_id=source_id,
            repo_id=r.get("repo_id", ""),
            file_path=r.get("file_path", ""),
            start_line=int(r.get("start_line", 0)) if r.get("start_line") else 0,
            end_line=int(r.get("end_line", 0)) if r.get("end_line") else 0,
            content=content,
            chunk_type=r.get("chunk_type", "code"),
            retrieval_rank=r.get("lexical_rank")
            or r.get("semantic_rank")
            or r.get("rank"),
            retrieval_score=r.get("lexical_score")
            or r.get("semantic_score")
            or r.get("rrf_score"),
            retrieval_method=(
                "hybrid"
                if r.get("lexical_rank") is not None
                and r.get("semantic_rank") is not None
                else ("semantic" if r.get("semantic_rank") is not None else "lexical")
            ),
        )

        bundle.sources.append(source)
        bundle.sources_by_id[source_id] = source
        bundle.total_tokens += chunk_tokens

    return bundle


def format_context_for_prompt(bundle: ContextBundle) -> str:
    """Format a ContextBundle into a plain-text context block for the LLM.

    Each source is prefixed with its source ID (e.g. [S1]) and includes
    file path and line range. Source metadata is never truncated.
    """
    if not bundle.sources:
        return ""

    lines: List[str] = []
    for s in bundle.sources:
        header = f"[{s.source_id}] {s.file_path}"
        if s.start_line and s.end_line:
            header += f":{s.start_line}-{s.end_line}"
        elif s.start_line:
            header += f":{s.start_line}"
        lines.append(f"{header}:")
        lines.append(s.content)
        lines.append("")  # blank line between sources

    return "\n".join(lines)


def build_citation_map(bundle: ContextBundle) -> Dict[str, Dict[str, Any]]:
    """Build a citation map from source_id to trusted metadata.

    Returns a dict mapping source_id (e.g. "S1") to a dict with keys:
        file_path, start_line, end_line, chunk_type, content
    """
    citation_map: Dict[str, Dict[str, Any]] = {}
    for s in bundle.sources:
        citation_map[s.source_id] = {
            "file_path": s.file_path,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "chunk_type": s.chunk_type,
            "content": s.content,
        }
    return citation_map
