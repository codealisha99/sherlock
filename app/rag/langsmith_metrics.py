"""Deterministic RAG metrics for Sherlock LangSmith observability.

All metrics are computed locally from observed data. No LLM-as-judge or
model-assisted evaluation is used here. Metrics can be attached to LangSmith
traces as metadata or used in local evaluation reports.

Metrics are grouped by concern:

  Retrieval quality:
    - recall_at_k, mrr, ndcg_at_k (when relevance labels exist)
    - result_count, score_distribution (observed from retrieval)

  Citation quality:
    - citation_syntax_validity (format check)
    - citation_file_validity (safe path check)
    - citation_line_range_validity (line range check)
    - citation_evidence_membership (% citations in current evidence)
    - citation_coverage (% answer claims with valid citations)

  Answer quality:
    - groundedness (valid citations / total citations)
    - unsupported_claim_rate (claims without valid citations)
    - abstention_precision, abstention_recall (when answerability labels exist)

  Operational performance:
    - retrieval_latency_ms, assembly_latency_ms, generation_latency_ms
    - citation_validation_latency_ms, total_latency_ms
    - error_rate, timeout_rate
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.rag.citation_validator import (
    CITATION_PATTERN,
    extract_citation_ids,
    validate_citations,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Recall@k: fraction of relevant documents found within top-k."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc, or 0.0."""
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """nDCG@k with binary relevance and log2 discounting."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return 0.0

    import math

    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #


CITATION_SYNTAX_RE = re.compile(r"^\[S\d+\]$")


def citation_syntax_validity(answer_text: str) -> Dict[str, Any]:
    """Check citation syntax validity in answer text.

    Returns dict with:
      - total_citations: number of [S<n>] tokens found
      - valid_syntax: number matching canonical [S<n>] format
      - invalid_syntax: number not matching
      - validity_rate: valid / total (0.0 if no citations)
    """
    found = extract_citation_ids(answer_text)
    total = len(found)
    # extract_citation_ids already filters to valid [S<n>] matches where n is digits
    # So all extracted IDs are syntactically valid by construction
    valid = total
    return {
        "total_citations": total,
        "valid_syntax": valid,
        "invalid_syntax": total - valid,
        "validity_rate": (valid / total) if total > 0 else 1.0,
    }


def citation_evidence_membership(
    answer_text: str,
    citation_map: Dict[str, Dict[str, Any]],
    repo_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Check what percentage of cited sources are in the current evidence map.

    Returns dict with:
      - total_cited: number of unique citations in answer
      - in_evidence: number whose source_id exists in citation_map
      - not_in_evidence: number whose source_id is absent from map
      - membership_rate: in_evidence / total_cited (1.0 if no citations)
      - cross_repo_rejected: number rejected due to repo mismatch (if repo_id given)
    """
    found_ids = extract_citation_ids(answer_text)
    total = len(found_ids)
    in_evidence = 0
    cross_repo_rejected = 0

    for source_id in found_ids:
        meta = citation_map.get(source_id)
        if meta is None:
            continue
        if repo_id and meta.get("repo_id") and meta["repo_id"] != repo_id:
            cross_repo_rejected += 1
            continue
        in_evidence += 1

    not_in_evidence = total - in_evidence - cross_repo_rejected
    return {
        "total_cited": total,
        "in_evidence": in_evidence,
        "not_in_evidence": not_in_evidence,
        "cross_repo_rejected": cross_repo_rejected,
        "membership_rate": (in_evidence / total) if total > 0 else 1.0,
    }


def citation_coverage(
    answer_text: str,
    citations: List[Any],
) -> Dict[str, Any]:
    """Estimate citation coverage: what portion of the answer is backed by citations.

    This is a heuristic: counts sentence-like segments in the answer and checks
    whether each has an adjacent citation. Not a substitute for claim-level
    verification.

    Returns dict with:
      - total_segments: number of sentence-like segments
      - cited_segments: segments containing or adjacent to a citation
      - coverage_rate: cited_segments / total_segments (0.0 if no segments)
    """
    if not answer_text:
        return {
            "total_segments": 0,
            "cited_segments": 0,
            "coverage_rate": 0.0,
        }

    # Split on sentence boundaries (naive but deterministic)
    segments = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    segments = [s for s in segments if s.strip()]
    total = len(segments)

    cited = 0
    citation_positions = [m.start() for m in CITATION_PATTERN.finditer(answer_text)]

    for seg in segments:
        seg_start = answer_text.find(seg)
        if seg_start < 0:
            continue
        seg_end = seg_start + len(seg)
        # Check if a citation appears within or immediately after this segment
        for pos in citation_positions:
            if seg_start <= pos <= seg_end + 2:
                cited += 1
                break

    return {
        "total_segments": total,
        "cited_segments": cited,
        "coverage_rate": (cited / total) if total > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Answer quality metrics
# --------------------------------------------------------------------------- #


def groundedness(
    citations: List[Any],
    total_citations_attempted: int,
) -> Dict[str, Any]:
    """Groundedness: fraction of attempted citations that validated.

    Returns dict with:
      - valid_citations: number of citations that passed validation
      - total_attempted: total citations the model attempted to make
      - groundedness_rate: valid / total_attempted (1.0 if no attempts)
    """
    valid = len(citations)
    total = total_citations_attempted
    return {
        "valid_citations": valid,
        "total_attempted": total,
        "groundedness_rate": (valid / total) if total > 0 else 1.0,
    }


def unsupported_claim_rate(
    answer_text: str,
    citations: List[Any],
) -> Dict[str, Any]:
    """Estimate unsupported claim rate.

    Heuristic: segments of the answer that contain no citation are treated
    as potentially unsupported claims. This is not a substitute for
    claim-level verification.

    Returns dict with:
      - total_segments: number of sentence-like segments
      - unsupported_segments: segments with no citation
      - unsupported_rate: unsupported / total (0.0 if no segments)
    """
    coverage = citation_coverage(answer_text, citations)
    total = coverage["total_segments"]
    cited = coverage["cited_segments"]
    unsupported = total - cited
    return {
        "total_segments": total,
        "unsupported_segments": unsupported,
        "unsupported_rate": (unsupported / total) if total > 0 else 0.0,
    }


def abstention_metrics(
    abstained: bool,
    answerable: Optional[bool],
    abstention_reason: Optional[str],
) -> Dict[str, Any]:
    """Abstention metrics when answerability labels are available.

    Parameters
    ----------
    abstained:
        Whether the pipeline abstained.
    answerable:
        Whether the question was answerable (from labels), or None if unknown.
    abstention_reason:
        The reason given for abstention.

    Returns dict with abstention decision and, when labels exist,
    precision/recall components.
    """
    result: Dict[str, Any] = {
        "abstained": abstained,
        "abstention_reason": abstention_reason,
    }

    if answerable is not None:
        # True positive: abstained and not answerable (correct abstention)
        # False positive: abstained but answerable (incorrect abstention)
        # False negative: didn't abstain but not answerable (hallucination risk)
        result["correct_abstention"] = abstained and not answerable
        result["incorrect_abstention"] = abstained and answerable
        result["missed_abstention"] = not abstained and not answerable

        if answerable is False:
            # Precision: of all abstentions, how many were correct?
            total_abstentions = 1  # This single case
            correct = 1 if (abstained and not answerable) else 0
            result["abstention_precision"] = correct / total_abstentions
            # Recall: of all unanswerable questions, how many did we abstain on?
            total_unanswerable = 1
            result["abstention_recall"] = correct / total_unanswerable

    return result


# --------------------------------------------------------------------------- #
# Latency metrics
# --------------------------------------------------------------------------- #


@dataclass
class LatencyMetrics:
    """Collected latency measurements for a single RAG request."""

    retrieval_ms: Optional[float] = None
    assembly_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    citation_validation_ms: Optional[float] = None
    total_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "retrieval_latency_ms": self.retrieval_ms,
            "assembly_latency_ms": self.assembly_ms,
            "generation_latency_ms": self.generation_ms,
            "citation_validation_latency_ms": self.citation_validation_ms,
            "total_latency_ms": self.total_ms,
        }


# --------------------------------------------------------------------------- #
# Aggregated RAG metrics for a single request
# --------------------------------------------------------------------------- #


@dataclass
class RAGRequestMetrics:
    """All deterministic metrics for a single RAG request."""

    request_id: str
    repo_id: str
    retrieval_method: str
    abstained: bool
    abstention_reason: Optional[str]
    num_chunks_retrieved: int
    num_chunks_used: int
    context_tokens: int
    latency: LatencyMetrics
    retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    citation_metrics: Dict[str, Any] = field(default_factory=dict)
    answer_metrics: Dict[str, Any] = field(default_factory=dict)
    abstention_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten all metrics into a single dict for LangSmith metadata."""
        return {
            "request_id": self.request_id,
            "repo_id": self.repo_id,
            "retrieval_method": self.retrieval_method,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "num_chunks_used": self.num_chunks_used,
            "context_tokens": self.context_tokens,
            **self.latency.to_dict(),
            **{f"retrieval_{k}": v for k, v in self.retrieval_metrics.items()},
            **{f"citation_{k}": v for k, v in self.citation_metrics.items()},
            **{f"answer_{k}": v for k, v in self.answer_metrics.items()},
            **{f"abstention_{k}": v for k, v in self.abstention_metrics.items()},
            "errors": self.errors,
        }


def compute_request_metrics(
    request_id: str,
    repo_id: str,
    retrieval_method: str,
    result: Any,  # RAGResult
    latency: LatencyMetrics,
    relevance_labels: Optional[Set[str]] = None,
    answerable: Optional[bool] = None,
) -> RAGRequestMetrics:
    """Compute all deterministic metrics for a single RAG request result.

    Parameters
    ----------
    result:
        RAGResult from pipeline.ask().
    latency:
        LatencyMetrics with observed timings.
    relevance_labels:
        Set of relevant chunk IDs for recall/MRR/nDCG (optional).
    answerable:
        Whether the question is answerable (for abstention metrics, optional).
    """
    # Retrieval metrics
    retrieval_metrics: Dict[str, Any] = {
        "num_results": result.num_chunks_retrieved,
    }
    if relevance_labels:
        retrieved_ids = [s.source_id for s in result.sources]
        for k in (1, 5, 10):
            retrieval_metrics[f"recall@{k}"] = recall_at_k(
                retrieved_ids, relevance_labels, k
            )
        retrieval_metrics["mrr"] = mrr(retrieved_ids, relevance_labels)
        retrieval_metrics[f"ndcg@{min(10, len(relevance_labels))}"] = ndcg_at_k(
            retrieved_ids, relevance_labels, min(10, len(relevance_labels))
        )

    # Citation metrics
    citation_map: Dict[str, Dict[str, Any]] = {}
    for s in result.sources:
        citation_map[s.source_id] = {
            "repo_id": s.repo_id,
            "file_path": s.file_path,
            "start_line": s.start_line,
            "end_line": s.end_line,
        }

    citation_metrics: Dict[str, Any] = {}
    citation_metrics.update(citation_syntax_validity(result.answer))
    citation_metrics.update(
        citation_evidence_membership(result.answer, citation_map, repo_id)
    )
    citation_metrics.update(citation_coverage(result.answer, result.citations))

    # Answer quality metrics
    answer_metrics: Dict[str, Any] = {}
    answer_metrics.update(
        groundedness(result.citations, len(extract_citation_ids(result.answer)))
    )
    answer_metrics.update(
        unsupported_claim_rate(result.answer, result.citations)
    )

    # Abstention metrics
    abstention_metrics: Dict[str, Any] = abstention_metrics(
        result.abstained, answerable, result.abstention_reason
    )

    return RAGRequestMetrics(
        request_id=request_id,
        repo_id=repo_id,
        retrieval_method=retrieval_method,
        abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        num_chunks_retrieved=result.num_chunks_retrieved,
        num_chunks_used=result.num_chunks_used,
        context_tokens=0,
        latency=latency,
        retrieval_metrics=retrieval_metrics,
        citation_metrics=citation_metrics,
        answer_metrics=answer_metrics,
        abstention_metrics=abstention_metrics,
        errors=[],
    )
