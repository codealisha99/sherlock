"""Deterministic citation validation for grounded RAG answers.

Validates ``[S<n>]`` citations found in generated answers against the
retrieved evidence (the context bundle assembled for the current question).

A citation is considered valid only when ALL of the following hold:

1. It uses the canonical ``[S<n>]`` format with a positive source number.
2. The referenced source exists in the retrieved evidence map (citations to
   source chunks that were not retrieved for the current question are invalid).
3. The source belongs to the repository being queried (no cross-repository
   citation leakage between requests).
4. The cited file path is a safe repository-relative path.
5. The cited line range is internally valid (``start_line >= 1`` and
   ``end_line >= start_line``).

Malformed citations (unparseable ``[S...]`` tokens) and duplicates are handled
by the extraction/deduplication step. Citations that fail any check are
skipped; callers decide how to react to rejected citations (via abstention).

This validation is deliberately deterministic. Deciding whether a *claim* is
actually supported by the cited code is a separate grounding concern handled
later; this module must never rely on the LLM's own assertion that its citations
are correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# Citation source IDs are assigned sequentially starting at "S1".
MIN_SOURCE_NUMBER = 1


@dataclass
class Citation:
    """A validated citation referencing a retrieved source chunk."""

    source_id: str  # e.g., "S1"
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str = "code"
    content: str = ""


def _is_safe_repo_relative_path(path: str) -> bool:
    """Return True if the path is a safe, non-empty repo-relative path."""
    cleaned = (path or "").replace("\\", "/").strip()
    if not cleaned:
        return False
    if cleaned.startswith("/"):
        # Absolute paths are never acceptable as in-repo citations.
        return False
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        return False
    if any(p == ".." for p in parts):
        # Path traversal outside the repository root is never acceptable.
        return False
    return True


def extract_citation_ids(answer_text: str) -> List[str]:
    """Extract unique source IDs referenced by an answer, in order of appearance.

    Citations use the ``[S<n>]`` format. Malformed tokens (e.g. ``[Sabc]`` or
    ``[S0]``) either do not parse or resolve to invalid source IDs and are
    excluded.
    """
    found: List[str] = []
    seen: set = set()
    if not answer_text:
        return found
    for match in CITATION_PATTERN.finditer(answer_text):
        source_id = f"S{match.group(1)}"
        if source_id in seen:
            continue
        seen.add(source_id)
        found.append(source_id)
    return found


def validate_citations(
    answer_text: str,
    citation_map: Dict[str, Dict[str, Any]],
    repo_id: Optional[str] = None,
) -> List[Citation]:
    """Validate citations in an answer against retrieved evidence.

    Parameters
    ----------
    answer_text:
        The generated answer text. Cited via ``[S<n>]`` tokens.
    citation_map:
        Mapping from source_id (e.g. ``"S1"``) to trusted metadata for the
        retrieved chunks that were assembled into context. Keys absent from
        this map were not retrieved for the current question and are invalid.
    repo_id:
        Optional repository id the answer was generated for. When provided,
        citations pointing at sources from a different repository are rejected.

    Returns
    -------
    List[Citation]
        Valid citations, in the order they were referenced in the answer.
        Invalid or unsupported citations are skipped.
    """
    valid: List[Citation] = []

    for source_id in extract_citation_ids(answer_text):
        meta = citation_map.get(source_id)
        if meta is None:
            # Citation references a source that was not retrieved for this question.
            continue

        source_repo_id = meta.get("repo_id")
        if repo_id and source_repo_id and source_repo_id != repo_id:
            # Cross-repository citation leakage guard.
            continue

        file_path = meta.get("file_path") or ""
        if not _is_safe_repo_relative_path(file_path):
            continue

        try:
            start_line = int(meta.get("start_line") or 0)
            end_line = int(meta.get("end_line") or 0)
        except (TypeError, ValueError):
            continue

        # Line-range sanity: lines are 1-based and end must not precede start.
        if start_line < 1 or end_line < 1 or end_line < start_line:
            continue

        valid.append(
            Citation(
                source_id=source_id,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type=meta.get("chunk_type") or "code",
                content=meta.get("content") or "",
            )
        )

    return valid
