"""Graph-aware chat context: a focused node's source, its neighbours, and its findings.

Built from the analysis artifacts (graph.json / findings.json) and the repo root.
Source is read through SafeFilesystem, so nothing is read outside the repository.
Chunks use the retrieval dict shape, so the pipeline cites them like any other chunk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from app.rag.context_builder import count_tokens
from app.saas.http import HTTPError
from app.saas.ingest_routes import _read_artifact, _repo_root
from app.tools.filesys import SafeFilesystem

logger = logging.getLogger(__name__)

FOCUS_TOKEN_BUDGET = 2500
WINDOW_LINES = 40
LEAD_LINES = 5
NEIGHBOUR_LIMIT = 6
FINDING_LIMIT = 5
EDGE_PRIORITY = {"calls": 0, "imports": 1}
SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")


def build_focus_context(repo_id: str, focus: Optional[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Chunks describing the focused node, within FOCUS_TOKEN_BUDGET. Empty when unavailable."""
    if not focus:
        return []
    try:
        graph = _read_artifact(repo_id, "graph.json")
        findings = _read_artifact(repo_id, "findings.json").get("findings", [])
        fs = SafeFilesystem(_repo_root(repo_id))
    except (HTTPError, OSError, ValueError, TypeError, AttributeError) as exc:
        logger.info("No graph context for repo %s: %s", repo_id, exc)
        return []
    node = _find_node(graph, focus)
    if node is None:
        return []
    candidates = _candidates(repo_id, fs, graph, node, findings)
    return _take_within_budget(candidates, FOCUS_TOKEN_BUDGET)


def _find_node(graph: Dict[str, Any], focus: Mapping[str, str]) -> Optional[Dict[str, Any]]:
    nodes = graph.get("nodes", [])
    node_id = focus.get("node_id") or ""
    if node_id:
        return next((n for n in nodes if n.get("id") == node_id), None)
    file_path = focus.get("file_path") or ""
    symbol = focus.get("symbol") or ""
    if symbol:
        for n in nodes:
            if file_path and n.get("file_path") != file_path:
                continue
            qualname = n.get("id", "").partition("::")[2]
            if qualname and (qualname == symbol or qualname.endswith("." + symbol)):
                return n
        return None
    if file_path:
        return next((n for n in nodes if n.get("id") == f"file:{file_path}"), None)
    return None


def _take_within_budget(candidates: Iterable[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    used = 0
    for chunk in candidates:
        cost = count_tokens(chunk["content"])
        if used + cost > budget:
            continue
        kept.append(chunk)
        used += cost
    return kept


def _candidates(
    repo_id: str,
    fs: SafeFilesystem,
    graph: Dict[str, Any],
    node: Dict[str, Any],
    findings: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Yield chunks in priority order: node source, its findings, then neighbours."""
    own = _window_chunk(repo_id, fs, node, "node")
    if own is not None:
        yield own
    summary = _findings_chunk(repo_id, node, findings)
    if summary is not None:
        yield summary
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    for neighbour in _neighbours(graph, nodes, node["id"]):
        chunk = _window_chunk(repo_id, fs, neighbour, "neighbour")
        if chunk is not None:
            yield chunk


def _neighbours(graph: Dict[str, Any], nodes: Dict[str, Dict[str, Any]], node_id: str) -> List[Dict[str, Any]]:
    edges = [
        e for e in graph.get("edges", [])
        if e.get("type") in EDGE_PRIORITY and node_id in (e.get("source"), e.get("target"))
    ]
    edges.sort(key=lambda e: EDGE_PRIORITY[e["type"]])  # stable: calls before imports
    seen = {node_id}
    picked: List[Dict[str, Any]] = []
    for edge in edges:
        other = edge["target"] if edge["source"] == node_id else edge["source"]
        if other in seen:
            continue
        seen.add(other)
        candidate = nodes.get(other)
        if candidate and candidate.get("file_path"):
            picked.append(candidate)
        if len(picked) == NEIGHBOUR_LIMIT:
            break
    return picked


def _window_chunk(repo_id: str, fs: SafeFilesystem, node: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    path = node.get("file_path")
    if not path:
        return None
    line = max(1, int(node.get("line") or 1))
    start = max(1, line - LEAD_LINES)
    try:
        lines = fs.read_file_lines(path, start, start + WINDOW_LINES - 1)
    except (OSError, ValueError):
        return None
    if not lines:
        return None
    return {
        "id": f"focus:{role}:{node['id']}",
        "repo_id": repo_id,
        "file_path": path,
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "content": "\n".join(lines),
        "chunk_type": "code",
        "source": "focus",
    }


def _findings_chunk(repo_id: str, node: Dict[str, Any], findings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    related = [
        f for f in findings
        if f.get("node_id") == node["id"]
        or (node.get("type") == "file" and f.get("file_path") == node.get("file_path"))
    ]
    if not related:
        return None
    related.sort(key=lambda f: SEVERITY_ORDER.index(f["severity"]) if f.get("severity") in SEVERITY_ORDER else len(SEVERITY_ORDER))
    lines = [f"Analysis findings for {node.get('label') or node['id']} ({len(related)} total):"]
    for f in related[:FINDING_LIMIT]:
        lines.append(
            f"- {f.get('title', 'Finding')} [{f.get('severity', 'info')}] line {f.get('line', 0)}: "
            f"{f.get('recommendation') or 'No recommendation recorded.'}"
        )
    line = int(node.get("line") or 1)
    return {
        "id": f"focus:findings:{node['id']}",
        "repo_id": repo_id,
        "file_path": node.get("file_path") or "",
        "start_line": line,
        "end_line": line,
        "content": "\n".join(lines),
        "chunk_type": "finding",
        "source": "focus",
    }
