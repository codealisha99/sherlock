"""Repository graph (files, symbols, imports, calls) built from the code index.

Node ids: ``file:<path>``, ``sym:<path>::<Qualified.Name>``, ``ext:<package>``.
Call edges are conservative: only bare-name or ``self.`` calls that resolve to
exactly one candidate symbol are emitted.
"""

from __future__ import annotations

import ast
import posixpath
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.saas.vuln_scanner import (
    JS_SUFFIXES,
    SymbolSpan,
    read_text,
    safe_text_path,
    symbol_spans,
)
from app.tools.filesys import SafeFilesystem

MAX_WINDOW_LINES = 200
WINDOW_LEAD = 5
WARN_COMPLEXITY = 15
SYMBOL_KINDS = {"class": "class", "function": "function", "async_function": "function", "method": "method"}
TYPE_PRIORITY = {"file": 0, "class": 1, "function": 1, "method": 1, "external": 2}
SEVERITY_KEYS = ("critical", "high", "medium", "low", "info")
JS_SPEC_RE = re.compile(
    r"""(?:import|export)\s[^'"`;]*?\bfrom\s*['"]([^'"]+)['"]"""
    r"""|\bimport\s*['"]([^'"]+)['"]"""
    r"""|\brequire\(\s*['"]([^'"]+)['"]\s*\)"""
    r"""|\bimport\(\s*['"]([^'"]+)['"]\s*\)"""
)
Edge = Tuple[str, str, str]


def _counts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = {k: 0 for k in SEVERITY_KEYS}
    for f in items:
        if f["severity"] in out:
            out[f["severity"]] += 1
    return out


def _health(counts: Dict[str, int], complexity: int) -> str:
    if counts["critical"] or counts["high"]:
        return "bad"
    if counts["medium"] or complexity > WARN_COMPLEXITY:
        return "warn"
    return "good"


def _read_rel(root: Path, rel: str) -> str:
    path = safe_text_path(root, rel)
    return (read_text(path) if path is not None else None) or ""


def _resolve_python(module: str, paths: Set[str]) -> Optional[str]:
    base = module.replace(".", "/")
    return next((c for c in (f"{base}.py", f"{base}/__init__.py") if c in paths), None)


def _resolve_js(rel: str, spec: str, paths: Set[str]) -> Optional[str]:
    base = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
    if base == ".." or base.startswith("../"):
        return None
    candidates = [base, *(base + e for e in JS_SUFFIXES), *(f"{base}/index{e}" for e in JS_SUFFIXES)]
    return next((c for c in candidates if c in paths), None)


def _package_name(spec: str) -> str:
    parts = spec.split("/")
    return "/".join(parts[:2]) if spec.startswith("@") else parts[0]


def _js_specs(text: str) -> List[str]:
    return [next(g for g in m.groups() if g) for m in JS_SPEC_RE.finditer(text)]


def _symbol_nodes(defs: List[Dict[str, Any]], path_of: Dict[int, str], lang: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for d in defs:
        kind = SYMBOL_KINDS.get(d["kind"])
        path = path_of.get(d["file_id"])
        if kind is None or path is None:
            continue
        nid = f"sym:{path}::{d['name']}"
        nodes[nid] = {"id": nid, "type": kind, "label": d["name"], "file_path": path,
                      "line": d["line"], "language": lang.get(path)}
    return nodes


def _file_nodes(files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        f"file:{f['path']}": {"id": f"file:{f['path']}", "type": "file", "label": Path(f["path"]).name,
                              "file_path": f["path"], "line": 1, "language": f["language"]}
        for f in files
    }


def _python_spans(root: Path, files: List[Dict[str, Any]]) -> Dict[str, List[SymbolSpan]]:
    spans: Dict[str, List[SymbolSpan]] = {}
    for f in files:
        if f["path"].endswith(".py"):
            try:
                spans[f["path"]] = symbol_spans(ast.parse(_read_rel(root, f["path"])))
            except (SyntaxError, ValueError, RecursionError):
                spans[f["path"]] = []
    return spans


def _caller_id(path: str, line: int, spans: Dict[str, List[SymbolSpan]]) -> Optional[str]:
    hits = [s for s in spans.get(path, []) if s[0] <= line <= s[1]]
    if not hits:
        return None
    return f"sym:{path}::{min(hits, key=lambda s: s[1] - s[0])[2]}"


def _callee_expr(evidence: str) -> str:
    """'Svc().run(...)' -> 'Svc().run'; 'helper(...)' -> 'helper'."""
    return evidence[: -len("(...)")] if evidence.endswith("(...)") else evidence


def _call_edges(
    refs: List[Dict[str, Any]], sym_nodes: Dict[str, Dict[str, Any]], spans: Dict[str, List[SymbolSpan]]
) -> Set[Edge]:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for nid, node in sym_nodes.items():
        if node["type"] in ("function", "method"):
            by_name[node["label"].split(".")[-1]].append(nid)
    edges: Set[Edge] = set()
    for ref in refs:
        callee_expr = _callee_expr(ref.get("evidence_detail") or "")
        if "." in callee_expr and not callee_expr.startswith("self."):
            continue
        caller = _caller_id(ref["file_path"], ref["line"], spans)
        cands = by_name.get(ref["symbol_name"], [])
        same_file = [c for c in cands if sym_nodes[c]["file_path"] == ref["file_path"]]
        target = same_file[0] if len(same_file) == 1 else (cands[0] if len(cands) == 1 else None)
        if caller and target and caller != target:
            edges.add((caller, target, "calls"))
    return edges


def _import_edges(
    root: Path, files: List[Dict[str, Any]], imports: List[Dict[str, Any]], paths: Set[str]
) -> Tuple[Set[Edge], Dict[str, Dict[str, Any]]]:
    edges: Set[Edge] = set()
    externals: Dict[str, Dict[str, Any]] = {}

    def add_ext(pkg: str, src_file: str) -> None:
        nid = f"ext:{pkg}"
        externals.setdefault(nid, {"id": nid, "type": "external", "label": pkg,
                                   "file_path": None, "line": 0, "language": None})
        edges.add((f"file:{src_file}", nid, "imports"))

    for imp in imports:
        module = imp["module"]
        if not module:
            continue
        target = _resolve_python(module, paths)
        if target:
            edges.add((f"file:{imp['file_path']}", f"file:{target}", "imports"))
        else:
            add_ext(module.split(".")[0], imp["file_path"])

    for f in files:
        if not f["path"].endswith(JS_SUFFIXES):
            continue
        for spec in _js_specs(_read_rel(root, f["path"])):
            if spec.startswith(("./", "../")):
                target = _resolve_js(f["path"], spec, paths)
                if target:
                    edges.add((f"file:{f['path']}", f"file:{target}", "imports"))
            elif spec and not spec.startswith("/"):
                add_ext(_package_name(spec), f["path"])
    return edges, externals


def build_graph(
    repo_id: str,
    store: Any,
    root: Path,
    findings: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, int]],
    max_nodes: int = 1500,
) -> Dict[str, Any]:
    """Assemble the graph dict per docs/SAAS_ARCHITECTURE.md, truncated to max_nodes."""
    files = store.get_files_by_repo(repo_id)
    paths = {f["path"] for f in files}
    lang = {f["path"]: f["language"] for f in files}
    path_of = {f["id"]: f["path"] for f in files}
    base = Path(root).resolve()

    nodes: Dict[str, Dict[str, Any]] = _file_nodes(files)
    sym_nodes = _symbol_nodes(store.get_definitions_by_repo(repo_id), path_of, lang)
    nodes.update(sym_nodes)

    edges: Set[Edge] = set()
    for nid in sym_nodes:
        path, qual = nid[len("sym:"):].split("::", 1)
        parent = f"sym:{path}::{qual.split('.')[0]}"
        if "." in qual and parent in sym_nodes:
            edges.add((parent, nid, "contains"))
        else:
            edges.add((f"file:{path}", nid, "contains"))

    imp_edges, externals = _import_edges(base, files, store.get_imports_by_repo(repo_id), paths)
    nodes.update(externals)
    edges |= imp_edges

    spans = _python_spans(base, files)
    edges |= _call_edges(store.get_symbol_references(repo_id), sym_nodes, spans)

    return _assemble(repo_id, nodes, edges, findings, metrics, files, max_nodes)


def _assemble(
    repo_id: str,
    nodes: Dict[str, Dict[str, Any]],
    edges: Set[Edge],
    findings: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, int]],
    files: List[Dict[str, Any]],
    max_nodes: int,
) -> Dict[str, Any]:
    edges = {e for e in edges if e[0] in nodes and e[1] in nodes}
    degree: Dict[str, int] = defaultdict(int)
    for src, dst, _ in edges:
        degree[src] += 1
        degree[dst] += 1

    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_node[f["node_id"]].append(f)
        by_file[f["file_path"]].append(f)
    line_counts = {f["path"]: f["line_count"] for f in files}

    built: List[Dict[str, Any]] = []
    for nid, node in nodes.items():
        is_file = node["type"] == "file"
        counts = _counts(by_file[node["file_path"]] if is_file else by_node[nid])
        default_loc = line_counts.get(node["file_path"] or "", 0) if is_file else 0
        m = dict(metrics.get(nid) or {"loc": default_loc, "complexity": 0})
        built.append({**node, "metrics": m, "findings": counts,
                      "health": _health(counts, m.get("complexity", 0))})

    ordered = sorted(built, key=lambda n: (TYPE_PRIORITY[n["type"]], -degree[n["id"]], n["id"]))
    kept = sorted(ordered[:max_nodes], key=lambda n: n["id"])
    kept_ids = {n["id"] for n in kept}
    kept_edges = sorted(
        ({"source": s, "target": t, "type": ty} for s, t, ty in edges if s in kept_ids and t in kept_ids),
        key=lambda e: (e["type"], e["source"], e["target"]),
    )
    return {
        "repo_id": repo_id,
        "nodes": kept,
        "edges": kept_edges,
        "stats": {"nodes": len(kept), "edges": len(kept_edges), "truncated": len(built) > max_nodes},
    }


def node_detail(graph: Dict[str, Any], node_id: str, findings: List[Dict[str, Any]], root: Path) -> Optional[Dict[str, Any]]:
    """Node + findings + neighbours + source window. None when node_id is unknown."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    node = nodes.get(node_id)
    if node is None:
        return None
    incoming = [_neighbour(nodes, e["source"], e) for e in graph["edges"] if e["target"] == node_id]
    outgoing = [_neighbour(nodes, e["target"], e) for e in graph["edges"] if e["source"] == node_id]
    related = [
        f for f in findings
        if f["node_id"] == node_id or (node["type"] == "file" and f["file_path"] == node["file_path"])
    ]
    return {
        "node": node,
        "findings": related,
        "neighbors": {"incoming": incoming, "outgoing": outgoing},
        "source": _source_window(root, node),
    }


def _neighbour(nodes: Dict[str, Dict[str, Any]], other_id: str, edge: Dict[str, Any]) -> Dict[str, Any]:
    other = nodes.get(other_id, {})
    return {"id": other_id, "label": other.get("label", other_id), "type": other.get("type", "external"),
            "edge": edge["type"]}


def _source_window(root: Path, node: Dict[str, Any]) -> Dict[str, Any]:
    empty = {"path": node.get("file_path"), "start_line": 0, "end_line": 0, "lines": []}
    if not node.get("file_path"):
        return empty
    start = max(1, int(node.get("line") or 1) - WINDOW_LEAD)
    try:
        lines = SafeFilesystem(root).read_file_lines(node["file_path"], start, start + MAX_WINDOW_LINES - 1)
    except ValueError:
        return empty
    return {
        "path": node["file_path"],
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "lines": [{"line": start + i, "content": text} for i, text in enumerate(lines)],
    }
