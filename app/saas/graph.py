"""Repository graph (files, symbols, imports, calls) built from the code index.

Node ids: ``file:<path>``, ``sym:<path>::<Qualified.Name>``, ``ext:<package>``.
Python symbols come from the code index; other languages come from tree-sitter
(``lang_parse``). Call edges are conservative: a ``self``/``this`` call resolves
to the enclosing class's method (then its same-repo bases); other calls must
resolve to exactly one candidate.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from app.saas.lang_parse import ParsedFile, parse_file
from app.saas.lang_queries import build_index, grammar_for, resolve_import
from app.saas.vuln_scanner import SymbolSpan, read_text, safe_text_path, symbol_spans
from app.tools.filesys import SafeFilesystem

MAX_WINDOW_LINES = 200
WINDOW_LEAD = 5
WARN_COMPLEXITY = 15
MAX_BASE_HOPS = 10
SYMBOL_KINDS = {"class": "class", "function": "function", "async_function": "function", "method": "method"}
CALLABLE = ("function", "method")
TYPE_PRIORITY = {"file": 0, "class": 1, "function": 1, "method": 1, "external": 2}
SEVERITY_KEYS = ("critical", "high", "medium", "low", "info")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
Edge = Tuple[str, str, str]


class _Ctx(NamedTuple):
    sym_nodes: Dict[str, Dict[str, Any]]
    bases: Dict[str, List[str]]  # Python class node id -> base class names
    by_class: Dict[str, List[str]]  # Python class name -> class node ids


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


def _fallback_metrics(node: Dict[str, Any], file_loc: int) -> Dict[str, Any]:
    """Metrics when quality analysis has none (non-Python symbols, externals).

    Complexity is omitted rather than reported as 0 when it was never measured.
    """
    if node["type"] == "file":
        return {"loc": file_loc, "complexity": 0}
    line, end = node.get("line") or 0, node.get("end_line") or 0
    return {"loc": end - line + 1} if line and end >= line else {}


def _counted(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Findings that should colour the graph: AI-dismissed ones are excluded."""
    return [f for f in findings if (f.get("triage") or {}).get("verdict") != "false_positive"]


def refresh_finding_counts(graph: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of the graph with per-node counts/health recomputed from findings.

    Used at read time so triage verdicts show up without rebuilding the graph.
    """
    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in _counted(findings):
        if f.get("node_id"):
            by_node[f["node_id"]].append(f)
        if f.get("file_path"):
            by_file[f["file_path"]].append(f)
    nodes = []
    for node in graph.get("nodes", []):
        hits = by_file[node.get("file_path") or ""] if node.get("type") == "file" else by_node[node["id"]]
        counts = _counts(hits)
        complexity = (node.get("metrics") or {}).get("complexity", 0)
        nodes.append({**node, "findings": counts, "health": _health(counts, complexity)})
    return {**graph, "nodes": nodes}


def _read_rel(root: Path, rel: str) -> str:
    path = safe_text_path(root, rel)
    return (read_text(path) if path is not None else None) or ""


def _resolve_python(module: str, paths: Set[str]) -> Optional[str]:
    base = module.replace(".", "/")
    return next((c for c in (f"{base}.py", f"{base}/__init__.py") if c in paths), None)


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


def _lang_symbol_nodes(parsed: Dict[str, ParsedFile], lang: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for path, pf in parsed.items():
        for s in pf.symbols:
            nid = f"sym:{path}::{s.qualname}"
            nodes[nid] = {"id": nid, "type": s.kind, "label": s.qualname, "file_path": path,
                          "line": s.line, "end_line": s.end_line, "language": lang.get(path)}
    return nodes


def _file_nodes(files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        f"file:{f['path']}": {"id": f"file:{f['path']}", "type": "file", "label": Path(f["path"]).name,
                              "file_path": f["path"], "line": 1, "language": f["language"]}
        for f in files
    }


def _parse_sources(root: Path, files: List[Dict[str, Any]]) -> Dict[str, ParsedFile]:
    out: Dict[str, ParsedFile] = {}
    for f in files:
        path = f["path"]
        if path.endswith(".py") or grammar_for(path) is None:
            continue
        out[path] = parse_file(path, _read_rel(root, path), f["language"])
    return out


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


def _class_bases(defs: List[Dict[str, Any]], path_of: Dict[int, str]) -> Dict[str, List[str]]:
    """Python base names from the class signature '(A, mod.B, metaclass=M)'; keyword args dropped."""
    out: Dict[str, List[str]] = {}
    for d in defs:
        path = path_of.get(d["file_id"])
        if d["kind"] != "class" or path is None or not d.get("signature"):
            continue
        names = [b.strip().split(".")[-1] for b in d["signature"].strip("()").split(",")]
        out[f"sym:{path}::{d['name']}"] = [n for n in names if _IDENT_RE.match(n)]
    return out


def _by_class(sym_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for nid, node in sym_nodes.items():
        if node["type"] == "class" and node["file_path"].endswith(".py"):
            out[node["label"].split(".")[-1]].append(nid)
    return out


def _find_base(name: str, path: str, ctx: _Ctx) -> Optional[str]:
    local = f"sym:{path}::{name}"
    if ctx.sym_nodes.get(local, {}).get("type") == "class":
        return local
    hits = ctx.by_class.get(name, [])
    return hits[0] if len(hits) == 1 else None


def _self_target(caller: str, name: str, ctx: _Ctx) -> Optional[str]:
    """Method `name` on the caller's enclosing class, then its same-repo bases (breadth-first)."""
    path, qual = caller[len("sym:"):].split("::", 1)
    if "." not in qual:
        return None
    queue = [f"sym:{path}::{qual.rsplit('.', 1)[0]}"]
    seen: Set[str] = set()
    while queue and len(seen) < MAX_BASE_HOPS:
        cls = queue.pop(0)
        if cls in seen:
            continue
        seen.add(cls)
        cand = f"{cls}.{name}"
        if ctx.sym_nodes.get(cand, {}).get("type") in CALLABLE:
            return cand
        queue.extend(b for b in (_find_base(n, path, ctx) for n in ctx.bases.get(cls, [])) if b)
    return None


def _call_edges(refs: List[Dict[str, Any]], ctx: _Ctx, spans: Dict[str, List[SymbolSpan]]) -> Set[Edge]:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for nid, node in ctx.sym_nodes.items():
        if node["type"] in CALLABLE and node["file_path"].endswith(".py"):
            by_name[node["label"].split(".")[-1]].append(nid)
    edges: Set[Edge] = set()
    for ref in refs:
        callee_expr = _callee_expr(ref.get("evidence_detail") or "")
        caller = _caller_id(ref["file_path"], ref["line"], spans)
        if caller and callee_expr.startswith("self."):
            target = _self_target(caller, ref["symbol_name"], ctx)
            if target:
                if target != caller:
                    edges.add((caller, target, "calls"))
                continue
        if "." in callee_expr and not callee_expr.startswith("self."):
            continue
        cands = by_name.get(ref["symbol_name"], [])
        same_file = [c for c in cands if ctx.sym_nodes[c]["file_path"] == ref["file_path"]]
        target = same_file[0] if len(same_file) == 1 else (cands[0] if len(cands) == 1 else None)
        if caller and target and caller != target:
            edges.add((caller, target, "calls"))
    return edges


def _unique_callee(path: str, name: str, by_name: Dict[str, List[str]], ctx: _Ctx, imported: Set[str]) -> Optional[str]:
    """Same file (must be unique), then same imported file, then unique in repo."""
    cands = by_name.get(name, [])
    same = [c for c in cands if ctx.sym_nodes[c]["file_path"] == path]
    if same:
        return same[0] if len(same) == 1 else None
    via = [c for c in cands if ctx.sym_nodes[c]["file_path"] in imported]
    if via:
        return via[0] if len(via) == 1 else None
    return cands[0] if len(cands) == 1 else None


def _lang_call_edges(parsed: Dict[str, ParsedFile], ctx: _Ctx, targets: Dict[str, Set[str]]) -> Set[Edge]:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for nid, node in ctx.sym_nodes.items():
        if node["type"] in CALLABLE and node["file_path"] in parsed:
            by_name[node["label"].split(".")[-1]].append(nid)
    edges: Set[Edge] = set()
    for path, pf in parsed.items():
        for call in pf.calls:
            if call.caller_qualname is None:
                continue
            caller = f"sym:{path}::{call.caller_qualname}"
            if caller not in ctx.sym_nodes:
                continue
            target = _self_target(caller, call.callee_name, ctx) if call.is_self else None
            target = target or _unique_callee(path, call.callee_name, by_name, ctx, targets.get(path, set()))
            if target and target != caller:
                edges.add((caller, target, "calls"))
    return edges


def _add_external(pkg: str, src_file: str, edges: Set[Edge], externals: Dict[str, Dict[str, Any]]) -> None:
    nid = f"ext:{pkg}"
    externals.setdefault(nid, {"id": nid, "type": "external", "label": pkg,
                               "file_path": None, "line": 0, "language": None})
    edges.add((f"file:{src_file}", nid, "imports"))


def _python_import_edges(imports: List[Dict[str, Any]], paths: Set[str]) -> Tuple[Set[Edge], Dict[str, Dict[str, Any]]]:
    edges: Set[Edge] = set()
    externals: Dict[str, Dict[str, Any]] = {}
    for imp in imports:
        module = imp["module"]
        if not module:
            continue
        target = _resolve_python(module, paths)
        if target:
            edges.add((f"file:{imp['file_path']}", f"file:{target}", "imports"))
        else:
            _add_external(module.split(".")[0], imp["file_path"], edges, externals)
    return edges, externals


def _lang_import_edges(
    parsed: Dict[str, ParsedFile], idx: Any
) -> Tuple[Set[Edge], Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    edges: Set[Edge] = set()
    externals: Dict[str, Dict[str, Any]] = {}
    targets: Dict[str, Set[str]] = defaultdict(set)
    for path, pf in parsed.items():
        for imp in pf.imports:
            internal, pkg = resolve_import(grammar_for(path) or "", path, imp.module_spec, idx)
            for target in internal:
                if target != path:
                    edges.add((f"file:{path}", f"file:{target}", "imports"))
                    targets[path].add(target)
            if pkg:
                _add_external(pkg, path, edges, externals)
    return edges, externals, targets


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
    defs = store.get_definitions_by_repo(repo_id)

    nodes: Dict[str, Dict[str, Any]] = _file_nodes(files)
    sym_nodes = _symbol_nodes(defs, path_of, lang)
    parsed = _parse_sources(base, files)
    sym_nodes.update(_lang_symbol_nodes(parsed, lang))
    nodes.update(sym_nodes)

    edges: Set[Edge] = set()
    for nid in sym_nodes:
        path, qual = nid[len("sym:"):].split("::", 1)
        parent = f"sym:{path}::{qual.rsplit('.', 1)[0]}" if "." in qual else None
        edges.add((parent, nid, "contains") if parent in sym_nodes else (f"file:{path}", nid, "contains"))

    idx = build_index(paths, _read_rel(base, "go.mod") if "go.mod" in paths else "")
    py_edges, py_ext = _python_import_edges(store.get_imports_by_repo(repo_id), paths)
    lang_edges, lang_ext, targets = _lang_import_edges(parsed, idx)
    nodes.update(py_ext)
    nodes.update(lang_ext)
    edges |= py_edges | lang_edges

    ctx = _Ctx(sym_nodes, _class_bases(defs, path_of), _by_class(sym_nodes))
    spans = _python_spans(base, files)
    edges |= _call_edges(store.get_symbol_references(repo_id), ctx, spans)
    edges |= _lang_call_edges(parsed, ctx, targets)

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
    for f in _counted(findings):
        by_node[f.get("node_id")].append(f)
        by_file[f.get("file_path")].append(f)
    line_counts = {f["path"]: f["line_count"] for f in files}

    built: List[Dict[str, Any]] = []
    for nid, node in nodes.items():
        is_file = node["type"] == "file"
        counts = _counts(by_file[node["file_path"]] if is_file else by_node[nid])
        default_loc = line_counts.get(node["file_path"] or "", 0) if is_file else 0
        m = dict(metrics.get(nid) or _fallback_metrics(node, default_loc))
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


def _window_bounds(node: Dict[str, Any]) -> Tuple[int, int]:
    """Symbols with end_line: exactly line..end_line (capped). Others: lead-in context around line."""
    line = max(1, int(node.get("line") or 1))
    if node.get("end_line"):
        return line, min(max(line, int(node["end_line"])), line + MAX_WINDOW_LINES - 1)
    start = max(1, line - WINDOW_LEAD)
    return start, start + MAX_WINDOW_LINES - 1


def _source_window(root: Path, node: Dict[str, Any]) -> Dict[str, Any]:
    empty = {"path": node.get("file_path"), "start_line": 0, "end_line": 0, "lines": []}
    if not node.get("file_path"):
        return empty
    start, end = _window_bounds(node)
    try:
        lines = SafeFilesystem(root).read_file_lines(node["file_path"], start, end)
    except ValueError:
        return empty
    return {
        "path": node["file_path"],
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "lines": [{"line": start + i, "content": text} for i, text in enumerate(lines)],
    }
