"""Maintainability and testing metrics: "where things are lagging".

Python: per-function cyclomatic complexity, length, nesting depth, missing
docstrings on public API. All text files: LOC and TODO/FIXME/HACK markers.
Repo-wide: modules with no matching test file (filename heuristic).
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from app.saas.vuln_scanner import (
    JS_SUFFIXES,
    SEVERITY_RANK,
    Meta,
    is_test_path,
    make_finding,
    read_text,
    safe_text_path,
)

SEVERITIES = tuple(SEVERITY_RANK)
COMPLEXITY_WARN = 10
COMPLEXITY_HIGH = 20
FUNCTION_LENGTH_WARN = 80
NESTING_WARN = 4
FILE_LOC_WARN = 800
FILE_LOC_VERY_LARGE = 1500
TODO_MIN_COUNT = 3
UNTESTED_MIN_LOC = 30
MIN_MODULE_STEM = 4
TODO_RE = re.compile(r"\b(?:TODO|FIXME|HACK)\b")
PENALTY = {"critical": 10.0, "high": 3.0, "medium": 1.0, "low": 0.1, "info": 0.0}

QUALITY_META: Dict[str, Meta] = {
    "q-complexity": Meta("maintainability", "High cyclomatic complexity", "low", "CWE-1121",
        "The function has many independent paths, which makes it hard to test and change.",
        "Split the function into smaller helpers; use early returns to flatten branches."),
    "q-function-length": Meta("maintainability", "Long function", "low", None,
        f"The function is longer than {FUNCTION_LENGTH_WARN} lines.",
        "Extract cohesive blocks into well-named helper functions."),
    "q-nesting": Meta("maintainability", "Deep nesting", "low", None,
        f"Control flow is nested more than {NESTING_WARN} levels deep.",
        "Use guard clauses or extract the inner blocks."),
    "q-file-large": Meta("maintainability", "Large file", "low", None,
        f"The file has more than {FILE_LOC_WARN} lines.",
        "Split the module by responsibility."),
    "q-file-very-large": Meta("maintainability", "Very large file", "medium", None,
        f"The file has more than {FILE_LOC_VERY_LARGE} lines and is likely doing too much.",
        "Split the module by responsibility and move shared code into utilities."),
    "q-todo": Meta("quality", "Outstanding TODO/FIXME/HACK markers", "info", None,
        "Several deferred-work markers accumulated in one file.",
        "Resolve the markers or convert them into tracked issues."),
    "q-docstring": Meta("quality", "Public API without docstring", "info", None,
        "A public module-level function or class has no docstring.",
        "Describe purpose, arguments, and return value in a docstring."),
    "q-no-tests": Meta("testing", "Module without matching tests", "low", None,
        "No test file appears to cover this module (filename heuristic).",
        "Add unit tests covering the module's public behaviour."),
}


class FunctionMetric(NamedTuple):
    qual: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


_NEST_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def cyclomatic_complexity(fn: ast.AST) -> int:
    """McCabe complexity: 1 + decision points (nested defs included)."""
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
        elif isinstance(node, ast.match_case):
            score += 1
    return score


def max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Deepest control-flow nesting under node. elif chains do not add depth."""
    best = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEF_NODES):
            continue
        is_elif = isinstance(node, ast.If) and isinstance(child, ast.If) and child in node.orelse
        step = 1 if isinstance(child, _NEST_NODES) and not is_elif else 0
        best = max(best, max_nesting(child, depth + step))
    return best


def _function_metrics(tree: ast.Module) -> List[FunctionMetric]:
    out: List[FunctionMetric] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(FunctionMetric(node.name, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(FunctionMetric(f"{node.name}.{item.name}", item))
    return out


def _line_text(lines: List[str], line: int) -> str:
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def _python_quality(rel: str, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]], int]:
    """Findings, per-symbol metrics, and max function complexity for one module."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return [], {}, 0
    findings: List[Dict[str, Any]] = []
    metrics: Dict[str, Dict[str, int]] = {}
    max_cc = 0
    lines = text.splitlines()
    for fm in _function_metrics(tree):
        node_id = f"sym:{rel}::{fm.qual}"
        cc = cyclomatic_complexity(fm.node)
        length = (fm.node.end_lineno or fm.node.lineno) - fm.node.lineno + 1
        metrics[node_id] = {"loc": length, "complexity": cc}
        max_cc = max(max_cc, cc)
        src = _line_text(lines, fm.node.lineno)
        if cc > COMPLEXITY_WARN:
            sev = "medium" if cc > COMPLEXITY_HIGH else "low"
            findings.append(_q("q-complexity", rel, fm.node.lineno, src, node_id, sev))
        if length > FUNCTION_LENGTH_WARN:
            findings.append(_q("q-function-length", rel, fm.node.lineno, src, node_id))
        if max_nesting(fm.node) > NESTING_WARN:
            findings.append(_q("q-nesting", rel, fm.node.lineno, src, node_id))
    findings.extend(_docstring_findings(rel, tree, lines))
    return findings, metrics, max_cc


def _docstring_findings(rel: str, tree: ast.Module, lines: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_") or ast.get_docstring(node) is not None:
            continue
        out.append(_q("q-docstring", rel, node.lineno, _line_text(lines, node.lineno), f"sym:{rel}::{node.name}"))
    return out


def _q(rule: str, rel: str, line: int, src: str, node_id: Optional[str], severity: Optional[str] = None) -> Dict[str, Any]:
    meta = QUALITY_META[rule]
    if severity is not None:
        meta = meta._replace(severity=severity)
    return make_finding(rule, meta, rel, line, src, node_id)


def _file_findings(rel: str, lines: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    loc = len(lines)
    if loc > FILE_LOC_VERY_LARGE:
        out.append(_q("q-file-very-large", rel, 1, f"{loc} lines", None))
    elif loc > FILE_LOC_WARN:
        out.append(_q("q-file-large", rel, 1, f"{loc} lines", None))
    todo_lines = [i for i, l in enumerate(lines, start=1) if TODO_RE.search(l)]
    if len(todo_lines) >= TODO_MIN_COUNT:
        first = todo_lines[0]
        out.append(_q("q-todo", rel, first, _line_text(lines, first), None))
    return out


def _test_stem(rel: str) -> str:
    """'tests/test_graph.py' and 'graph.test.ts' both map to 'graph'."""
    stem = Path(rel).name.split(".")[0].lower()
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    for suffix in ("_test", "_spec"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def untested_module_findings(files: List[str], loc_by_file: Dict[str, int]) -> List[Dict[str, Any]]:
    """Flag sizeable source modules whose name appears in no test file name."""
    test_stems = [_test_stem(f) for f in files if is_test_path(f)]
    out: List[Dict[str, Any]] = []
    for rel in files:
        name = Path(rel).name
        if is_test_path(rel) or name.startswith(("_", "conftest", "index.")):
            continue
        if not rel.endswith((".py", *JS_SUFFIXES)):
            continue
        stem = _test_stem(rel)
        if len(stem) < MIN_MODULE_STEM or loc_by_file.get(rel, 0) < UNTESTED_MIN_LOC:
            continue
        if any(stem in t for t in test_stems):
            continue
        out.append(_q("q-no-tests", rel, 1, name, None))
    return out


def analyze_quality(root: Path, files: Iterable[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Return (findings, metrics). metrics is keyed by graph node id."""
    base = Path(root).resolve()
    rels = list(files)
    findings: List[Dict[str, Any]] = []
    metrics: Dict[str, Dict[str, int]] = {}
    loc_by_file: Dict[str, int] = {}
    for rel in rels:
        path = safe_text_path(base, rel)
        text = read_text(path) if path is not None else None
        if text is None:
            continue
        lines = text.splitlines()
        loc_by_file[rel] = len(lines)
        file_cc = 0
        if rel.endswith(".py") and not is_test_path(rel):
            py_findings, py_metrics, file_cc = _python_quality(rel, text)
            findings.extend(py_findings)
            metrics.update(py_metrics)
        findings.extend(_file_findings(rel, lines))
        metrics[f"file:{rel}"] = {"loc": len(lines), "complexity": file_cc}
    findings.extend(untested_module_findings(rels, loc_by_file))
    return findings, metrics


def summarize(findings: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Score 0-100 (100 = no weighted penalty) plus severity/category counts."""
    items = list(findings)
    by_sev = Counter(f["severity"] for f in items)
    by_cat = Counter(f["category"] for f in items)
    penalty = sum(PENALTY.get(sev, 0.0) * n for sev, n in by_sev.items())
    return {
        "score": int(max(0, min(100, round(100 - penalty)))),
        "by_severity": {sev: by_sev.get(sev, 0) for sev in SEVERITIES},
        "by_category": dict(sorted(by_cat.items())),
        "total": len(items),
    }
