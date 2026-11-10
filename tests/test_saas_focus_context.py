"""Graph-aware chat context built from graph.json / findings.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.rag.context_builder import count_tokens
from app.saas.focus_context import FOCUS_TOKEN_BUDGET, NEIGHBOUR_LIMIT, WINDOW_LINES, build_focus_context

REPO_ID = "r1"


def _sym(path: str, qual: str, line: int, kind: str = "function") -> Dict[str, Any]:
    return {"id": f"sym:{path}::{qual}", "type": kind, "label": qual.split(".")[-1], "file_path": path, "line": line}


@pytest.fixture()
def artifacts(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("".join(f"line {i}\n" for i in range(1, 101)))
    (repo / "x.py").write_text("class Caller:\n    def run(self):\n        return add()\n")
    for i in range(7):
        (repo / f"u{i}.py").write_text(f"def f{i}():\n    pass\n")
    base = tmp_path / "analyses"
    monkeypatch.setattr("app.saas.ingest_routes.analyses_dir", lambda repo_id: base / repo_id)
    monkeypatch.setattr("app.saas.focus_context._repo_root", lambda repo_id: repo)

    nodes = [
        {"id": "file:calc.py", "type": "file", "label": "calc.py", "file_path": "calc.py", "line": 1},
        _sym("calc.py", "add", 10),
        _sym("x.py", "Caller.run", 2, "method"),
        {"id": "ext:requests", "type": "external", "label": "requests"},
        *[_sym(f"u{i}.py", f"f{i}", 1) for i in range(7)],
    ]
    edges = [
        {"source": "file:calc.py", "target": "sym:calc.py::add", "type": "contains"},
        {"source": "sym:x.py::Caller.run", "target": "sym:calc.py::add", "type": "calls"},
        {"source": "sym:calc.py::add", "target": "ext:requests", "type": "imports"},
        *[{"source": "sym:calc.py::add", "target": f"sym:u{i}.py::f{i}", "type": "calls"} for i in range(7)],
    ]
    findings = [{
        "id": "f1", "node_id": "sym:calc.py::add", "file_path": "calc.py", "line": 12,
        "title": "Hardcoded token", "severity": "high", "recommendation": "Load it from the environment.",
    }]
    folder = base / REPO_ID
    folder.mkdir(parents=True)
    (folder / "graph.json").write_text(json.dumps({"repo_id": REPO_ID, "nodes": nodes, "edges": edges}))
    (folder / "findings.json").write_text(json.dumps({"repo_id": REPO_ID, "summary": {}, "findings": findings}))
    return repo


def _ids(chunks: List[Dict[str, Any]], prefix: str) -> List[str]:
    return [c["id"] for c in chunks if c["id"].startswith(prefix)]


def test_node_focus_has_window_findings_and_capped_call_neighbours(artifacts):
    chunks = build_focus_context(REPO_ID, {"file_path": "calc.py", "symbol": "add", "node_id": "sym:calc.py::add"})

    node = next(c for c in chunks if c["id"] == "focus:node:sym:calc.py::add")
    assert node["file_path"] == "calc.py"
    assert node["start_line"] == 5
    assert node["end_line"] == 5 + WINDOW_LINES - 1
    assert node["content"].splitlines()[0] == "line 5"

    summary = next(c for c in chunks if c["chunk_type"] == "finding")
    assert "Hardcoded token" in summary["content"]
    assert "Load it from the environment." in summary["content"]

    neighbours = _ids(chunks, "focus:neighbour:")
    assert len(neighbours) == NEIGHBOUR_LIMIT
    assert "focus:neighbour:sym:x.py::Caller.run" in neighbours  # calls edge, incoming
    assert not any("requests" in c for c in neighbours)  # external nodes have no source
    assert all(len(c["content"].splitlines()) <= WINDOW_LINES for c in chunks)


def test_symbol_focus_matches_qualified_name(artifacts):
    chunks = build_focus_context(REPO_ID, {"file_path": "x.py", "symbol": "run"})
    assert chunks[0]["id"] == "focus:node:sym:x.py::Caller.run"
    assert chunks[0]["content"].startswith("class Caller:")


def test_file_focus_uses_file_node(artifacts):
    chunks = build_focus_context(REPO_ID, {"file_path": "calc.py", "symbol": ""})
    assert chunks and chunks[0]["id"] == "focus:node:file:calc.py"


def test_total_is_within_token_budget(artifacts):
    long_line = "token_" + "x" * 300
    (artifacts / "calc.py").write_text("\n".join(long_line for _ in range(200)) + "\n")
    chunks = build_focus_context(REPO_ID, {"node_id": "sym:calc.py::add"})
    assert chunks
    assert sum(count_tokens(c["content"]) for c in chunks) <= FOCUS_TOKEN_BUDGET


def test_path_escape_is_never_read(artifacts, tmp_path, monkeypatch):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n")
    graph_path = tmp_path / "analyses" / REPO_ID / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"].append(_sym("../secret.txt", "leak", 1))
    graph["edges"].append({"source": "sym:calc.py::add", "target": "sym:../secret.txt::leak", "type": "calls"})
    graph_path.write_text(json.dumps(graph))

    chunks = build_focus_context(REPO_ID, {"node_id": "sym:calc.py::add"})
    assert not any("TOP SECRET" in c["content"] for c in chunks)


def test_missing_artifacts_or_unknown_node_yield_nothing(artifacts, tmp_path):
    assert build_focus_context(REPO_ID, {"node_id": "sym:calc.py::nope"}) == []
    assert build_focus_context(REPO_ID, None) == []
    assert build_focus_context("missing-repo", {"node_id": "sym:calc.py::add"}) == []
