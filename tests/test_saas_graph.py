"""Graph builder: node ids, edges, health, truncation, and node detail on a real index."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.indexer import RepositoryIndexer
from app.saas.graph import build_graph, node_detail
from app.saas.vuln_scanner import make_finding, META
from config import config

ID_RE = re.compile(r"^(file:[^:]+|sym:[^:]+::[A-Za-z_][\w.]*|ext:[^:]+)$")

FILES = {
    "app/__init__.py": "",
    "app/core.py": (
        "def helper():\n"
        "    return 1\n"
        "\n"
        "class Svc:\n"
        "    def run(self):\n"
        "        return helper()\n"
    ),
    "app/main.py": (
        "import requests\n"
        "from app.core import Svc\n"
        "\n"
        "def main():\n"
        "    helper()\n"
        "    return Svc().run()\n"
    ),
    "web/util.js": "export const x = 1;\n",
    "web/app.js": "import { x } from './util';\nconst React = require('react');\n",
}


@pytest.fixture()
def indexed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    monkeypatch.setattr(config, "index_db_path", tmp_path / "idx.db")
    indexer = RepositoryIndexer()
    result = indexer.index_repository(root, repo_name="t/repo")
    return indexer, result.repo_id, root


def _edges(graph, kind):
    return {(e["source"], e["target"]) for e in graph["edges"] if e["type"] == kind}


def test_nodes_have_contract_ids_and_types(indexed):
    indexer, repo_id, root = indexed
    graph = build_graph(repo_id, indexer.store, root, [], {})
    ids = {n["id"]: n for n in graph["nodes"]}
    for expected in (
        "file:app/core.py", "sym:app/core.py::helper", "sym:app/core.py::Svc",
        "sym:app/core.py::Svc.run", "sym:app/main.py::main", "ext:requests", "ext:react",
    ):
        assert expected in ids
    assert ids["sym:app/core.py::Svc.run"]["type"] == "method"
    assert ids["sym:app/core.py::Svc"]["type"] == "class"
    assert ids["ext:requests"]["type"] == "external"
    assert all(ID_RE.match(n["id"]) for n in graph["nodes"])
    assert graph["stats"] == {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "truncated": False}


def test_contains_imports_and_calls_edges(indexed):
    indexer, repo_id, root = indexed
    graph = build_graph(repo_id, indexer.store, root, [], {})
    assert ("file:app/core.py", "sym:app/core.py::helper") in _edges(graph, "contains")
    assert ("sym:app/core.py::Svc", "sym:app/core.py::Svc.run") in _edges(graph, "contains")
    imports = _edges(graph, "imports")
    assert ("file:app/main.py", "file:app/core.py") in imports
    assert ("file:app/main.py", "ext:requests") in imports
    assert ("file:web/app.js", "file:web/util.js") in imports
    assert ("file:web/app.js", "ext:react") in imports
    calls = _edges(graph, "calls")
    assert ("sym:app/core.py::Svc.run", "sym:app/core.py::helper") in calls
    assert ("sym:app/main.py::main", "sym:app/core.py::helper") in calls
    # Svc().run() has a non-bare receiver: intentionally not an edge.
    assert ("sym:app/main.py::main", "sym:app/core.py::Svc.run") not in calls


def test_health_from_findings_and_complexity(indexed):
    indexer, repo_id, root = indexed
    findings = [
        make_finding("py-eval-exec", META["py-eval-exec"], "app/core.py", 3, "x", "sym:app/core.py::helper"),
        make_finding("py-yaml-unsafe", META["py-yaml-unsafe"], "app/main.py", 9, "y"),
    ]
    metrics = {"sym:app/core.py::Svc.run": {"loc": 2, "complexity": 16}}
    graph = build_graph(repo_id, indexer.store, root, findings, metrics)
    ids = {n["id"]: n for n in graph["nodes"]}
    assert ids["sym:app/core.py::helper"]["health"] == "bad"
    assert ids["sym:app/core.py::helper"]["findings"]["high"] == 1
    assert ids["sym:app/core.py::Svc.run"]["health"] == "warn"
    assert ids["file:app/main.py"]["health"] == "bad"
    assert ids["file:web/util.js"]["health"] == "good"


def test_truncation_keeps_files_and_drops_edges_to_removed_nodes(indexed):
    indexer, repo_id, root = indexed
    graph = build_graph(repo_id, indexer.store, root, [], {}, max_nodes=3)
    assert graph["stats"]["truncated"] is True
    assert len(graph["nodes"]) == 3
    kept = {n["id"] for n in graph["nodes"]}
    assert all(n["id"].startswith("file:") for n in graph["nodes"] if n["type"] == "file")
    assert all(e["source"] in kept and e["target"] in kept for e in graph["edges"])


def test_truncation_is_deterministic(indexed):
    indexer, repo_id, root = indexed
    a = build_graph(repo_id, indexer.store, root, [], {}, max_nodes=4)
    b = build_graph(repo_id, indexer.store, root, [], {}, max_nodes=4)
    assert a == b


def test_node_detail_neighbours_source_and_findings(indexed):
    indexer, repo_id, root = indexed
    finding = make_finding("py-eval-exec", META["py-eval-exec"], "app/core.py", 2, "return 1",
                           "sym:app/core.py::helper")
    graph = build_graph(repo_id, indexer.store, root, [finding], {})
    detail = node_detail(graph, "sym:app/core.py::helper", [finding], root)
    assert detail is not None
    assert detail["node"]["id"] == "sym:app/core.py::helper"
    assert [f["id"] for f in detail["findings"]] == [finding["id"]]
    incoming = {(n["id"], n["edge"]) for n in detail["neighbors"]["incoming"]}
    assert ("sym:app/core.py::Svc.run", "calls") in incoming
    assert ("file:app/core.py", "contains") in incoming
    lines = [row["content"] for row in detail["source"]["lines"]]
    assert "def helper():" in lines
    assert detail["source"]["start_line"] == 1
    assert detail["source"]["end_line"] == detail["source"]["start_line"] + len(lines) - 1


def test_node_detail_unknown_id_and_window_cap(indexed, tmp_path):
    indexer, repo_id, root = indexed
    graph = build_graph(repo_id, indexer.store, root, [], {})
    assert node_detail(graph, "sym:nope.py::x", [], root) is None
    big = root / "app" / "big.py"
    big.write_text("".join(f"v{i} = {i}\n" for i in range(500)))
    indexer.index_repository(root, repo_name="t/repo")
    graph = build_graph(repo_id, indexer.store, root, [], {})
    detail = node_detail(graph, "file:app/big.py", [], root)
    assert detail is not None
    assert len(detail["source"]["lines"]) <= 200


def _build(tmp_path, monkeypatch, files, name="repo"):
    root = tmp_path / name
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    monkeypatch.setattr(config, "index_db_path", tmp_path / f"{name}.db")
    indexer = RepositoryIndexer()
    result = indexer.index_repository(root, repo_name=f"t/{name}")
    return indexer, result.repo_id, root


MIXED_FILES = {
    "go.mod": "module example.com/mod\n\ngo 1.21\n",
    "web/fmt.ts": "export function fmt(s: string) {\n  return s.trim();\n}\n",
    "web/api.ts": (
        "import { fmt } from './fmt';\n"
        "import axios from 'axios';\n"
        "export class Client {\n"
        "  request(url: string) {\n"
        "    return this.send(url);\n"
        "  }\n"
        "  send(url: string) {\n"
        "    return fmt(axios.get(url));\n"
        "  }\n"
        "}\n"
    ),
    "web/main.js": "import { Client } from './api';\nexport const start = () => new Client().request('x');\n",
    "svc/s.go": (
        "package svc\n"
        "\n"
        "import (\n"
        "\t\"fmt\"\n"
        "\t\"example.com/mod/util\"\n"
        ")\n"
        "\n"
        "type Server struct{}\n"
        "\n"
        "func (s *Server) Run() {\n"
        "\tutil.Help()\n"
        "\thelper()\n"
        "\tfmt.Println(s)\n"
        "}\n"
        "\n"
        "func helper() {}\n"
    ),
    "util/help.go": "package util\n\nfunc Help() {}\n",
}


@pytest.fixture()
def mixed(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, MIXED_FILES, "mixed")


def test_mixed_js_ts_go_symbols_contains_imports_calls(mixed):
    indexer, repo_id, root = mixed
    graph = build_graph(repo_id, indexer.store, root, [], {})
    ids = {n["id"]: n for n in graph["nodes"]}
    for expected, kind in (
        ("sym:web/api.ts::Client", "class"), ("sym:web/api.ts::Client.request", "method"),
        ("sym:web/api.ts::Client.send", "method"), ("sym:web/fmt.ts::fmt", "function"),
        ("sym:svc/s.go::Server", "class"), ("sym:svc/s.go::Server.Run", "method"),
        ("sym:svc/s.go::helper", "function"), ("sym:util/help.go::Help", "function"),
    ):
        assert ids[expected]["type"] == kind, expected
    assert all(ID_RE.match(n["id"]) for n in graph["nodes"])

    contains = _edges(graph, "contains")
    assert ("file:web/api.ts", "sym:web/api.ts::Client") in contains
    assert ("sym:web/api.ts::Client", "sym:web/api.ts::Client.request") in contains
    assert ("sym:svc/s.go::Server", "sym:svc/s.go::Server.Run") in contains

    imports = _edges(graph, "imports")
    assert ("file:web/api.ts", "file:web/fmt.ts") in imports
    assert ("file:web/api.ts", "ext:axios") in imports
    assert ("file:web/main.js", "file:web/api.ts") in imports
    assert ("file:svc/s.go", "file:util/help.go") in imports
    assert ("file:svc/s.go", "ext:fmt") in imports

    calls = _edges(graph, "calls")
    assert ("sym:web/api.ts::Client.request", "sym:web/api.ts::Client.send") in calls
    assert ("sym:svc/s.go::Server.Run", "sym:svc/s.go::helper") in calls
    assert ("sym:svc/s.go::Server.Run", "sym:util/help.go::Help") in calls


def test_node_detail_for_typescript_method(mixed):
    indexer, repo_id, root = mixed
    graph = build_graph(repo_id, indexer.store, root, [], {})
    detail = node_detail(graph, "sym:web/api.ts::Client.request", [], root)
    assert detail is not None
    assert detail["node"]["type"] == "method"
    assert detail["source"]["start_line"] == 4
    assert detail["source"]["end_line"] == 6
    lines = [row["content"] for row in detail["source"]["lines"]]
    assert lines[0].strip().startswith("request(url")
    outgoing = {(n["id"], n["edge"]) for n in detail["neighbors"]["outgoing"]}
    assert ("sym:web/api.ts::Client.send", "calls") in outgoing
    incoming = {(n["id"], n["edge"]) for n in detail["neighbors"]["incoming"]}
    assert ("sym:web/api.ts::Client", "contains") in incoming


# Regression: self.x() must bind to the enclosing class's method even when the
# same name exists on other classes (previously dropped as ambiguous).
SELF_FILES = {
    "auth/signer.py": (
        "class SigningAlgorithm:\n"
        "    def get_signature(self, key, value):\n"
        "        return b''\n"
        "\n"
        "class HMACAlgorithm(SigningAlgorithm):\n"
        "    def get_signature(self, key, value):\n"
        "        return b'x'\n"
        "\n"
        "class Signer:\n"
        "    def sign(self, value):\n"
        "        return self.get_signature(value)\n"
        "\n"
        "    def get_signature(self, value):\n"
        "        return b'y'\n"
        "\n"
        "class Base:\n"
        "    def helper(self):\n"
        "        return 1\n"
        "\n"
        "class Other:\n"
        "    def helper(self):\n"
        "        return 2\n"
        "\n"
        "class Child(Base):\n"
        "    def run(self):\n"
        "        return self.helper()\n"
    ),
}


def test_self_call_links_to_enclosing_class_method(tmp_path, monkeypatch):
    indexer, repo_id, root = _build(tmp_path, monkeypatch, SELF_FILES, "selfcalls")
    graph = build_graph(repo_id, indexer.store, root, [], {})
    calls = _edges(graph, "calls")
    assert ("sym:auth/signer.py::Signer.sign", "sym:auth/signer.py::Signer.get_signature") in calls
    assert ("sym:auth/signer.py::Signer.sign", "sym:auth/signer.py::SigningAlgorithm.get_signature") not in calls
    assert ("sym:auth/signer.py::Child.run", "sym:auth/signer.py::Base.helper") in calls
    assert ("sym:auth/signer.py::Child.run", "sym:auth/signer.py::Other.helper") not in calls


def test_refresh_finding_counts_ignores_false_positives():
    from app.saas.graph import refresh_finding_counts

    graph = {"nodes": [
        {"id": "file:a.py", "type": "file", "file_path": "a.py", "metrics": {"complexity": 0}},
        {"id": "sym:a.py::f", "type": "function", "file_path": "a.py", "metrics": {"complexity": 2}},
    ], "edges": []}
    findings = [
        {"node_id": "sym:a.py::f", "file_path": "a.py", "severity": "high",
         "triage": {"verdict": "false_positive"}},
        {"node_id": "sym:a.py::f", "file_path": "a.py", "severity": "medium"},
    ]
    out = refresh_finding_counts(graph, findings)
    by_id = {n["id"]: n for n in out["nodes"]}
    assert by_id["sym:a.py::f"]["findings"]["high"] == 0
    assert by_id["sym:a.py::f"]["health"] == "warn"
    assert by_id["file:a.py"]["findings"]["medium"] == 1
    assert "findings" not in graph["nodes"][0]  # input not mutated


def test_fallback_metrics_use_line_span_and_omit_unknown_complexity():
    from app.saas.graph import _fallback_metrics

    assert _fallback_metrics({"type": "method", "line": 10, "end_line": 24}, 0) == {"loc": 15}
    assert _fallback_metrics({"type": "external", "line": 0}, 0) == {}
    assert _fallback_metrics({"type": "file", "line": 1}, 120) == {"loc": 120, "complexity": 0}
