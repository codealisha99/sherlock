"""Ingest job runner and SaaS handlers. Git is faked; the thread pool is never started."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app.indexer import RepositoryIndexer
from app.saas import ingest_routes as ir
from app.saas.db import SaasStore, set_store
from app.saas.github_ingest import CloneError
from app.saas.http import HTTPError, Request
from app.saas.plans import EVENT_ANALYSIS
from config import config

URL = "https://github.com/octo/demo"
APP_PY = (
    "import os\n"
    "import yaml\n"
    "\n"
    "def run(cmd):\n"
    "    os.system(cmd)\n"
    "    return eval(cmd)\n"
    "\n"
    "def load(blob):\n"
    "    return yaml.load(blob)\n"
)


@pytest.fixture(autouse=True)
def _no_network_audit(monkeypatch):
    """Jobs never reach OSV.dev in tests; audit-specific tests inject an auditor instead."""
    monkeypatch.setenv("SHERLOCK_DEP_AUDIT", "false")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "index_db_path", tmp_path / "idx.db")
    monkeypatch.setattr(config, "workspace_dir", tmp_path / "ws")
    db = SaasStore(tmp_path / "s.db")
    set_store(db)
    monkeypatch.setattr(ir, "_dispatch", lambda *a, **k: None)  # never start real threads
    yield db
    set_store(None)


@pytest.fixture()
def owner(store):
    return store.create_user(f"{uuid.uuid4().hex[:8]}@example.com", "hash")


def _other(store):
    return store.create_user(f"{uuid.uuid4().hex[:8]}@example.com", "hash")


def _user(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": row["id"], "email": row["email"], "plan": row.get("plan", "free")}


def _req(user: Optional[Dict[str, Any]], body: Optional[Dict[str, Any]] = None,
         path_params: Optional[Dict[str, str]] = None, query: Optional[Dict[str, List[str]]] = None,
         method: str = "POST") -> Request:
    raw = json.dumps(body).encode() if body is not None else b""
    return Request(method=method, path="/x", query=query or {}, headers={}, body=raw,
                   client_ip="127.0.0.1", user=user, path_params=path_params or {})


def _fake_cloner(files: Optional[Dict[str, str]] = None):
    calls: List[tuple] = []

    def cloner(owner_name, repo, branch, dest_root, max_mb):
        calls.append((owner_name, repo, branch, max_mb))
        dest = Path(dest_root) / f"{owner_name}__{repo}__{uuid.uuid4().hex}"
        for rel, text in (files or {"app/main.py": APP_PY, "README.md": "# demo\n"}).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return dest

    cloner.calls = calls  # type: ignore[attr-defined]
    return cloner


def _run(store, owner_row, cloner, *, embed=False, indexer=None, url=URL, tmp=None):
    job = store.create_job(owner_row["id"], url)
    ir.run_job(job["id"], owner_row["id"], "octo", "demo", None, url, 50,
               cloner=cloner, indexer=indexer, embed=embed, repos_dir=tmp)
    return store.get_job(job["id"])


def test_full_job_run_produces_artifacts_and_ownership(store, owner, tmp_path):
    job = _run(store, owner, _fake_cloner(), tmp=tmp_path / "repos")
    assert job["status"] == "done" and job["stage"] == "done" and job["progress"] == 100
    result = job["result"]
    assert result["repo_name"] == "octo/demo"
    assert result["stats"]["files"] == 2
    assert result["summary"]["total"] >= 3  # os.system, eval, yaml.load
    repo_id = result["repo_id"]
    assert store.user_owns_repo(owner["id"], repo_id)
    out = ir.analyses_dir(repo_id)
    graph = json.loads((out / "graph.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    assert graph["repo_id"] == repo_id and graph["nodes"]
    assert findings["summary"] == result["summary"]
    assert any(f["rule_id"] == "py-eval-exec" for f in findings["findings"])


def test_auditing_stage_merges_dependency_findings(store, owner, tmp_path, monkeypatch):
    dep = {
        "id": "dep1", "category": "dependency", "rule_id": "dep-vuln", "title": "requests 2.19.0 has 1 known vulnerability",
        "severity": "high", "file_path": "requirements.txt", "line": 1, "snippet": "requests==2.19.0",
        "description": "", "recommendation": "", "cwe": None, "node_id": "file:requirements.txt",
        "confidence": "high", "package": "requests", "version": "2.19.0", "ecosystem": "PyPI",
        "advisory_ids": ["GHSA-x"], "fixed_versions": ["2.20.0"], "references": [],
    }
    stages: List[str] = []
    real_advance = ir._advance

    def record(job_id, stage, progress, **extra):
        stages.append((stage, progress))
        real_advance(job_id, stage, progress, **extra)

    seen: List[Any] = []

    def auditor(path, files):
        seen.append(sorted(files))
        return [dep], {"scanned": 1, "vulnerable": 1, "ecosystems": ["PyPI"],
                       "manifests": ["requirements.txt"], "error": None}

    monkeypatch.setattr(ir, "_advance", record)
    job = store.create_job(owner["id"], URL)
    ir.run_job(job["id"], owner["id"], "octo", "demo", None, URL, 50, cloner=_fake_cloner(),
               indexer=None, embed=False, repos_dir=tmp_path / "repos", audit=auditor)
    job = store.get_job(job["id"])
    assert job["status"] == "done"
    progress = [p for _, p in stages]
    assert progress == sorted(progress)
    assert ("auditing", 82) in stages and stages.index(("scanning", 75)) < stages.index(("auditing", 82)) < stages.index(("graphing", 90))
    assert seen and "app/main.py" in seen[0]
    summary = job["result"]["summary"]
    assert summary["dependencies"]["vulnerable"] == 1 and summary["by_category"]["dependency"] == 1
    findings = json.loads((ir.analyses_dir(job["result"]["repo_id"]) / "findings.json").read_text())
    assert any(f["id"] == "dep1" for f in findings["findings"])
    assert findings["summary"]["dependencies"]["manifests"] == ["requirements.txt"]


def test_audit_failure_is_warning_not_failure(store, owner, tmp_path):
    def crashing(path, files):
        raise RuntimeError("boom")

    job = store.create_job(owner["id"], URL)
    ir.run_job(job["id"], owner["id"], "octo", "demo", None, URL, 50, cloner=_fake_cloner(),
               embed=False, repos_dir=tmp_path / "repos", audit=crashing)
    job = store.get_job(job["id"])
    assert job["status"] == "done"
    assert job["result"]["summary"]["dependencies"]["error"] == "Dependency audit failed"


def test_env_disables_dependency_audit(store, owner, tmp_path, monkeypatch):
    called: List[Any] = []
    monkeypatch.setattr(ir, "audit_dependencies", lambda *a, **k: called.append(a) or ([], {}))
    job = _run(store, owner, _fake_cloner(), tmp=tmp_path / "repos")
    assert called == []
    deps = job["result"]["summary"]["dependencies"]
    assert deps["scanned"] == 0 and deps["error"] == ir.DEP_AUDIT_DISABLED


def test_embedding_failure_is_warning_not_failure(store, owner, tmp_path):
    indexer = RepositoryIndexer()

    def broken(repo_id, **kwargs):
        raise RuntimeError("model not available")

    indexer.embed_chunks = broken  # type: ignore[method-assign]
    job = _run(store, owner, _fake_cloner(), embed=True, indexer=indexer, tmp=tmp_path / "repos")
    assert job["status"] == "done"
    assert job["result"]["stats"]["embeddings"] == 0


def test_clone_failure_surfaces_friendly_message(store, owner, tmp_path):
    def cloner(*args):
        raise CloneError("Could not clone that repository. Check that it exists and is public.")

    job = _run(store, owner, cloner, tmp=tmp_path / "repos")
    assert job["status"] == "failed" and job["stage"] == "failed"
    assert job["error"].startswith("Could not clone")


def test_unexpected_error_hides_internals(store, owner, tmp_path):
    def cloner(*args):
        raise OSError(f"disk full at {tmp_path}/secret/path")

    job = _run(store, owner, cloner, tmp=tmp_path / "repos")
    assert job["status"] == "failed"
    assert job["error"] == ir.GENERIC_FAILURE
    assert str(tmp_path) not in job["error"]


def test_start_ingest_accepts_without_charging_quota_until_success(store, owner, monkeypatch):
    dispatched: List[tuple] = []
    monkeypatch.setattr(ir, "_dispatch", lambda *a: dispatched.append(a))
    resp = ir.start_ingest(_req(_user(owner), {"url": "https://github.com/octo/demo.git"}))
    assert resp.status == 202
    job = resp.body["job"]
    assert "user_id" not in job and job["status"] == "queued"
    assert job["github_url"] == URL
    assert dispatched and dispatched[0][0] == job["id"]
    # Charged on success (see test_success_is_charged_and_failure_is_free), not on submit.
    assert store.count_usage_since(owner["id"], EVENT_ANALYSIS, datetime.now(timezone.utc) - timedelta(days=1)) == 0


def test_start_ingest_rejects_bad_url(store, owner):
    with pytest.raises(HTTPError) as exc:
        ir.start_ingest(_req(_user(owner), {"url": "http://evil.example/x"}))
    assert exc.value.status == 400


def test_quota_exceeded_returns_402(store, owner):
    for _ in range(3):  # free plan allows 3 analyses per month
        store.record_usage(owner["id"], EVENT_ANALYSIS)
    with pytest.raises(HTTPError) as exc:
        ir.start_ingest(_req(_user(owner), {"url": URL}))
    assert exc.value.status == 402
    assert "Monthly analysis limit" in exc.value.message


def test_two_active_jobs_returns_429(store, owner):
    store.create_job(owner["id"], URL)
    store.create_job(owner["id"], URL)
    with pytest.raises(HTTPError) as exc:
        ir.start_ingest(_req(_user(owner), {"url": URL}))
    assert exc.value.status == 429


def test_dispatch_failure_marks_job_failed(store, owner, monkeypatch):
    def boom(*args):
        raise RuntimeError("executor down")

    monkeypatch.setattr(ir, "_dispatch", boom)
    resp = ir.start_ingest(_req(_user(owner), {"url": URL}))
    assert store.get_job(resp.body["job"]["id"])["status"] == "failed"


def test_jobs_are_private_to_owner(store, owner):
    job = store.create_job(owner["id"], URL)
    intruder = _other(store)
    with pytest.raises(HTTPError) as exc:
        ir.get_job(_req(_user(intruder), path_params={"job_id": job["id"]}, method="GET"))
    assert exc.value.status == 404
    assert ir.get_job(_req(_user(owner), path_params={"job_id": job["id"]}, method="GET")).body["job"]["id"] == job["id"]
    assert [j["id"] for j in ir.list_jobs(_req(_user(intruder), method="GET")).body["jobs"]] == []


def test_repo_endpoints_404_for_non_owner(store, owner, tmp_path):
    job = _run(store, owner, _fake_cloner(), tmp=tmp_path / "repos")
    repo_id = job["result"]["repo_id"]
    intruder = _other(store)
    for handler in (ir.get_graph, ir.get_findings):
        with pytest.raises(HTTPError) as exc:
            handler(_req(_user(intruder), path_params={"repo_id": repo_id}, method="GET"))
        assert exc.value.status == 404
    with pytest.raises(HTTPError) as exc:
        ir.get_node(_req(_user(intruder), path_params={"repo_id": repo_id}, query={"id": ["file:app/main.py"]}, method="GET"))
    assert exc.value.status == 404


def test_repo_id_traversal_is_rejected(store, owner):
    with pytest.raises(HTTPError) as exc:
        ir.get_graph(_req(_user(owner), path_params={"repo_id": ".."}, method="GET"))
    assert exc.value.status == 404


def test_owner_reads_graph_findings_and_node(store, owner, tmp_path):
    job = _run(store, owner, _fake_cloner(), tmp=tmp_path / "repos")
    repo_id = job["result"]["repo_id"]
    params = {"repo_id": repo_id}
    graph = ir.get_graph(_req(_user(owner), path_params=params, method="GET")).body
    assert any(n["id"] == "sym:app/main.py::run" for n in graph["nodes"])
    findings = ir.get_findings(_req(_user(owner), path_params=params, method="GET")).body
    assert findings["summary"]["total"] == len(findings["findings"])
    node = ir.get_node(_req(_user(owner), path_params=params, query={"id": ["sym:app/main.py::run"]}, method="GET")).body
    assert node["node"]["type"] == "function"
    assert any(f["rule_id"] == "py-os-shell" for f in node["findings"])
    assert any("def run" in row["content"] for row in node["source"]["lines"])
    with pytest.raises(HTTPError) as exc:
        ir.get_node(_req(_user(owner), path_params=params, query={"id": ["sym:app/main.py::missing"]}, method="GET"))
    assert exc.value.status == 404
    with pytest.raises(HTTPError) as exc:
        ir.get_node(_req(_user(owner), path_params=params, method="GET"))
    assert exc.value.status == 400


def test_list_repos_only_caller_and_no_server_path(store, owner, tmp_path):
    _run(store, owner, _fake_cloner(), tmp=tmp_path / "repos")
    body = ir.list_repos(_req(_user(owner), method="GET")).body
    assert body["count"] == 1
    assert "path" not in body["repositories"][0]
    assert body["repositories"][0]["name"] == "octo/demo"
    assert ir.list_repos(_req(_user(_other(store)), method="GET")).body["count"] == 0


def test_anonymous_list_repos_is_empty(store):
    assert ir.list_repos(_req(None, method="GET")).body == {"repositories": [], "count": 0}


def test_missing_artifacts_return_404(store, owner):
    repo_id = str(uuid.uuid4())
    store.add_repo_owner(repo_id, owner["id"], source_url=URL)
    with pytest.raises(HTTPError) as exc:
        ir.get_graph(_req(_user(owner), path_params={"repo_id": repo_id}, method="GET"))
    assert exc.value.status == 404



def _month_usage(store, user_id):
    return store.count_usage_since(user_id, EVENT_ANALYSIS, datetime.now(timezone.utc) - timedelta(days=1))


def test_restart_recovery_unblocks_user_and_refunds_quota(store, owner, monkeypatch):
    """Reproduces the production failure: 2 jobs orphaned by a restart blocked the user forever."""
    monkeypatch.setattr(ir, "_dispatch", lambda *a: None)  # jobs never run: simulates the killed process
    for _ in range(2):
        assert ir.start_ingest(_req(_user(owner), {"url": URL})).status == 202
    with pytest.raises(HTTPError) as exc:
        ir.start_ingest(_req(_user(owner), {"url": URL}))
    assert exc.value.status == 429  # blocked, as observed

    assert ir.recover_interrupted_jobs() == 2
    jobs = store.list_jobs(owner["id"])
    assert {j["status"] for j in jobs} == {"failed"} and all(j["error"] == ir.INTERRUPTED for j in jobs)
    assert _month_usage(store, owner["id"]) == 0  # interrupted runs cost nothing
    assert ir.start_ingest(_req(_user(owner), {"url": URL})).status == 202  # unblocked


def test_in_flight_jobs_count_toward_monthly_quota(store, owner, monkeypatch):
    monkeypatch.setattr(ir, "_dispatch", lambda *a: None)
    store.record_usage(owner["id"], EVENT_ANALYSIS)  # free plan: 3/month; 1 completed
    ir.start_ingest(_req(_user(owner), {"url": URL}))
    ir.start_ingest(_req(_user(owner), {"url": URL}))  # 1 completed + 2 in flight = 3
    with pytest.raises(HTTPError) as exc:
        ir.start_ingest(_req(_user(owner), {"url": URL}))
    assert exc.value.status == 402  # quota is checked before the active-job cap


def _no_audit(path, files):
    return [], {"scanned": 0, "vulnerable": 0, "ecosystems": [], "manifests": [], "error": None}


def test_success_is_charged_and_failure_is_free(store, owner, tmp_path):
    ok = store.create_job(owner["id"], URL)
    ir.run_job(ok["id"], owner["id"], "octo", "demo", None, URL, 50, cloner=_fake_cloner(),
               indexer=None, embed=False, repos_dir=tmp_path / "repos", audit=_no_audit)
    assert store.get_job(ok["id"])["status"] == "done"
    assert _month_usage(store, owner["id"]) == 1

    def broken_cloner(*_a, **_k):
        raise RuntimeError("boom")

    bad = store.create_job(owner["id"], URL)
    ir.run_job(bad["id"], owner["id"], "octo", "demo", None, URL, 50, cloner=broken_cloner,
               indexer=None, embed=False, repos_dir=tmp_path / "repos2", audit=_no_audit)
    assert store.get_job(bad["id"])["status"] == "failed"
    assert _month_usage(store, owner["id"]) == 1
