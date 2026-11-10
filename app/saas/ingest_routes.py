"""SaaS repo-ingest routes and the background analysis job runner.

Stages: queued -> cloning(10) -> indexing(35) -> embedding(55) -> scanning(75)
-> auditing(82) -> graphing(90) -> done(100). Embedding and dependency-audit failures
are warnings, not job failures. SHERLOCK_DEP_AUDIT=false disables the network audit.
Artifacts live at <workspace_dir>/analyses/<repo_id>/{findings.json,graph.json}.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.indexer import RepositoryIndexer
from app.saas.db import get_store
from app.saas.dep_audit import audit_dependencies, empty_summary
from app.saas.github_ingest import CloneError, clone_repo, parse_github_url, repos_root
from app.saas.graph import refresh_finding_counts, build_graph, node_detail
from app.saas.http import HTTPError, Request, Response
from app.saas.plans import EVENT_ANALYSIS, plan_limits
from app.saas.quality import analyze_quality, summarize
from app.saas.vuln_scanner import scan_repository
from app.storage import IndexStore
from config import config

logger = logging.getLogger(__name__)

MAX_ACTIVE_JOBS = 2
MAX_GRAPH_NODES = 1500
REPO_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
GENERIC_FAILURE = "Analysis failed. Please try again, or try a smaller repository."
Cloner = Callable[[str, str, Optional[str], Path, int], Path]
# (repo_root, scannable_files) -> (dependency findings, summary["dependencies"])
Auditor = Callable[[Path, List[str]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]
DEP_AUDIT_DISABLED = "Dependency audit disabled by SHERLOCK_DEP_AUDIT"

_quota_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sherlock-ingest")


def analyses_dir(repo_id: str) -> Path:
    if not REPO_ID_RE.fullmatch(repo_id):
        raise ValueError("invalid repo id")
    return Path(config.workspace_dir) / "analyses" / repo_id


def _index_store() -> IndexStore:
    return IndexStore(cfg=config)


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if k != "user_id"}


def _month_start() -> datetime:
    return datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------- job runner
def _advance(job_id: str, stage: str, progress: int, **extra: Any) -> None:
    get_store().update_job(job_id, status="running", stage=stage, progress=progress, **extra)


def _embed(indexer: RepositoryIndexer, repo_id: str, enabled: bool) -> int:
    if not enabled:
        return 0
    try:
        return int(indexer.embed_chunks(repo_id).get("embedded_count", 0))
    except Exception as exc:  # optional stage: never fail the job
        logger.warning("Embedding skipped for repo %s: %s", repo_id, exc)
        return 0


def _scannable_files(index: IndexStore, repo_id: str) -> List[str]:
    return [f["path"] for f in index.get_files_by_repo(repo_id) if not f["is_binary"] and not f["is_secret"]]


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _unique(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list({f["id"]: f for f in findings}.values())


def _dep_audit_enabled() -> bool:
    return os.environ.get("SHERLOCK_DEP_AUDIT", "true").strip().lower() not in ("false", "0", "no", "off")


def _run_audit(path: Path, files: List[str], audit: Optional[Auditor]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Dependency stage. Never raises: an auditor failure is recorded in the summary."""
    if audit is None:
        if not _dep_audit_enabled():
            return [], empty_summary(DEP_AUDIT_DISABLED)
        audit = audit_dependencies
    try:
        return audit(path, files)
    except Exception:
        logger.exception("Dependency audit crashed for %s", path)
        return [], empty_summary("Dependency audit failed")


def _run_stages(
    job_id: str, user_id: str, owner: str, repo: str, branch: Optional[str], url: str, max_mb: int,
    cloner: Cloner, indexer: RepositoryIndexer, embed: bool, repos_dir: Path, audit: Optional[Auditor],
) -> Dict[str, Any]:
    store = get_store()
    _advance(job_id, "cloning", 10)
    path = cloner(owner, repo, branch, repos_dir, max_mb)

    _advance(job_id, "indexing", 35)
    res = indexer.index_repository(path, repo_name=f"{owner}/{repo}")
    if res.status != "complete":
        raise RuntimeError(f"indexing ended with status {res.status}: {res.error}")
    repo_id = res.repo_id
    store.update_job(job_id, repo_id=repo_id)

    _advance(job_id, "embedding", 55)
    embedded = _embed(indexer, repo_id, embed)

    _advance(job_id, "scanning", 75)
    files = _scannable_files(indexer.store, repo_id)
    quality_findings, metrics = analyze_quality(path, files)
    scan_findings = scan_repository(path, files)

    _advance(job_id, "auditing", 82)
    dep_findings, dep_summary = _run_audit(path, files, audit)
    findings = _unique([*scan_findings, *quality_findings, *dep_findings])
    summary = {**summarize(findings), "dependencies": dep_summary}

    _advance(job_id, "graphing", 90)
    graph = build_graph(repo_id, indexer.store, path, findings, metrics, MAX_GRAPH_NODES)
    out_dir = analyses_dir(repo_id)
    _write_json_atomic(out_dir / "findings.json", {"repo_id": repo_id, "summary": summary, "findings": findings})
    _write_json_atomic(out_dir / "graph.json", graph)

    store.add_repo_owner(repo_id, user_id, source_url=url)
    result = {
        "repo_id": repo_id,
        "repo_name": f"{owner}/{repo}",
        "stats": {
            "files": res.file_count, "lines": res.total_lines, "symbols": res.definition_count,
            "chunks": res.chunk_count, "embeddings": embedded,
        },
        "summary": summary,
    }
    # Charge the analysis only once it succeeds. Record before marking done so the
    # quota check never sees it as neither in-flight nor used.
    store.record_usage(user_id, EVENT_ANALYSIS)
    store.update_job(job_id, status="done", stage="done", progress=100, repo_id=repo_id, result=result)
    return result


def run_job(
    job_id: str,
    user_id: str,
    owner: str,
    repo: str,
    branch: Optional[str],
    url: str,
    max_mb: int,
    *,
    cloner: Cloner = clone_repo,
    indexer: Optional[RepositoryIndexer] = None,
    embed: bool = True,
    repos_dir: Optional[Path] = None,
    audit: Optional[Auditor] = None,
) -> None:
    """Run one analysis synchronously. Never raises: failures are recorded on the job.

    audit=None uses audit_dependencies (unless SHERLOCK_DEP_AUDIT=false). Tests inject a fake.
    """
    store = get_store()
    try:
        _run_stages(
            job_id, user_id, owner, repo, branch, url, max_mb, cloner,
            indexer or RepositoryIndexer(), embed, repos_dir or repos_root(), audit,
        )
    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        message = str(exc) if isinstance(exc, CloneError) else GENERIC_FAILURE
        try:
            store.update_job(job_id, status="failed", stage="failed", error=message)
        except Exception:
            logger.exception("Could not record failure for job %s", job_id)


INTERRUPTED = "Interrupted by a server restart. Please run the analysis again."


def recover_interrupted_jobs() -> int:
    """Fail jobs orphaned by the previous process (deploy/crash) so users are not
    blocked by the active-job cap and the UI stops polling them forever."""
    count = get_store().fail_interrupted_jobs(INTERRUPTED)
    if count:
        logger.warning("Marked %d interrupted analysis job(s) as failed after restart", count)
    return count


def _log_worker_crash(fut: "Future[None]") -> None:
    exc = fut.exception()
    if exc is not None:
        logger.error("Ingest worker crashed", exc_info=exc)


def _dispatch(job_id: str, user_id: str, owner: str, repo: str, branch: Optional[str], url: str, max_mb: int) -> None:
    fut = _executor.submit(run_job, job_id, user_id, owner, repo, branch, url, max_mb)
    fut.add_done_callback(_log_worker_crash)


# ------------------------------------------------------------------ handlers
def start_ingest(req: Request) -> Response:
    user = req.require_user()
    try:
        owner, repo, branch = parse_github_url(req.json().get("url"))  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPError(400, str(exc))
    canonical = f"https://github.com/{owner}/{repo}" + (f"/tree/{branch}" if branch else "")
    limits = plan_limits(user.get("plan", "free"))
    store = get_store()
    with _quota_lock:
        # Completed analyses this month + ones in flight: failed/interrupted runs are free,
        # but concurrent submissions still cannot exceed the plan.
        used = store.count_usage_since(user["id"], EVENT_ANALYSIS, _month_start())
        if used + store.count_active_jobs(user["id"]) >= limits["analyses_per_month"]:
            raise HTTPError(
                402,
                f"Monthly analysis limit reached ({limits['analyses_per_month']} on the "
                f"{limits['name']} plan). Upgrade to analyze more repositories.",
            )
        if store.count_active_jobs(user["id"]) >= MAX_ACTIVE_JOBS:
            raise HTTPError(429, f"You already have {MAX_ACTIVE_JOBS} analyses running. Wait for one to finish.")
        job = store.create_job(user["id"], canonical)
    try:
        _dispatch(job["id"], user["id"], owner, repo, branch, canonical, limits["max_repo_mb"])
    except Exception:
        logger.exception("Could not dispatch job %s", job["id"])
        store.update_job(job["id"], status="failed", stage="failed", error=GENERIC_FAILURE)
    return Response(status=202, body={"job": _public_job(job)})


def _owned_job(req: Request) -> Dict[str, Any]:
    job = get_store().get_job(req.path_params.get("job_id", ""))
    if job is None or job["user_id"] != req.require_user()["id"]:
        raise HTTPError(404, "Job not found")
    return job


def list_jobs(req: Request) -> Response:
    return Response(body={"jobs": get_store().list_jobs(req.require_user()["id"])})


def get_job(req: Request) -> Response:
    return Response(body={"job": _public_job(_owned_job(req))})


def list_repos(req: Request) -> Response:
    if req.user is None:
        return Response(body={"repositories": [], "count": 0})
    ids = get_store().list_user_repo_ids(req.user["id"])
    index = _index_store()
    try:
        rows = [index.get_repository(rid) for rid in ids]
    finally:
        index.close()
    repos = [{k: v for k, v in row.items() if k != "path"} for row in rows if row]
    return Response(body={"repositories": repos, "count": len(repos)})


def _owned_repo_id(req: Request) -> str:
    user = req.require_user()
    repo_id = req.path_params.get("repo_id", "")
    if not REPO_ID_RE.fullmatch(repo_id) or not get_store().user_owns_repo(user["id"], repo_id):
        raise HTTPError(404, "Repository not found")
    return repo_id


def _read_artifact(repo_id: str, name: str) -> Dict[str, Any]:
    path = analyses_dir(repo_id) / name
    if not path.is_file():
        raise HTTPError(404, "Analysis results not found. The analysis may still be running.")
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root(repo_id: str) -> Path:
    index = _index_store()
    try:
        row = index.get_repository(repo_id)
    finally:
        index.close()
    if row is None:
        raise HTTPError(404, "Repository not found")
    return Path(row["path"])


def get_graph(req: Request) -> Response:
    repo_id = _owned_repo_id(req)
    graph = _read_artifact(repo_id, "graph.json")
    findings = _read_artifact(repo_id, "findings.json").get("findings", [])
    return Response(body=refresh_finding_counts(graph, findings))


def get_findings(req: Request) -> Response:
    return Response(body=_read_artifact(_owned_repo_id(req), "findings.json"))


def get_node(req: Request) -> Response:
    repo_id = _owned_repo_id(req)
    node_id = req.q("id")
    if not node_id:
        raise HTTPError(400, "The id query parameter is required")
    graph = _read_artifact(repo_id, "graph.json")
    findings = _read_artifact(repo_id, "findings.json")["findings"]
    detail = node_detail(graph, node_id, findings, _repo_root(repo_id))
    if detail is None:
        raise HTTPError(404, "Node not found")
    return Response(body=detail)
