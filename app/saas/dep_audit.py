"""Dependency vulnerability audit against OSV.dev.

Produces one "dep-vuln" finding per (package, version) that has known advisories.
Network failures never raise: they are reported in summary["error"].
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx

from app.saas.cvss import base_score, severity_for_score
from app.saas.dep_manifests import ECO_PYPI, Dependency, discover_dependencies, normalize_pypi
from app.saas.vuln_scanner import SEVERITY_RANK, Meta, make_finding

logger = logging.getLogger(__name__)

OSV_BASE = "https://api.osv.dev/v1"
BATCH_SIZE = 1000  # querybatch limit
DETAIL_LIMIT = 200  # max vulnerability records fetched per run
DETAIL_WORKERS = 8
BUDGET_SECONDS = 20.0  # total network budget for one audit
ENDPOINT_TIMEOUT = 10.0  # per request

MSG_UNREACHABLE = "Dependency vulnerability database unreachable"
MSG_TIMEOUT = "Dependency vulnerability database timed out"
MSG_ERROR = "Dependency vulnerability database returned an error"
MSG_FAILED = "Dependency audit failed"

_DB_SEVERITY = {"CRITICAL": "critical", "HIGH": "high", "MODERATE": "medium", "MEDIUM": "medium", "LOW": "low"}
_CORE_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")
_VERSION_PAD = 6


class _AuditError(Exception):
    """Network or protocol failure with a user-safe message."""


def empty_summary(error: Optional[str] = None) -> Dict[str, Any]:
    return {"scanned": 0, "vulnerable": 0, "ecosystems": [], "manifests": [], "error": error}


# ------------------------------------------------------------------ versions
def version_key(version: str) -> Optional[Tuple[int, ...]]:
    """Numeric sort key for release-style versions. Pre-release suffixes are ignored."""
    m = _CORE_VERSION.match(version.strip())
    if m is None:
        return None
    parts = tuple(int(p) for p in m.group(1).split("."))
    return parts + (0,) * (_VERSION_PAD - len(parts))


def next_fixed(current: str, fixed: Iterable[str]) -> Optional[str]:
    """Smallest fixed version strictly greater than current."""
    cur = version_key(current)
    if cur is None:
        return None
    greater = [(key, v) for v in fixed if (key := version_key(v)) is not None and key > cur]
    return min(greater)[1] if greater else None


# ------------------------------------------------------------- advisory data
def _same_package(ecosystem: str, a: str, b: str) -> bool:
    if ecosystem == ECO_PYPI:
        return normalize_pypi(a) == normalize_pypi(b)
    return a.lower() == b.lower()


def severity_of(record: Dict[str, Any]) -> str:
    """database_specific severity first, then the CVSS v3 vector, else medium."""
    db = (record.get("database_specific") or {}).get("severity")
    if isinstance(db, str) and db.upper() in _DB_SEVERITY:
        return _DB_SEVERITY[db.upper()]
    for sev in record.get("severity") or []:
        if isinstance(sev, dict) and str(sev.get("type", "")).startswith("CVSS_V3"):
            score = base_score(str(sev.get("score", "")))
            if score is not None:
                return severity_for_score(score)
    return "medium"


def _fixed_versions(record: Dict[str, Any], dep: Dependency) -> List[str]:
    fixed: List[str] = []
    for aff in record.get("affected") or []:
        pkg = aff.get("package") or {}
        if pkg.get("ecosystem") != dep.ecosystem or not _same_package(dep.ecosystem, str(pkg.get("name", "")), dep.name):
            continue
        for rng in aff.get("ranges") or []:
            if rng.get("type") not in ("ECOSYSTEM", "SEMVER"):
                continue  # GIT ranges are commit ids, not comparable versions
            for ev in rng.get("events") or []:
                if isinstance(ev, dict) and isinstance(ev.get("fixed"), str):
                    fixed.append(ev["fixed"])
    return fixed


def _cwes(record: Dict[str, Any]) -> List[str]:
    ids = (record.get("database_specific") or {}).get("cwe_ids") or []
    return [c for c in ids if isinstance(c, str)]


def _advisory(vuln_id: str, record: Dict[str, Any], dep: Dependency) -> Dict[str, Any]:
    aliases = [a for a in record.get("aliases") or [] if isinstance(a, str)]
    refs = [r["url"] for r in record.get("references") or [] if isinstance(r, dict) and isinstance(r.get("url"), str)]
    summary = record.get("summary") or str(record.get("details") or "")[:200]
    return {
        "id": vuln_id,
        "aliases": aliases,
        "summary": str(summary),
        "severity": severity_of(record),
        "cwe": _cwes(record),
        "fixed": _fixed_versions(record, dep),
        "references": refs,
    }


def _recommendation(dep: Dependency, advisories: List[Dict[str, Any]]) -> str:
    targets = [next_fixed(dep.version, a["fixed"]) for a in advisories]
    if all(targets):
        best = max(targets, key=lambda t: version_key(t) or ())
        return f"Upgrade to ≥ {best}"
    return "Upgrade to a patched release; see the advisories for details."


def _dep_finding(dep: Dependency, advisories: List[Dict[str, Any]]) -> Dict[str, Any]:
    severity = min((a["severity"] for a in advisories), key=SEVERITY_RANK.__getitem__)
    count = len(advisories)
    noun = "vulnerability" if count == 1 else "vulnerabilities"
    top = next(a for a in advisories if a["severity"] == severity)
    cwes = sorted({c for a in advisories for c in a["cwe"]})
    meta = Meta(
        "dependency", f"{dep.name} {dep.version} has {count} known {noun}", severity,
        cwes[0] if cwes else None,
        f"OSV.dev lists {count} advisories affecting this version. {top['summary']}".strip(),
        _recommendation(dep, advisories),
    )
    finding = make_finding("dep-vuln", meta, dep.manifest_path, dep.line, f"{dep.name}=={dep.version}")
    advisory_ids = list(dict.fromkeys(i for a in advisories for i in [a["id"], *a["aliases"]]))
    fixed = sorted({v for a in advisories for v in a["fixed"] if (version_key(v) or ()) > (version_key(dep.version) or ())},
                   key=lambda v: version_key(v) or ())
    refs = list(dict.fromkeys([f"https://osv.dev/vulnerability/{a['id']}" for a in advisories]
                              + [u for a in advisories for u in a["references"]]))[:10]
    digest = hashlib.sha1(f"dep-vuln|{dep.ecosystem}|{dep.name}|{dep.version}|{dep.manifest_path}".encode("utf-8"),
                          usedforsecurity=False).hexdigest()[:16]
    return {
        **finding, "id": digest, "confidence": "high",
        "package": dep.name, "version": dep.version, "ecosystem": dep.ecosystem,
        "advisory_ids": advisory_ids, "fixed_versions": fixed, "references": refs,
    }


# --------------------------------------------------------------- OSV client
def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise _AuditError(MSG_TIMEOUT)
    return left


def _request(http: httpx.Client, method: str, url: str, body: Optional[Dict[str, Any]], deadline: float) -> Dict[str, Any]:
    timeout = min(ENDPOINT_TIMEOUT, _remaining(deadline))
    try:
        resp = http.request(method, url, json=body, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise _AuditError(MSG_TIMEOUT) from exc
    except httpx.HTTPError as exc:
        raise _AuditError(MSG_UNREACHABLE) from exc
    if resp.status_code != 200:
        raise _AuditError(MSG_ERROR)
    try:
        data = resp.json()
    except ValueError as exc:
        raise _AuditError(MSG_ERROR) from exc
    if not isinstance(data, dict):
        raise _AuditError(MSG_ERROR)
    return data


def _query_batches(http: httpx.Client, deps: Sequence[Dependency], deadline: float) -> Dict[int, List[str]]:
    """Map dependency index -> advisory ids. Raises _AuditError on any failed batch."""
    hits: Dict[int, List[str]] = {}
    for start in range(0, len(deps), BATCH_SIZE):
        chunk = deps[start:start + BATCH_SIZE]
        body = {"queries": [{"package": {"name": d.name, "ecosystem": d.ecosystem}, "version": d.version} for d in chunk]}
        # ponytail: querybatch pages results past 1000 advisories per query; not followed.
        data = _request(http, "POST", f"{OSV_BASE}/querybatch", body, deadline)
        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(chunk):
            raise _AuditError(MSG_ERROR)
        for offset, res in enumerate(results):
            vulns = (res or {}).get("vulns") or []
            ids = [v["id"] for v in vulns if isinstance(v, dict) and isinstance(v.get("id"), str)]
            if ids:
                hits[start + offset] = ids
    return hits


def _fetch_details(http: httpx.Client, ids: Sequence[str], deadline: float) -> Dict[str, Dict[str, Any]]:
    """Fetch vulnerability records concurrently. A failed record is absent and degrades to id-only data."""

    def one(vuln_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        url = f"{OSV_BASE}/vulns/{quote(vuln_id, safe='')}"
        try:
            return vuln_id, _request(http, "GET", url, None, deadline)
        except _AuditError as exc:
            logger.warning("OSV record %s unavailable: %s", vuln_id, exc)
            return vuln_id, None

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        pairs = list(pool.map(one, ids))
    return {vid: rec for vid, rec in pairs if rec}


def _findings_for(http: httpx.Client, deps: Sequence[Dependency], deadline: float) -> List[Dict[str, Any]]:
    hits = _query_batches(http, deps, deadline)
    if not hits:
        return []
    ids = sorted({i for found in hits.values() for i in found})
    details = _fetch_details(http, ids[:DETAIL_LIMIT], deadline)
    findings: List[Dict[str, Any]] = []
    for index, found in sorted(hits.items()):
        dep = deps[index]
        live = [(i, details.get(i) or {}) for i in found]
        advisories = [_advisory(i, rec, dep) for i, rec in live if not rec.get("withdrawn")]
        if advisories:
            findings.append(_dep_finding(dep, advisories))
    return sorted(findings, key=lambda f: (SEVERITY_RANK[f["severity"]], f["file_path"], f["line"]))


# ------------------------------------------------------------------ entry point
def audit_dependencies(
    root: Path,
    files: Iterable[str],
    *,
    client: Optional[httpx.Client] = None,
    budget_seconds: float = BUDGET_SECONDS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (dependency findings, summary["dependencies"]). Never raises."""
    try:
        deps, manifests = discover_dependencies(Path(root).resolve(), files)
    except Exception:
        logger.exception("Dependency manifest discovery failed under %s", root)
        return [], empty_summary(MSG_FAILED)
    summary = empty_summary()
    summary.update(scanned=len(deps), ecosystems=sorted({d.ecosystem for d in deps}), manifests=manifests)
    if not deps:
        return [], summary
    owned = client is None
    http = client if client is not None else httpx.Client()
    deadline = time.monotonic() + budget_seconds
    try:
        findings = _findings_for(http, deps, deadline)
    except _AuditError as exc:
        logger.warning("Dependency audit stopped: %s", exc)
        return [], {**summary, "error": str(exc)}
    except Exception:
        logger.exception("Dependency audit failed")
        return [], {**summary, "error": MSG_FAILED}
    finally:
        if owned:
            http.close()
    summary["vulnerable"] = len(findings)
    return findings, summary
