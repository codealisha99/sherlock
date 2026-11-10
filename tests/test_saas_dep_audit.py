"""Dependency audit: CVSS math, manifest/lockfile parsing, OSV batching, severity, and failure handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pytest

from app.saas import dep_audit as da
from app.saas.cvss import base_score, severity_for_score
from app.saas.dep_manifests import (
    discover_dependencies,
    parse_cargo_lock,
    parse_go_mod,
    parse_package_json,
    parse_package_lock,
    parse_pipfile_lock,
    parse_poetry_lock,
    parse_pyproject,
    parse_requirements,
    parse_yarn_lock,
)

# ----------------------------------------------------------------------- CVSS


@pytest.mark.parametrize("vector, expected", [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:H/I:H/A:H", 9.6),
    ("CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N", 5.5),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),
])
def test_cvss_known_vectors(vector, expected):
    assert base_score(vector) == expected


def test_cvss_malformed_vector_is_none():
    assert base_score("not-a-vector") is None
    assert base_score("CVSS:3.1/AV:N/AC:L") is None
    assert base_score("CVSS:2.0/AV:N/AC:L/Au:N/C:P/I:P/A:P") is None


@pytest.mark.parametrize("score, sev", [(9.0, "critical"), (8.9, "high"), (7.0, "high"),
                                        (6.9, "medium"), (4.0, "medium"), (3.9, "low"), (0.0, "low")])
def test_severity_bands(score, sev):
    assert severity_for_score(score) == sev


# ------------------------------------------------------------ requirements.txt


def _pins(rel: str, text: str) -> List[Tuple[str, str, int]]:
    return [(d.name, d.version, d.line) for d in parse_requirements(rel, text)]


def test_requirements_exact_pins_only_and_comments_ignored():
    text = (
        "# top comment\n"
        "requests==2.19.0  # pinned for the legacy client\n"
        "-r base.txt\n"
        "-c constraints.txt\n"
        "-e git+https://example.com/x.git#egg=x\n"
        "flask>=2.0\n"
        "django~=4.2\n"
        "numpy==1.*\n"
        "\n"
        "pillow==9.0.0 ; python_version >= '3.8'\n"
    )
    assert _pins("requirements.txt", text) == [("requests", "2.19.0", 2), ("pillow", "9.0.0", 10)]


def test_requirements_extras_hashes_and_continuations():
    text = (
        "Flask_SQLAlchemy[mysql]==2.5.1 \\\n"
        "    --hash=sha256:abc\n"
        "urllib3==1.26.5 --hash=sha256:def\n"
    )
    assert _pins("requirements-dev.txt", text) == [
        ("flask-sqlalchemy", "2.5.1", 1),
        ("urllib3", "1.26.5", 3),
    ]


# -------------------------------------------------------------- pyproject.toml


def test_pyproject_pep621_and_poetry_exact_versions_only():
    text = (
        '[project]\n'
        'name = "demo"\n'
        'dependencies = [\n'
        '  "requests==2.19.0",\n'
        '  "httpx>=0.27",\n'
        ']\n'
        '[project.optional-dependencies]\n'
        'dev = ["pytest==7.0.0"]\n'
        '[tool.poetry.dependencies]\n'
        'python = "^3.11"\n'
        'Jinja2 = "2.10"\n'
        'click = "^8.0"\n'
        'rich = { version = "13.7.0", extras = ["jupyter"] }\n'
    )
    got = {(d.name, d.version) for d in parse_pyproject("pyproject.toml", text)}
    assert got == {("requests", "2.19.0"), ("pytest", "7.0.0"), ("jinja2", "2.10"), ("rich", "13.7.0")}


# ------------------------------------------------------------------ lockfiles


def test_pipfile_lock_default_and_develop():
    text = json.dumps({
        "_meta": {},
        "default": {"Flask": {"version": "==0.12"}, "Click": {"version": "*"}},
        "develop": {"pytest": {"version": "==7.0.0"}},
    }, indent=2)
    got = {(d.name, d.version) for d in parse_pipfile_lock("Pipfile.lock", text)}
    assert got == {("flask", "0.12"), ("pytest", "7.0.0")}


def test_poetry_lock_packages():
    text = (
        '[[package]]\nname = "requests"\nversion = "2.19.0"\n\n'
        '[[package]]\nname = "Jinja_2"\nversion = "2.10"\n'
    )
    got = [(d.name, d.version, d.line) for d in parse_poetry_lock("poetry.lock", text)]
    assert got == [("requests", "2.19.0", 2), ("jinja-2", "2.10", 6)]


def test_package_lock_v3_scoped_nested_and_links():
    data = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app", "version": "1.0.0"},
            "node_modules/lodash": {"version": "4.17.20"},
            "node_modules/@babel/core": {"version": "7.12.0"},
            "node_modules/a/node_modules/semver": {"version": "5.7.1"},
            "node_modules/local": {"link": True, "resolved": "packages/local"},
        },
    }
    got = {(d.name, d.version) for d in parse_package_lock("package-lock.json", json.dumps(data))}
    assert got == {("lodash", "4.17.20"), ("@babel/core", "7.12.0"), ("semver", "5.7.1")}


def test_package_lock_v1_recursive_dependencies():
    data = {
        "lockfileVersion": 1,
        "dependencies": {
            "express": {"version": "4.17.1", "dependencies": {"qs": {"version": "6.7.0"}}},
            "@scope/pkg": {"version": "1.2.3"},
        },
    }
    got = {(d.name, d.version) for d in parse_package_lock("package-lock.json", json.dumps(data))}
    assert got == {("express", "4.17.1"), ("qs", "6.7.0"), ("@scope/pkg", "1.2.3")}


def test_yarn_v1_multi_spec_headers_and_scoped():
    text = (
        '# yarn lockfile v1\n\n\n'
        'lodash@^4.17.21:\n'
        '  version "4.17.21"\n'
        '  resolved "https://registry.yarnpkg.com/lodash/-/lodash-4.17.21.tgz"\n\n'
        '"@babel/core@^7.0.0", "@babel/core@^7.1.0":\n'
        '  version "7.12.0"\n'
    )
    got = [(d.name, d.version, d.line) for d in parse_yarn_lock("yarn.lock", text)]
    assert got == [("lodash", "4.17.21", 4), ("@babel/core", "7.12.0", 8)]


def test_yarn_berry_format():
    text = (
        '__metadata:\n  version: 6\n\n'
        '"minimist@npm:^1.2.5":\n  version: 1.2.6\n  resolution: "minimist@npm:1.2.6"\n'
    )
    assert [(d.name, d.version) for d in parse_yarn_lock("yarn.lock", text)] == [("minimist", "1.2.6")]


def test_cargo_lock_skips_workspace_members():
    text = (
        '[[package]]\nname = "demo"\nversion = "0.1.0"\n\n'
        '[[package]]\nname = "serde"\nversion = "1.0.100"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
    )
    assert [(d.name, d.version) for d in parse_cargo_lock("Cargo.lock", text)] == [("serde", "1.0.100")]


def test_go_mod_block_and_single_require():
    text = (
        "module example.com/app\n\n"
        "require github.com/gin-gonic/gin v1.6.0\n\n"
        "require (\n"
        "\tgolang.org/x/text v0.3.0 // indirect\n"
        "\tgithub.com/x/y v1.2.3\n"
        ")\n"
    )
    got = [(d.name, d.version, d.line) for d in parse_go_mod("go.mod", text)]
    assert got == [("github.com/gin-gonic/gin", "v1.6.0", 3), ("golang.org/x/text", "v0.3.0", 6),
                   ("github.com/x/y", "v1.2.3", 7)]


# -------------------------------------------------------------- package.json


def test_package_json_exact_pins_only():
    text = (
        '{\n  "dependencies": {\n    "left-pad": "1.3.0",\n    "lodash": "^4.17.21"\n  },\n'
        '  "devDependencies": {"x": "*", "y": "npm:other@2.0.0"}\n}\n'
    )
    got = [(d.name, d.version, d.line) for d in parse_package_json("package.json", text)]
    assert got == [("left-pad", "1.3.0", 3)]


# ---------------------------------------------------------------- discovery


def test_discovery_prefers_lockfile_and_dedupes(tmp_path):
    (tmp_path / "package-lock.json").write_text(json.dumps({"packages": {
        "node_modules/lodash": {"version": "4.17.20"}}}))
    (tmp_path / "package.json").write_text('{"dependencies": {"lodash": "4.17.20", "left-pad": "1.3.0"}}')
    (tmp_path / "requirements.txt").write_text("requests==2.19.0\nrequests==2.19.0\n")
    deps, manifests = discover_dependencies(tmp_path, ["package.json", "package-lock.json", "requirements.txt"])
    keys = sorted((d.ecosystem, d.name, d.version, d.manifest_path) for d in deps)
    assert keys == [
        ("PyPI", "requests", "2.19.0", "requirements.txt"),
        ("npm", "left-pad", "1.3.0", "package.json"),
        ("npm", "lodash", "4.17.20", "package-lock.json"),
    ]
    assert manifests == ["package-lock.json", "package.json", "requirements.txt"]


def test_discovery_ignores_symlinks_and_missing_files(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("evil==1.0.0\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").symlink_to(outside)
    deps, manifests = discover_dependencies(repo, ["requirements.txt", "missing/requirements.txt"])
    assert deps == [] and manifests == []


# --------------------------------------------------------------- OSV client

REQUESTS_ID, REQUESTS_CVSS_ID, FLASK_ID = "GHSA-aaaa-1111-aaaa", "GHSA-bbbb-2222-bbbb", "GHSA-cccc-3333-cccc"


def _records() -> Dict[str, Any]:
    return {
        REQUESTS_ID: {
            "id": REQUESTS_ID, "aliases": ["CVE-2018-18074"], "summary": "Credentials leak on redirect",
            "database_specific": {"severity": "HIGH", "cwe_ids": ["CWE-522"]},
            "affected": [{"package": {"name": "requests", "ecosystem": "PyPI"},
                          "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.20.0"}]}]}],
            "references": [{"type": "ADVISORY", "url": "https://example.test/advisory"}],
        },
        REQUESTS_CVSS_ID: {
            "id": REQUESTS_CVSS_ID, "aliases": [], "summary": "Remote code path",
            "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
            "affected": [{"package": {"name": "requests", "ecosystem": "PyPI"},
                          "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "2.19.1"}]}]}],
        },
        FLASK_ID: {
            "id": FLASK_ID, "aliases": ["CVE-2018-1000656"], "summary": "Denial of service",
            "database_specific": {"severity": "MODERATE"},
            "affected": [{"package": {"name": "Flask", "ecosystem": "PyPI"},
                          "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "0.12.3"}]}]}],
        },
    }


def _handler(db: Dict[Tuple[str, str], List[str]], records: Dict[str, Any], calls: List[str]):
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v1/querybatch":
            body = json.loads(request.content)
            results = [{"vulns": [{"id": i} for i in db.get((q["package"]["name"], q["version"]), [])]}
                       for q in body["queries"]]
            return httpx.Response(200, json={"results": results})
        vid = request.url.path.rsplit("/", 1)[-1]
        if request.url.path.startswith("/v1/vulns/") and vid in records:
            return httpx.Response(200, json=records[vid])
        return httpx.Response(404, json={"code": 5})

    return handle


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _repo(tmp_path: Path, files: Dict[str, str]) -> Tuple[Path, List[str]]:
    for rel, text in files.items():
        (tmp_path / rel).write_text(text)
    return tmp_path, list(files)


def test_dependency_finding_aggregates_advisories(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "requests==2.19.0\n"})
    db = {("requests", "2.19.0"): [REQUESTS_ID, REQUESTS_CVSS_ID]}
    calls: List[str] = []
    findings, summary = da.audit_dependencies(root, files, client=_client(_handler(db, _records(), calls)))

    assert summary == {"scanned": 1, "vulnerable": 1, "ecosystems": ["PyPI"],
                       "manifests": ["requirements.txt"], "error": None}
    assert calls[0] == "POST /v1/querybatch"
    assert len(findings) == 1
    f = findings[0]
    assert f["rule_id"] == "dep-vuln" and f["category"] == "dependency"
    assert f["title"] == "requests 2.19.0 has 2 known vulnerabilities"
    assert f["severity"] == "critical"  # CVSS 9.8 beats the HIGH advisory
    assert f["file_path"] == "requirements.txt" and f["line"] == 1 and f["node_id"] == "file:requirements.txt"
    assert f["cwe"] == "CWE-522"
    assert f["recommendation"] == "Upgrade to ≥ 2.20.0"
    assert f["fixed_versions"] == ["2.19.1", "2.20.0"]
    assert f["advisory_ids"] == [REQUESTS_ID, "CVE-2018-18074", REQUESTS_CVSS_ID]
    assert f["references"][0] == f"https://osv.dev/vulnerability/{REQUESTS_ID}"
    assert f["confidence"] == "high" and f["package"] == "requests" and f["ecosystem"] == "PyPI"
    assert "requests==2.19.0" in f["snippet"]


def test_moderate_maps_to_medium_and_singular_title(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "flask==0.12\n"})
    db = {("flask", "0.12"): [FLASK_ID]}
    findings, _ = da.audit_dependencies(root, files, client=_client(_handler(db, _records(), [])))
    assert findings[0]["severity"] == "medium"
    assert findings[0]["title"] == "flask 0.12 has 1 known vulnerability"
    assert findings[0]["recommendation"] == "Upgrade to ≥ 0.12.3"


def test_clean_dependencies_produce_no_findings(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "requests==2.32.0\n"})
    findings, summary = da.audit_dependencies(root, files, client=_client(_handler({}, {}, [])))
    assert findings == [] and summary["scanned"] == 1 and summary["vulnerable"] == 0 and summary["error"] is None


def test_querybatch_chunks_at_1000_and_details_are_cached(tmp_path):
    lines = "".join(f"pkg{i}==1.0.{i}\n" for i in range(1001))
    root, files = _repo(tmp_path, {"requirements.txt": lines})
    # every package shares one advisory id, so details must be fetched once
    db = {(f"pkg{i}", f"1.0.{i}"): [FLASK_ID] for i in range(1001)}
    calls: List[str] = []
    findings, summary = da.audit_dependencies(root, files, client=_client(_handler(db, _records(), calls)))
    assert summary["scanned"] == 1001 and len(findings) == 1001
    assert calls.count("POST /v1/querybatch") == 2
    assert calls.count(f"GET /v1/vulns/{FLASK_ID}") == 1


def test_network_failure_never_raises(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "requests==2.19.0\n"})

    def down(request):
        raise httpx.ConnectError("no route", request=request)

    findings, summary = da.audit_dependencies(root, files, client=_client(down))
    assert findings == []
    assert summary["error"] == "Dependency vulnerability database unreachable"
    assert summary["scanned"] == 1


def test_timeout_and_http_error_are_reported(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "requests==2.19.0\n"})

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    _, summary = da.audit_dependencies(root, files, client=_client(slow))
    assert summary["error"] == "Dependency vulnerability database timed out"

    _, summary = da.audit_dependencies(root, files, client=_client(lambda r: httpx.Response(503)))
    assert summary["error"] == "Dependency vulnerability database returned an error"


def test_budget_exhaustion_reports_timeout_without_network(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "requests==2.19.0\n"})
    calls: List[str] = []
    _, summary = da.audit_dependencies(root, files, client=_client(_handler({}, {}, calls)), budget_seconds=0)
    assert calls == []
    assert summary["error"] == "Dependency vulnerability database timed out"


def test_withdrawn_advisory_is_ignored(tmp_path):
    root, files = _repo(tmp_path, {"requirements.txt": "flask==0.12\n"})
    records = _records()
    records[FLASK_ID]["withdrawn"] = "2024-01-01T00:00:00Z"
    db = {("flask", "0.12"): [FLASK_ID]}
    findings, summary = da.audit_dependencies(root, files, client=_client(_handler(db, records, [])))
    assert findings == [] and summary["vulnerable"] == 0


def test_no_manifests_skips_network(tmp_path):
    (tmp_path / "README.md").write_text("hi")

    def forbidden(request):
        raise AssertionError("network must not be used without dependencies")

    findings, summary = da.audit_dependencies(tmp_path, ["README.md"], client=_client(forbidden))
    assert findings == [] and summary == da.empty_summary()


def test_severity_prefers_database_specific_then_cvss():
    assert da.severity_of({"database_specific": {"severity": "LOW"}}) == "low"
    assert da.severity_of({"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]}) == "critical"
    assert da.severity_of({"severity": [{"type": "CVSS_V4", "score": "x"}]}) == "medium"
    assert da.severity_of({}) == "medium"


def test_next_fixed_picks_smallest_greater_version():
    assert da.next_fixed("2.19.0", ["2.20.0", "2.19.1", "1.0.0"]) == "2.19.1"
    assert da.next_fixed("2.19.0", ["2.19.0"]) is None
    assert da.next_fixed("v1.2.3", ["v1.2.4"]) == "v1.2.4"
