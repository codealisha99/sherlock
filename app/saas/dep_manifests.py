"""Dependency manifest and lockfile parsers for the OSV audit.

Each parser returns (ecosystem, name, version, manifest_path, line) records for
exact versions only. Ranges and wildcards are skipped because they do not name
one version to check. Lockfiles are parsed before manifests so that a lock
version wins when the same package appears in both.
"""

from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from app.saas.vuln_scanner import safe_text_path

logger = logging.getLogger(__name__)

DEP_MAX_BYTES = 10 * 1024 * 1024  # lockfiles are large; the scanner's 512 KB cap is too small
ECO_PYPI, ECO_NPM, ECO_GO, ECO_CRATES = "PyPI", "npm", "Go", "crates.io"

_PEP508_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*===?\s*([A-Za-z0-9][A-Za-z0-9.+!_-]*)$")
_PY_EXACT = re.compile(r"\d[\w.+!-]*")
_NPM_EXACT = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]*)?")
_LOCKFILES = frozenset({"Pipfile.lock", "poetry.lock", "package-lock.json", "yarn.lock", "Cargo.lock"})


class Dependency(NamedTuple):
    ecosystem: str
    name: str
    version: str
    manifest_path: str
    line: int


def normalize_pypi(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _line_index(lines: List[str], pattern: str) -> Dict[str, int]:
    """First 1-based line number for each group(1) match of pattern."""
    rx = re.compile(pattern)
    found: Dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        m = rx.match(line)
        if m:
            found.setdefault(m.group(1), number)
    return found


def _pep508_exact(spec: str) -> Optional[Tuple[str, str]]:
    """(name, version) for 'name==1.2.3' style requirements; None otherwise."""
    spec = spec.split(";", 1)[0].split(" --hash", 1)[0].strip()
    m = _PEP508_NAME.match(spec)
    if m is None or not _PY_EXACT.fullmatch(m.group(2)):
        return None
    return normalize_pypi(m.group(1)), m.group(2)


# ------------------------------------------------------------- requirements
def parse_requirements(rel: str, text: str) -> List[Dependency]:
    out: List[Dependency] = []
    logical: List[Tuple[int, str]] = []
    pending, start = "", 1
    for number, raw in enumerate(text.splitlines(), start=1):
        if not pending:
            start = number
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical.append((start, (pending + line).split(" #", 1)[0].split("\t#", 1)[0].strip()))
        pending = ""
    for number, spec in logical:
        if not spec or spec.startswith("#") or spec.startswith("-"):
            continue  # blank, comment, option, -r/-c/-e include: nothing to check here
        parsed = _pep508_exact(spec)
        if parsed:
            out.append(Dependency(ECO_PYPI, parsed[0], parsed[1], rel, number))
    return out


# ---------------------------------------------------------------- pyproject
def _poetry_exact(spec: object) -> Optional[str]:
    """Poetry treats a bare version as exact; '^', '~', '*', ranges are skipped."""
    if isinstance(spec, dict):
        spec = spec.get("version")
    if not isinstance(spec, str):
        return None
    spec = spec.strip()
    if spec.startswith("=="):
        spec = spec[2:]
    return spec if _PY_EXACT.fullmatch(spec) else None


def parse_pyproject(rel: str, text: str) -> List[Dependency]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    lines = text.splitlines()
    out: List[Dependency] = []
    project = data.get("project") or {}
    pep621: List[str] = list(project.get("dependencies") or [])
    for group in (project.get("optional-dependencies") or {}).values():
        pep621.extend(g for g in group if isinstance(g, str))
    for group in (data.get("dependency-groups") or {}).values():
        pep621.extend(g for g in group if isinstance(g, str))
    for raw in pep621:
        parsed = _pep508_exact(raw)
        if parsed:
            line = next((i for i, l in enumerate(lines, start=1) if raw in l), 1)
            out.append(Dependency(ECO_PYPI, parsed[0], parsed[1], rel, line))
    poetry = (data.get("tool") or {}).get("poetry") or {}
    poetry_tables = [poetry.get("dependencies") or {}, poetry.get("dev-dependencies") or {}]
    poetry_tables += [g.get("dependencies") or {} for g in (poetry.get("group") or {}).values()]
    line_of = _line_index(lines, r'^\s*["\']?([A-Za-z0-9][A-Za-z0-9._-]*)["\']?\s*=')
    for table in poetry_tables:
        for name, spec in table.items():
            version = _poetry_exact(spec)
            if name.lower() == "python" or version is None:
                continue
            out.append(Dependency(ECO_PYPI, normalize_pypi(name), version, rel, line_of.get(name, 1)))
    return out


# -------------------------------------------------------------- lock files
def parse_pipfile_lock(rel: str, text: str) -> List[Dependency]:
    data = _load_json(text)
    if data is None:
        return []
    lines = text.splitlines()
    line_of = _line_index(lines, r'^\s*"([^"]+)"\s*:')
    out: List[Dependency] = []
    for section in ("default", "develop"):
        for name, info in (data.get(section) or {}).items():
            version = str((info or {}).get("version", "")).removeprefix("==")
            if _PY_EXACT.fullmatch(version):
                out.append(Dependency(ECO_PYPI, normalize_pypi(name), version, rel, line_of.get(name, 1)))
    return out


def parse_poetry_lock(rel: str, text: str) -> List[Dependency]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    line_of = _line_index(text.splitlines(), r'^name = "([^"]+)"')
    out: List[Dependency] = []
    for pkg in data.get("package") or []:
        name, version = pkg.get("name"), pkg.get("version")
        if isinstance(name, str) and isinstance(version, str):
            out.append(Dependency(ECO_PYPI, normalize_pypi(name), version, rel, line_of.get(name, 1)))
    return out


def parse_cargo_lock(rel: str, text: str) -> List[Dependency]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    line_of = _line_index(text.splitlines(), r'^name = "([^"]+)"')
    out: List[Dependency] = []
    for pkg in data.get("package") or []:
        # Workspace members have no 'source'; only registry crates have a version to audit.
        if "source" not in pkg:
            continue
        name, version = pkg.get("name"), pkg.get("version")
        if isinstance(name, str) and isinstance(version, str):
            out.append(Dependency(ECO_CRATES, name, version, rel, line_of.get(name, 1)))
    return out


def _load_json(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_package_lock(rel: str, text: str) -> List[Dependency]:
    """npm lockfile v2/v3 ('packages') or v1 ('dependencies', recursive)."""
    data = _load_json(text)
    if data is None:
        return []
    lines = text.splitlines()
    out: List[Dependency] = []
    packages = data.get("packages")
    if isinstance(packages, dict):
        line_of = _line_index(lines, r'^\s*"([^"]*)"\s*:')
        for key, info in packages.items():
            if "node_modules/" not in key or not isinstance(info, dict) or info.get("link"):
                continue
            name, version = key.rsplit("node_modules/", 1)[1], info.get("version")
            if isinstance(version, str):
                out.append(Dependency(ECO_NPM, name, version, rel, line_of.get(key, 1)))
        return out
    line_of = _line_index(lines, r'^\s*"([^"]+)"\s*:')
    _walk_v1(data.get("dependencies") or {}, rel, line_of, out)
    return out


def _walk_v1(deps: dict, rel: str, line_of: Dict[str, int], out: List[Dependency]) -> None:
    for name, info in deps.items():
        if not isinstance(info, dict):
            continue
        if isinstance(info.get("version"), str) and not info.get("version", "").startswith(("file:", "link:")):
            out.append(Dependency(ECO_NPM, name, info["version"], rel, line_of.get(name, 1)))
        if isinstance(info.get("dependencies"), dict):
            _walk_v1(info["dependencies"], rel, line_of, out)


_YARN_VERSION = re.compile(r'^\s+version:?\s+"?([^"\s]+)"?\s*$')


def parse_yarn_lock(rel: str, text: str) -> List[Dependency]:
    """yarn v1 and berry lockfiles. Headers look like: "@scope/pkg@^1.0.0", "@scope/pkg@^1.1.0":"""
    lines = text.splitlines()
    out: List[Dependency] = []
    name: Optional[str] = None
    header_line = 1
    for number, line in enumerate(lines, start=1):
        if line and not line[0].isspace() and line.rstrip().endswith(":") and not line.startswith("#"):
            first = line.split(",", 1)[0].strip().rstrip(":").strip('"').strip()
            at = first.find("@", 1)
            name = first[:at] if at > 0 else None
            header_line = number
            continue
        m = _YARN_VERSION.match(line)
        if m and name and name != "__metadata":
            out.append(Dependency(ECO_NPM, name, m.group(1), rel, header_line))
            name = None
    return out


def parse_package_json(rel: str, text: str) -> List[Dependency]:
    """Direct dependencies pinned with an exact version. Ranges are covered by the lockfile."""
    data = _load_json(text)
    if data is None:
        return []
    lines = text.splitlines()
    line_of = _line_index(lines, r'^\s*"([^"]+)"\s*:')
    out: List[Dependency] = []
    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        for name, spec in (data.get(section) or {}).items():
            if isinstance(spec, str) and _NPM_EXACT.fullmatch(spec.strip()):
                out.append(Dependency(ECO_NPM, name, spec.strip(), rel, line_of.get(name, 1)))
    return out


def parse_go_mod(rel: str, text: str) -> List[Dependency]:
    out: List[Dependency] = []
    in_block = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("//", 1)[0].strip()
        if line.startswith("require (") or line == "require (":
            in_block = True
            continue
        if in_block and line == ")":
            in_block = False
            continue
        body = line if in_block else (line[len("require "):] if line.startswith("require ") else None)
        if body is None:
            continue
        parts = body.split()
        if len(parts) >= 2 and parts[1].startswith("v"):
            out.append(Dependency(ECO_GO, parts[0], parts[1], rel, number))
    return out


# ----------------------------------------------------------------- discovery
Parser = Callable[[str, str], List[Dependency]]
_PARSERS: Dict[str, Parser] = {
    "Pipfile.lock": parse_pipfile_lock,
    "poetry.lock": parse_poetry_lock,
    "package-lock.json": parse_package_lock,
    "yarn.lock": parse_yarn_lock,
    "Cargo.lock": parse_cargo_lock,
    "pyproject.toml": parse_pyproject,
    "package.json": parse_package_json,
    "go.mod": parse_go_mod,
}


def _parser_for(name: str) -> Optional[Parser]:
    if name.startswith("requirements") and name.endswith(".txt"):
        return parse_requirements
    return _PARSERS.get(name)


def _read_manifest(base: Path, rel: str) -> Optional[str]:
    path = safe_text_path(base, rel)
    if path is None:
        return None
    try:
        if path.stat().st_size > DEP_MAX_BYTES:
            logger.info("Skipping oversized dependency file %s", rel)
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def discover_dependencies(base: Path, files: Iterable[str]) -> Tuple[List[Dependency], List[str]]:
    """Unique dependencies across all supported manifests, and the manifests that produced them.

    Dedupe key is (ecosystem, name, version); the first occurrence wins, lockfiles first.
    """
    rels = [r for r in files if _parser_for(Path(r).name) is not None]
    rels.sort(key=lambda r: (Path(r).name not in _LOCKFILES, r))
    seen: Dict[Tuple[str, str, str], Dependency] = {}
    used: List[str] = []
    for rel in rels:
        text = _read_manifest(base, rel)
        parser = _parser_for(Path(rel).name)
        if text is None or parser is None:
            continue
        found = parser(rel, text)
        if found:
            used.append(rel)
        for dep in found:
            seen.setdefault((dep.ecosystem, dep.name, dep.version), dep)
    return list(seen.values()), sorted(used)
