"""GitHub URL validation and hardened shallow clone for the SaaS ingest path.

Only ``https://github.com/<owner>/<repo>`` (optionally ``/tree/<branch>``) is
accepted. Git runs via argv lists (never a shell) with hooks disabled, no
credential prompts, and a size cap enforced on the checked-out tree.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from config import config

logger = logging.getLogger(__name__)

OWNER_RE = re.compile(r"[A-Za-z0-9-]{1,39}")
REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
MAX_URL_LENGTH = 300
CLONE_TIMEOUT_SECONDS = 120
CLONE_FILTER = "--filter=blob:limit=2m"
GIT_HARDENING = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "protocol.file.allow=never",
    "-c", "core.symlinks=false",
)
URL_HELP = "Enter a public GitHub repository URL like https://github.com/owner/repo"
MB = 1024 * 1024

Runner = Callable[[Sequence[str], Dict[str, str], int], "subprocess.CompletedProcess[str]"]


class CloneError(RuntimeError):
    """Clone failed. The message is user-facing and safe to show."""


class RepoTooLargeError(CloneError):
    """Checked-out tree exceeds the plan's size cap."""


def parse_github_url(url: str) -> Tuple[str, str, Optional[str]]:
    """Return (owner, repo, branch) for a GitHub URL, or raise ValueError."""
    if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_LENGTH:
        raise ValueError(URL_HELP)
    raw = url.strip()
    if any(ord(ch) <= 32 or ord(ch) == 127 for ch in raw):
        raise ValueError(URL_HELP)
    try:
        parts = urlsplit(raw)
    except ValueError:
        raise ValueError(URL_HELP)
    # netloc must be exactly github.com: rejects userinfo, ports, other hosts.
    if parts.scheme != "https" or parts.netloc.lower() != "github.com":
        raise ValueError(URL_HELP)
    if parts.query or parts.fragment:
        raise ValueError(URL_HELP)

    segments = parts.path.strip("/").split("/") if parts.path.strip("/") else []
    if len(segments) < 2 or any(s in ("", ".", "..") for s in segments):
        raise ValueError(URL_HELP)
    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    branch = _parse_tree_branch(segments[2:])
    if not OWNER_RE.fullmatch(owner) or not REPO_RE.fullmatch(repo) or repo in (".", ".."):
        raise ValueError(URL_HELP)
    return owner, repo, branch


def _parse_tree_branch(rest: Sequence[str]) -> Optional[str]:
    if not rest:
        return None
    if rest[0] != "tree" or len(rest) < 2:
        raise ValueError(URL_HELP)
    branch = "/".join(rest[1:])
    if not BRANCH_RE.fullmatch(branch) or ".." in branch:
        raise ValueError(URL_HELP)
    return branch


def repos_root() -> Path:
    """Workspace directory for clones: SHERLOCK_REPOS_DIR or workspace/repos."""
    env = os.getenv("SHERLOCK_REPOS_DIR")
    root = Path(env) if env else Path(config.workspace_dir) / "repos"
    root.mkdir(parents=True, exist_ok=True)
    return root


def clone_repo(
    owner: str,
    repo: str,
    branch: Optional[str],
    dest_root: Path,
    max_mb: int,
    timeout: int = CLONE_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> Path:
    """Shallow-clone github.com/owner/repo into a fresh dir under dest_root.

    Returns the clone path. On any failure the partial clone is removed and
    CloneError (or RepoTooLargeError) is raised.
    """
    if not OWNER_RE.fullmatch(owner) or not REPO_RE.fullmatch(repo) or repo in (".", ".."):
        raise ValueError(URL_HELP)
    dest = Path(dest_root) / f"{owner}__{repo}__{uuid.uuid4().hex}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = _clone_command(owner, repo, branch, dest)
    run = runner or _run_git
    try:
        proc = run(cmd, _git_env(), timeout)
    except subprocess.TimeoutExpired:
        _discard(dest)
        raise CloneError("Cloning timed out. The repository may be too large or unavailable.")
    if proc.returncode != 0:
        logger.warning("git clone failed for %s/%s: %s", owner, repo, (proc.stderr or "")[-500:])
        _discard(dest)
        raise CloneError("Could not clone that repository. Check that it exists and is public.")
    try:
        _enforce_size(dest, max_mb)
        shutil.rmtree(dest / ".git", ignore_errors=True)
        _drop_escaping_symlinks(dest)
    except CloneError:
        _discard(dest)
        raise
    return dest


def _clone_command(owner: str, repo: str, branch: Optional[str], dest: Path) -> list[str]:
    args = ["git", *GIT_HARDENING, "clone", "--depth", "1", "--no-tags", "--single-branch"]
    if branch:
        args += ["--branch", branch]
    args += [CLONE_FILTER, f"https://github.com/{owner}/{repo}.git", str(dest)]
    return args


def _git_env() -> Dict[str, str]:
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _run_git(cmd: Sequence[str], env: Dict[str, str], timeout: int) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(cmd), env=env, capture_output=True, text=True, timeout=timeout, check=False, shell=False
    )


def _enforce_size(root: Path, max_mb: int) -> None:
    """Sum apparent sizes without following symlinks (lstat)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    if total > max_mb * MB:
        raise RepoTooLargeError(f"Repository is larger than the {max_mb} MB limit for your plan.")


def _drop_escaping_symlinks(root: Path) -> None:
    """Unlink symlinks whose real target lies outside the clone."""
    real_root = Path(os.path.realpath(root))
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            link = os.path.join(dirpath, name)
            if not os.path.islink(link):
                continue
            target = Path(os.path.realpath(link))
            if not target.is_relative_to(real_root):
                logger.info("Removing symlink that escapes the repository: %s", name)
                os.unlink(link)


def _discard(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
