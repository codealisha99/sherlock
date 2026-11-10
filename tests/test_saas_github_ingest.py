"""URL validation and hardened clone for the SaaS ingest path. No network: git is faked."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from app.saas import github_ingest as gi
from app.saas.github_ingest import CloneError, RepoTooLargeError, clone_repo, parse_github_url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/octocat/hello-world", ("octocat", "hello-world", None)),
        ("https://github.com/octocat/hello-world.git", ("octocat", "hello-world", None)),
        ("https://github.com/octocat/hello-world/", ("octocat", "hello-world", None)),
        ("  https://GitHub.com/a/b  ", ("a", "b", None)),
        ("https://github.com/o/r/tree/main", ("o", "r", "main")),
        ("https://github.com/o/r/tree/feature/x-1.2", ("o", "r", "feature/x-1.2")),
    ],
)
def test_valid_urls(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
        42,
        "http://github.com/a/b",
        "ftp://github.com/a/b",
        "javascript:alert(1)",
        "https://evil.com/a/b",
        "https://github.com.evil.com/a/b",
        "https://user:pass@github.com/a/b",
        "https://github.com@evil.com/a/b",
        "https://github.com:8443/a/b",
        "https://github.com/a/b?x=1",
        "https://github.com/a/b#frag",
        "https://github.com/a",
        "https://github.com/",
        "https://github.com/../b",
        "https://github.com/a/..",
        "https://github.com/a/.",
        "https://github.com/a/b/../../c",
        "https://github.com/a//b",
        "https://github.com/a%2Fb/c",
        "https://github.com/a/b;id",
        "https://github.com/a/b$(id)",
        "https://github.com/a/`id`",
        "https://github.com/a/b c",
        "https://github.com/a/b\nhttps://evil.com",
        "https://github.com/a/b\x00",
        "https://github.com/a/b/blob/main/x.py",
        "https://github.com/a/b/tree/../x",
        "https://github.com/a/b/tree/main/../../x",
        "https://github.com/a/b/tree/-evil",
        "https://github.com/" + "a" * 40 + "/b",
        "https://github.com/a/" + "b" * 101,
        "https://github.com/a/" + "b" * 300,
    ],
)
def test_attack_urls_rejected(url):
    with pytest.raises(ValueError) as exc:
        parse_github_url(url)  # type: ignore[arg-type]
    assert "GitHub" in str(exc.value)


def _fake_git(files=None, returncode=0, stderr="", sleep_timeout=False):
    """Runner that pretends to clone: writes files into the destination (last argv)."""
    calls = []

    def runner(cmd, env, timeout):
        calls.append((list(cmd), dict(env), timeout))
        if sleep_timeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        for rel, content in (files or {}).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                p.write_text(content)
            else:
                p.write_bytes(content)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_clone_command_is_hardened_argv(tmp_path):
    runner = _fake_git({"a.py": "x = 1\n"})
    dest = clone_repo("octo", "demo", "dev", tmp_path, 10, runner=runner)
    cmd, env, timeout = runner.calls[0]
    assert cmd[0] == "git"
    assert "shell" not in " ".join(cmd)
    for flag in ("core.hooksPath=/dev/null", "protocol.file.allow=never", "core.symlinks=false"):
        assert flag in cmd
    for arg in ("clone", "--depth", "1", "--no-tags", "--single-branch", "--branch", "dev", "--filter=blob:limit=2m"):
        assert arg in cmd
    assert cmd[-2] == "https://github.com/octo/demo.git"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert timeout == 120
    assert dest.parent == tmp_path and dest.is_dir()
    assert (dest / "a.py").read_text() == "x = 1\n"


def test_each_clone_gets_unique_directory(tmp_path):
    runner = _fake_git({"a.py": "x"})
    first = clone_repo("o", "r", None, tmp_path, 10, runner=runner)
    second = clone_repo("o", "r", None, tmp_path, 10, runner=runner)
    assert first != second


def test_size_cap_deletes_clone_and_raises(tmp_path):
    runner = _fake_git({"big.bin": b"\0" * (2 * 1024 * 1024)})
    with pytest.raises(RepoTooLargeError):
        clone_repo("o", "r", None, tmp_path, 1, runner=runner)
    assert list(tmp_path.iterdir()) == []


def test_size_cap_counts_under_limit_ok(tmp_path):
    runner = _fake_git({"small.txt": "x" * 1000})
    dest = clone_repo("o", "r", None, tmp_path, 1, runner=runner)
    assert dest.is_dir()


def test_git_metadata_removed_and_escaping_symlinks_dropped(tmp_path):
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("top secret")

    def runner(cmd, env, timeout):
        dest = Path(cmd[-1])
        (dest / ".git" / "hooks").mkdir(parents=True)
        (dest / ".git" / "config").write_text("[core]")
        (dest / "ok.py").write_text("x = 1")
        (dest / "leak.txt").symlink_to(outside)
        (dest / "inside.py").symlink_to(dest / "ok.py")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    root = tmp_path / "root"
    dest = clone_repo("o", "r", None, root, 10, runner=runner)
    assert not (dest / ".git").exists()
    assert not os.path.lexists(dest / "leak.txt")
    assert os.path.lexists(dest / "inside.py")
    assert outside.read_text() == "top secret"


def test_nonzero_exit_gives_friendly_error_without_stderr(tmp_path):
    runner = _fake_git(returncode=128, stderr="fatal: could not read /home/secret/path")
    with pytest.raises(CloneError) as exc:
        clone_repo("o", "r", None, tmp_path, 10, runner=runner)
    assert "/home" not in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_clone_error(tmp_path):
    runner = _fake_git(sleep_timeout=True)
    with pytest.raises(CloneError, match="timed out"):
        clone_repo("o", "r", None, tmp_path, 10, runner=runner, timeout=1)


def test_invalid_owner_never_reaches_git(tmp_path):
    runner = _fake_git({})
    with pytest.raises(ValueError):
        clone_repo("../etc", "r", None, tmp_path, 10, runner=runner)
    assert runner.calls == []  # type: ignore[attr-defined]


def test_repos_root_honours_env(tmp_path, monkeypatch):
    target = tmp_path / "custom-repos"
    monkeypatch.setenv("SHERLOCK_REPOS_DIR", str(target))
    assert gi.repos_root() == target
    assert target.is_dir()
