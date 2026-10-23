"""CLI tests for embed and semantic-search commands.

Uses isolated temporary repositories and databases. Never touches
index/sherlock.db.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")

from app.indexer import RepositoryIndexer  # noqa: E402
from app.main import cmd_embed, cmd_hybrid_search, cmd_semantic_search  # noqa: E402
from app.main import main as cli_main  # noqa: E402
from config import config  # noqa: E402


class _Args:
    """Simple namespace for CLI arguments."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_repo(files):
    """Create a temp repo from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix="sherlock_cli_test_"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


class TestCliEmbedCommands(unittest.TestCase):
    """Tests for the embed and semantic-search CLI commands."""

    def setUp(self):
        # Redirect the global config's index_db_path to a temporary database
        # so CLI commands (which use _get_indexer() -> config.index_db_path)
        # never write to index/sherlock.db.
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_cli_db_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self._orig_index_db_path = config.index_db_path
        config.index_db_path = self.db_path

        self.indexer = RepositoryIndexer()
        self.root = _make_repo(
            {
                "src/calculator.py": "def add(a, b):\n    return a + b\n",
                "src/greeting.py": "def greet(name):\n    return f\"Hi, {name}\"\n",
            }
        )
        self.res = self.indexer.index_repository(self.root)
        self.repo_id = self.res.repo_id

    def tearDown(self):
        try:
            self.indexer.remove_repository(self.repo_id)
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        # Restore the original config and remove the temp database.
        config.index_db_path = self._orig_index_db_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_embed_command_success(self):
        """The embed command reports embedded chunks."""
        args = _Args(repo_id=self.repo_id)
        rc = cmd_embed(args)
        self.assertEqual(rc, 0)

    def test_embed_command_nonexistent_repo(self):
        """The embed command returns error for nonexistent repo."""
        args = _Args(repo_id="nonexistent-repo-id")
        rc = cmd_embed(args)
        self.assertEqual(rc, 1)

    def test_semantic_search_command_before_embedding(self):
        """semantic-search reports no embeddings before embed is run."""
        args = _Args(repo_id=self.repo_id, query="add function", limit=5)
        rc = cmd_semantic_search(args)
        # Returns 1 with error message because no embeddings exist
        self.assertEqual(rc, 1)

    def test_semantic_search_command_after_embedding(self):
        """semantic-search returns results after embed is run."""
        # First embed
        embed_args = _Args(repo_id=self.repo_id)
        rc = cmd_embed(embed_args)
        self.assertEqual(rc, 0)

        # Then search
        args = _Args(repo_id=self.repo_id, query="add function", limit=5)
        rc = cmd_semantic_search(args)
        self.assertEqual(rc, 0)

    def test_semantic_search_command_empty_query(self):
        """semantic-search returns error for empty query."""
        self.indexer.embed_chunks(self.repo_id)
        args = _Args(repo_id=self.repo_id, query="", limit=5)
        rc = cmd_semantic_search(args)
        self.assertEqual(rc, 1)

    def test_semantic_search_command_nonexistent_repo(self):
        """semantic-search returns error for nonexistent repo."""
        args = _Args(repo_id="nonexistent-repo-id", query="test", limit=5)
        rc = cmd_semantic_search(args)
        self.assertEqual(rc, 1)


class TestCliHybridSearchCommands(unittest.TestCase):
    """Tests for the hybrid-search CLI command."""

    def setUp(self):
        # Redirect the global config's index_db_path to a temporary database
        # so CLI commands never write to index/sherlock.db.
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_cli_db_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self._orig_index_db_path = config.index_db_path
        config.index_db_path = self.db_path

        self.indexer = RepositoryIndexer()
        self.root = _make_repo(
            {
                "src/calculator.py": "def add(a, b):\n    return a + b\n",
                "src/greeting.py": "def greet(name):\n    return f\"Hi, {name}\"\n",
            }
        )
        self.res = self.indexer.index_repository(self.root)
        self.repo_id = self.res.repo_id

    def tearDown(self):
        try:
            self.indexer.remove_repository(self.repo_id)
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        config.index_db_path = self._orig_index_db_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Parser registration
    # ------------------------------------------------------------------ #
    def test_hybrid_search_subcommand_registered(self):
        """hybrid-search is registered as a CLI subcommand (--help works)."""
        buf = io.StringIO()
        try:
            with patch.object(sys, "argv", ["app.main", "hybrid-search", "--help"]):
                with redirect_stdout(buf):
                    cli_main()
        except SystemExit as exc:
            self.assertEqual(exc.code, 0)
        help_text = buf.getvalue()
        self.assertIn("hybrid-search", help_text)
        self.assertIn("RRF ranking constant", help_text)

    def test_hybrid_search_help_includes_required_args(self):
        """hybrid-search --help lists --repo-id and --query as required."""
        buf = io.StringIO()
        try:
            with patch.object(sys, "argv", ["app.main", "hybrid-search", "--help"]):
                with redirect_stdout(buf):
                    cli_main()
        except SystemExit:
            pass
        help_text = buf.getvalue()
        self.assertIn("--repo-id REPO_ID", help_text)
        self.assertIn("--query QUERY", help_text)
        self.assertIn("--limit LIMIT", help_text)
        self.assertIn("--rrf-k RRF_K", help_text)

    def test_hybrid_search_missing_repo_id(self):
        """Missing --repo-id produces a clear error and exit code 2."""
        buf = io.StringIO()
        try:
            with patch.object(
                sys, "argv", ["app.main", "hybrid-search", "--query", "add function"]
            ):
                with redirect_stderr(buf):
                    cli_main()
        except SystemExit as exc:
            self.assertEqual(exc.code, 2)
        help_text = buf.getvalue()
        self.assertIn("required", help_text.lower())

    def test_hybrid_search_missing_query(self):
        """Missing --query produces a clear error and exit code 2."""
        buf = io.StringIO()
        try:
            with patch.object(
                sys, "argv", ["app.main", "hybrid-search", "--repo-id", self.repo_id]
            ):
                with redirect_stderr(buf):
                    cli_main()
        except SystemExit as exc:
            self.assertEqual(exc.code, 2)
        help_text = buf.getvalue()
        self.assertIn("required", help_text.lower())

    def test_hybrid_search_invalid_limit(self):
        """--limit 0 produces a clear error and exit code 1."""
        args = _Args(repo_id=self.repo_id, query="add", limit=0, rrf_k=60)
        rc = cmd_hybrid_search(args)
        self.assertEqual(rc, 1)

    def test_hybrid_search_negative_limit(self):
        """--limit -1 produces a clear error and exit code 1."""
        args = _Args(repo_id=self.repo_id, query="add", limit=-1, rrf_k=60)
        rc = cmd_hybrid_search(args)
        self.assertEqual(rc, 1)

    def test_hybrid_search_invalid_rrf_k(self):
        """--rrf-k 0 produces a clear error and exit code 1."""
        args = _Args(repo_id=self.repo_id, query="add", limit=5, rrf_k=0)
        rc = cmd_hybrid_search(args)
        self.assertEqual(rc, 1)

    def test_hybrid_search_negative_rrf_k(self):
        """--rrf-k -5 produces a clear error and exit code 1."""
        args = _Args(repo_id=self.repo_id, query="add", limit=5, rrf_k=-5)
        rc = cmd_hybrid_search(args)
        self.assertEqual(rc, 1)

    def test_hybrid_search_nonexistent_repo(self):
        """Nonexistent repo produces a clear error and exit code 1."""
        args = _Args(repo_id="nonexistent-repo-id", query="add", limit=5, rrf_k=60)
        rc = cmd_hybrid_search(args)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
