"""CLI tests for embed and semantic-search commands.

Uses isolated temporary repositories and databases. Never touches
index/sherlock.db.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from app.indexer import RepositoryIndexer  # noqa: E402
from app.main import cmd_embed, cmd_semantic_search  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
