"""Tests for the model-free repository indexer (no LLM required).

Uses synthetic temporary repositories only. Never touches the user's private
projects and never executes repository content during indexing.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path

from app.indexer import RepositoryIndexer


def _make_repo(files):
    """Create a temp repo from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix="sherlock_indexer_test_"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


class TestRepositoryIndexer(unittest.TestCase):
    def setUp(self):
        self.idx = RepositoryIndexer()

    def _cleanup(self, root):
        import shutil
        shutil.rmtree(root, ignore_errors=True)

    def test_index_and_search_without_ollama(self):
        root = _make_repo({"src/app.py": "def greet(name):\n    return \"Hi, \" + name\n"})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.status, "complete")
            self.assertEqual(res.file_count, 1)
            self.assertEqual(res.definition_count, 1)

            results = self.idx.store.search_chunks(res.repo_id, "greet")
            self.assertGreaterEqual(len(results), 1)
            # Every result carries a valid file path and line range.
            for r in results:
                self.assertTrue(r["file_path"])
                self.assertIsInstance(r["start_line"], int)
                self.assertIsInstance(r["end_line"], int)

            defs = self.idx.store.get_definitions_by_repo(res.repo_id)
            self.assertEqual([d["name"] for d in defs], ["greet"])
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_idempotent_reindex_no_duplicates(self):
        root = _make_repo({"src/app.py": "def greet(name):\n    return name\n"})
        try:
            res1 = self.idx.index_repository(root)
            res2 = self.idx.index_repository(root)
            defs = self.idx.store.get_definitions_by_repo(res2.repo_id)
            self.assertEqual(len(defs), 1)
            self.assertEqual(res2.file_count, 1)
        finally:
            self.idx.remove_repository(res1.repo_id)
            self._cleanup(root)

    def test_modification_reflected(self):
        root = _make_repo({"src/app.py": "def a():\n    return 1\n"})
        try:
            res1 = self.idx.index_repository(root)
            (root / "src" / "app.py").write_text("def a():\n    return 1\n\ndef b():\n    return 2\n")
            res2 = self.idx.index_repository(root)
            names = [d["name"] for d in self.idx.store.get_definitions_by_repo(res2.repo_id)]
            self.assertIn("a", names)
            self.assertIn("b", names)
        finally:
            self.idx.remove_repository(res1.repo_id)
            self._cleanup(root)

    def test_deletion_reflected_and_stale_cleaned(self):
        root = _make_repo(
            {"keep.py": "x = 1\n", "remove.py": "y = 2\n"}
        )
        try:
            res1 = self.idx.index_repository(root)
            self.assertEqual(res1.file_count, 2)
            os.remove(root / "remove.py")
            res2 = self.idx.index_repository(root)
            self.assertEqual(res2.file_count, 1)
            paths = [f["path"] for f in self.idx.store.get_files_by_repo(res2.repo_id)]
            self.assertEqual(paths, ["keep.py"])
            self.assertTrue(self.idx.consistency_check(res2.repo_id)["consistent"])
        finally:
            self.idx.remove_repository(res1.repo_id)
            self._cleanup(root)

    def test_repository_removal_cleans_derived_data(self):
        root = _make_repo({"src/a.py": "def f(): pass\n"})
        try:
            res = self.idx.index_repository(root)
            repo_id = res.repo_id
            self.idx.remove_repository(repo_id)
            self.assertIsNone(self.idx.get_repository(repo_id))
            self.assertEqual(self.idx.list_repositories(), [])
            # Store connection is closed; reopen to verify derived rows are gone.
            from app.storage import IndexStore
            store = IndexStore()
            store.connect()
            self.assertEqual(store.get_repository(repo_id), None)
            store.close()
        finally:
            self._cleanup(root)

    def test_path_traversal_rejected(self):
        # A symlink that escapes the repo root must be skipped, not indexed.
        root = _make_repo({"src/app.py": "def f(): pass\n"})
        try:
            escape = root / "escape_link"
            os.symlink("/etc", escape)
            res = self.idx.index_repository(root)
            # The symlink itself should not appear as an indexed file.
            paths = [f["path"] for f in self.idx.store.get_files_by_repo(res.repo_id)]
            self.assertNotIn("escape_link", paths)
            self.assertEqual(res.file_count, 1)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_external_symlink_not_followed(self):
        root = _make_repo({"src/app.py": "def f(): pass\n"})
        try:
            # Symlink to a file outside the repo
            link = root / "src" / "outside.py"
            os.symlink("/etc/hostname", link)
            res = self.idx.index_repository(root)
            paths = [f["path"] for f in self.idx.store.get_files_by_repo(res.repo_id)]
            self.assertNotIn("src/outside.py", paths)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_malformed_python_handled_safely(self):
        root = _make_repo({"src/broken.py": "def f(\n    : not valid python :(\n"})
        try:
            res = self.idx.index_repository(root)
            # Should not crash; malformed Python is indexed as text without defs.
            self.assertEqual(res.status, "complete")
            self.assertEqual(res.file_count, 1)
            self.assertEqual(res.definition_count, 0)
            # It should still be searchable by text present in the content.
            results = self.idx.store.search_chunks(res.repo_id, "python")
            self.assertGreaterEqual(len(results), 1)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_oversized_file_skipped(self):
        root = _make_repo({"src/big.py": "x = 1\n" * 200000})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.file_count, 0)
            self.assertTrue(len(res.skipped_files) == 1)
            self.assertIn("src/big.py", res.skipped_files)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_binary_file_skipped(self):
        root = _make_repo({"src/data.bin": "\x00\x01\x02\x03binary"})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.file_count, 0)
            self.assertEqual(len(res.skipped_files), 1)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_secret_file_skipped(self):
        root = _make_repo({".env": "SECRET_KEY=supersecret\n"})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.file_count, 0)
            self.assertIn(".env", res.skipped_files)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_vcs_and_cache_dirs_ignored(self):
        root = _make_repo(
            {
                ".git/config": "x\n",
                "__pycache__/mod.cpython-312.pyc": "\x00\x01",
                "src/app.py": "def f(): pass\n",
            }
        )
        try:
            res = self.idx.index_repository(root)
            paths = [f["path"] for f in self.idx.store.get_files_by_repo(res.repo_id)]
            self.assertIn("src/app.py", paths)
            self.assertNotIn(".git/config", paths)
            self.assertNotIn("__pycache__/mod.cpython-312.pyc", paths)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_content_never_executed(self):
        # Even a file that looks executable-ish is only parsed as text/AST,
        # never executed. Writing to disk is the only side effect of indexing.
        root = _make_repo({"src/app.py": "print('indexed as text only')\n"})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.status, "complete")
            self.assertEqual(res.file_count, 1)
            # No definitions extracted from a bare print statement.
            self.assertEqual(res.definition_count, 0)
            # But searchable as text.
            results = self.idx.store.search_chunks(res.repo_id, "indexed")
            self.assertGreaterEqual(len(results), 1)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_nonexistent_path_reported(self):
        with self.assertRaises(ValueError):
            self.idx.index_repository("/tmp/does_not_exist_xyz_12345")

    def test_multiple_repositories_isolated(self):
        r1 = _make_repo({"a.py": "def alpha(): pass\n"})
        r2 = _make_repo({"b.py": "def beta(): pass\n"})
        try:
            res1 = self.idx.index_repository(r1)
            res2 = self.idx.index_repository(r2)
            self.assertEqual(
                [d["name"] for d in self.idx.store.get_definitions_by_repo(res1.repo_id)],
                ["alpha"],
            )
            self.assertEqual(
                [d["name"] for d in self.idx.store.get_definitions_by_repo(res2.repo_id)],
                ["beta"],
            )
        finally:
            self.idx.remove_repository(res1.repo_id)
            self.idx.remove_repository(res2.repo_id)
            self._cleanup(r1)
            self._cleanup(r2)


if __name__ == "__main__":
    unittest.main()
