"""Integration tests for the Sherlock indexer using synthetic fixtures.

These tests verify that the indexer correctly discovers, indexes, and
persists repository files, symbols, imports, and search chunks using
the actual app.indexer.RepositoryIndexer and app.storage.IndexStore
classes. All tests use temporary directories and databases so they
never touch the application's production index.
"""

import tempfile
import unittest
from pathlib import Path

from app.indexer import RepositoryIndexer
from app.storage import IndexStore


def make_repo(files: dict) -> Path:
    """Create a temporary repository from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix="sherlock_idx_test_"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


class TestRepositoryIndexerIntegration(unittest.TestCase):
    """Integration tests for RepositoryIndexer using real SQLite storage."""

    def setUp(self):
        self.idx = RepositoryIndexer()
        self.created_repos = []

    def tearDown(self):
        # Clean up all repositories created during the test
        for repo_id in self.created_repos:
            try:
                self.idx.remove_repository(repo_id)
            except Exception:
                pass
        self.idx.store.close()

    def _index(self, files: dict, repo_name: str = "test_repo"):
        """Helper to create a temp repo and index it."""
        root = make_repo(files)
        repo_id = f"eval_{repo_name}"
        result = self.idx.index_repository(root, repo_id=repo_id, repo_name=repo_name)
        self.created_repos.append(repo_id)
        return result, root

    def test_file_discovery(self):
        """All eligible files are discovered during indexing."""
        files = {
            "src/app.py": "def main():\n    pass\n",
            "src/utils.py": "def helper():\n    pass\n",
            "config.py": "DEBUG = True\n",
            "README.md": "# Project\n",
        }
        result, _ = self._index(files, "test_discovery")
        self.assertEqual(result.status, "complete", f"Indexing failed: {result.errors}")
        self.assertEqual(result.file_count, 4, f"Expected 4 files, got {result.file_count}")
        self.assertEqual(result.errors, [], f"Unexpected errors: {result.errors}")

        # All files should be in the index
        indexed_paths = {f["path"] for f in self.idx.store.get_files_by_repo(result.repo_id)}
        self.assertEqual(
            indexed_files := indexed_paths,
            {"src/app.py", "src/utils.py", "config.py", "README.md"},
            f"Expected all 4 files indexed, got {indexed_files}",
        )

    def test_ast_symbol_extraction(self):
        """Python classes and functions are extracted as definitions."""
        files = {
            "src/math.py": "def add(a, b):\n    return a + b\n\nclass Calculator:\n    def multiply(self, a, b):\n        return a * b\n",
        }
        result, _ = self._index(files, "test_ast")
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.definition_count, 3, f"Expected 3 defs, got {result.definition_count}")

        defs = self.idx.store.get_definitions_by_repo(result.repo_id)
        def_names = {d["name"] for d in defs}
        self.assertIn("add", def_names)
        self.assertIn("Calculator", def_names)
        self.assertIn("Calculator.multiply", def_names)

    def test_import_extraction(self):
        """Import statements are extracted."""
        files = {
            "src/main.py": "import os\nfrom sys import path\n",
        }
        result, _ = self._index(files, "test_imports")
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.import_count, 2, f"Expected 2 imports, got {result.import_count}")

        imports = self.idx.store.get_imports_by_repo(result.repo_id)
        import_modules = {i["module"] for i in imports}
        self.assertIn("os", import_modules)
        self.assertIn("sys", import_modules)

    def test_idempotent_reindexing(self):
        """Re-indexing the same repository does not create duplicates."""
        files = {"src/app.py": "def foo():\n    pass\n"}
        result1, root = self._index(files, "test_idempotent")
        self.assertEqual(result1.file_count, 1)

        # Re-index the same repository
        result2 = self.idx.index_repository(root, repo_id=result1.repo_id, repo_name="test_repo")
        self.assertEqual(result2.file_count, 1, f"Expected 1 file after re-index, got {result2.file_count}")

        defs = self.idx.store.get_definitions_by_repo(result2.repo_id)
        self.assertEqual(len(defs), 1, f"Expected 1 definition after re-index, got {len(defs)}")

    def test_stale_file_cleanup(self):
        """Files deleted from disk are removed from the index on re-index."""
        files = {
            "src/keep.py": "x = 1\n",
            "src/remove.py": "y = 2\n",
        }
        result1, root = self._index(files, "test_stale")
        self.assertEqual(result1.file_count, 2)

        # Delete remove.py and re-index
        (root / "src/remove.py").unlink()
        result2 = self.idx.index_repository(root, repo_id=result1.repo_id, repo_name="test_stale")
        self.assertEqual(result2.file_count, 1, f"Expected 1 file after deletion, got {result2.file_count}")

        indexed_paths = {f["path"] for f in self.idx.store.get_files_by_repo(result2.repo_id)}
        self.assertEqual(indexed_paths, {"src/keep.py"}, f"Expected only keep.py, got {indexed_paths}")

    def test_consistency_check_passes(self):
        """consistency_check returns consistent=True after clean indexing."""
        files = {"src/app.py": "def f():\n    pass\n"}
        result, _ = self._index(files, "test_consistency")
        consistency = self.idx.consistency_check(result.repo_id)
        self.assertTrue(consistency["consistent"], f"Index inconsistent: {consistency}")
        self.assertEqual(consistency["orphan_chunks"], 0)
        self.assertEqual(consistency["orphan_definitions"], 0)
        self.assertEqual(consistency["orphan_imports"], 0)

    def test_path_relative_to_repo_root(self):
        """File paths are stored relative to the repository root, not the scanner root."""
        files = {"src/deep/nested.py": "def deep():\n    pass\n"}
        result, root = self._index(files, "test_path")
        self.assertEqual(result.status, "complete")

        indexed_paths = {f["path"] for f in self.idx.store.get_files_by_repo(result.repo_id)}
        self.assertIn("src/deep/nested.py", indexed_paths, f"Expected src/deep/nested.py, got {indexed_paths}")


class TestIndexStoreIntegration(unittest.TestCase):
    """Integration tests for IndexStore using a temporary database."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sherlock_store_test_"))
        self.db_path = self.tmpdir / "test.db"
        self.store = IndexStore(db_path=self.db_path)
        self.store.connect()
        self.repo_id = "test_repo_1"
        self.store.register_repository(self.repo_id, "/tmp/test_repo", "test_repo")

    def tearDown(self):
        self.store.close()

    def test_upsert_and_get_file(self):
        """Files can be upserted and retrieved."""
        self.store.upsert_file(
            repo_id=self.repo_id,
            file_id=f"{self.repo_id}:src/main.py",
            path="src/main.py",
            language="Python",
            size_bytes=100,
            line_count=10,
            is_binary=False,
            is_secret=False,
            mtime=1234567890.0,
            content_hash="abc123",
        )
        f = self.store.get_file(f"{self.repo_id}:src/main.py")
        self.assertIsNotNone(f)
        self.assertEqual(f["path"], "src/main.py")
        self.assertEqual(f["language"], "Python")

    def test_add_and_search_chunks(self):
        """Chunks can be added and found via FTS search."""
        self.store.upsert_file(
            repo_id=self.repo_id,
            file_id=f"{self.repo_id}:src/main.py",
            path="src/main.py",
            language="Python",
            size_bytes=50,
            line_count=5,
            is_binary=False,
            is_secret=False,
            mtime=1234567890.0,
            content_hash="abc123",
        )
        self.store.add_chunk(
            repo_id=self.repo_id,
            file_id=f"{self.repo_id}:src/main.py",
            content="def add(a, b):\n    return a + b\n",
            start_line=1,
            end_line=2,
        )
        results = self.store.search_chunks(self.repo_id, "add", limit=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["file_path"], "src/main.py")


if __name__ == "__main__":
    unittest.main()