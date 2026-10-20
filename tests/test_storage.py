"""Tests for the model-free SQLite index store (no LLM required)."""

import os
import tempfile
import unittest
from pathlib import Path

from app.storage import IndexStore


class TestIndexStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="sherlock_store_test_"))
        self.db_path = self.test_dir / "index" / "sherlock.db"
        self.store = IndexStore(db_path=self.db_path)
        self.store.connect()
        self.repo_id = "repo-test-1"
        self.store.register_repository(self.repo_id, str(self.test_dir / "repo"), "repo")

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_repository_crud(self):
        self.assertIsNotNone(self.store.get_repository(self.repo_id))
        repos = self.store.list_repositories()
        self.assertEqual(len(repos), 1)
        self.assertTrue(self.store.remove_repository(self.repo_id))
        self.assertEqual(len(self.store.list_repositories()), 0)

    def test_file_upsert_and_get(self):
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        f = self.store.get_file("f1")
        self.assertIsNotNone(f)
        self.assertEqual(f["path"], "src/a.py")
        self.assertEqual(f["language"], "Python")

    def test_chunks_and_fts_search(self):
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.store.add_chunk(self.repo_id, "f1", "def greet(name): return name", 1, 1)
        self.store.add_chunk(self.repo_id, "f1", "class Calculator: pass", 3, 3)

        results = self.store.search_chunks(self.repo_id, "greet")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file_path"], "src/a.py")
        self.assertEqual(results[0]["start_line"], 1)
        self.assertEqual(results[0]["end_line"], 1)
        self.assertIn("rank", results[0])

    def test_definitions_and_imports(self):
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.store.add_definition(self.repo_id, "f1", "greet", "function", 1, "greet(name)")
        self.store.add_definition(self.repo_id, "f1", "Calculator", "class", 3, None)
        self.store.add_import(self.repo_id, "f1", "os", 1, "import")

        defs = self.store.get_definitions_by_repo(self.repo_id)
        self.assertEqual(len(defs), 2)
        names = [d["name"] for d in defs]
        self.assertIn("greet", names)
        self.assertIn("Calculator", names)

        found = self.store.search_definitions(self.repo_id, "greet")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "greet")

        imports = self.store.get_imports_by_file("f1")
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["module"], "os")

    def test_remove_file_cascades(self):
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.store.add_chunk(self.repo_id, "f1", "content", 1, 1)
        self.store.add_definition(self.repo_id, "f1", "greet", "function", 1, None)
        self.store.delete_file("f1")
        self.assertIsNone(self.store.get_file("f1"))
        self.assertEqual(len(self.store.get_definitions_by_file("f1")), 0)

    def test_consistency_check(self):
        self.store.upsert_file(
            self.repo_id, "f1", "src/a.py", "Python", 100, 10, False, False, 1.0, "abc"
        )
        self.store.add_chunk(self.repo_id, "f1", "content", 1, 1)
        c = self.store.consistency_check(self.repo_id)
        self.assertTrue(c["consistent"])

    def test_index_log(self):
        self.store.log_action(self.repo_id, "index_start", "path=/tmp/repo")
        logs = self.store.get_index_log(self.repo_id)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"], "index_start")


if __name__ == "__main__":
    unittest.main()
