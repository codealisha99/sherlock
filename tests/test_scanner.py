"""Tests for SafeFilesystem, RepositoryScanner, and CodeSearch."""

import unittest
from pathlib import Path
import tempfile
import shutil

from config import SherlockConfig
from app.tools.filesys import SafeFilesystem
from app.tools.repository_scanner import RepositoryScanner
from app.tools.code_search import CodeSearch


class TestScannerAndFilesys(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sherlock_test_")
        self.root = Path(self.test_dir)
        self.cfg = SherlockConfig(workspace_dir=self.root / "workspace")

        # Create dummy structure
        (self.root / "app").mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".secret_dir").mkdir()

        # Create python file with AST structures
        sample_code = '''"""Sample module docstring."""
import os
from pydantic import BaseModel

class User(BaseModel):
    """User representation."""
    id: int
    name: str

    def get_display_name(self) -> str:
        return self.name

def calculate_total(items: list) -> float:
    # TODO: Implement discount logic
    # FIXME: Handle zero items edge case
    total = sum(items)
    return total

def api_route_handler():
    # @app.get('/users')
    pass
'''
        with open(self.root / "app" / "users.py", "w") as f:
            f.write(sample_code)

        # Create secret file
        with open(self.root / ".env", "w") as f:
            f.write("SECRET_KEY=supersecret123\n")

        # Create ignored git file
        with open(self.root / ".git" / "config", "w") as f:
            f.write("[core]\nrepositoryformatversion = 0\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_safe_filesystem_boundaries(self):
        fs = SafeFilesystem(self.root, self.cfg)
        # Inside root
        resolved = fs.resolve_path("app/users.py")
        self.assertTrue(resolved.is_file())

        # Path traversal should raise PermissionError
        with self.assertRaises(PermissionError):
            fs.resolve_path("../../etc/passwd")

    def test_secret_file_redaction(self):
        fs = SafeFilesystem(self.root, self.cfg)
        content, _ = fs.read_file(".env")
        self.assertIn("REDACTED", content)

    def test_repository_scanner_ast(self):
        scanner = RepositoryScanner(self.root, self.cfg)
        index = scanner.scan()

        self.assertEqual(index.project_name, self.root.name)
        # .git should be ignored
        git_files = [f for f in index.files if ".git" in f.path]
        self.assertEqual(len(git_files), 0)

        # Python AST checks
        user_file = next((f for f in index.files if f.name == "users.py"), None)
        self.assertIsNotNone(user_file)
        self.assertIsNotNone(user_file.ast_info)

        # Verify class extraction
        classes = user_file.ast_info.classes
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].name, "User")
        self.assertEqual(len(classes[0].methods), 1)
        self.assertEqual(classes[0].methods[0].name, "get_display_name")

        # Verify functions
        funcs = user_file.ast_info.functions
        func_names = [fn.name for fn in funcs]
        self.assertIn("calculate_total", func_names)

        # Verify TODOs/FIXMEs
        self.assertTrue(len(index.todos_and_fixmes) >= 2)
        todo_texts = [t["text"] for t in index.todos_and_fixmes]
        self.assertTrue(any("discount logic" in t for t in todo_texts))
        self.assertTrue(any("zero items" in t for t in todo_texts))

    def test_code_search(self):
        scanner = RepositoryScanner(self.root, self.cfg)
        index = scanner.scan()
        fs = SafeFilesystem(self.root, self.cfg)
        search = CodeSearch(index, fs)

        matches = search.search_pattern("calculate_total")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].line_number, 13)


if __name__ == "__main__":
    unittest.main()
