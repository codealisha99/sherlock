"""Frontend static checks for the SaaS workspace (frontend/index.html + JS modules).

Sherlock has no JavaScript test runner. These tests read the source and assert:

- The legacy local-folder UI ("Repository Path", "Index Workspace", /api/analyze,
  /api/index) is gone from the default SaaS workspace.
- The frontend never starts Ollama or pulls models.
- Every JS file parses with `node --check` when Node.js is available.

The contract tests (endpoints, CSP constraints, element ids) live in
tests/test_frontend_workspace.py.
"""

import shutil
import subprocess
import unittest
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
JS_FILES = sorted(FRONTEND.glob("*.js"))


def _read(name: str) -> str:
    return (FRONTEND / name).read_text(encoding="utf-8")


class TestLegacyLocalPathUiRemoved(unittest.TestCase):
    """SaaS mode disables local paths; the workspace must not expose them."""

    @classmethod
    def setUpClass(cls):
        cls.index_html = _read("index.html")
        cls.js_sources = {path.name: path.read_text(encoding="utf-8") for path in JS_FILES}
        cls.all_js = "\n".join(cls.js_sources.values())

    def test_repository_path_input_removed(self):
        self.assertNotIn("project-path-input", self.index_html)
        self.assertNotIn("Repository Path", self.index_html)

    def test_index_workspace_button_removed(self):
        self.assertNotIn('id="btn-index"', self.index_html)
        self.assertNotIn("Index Workspace", self.index_html)

    def test_no_local_path_endpoints_called(self):
        self.assertNotIn('"/api/analyze"', self.all_js)
        self.assertNotIn('"/api/index"', self.all_js)
        self.assertNotIn("/api/index/repos", self.all_js)
        self.assertNotIn("project_path", self.all_js)


class TestFrontendNoOllama(unittest.TestCase):
    """Frontend never attempts to start Ollama or pull models."""

    @classmethod
    def setUpClass(cls):
        cls.all_js = "\n".join(p.read_text(encoding="utf-8") for p in JS_FILES)

    def test_no_ollama_serve(self):
        self.assertNotIn("ollama serve", self.all_js, "Frontend must not start Ollama server")

    def test_no_ollama_pull(self):
        self.assertNotIn("ollama pull", self.all_js, "Frontend must not pull Ollama models")

    def test_no_ollama_startup_function(self):
        self.assertNotIn("startOllama", self.all_js, "Frontend must not have Ollama startup function")


class TestFrontendJSParse(unittest.TestCase):
    """Every frontend JS file parses with Node.js when Node is available."""

    def test_all_js_files_parse_with_node(self):
        if shutil.which("node") is None:
            self.skipTest("Node.js not available for JS syntax check")
        self.assertTrue(JS_FILES, "frontend/*.js must contain at least one file")
        for path in JS_FILES:
            with self.subTest(file=path.name):
                result = subprocess.run(
                    ["node", "--check", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, f"{path.name} has JS syntax errors: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
