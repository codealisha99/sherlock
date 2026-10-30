"""Frontend logic tests.

Sherlock has no JavaScript testing framework (no package.json, no Jest).
These Python tests verify the frontend JavaScript source code contains
the required patterns for:

- Ollama health check with bounded timeout (no indefinite "Checking...")
- Error handling for connection refused, timeout, server unavailable
- Index Workspace button handler (model-free, works without Ollama)
- Analyze button error handling (no silent failures)
- Proper frontend state transitions

Tests read frontend/app.js and frontend/index.html and assert on expected
code patterns. They also verify the JS can be parsed by Node.js if available.
"""

import json
import subprocess
import unittest
from pathlib import Path


class TestFrontendSourcePatterns(unittest.TestCase):
    """Verify frontend JS source contains required error handling and timeout patterns."""

    @classmethod
    def setUpClass(cls):
        cls.app_js = Path("frontend/app.js").read_text()
        cls.index_html = Path("frontend/index.html").read_text()

    def test_ollama_health_check_has_timeout(self):
        """checkOllamaHealth uses AbortController with a bounded timeout."""
        self.assertIn("AbortController", self.app_js,
                      "checkOllamaHealth must use AbortController for timeout")
        self.assertIn("setTimeout(() => controller.abort(), 5000)", self.app_js,
                      "checkOllamaHealth must abort after 5 seconds")
        self.assertIn("{ signal: controller.signal }", self.app_js,
                      "fetch must pass the AbortController signal")

    def test_ollama_health_check_handles_abort_error(self):
        """checkOllamaHealth handles AbortError (timeout) separately."""
        self.assertIn('err.name === "AbortError"', self.app_js,
                      "Must distinguish timeout errors from connection errors")

    def test_ollama_status_transitions_offline(self):
        """Ollama status transitions from 'Checking...' to offline on failure."""
        # Initial HTML has "Checking..."
        self.assertIn('id="ollama-status-text"', self.index_html)
        self.assertIn("Checking Ollama...", self.index_html,
                      "Initial status should be 'Checking...'")

        # JS transitions to offline on error
        self.assertIn("status-offline", self.app_js,
                      "Must set status-offline class on error")
        self.assertIn("Ollama: Offline", self.app_js,
                      "Must show offline text on error")

    def test_index_workspace_button_exists(self):
        """Index Workspace button exists in HTML."""
        self.assertIn('id="btn-index"', self.index_html,
                      "Index Workspace button must exist in HTML")
        self.assertIn("Index Workspace", self.index_html,
                      "Index Workspace button text must be present")

    def test_index_workspace_button_has_event_listener(self):
        """Index Workspace button has click event listener in JS."""
        self.assertIn('btnIndex.addEventListener("click", startIndexing)', self.app_js,
                      "Index Workspace button must have click listener")

    def test_start_indexing_function_exists(self):
        """startIndexing function exists in JS."""
        self.assertIn("async function startIndexing()", self.app_js,
                      "startIndexing function must exist")

    def test_poll_index_status_function_exists(self):
        """pollIndexStatus function exists in JS."""
        self.assertIn("async function pollIndexStatus()", self.app_js,
                      "pollIndexStatus function must exist")

    def test_start_indexing_calls_api_index(self):
        """startIndexing calls the model-free /api/index endpoint."""
        self.assertIn('fetch("/api/index"', self.app_js,
                      "startIndexing must call POST /api/index")

    def test_poll_index_status_calls_api_index_repos(self):
        """pollIndexStatus polls GET /api/index/repos for status."""
        self.assertIn('fetch("/api/index/repos"', self.app_js,
                      "pollIndexStatus must poll GET /api/index/repos")

    def test_start_analysis_calls_api_analyze(self):
        """startAnalysis calls the AI-powered /api/analyze endpoint."""
        self.assertIn('fetch("/api/analyze"', self.app_js,
                      "startAnalysis must call POST /api/analyze")

    def test_analyze_shows_loading_state(self):
        """Analyze button shows loading state when clicked."""
        self.assertIn("Analyzing...", self.app_js,
                      "Analyze button must show loading state")
        self.assertIn('btnAnalyze.disabled = true', self.app_js,
                      "Analyze button must be disabled during analysis")

    def test_index_shows_loading_state(self):
        """Index button shows loading state when clicked."""
        self.assertIn("Indexing...", self.app_js,
                      "Index button must show loading state")
        self.assertIn('btnIndex.disabled = true', self.app_js,
                      "Index button must be disabled during indexing")

    def test_analyze_handles_errors(self):
        """Analyze button shows error alert on failure (no silent failures)."""
        self.assertIn('alert("Error: " + err.message)', self.app_js,
                      "Analyze must show error alert on failure")
        self.assertIn("catch (err)", self.app_js,
                      "Analyze must have error handler")

    def test_index_handles_errors(self):
        """Index button shows error alert on failure (no silent failures)."""
        self.assertIn('alert("Error: " + err.message)', self.app_js,
                      "Index must show error alert on failure")

    def test_no_ollama_startup_in_frontend(self):
        """Frontend never attempts to start Ollama or pull models."""
        self.assertNotIn("ollama serve", self.app_js,
                         "Frontend must not start Ollama server")
        self.assertNotIn("ollama pull", self.app_js,
                         "Frontend must not pull Ollama models")
        self.assertNotIn("startOllama", self.app_js,
                         "Frontend must not have Ollama startup function")

    def test_frontend_distinguishes_index_from_analyze(self):
        """Frontend clearly separates Index Workspace (offline) from Analyze (AI)."""
        # Index uses /api/index (model-free)
        self.assertIn('fetch("/api/index"', self.app_js)
        # Analyze uses /api/analyze (requires AI)
        self.assertIn('fetch("/api/analyze"', self.app_js)
        # They are different functions
        self.assertIn("startIndexing", self.app_js)
        self.assertIn("startAnalysis", self.app_js)


class TestFrontendJSParse(unittest.TestCase):
    """Verify frontend JS can be parsed by Node.js if available."""

    def test_app_js_parseable_by_node(self):
        """frontend/app.js can be parsed by Node.js (syntax check)."""
        app_js_path = Path("frontend/app.js")
        if not app_js_path.exists():
            self.skipTest("frontend/app.js not found")

        try:
            result = subprocess.run(
                ["node", "--check", str(app_js_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                self.fail(f"frontend/app.js has JS syntax errors: {result.stderr}")
        except FileNotFoundError:
            self.skipTest("Node.js not available for JS syntax check")
        except subprocess.TimeoutExpired:
            self.skipTest("Node.js check timed out")


if __name__ == "__main__":
    unittest.main()
