"""Static contract checks for the SaaS workspace at /app (frontend/index.html + JS modules).

Covers the CSP constraints (no inline scripts, no on*= handlers, scripts only from
same origin or cdn.jsdelivr.net), the element ids the JS depends on, and the API
endpoints from docs/SAAS_ARCHITECTURE.md.
"""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

WORKSPACE_JS = ["util.js", "api.js", "node-panel.js", "graph.js", "findings.js", "chat.js", "report.js", "app.js"]
MAX_JS_LINES = 600

INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>", re.IGNORECASE)
INLINE_HANDLER_RE = re.compile(r"\son[a-z]+\s*=", re.IGNORECASE)
SCRIPT_SRC_RE = re.compile(r'<script[^>]*\bsrc="([^"]+)"', re.IGNORECASE)
GET_BY_ID_RE = re.compile(r'getElementById\(\s*"([^"]+)"\s*\)')


def _read(name: str) -> str:
    return (FRONTEND / name).read_text(encoding="utf-8")


class TestWorkspaceHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read("index.html")

    def test_cytoscape_loaded_from_jsdelivr(self):
        self.assertIn(
            'src="https://cdn.jsdelivr.net/npm/cytoscape@3/dist/cytoscape.min.js"',
            self.html,
        )

    def test_no_inline_script_blocks(self):
        self.assertIsNone(INLINE_SCRIPT_RE.search(self.html), "index.html has an inline <script> block")

    def test_no_inline_event_handlers(self):
        self.assertIsNone(INLINE_HANDLER_RE.search(self.html), "index.html has an on*= attribute")

    def test_scripts_only_from_same_origin_or_jsdelivr(self):
        for src in SCRIPT_SRC_RE.findall(self.html):
            ok = src.startswith("/") or src.startswith("https://cdn.jsdelivr.net/")
            self.assertTrue(ok, f"script src not allowed by CSP: {src}")

    def test_workspace_scripts_load_in_order(self):
        srcs = [s.lstrip("/") for s in SCRIPT_SRC_RE.findall(self.html) if s.startswith("/")]
        positions = [srcs.index(name) for name in WORKSPACE_JS]
        self.assertEqual(positions, sorted(positions), "workspace JS modules must load in dependency order")
        self.assertEqual(srcs[-1], "app.js", "app.js must load last")

    def test_required_element_ids_exist(self):
        required = [
            # auth + top bar
            "plan-badge", "user-email", "btn-logout", "global-error", "auth-loading",
            # start screen
            "view-start", "analyze-form", "repo-url-input", "btn-analyze", "analyze-error",
            "repo-list", "job-list", "repos-empty", "jobs-empty",
            # progress
            "view-progress", "stage-stepper", "progress-fill", "progress-track", "progress-status",
            "progress-error", "progress-error-text", "btn-retry", "btn-back-start",
            # workspace header + tabs
            "view-workspace", "repo-name", "repo-url", "repo-score", "severity-counts",
            "btn-back-repos", "tab-graph", "tab-findings", "tab-report",
            "panel-graph", "panel-findings", "panel-report",
            # graph
            "graph-canvas", "graph-status", "graph-search", "graph-layout", "btn-layout",
            "btn-fit", "filter-findings-only", "node-panel", "node-panel-content", "btn-close-node",
            # findings
            "findings-summary", "filter-severity", "filter-category", "findings-sort",
            "findings-list", "findings-empty", "findings-status",
            # report
            "report-body", "report-title", "report-subtitle", "report-indicator", "report-prev",
            "report-next", "report-tabs", "code-modal", "modal-filepath", "modal-code-content",
            "btn-close-modal",
            # chat
            "chat-widget", "chat-bubble", "chat-panel", "chat-session-title", "chat-messages",
            "chat-input", "chat-send", "chat-status", "chat-close", "chat-new-chat",
            "chat-focus", "chat-focus-label", "chat-focus-clear",
        ]
        for element_id in required:
            with self.subTest(id=element_id):
                self.assertIn(f'id="{element_id}"', self.html)

    def test_legacy_local_path_ui_absent(self):
        self.assertNotIn("Index Workspace", self.html)
        self.assertNotIn("project-path-input", self.html)

    def test_report_tab_hidden_by_default(self):
        self.assertRegex(self.html, r'id="tab-report"[^>]*class="ws-tab hidden"')


class TestWorkspaceJsIds(unittest.TestCase):
    """Every getElementById("x") in the workspace JS must exist in index.html."""

    def test_referenced_ids_exist_in_html(self):
        html = _read("index.html")
        for name in WORKSPACE_JS:
            for element_id in GET_BY_ID_RE.findall(_read(name)):
                with self.subTest(file=name, id=element_id):
                    self.assertIn(f'id="{element_id}"', html)


class TestWorkspaceJsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: _read(name) for name in WORKSPACE_JS}
        cls.all_js = "\n".join(cls.sources.values())

    def test_repo_endpoints_used(self):
        self.assertIn('"/api/repos/github"', self.all_js)
        self.assertIn("/api/repos/jobs", self.all_js)
        self.assertIn('repos: () => apiGet("/api/repos")', self.sources["api.js"])
        self.assertIn("/graph`", self.sources["api.js"])
        self.assertIn("/findings`", self.sources["api.js"])
        self.assertIn("/node?id=", self.sources["api.js"])

    def test_auth_endpoints_used(self):
        self.assertIn('"/api/auth/me"', self.sources["api.js"])
        self.assertIn('"/api/auth/logout"', self.sources["api.js"])
        self.assertIn("/login?next=", self.sources["app.js"])

    def test_job_polling_every_1500ms(self):
        self.assertIn("POLL_INTERVAL_MS = 1500", self.sources["app.js"])
        self.assertIn("/api/repos/jobs/", self.sources["api.js"])

    def test_chat_session_and_focus_contract(self):
        chat = self.sources["chat.js"]
        self.assertIn('"/api/chat/sessions"', self.sources["api.js"])
        self.assertIn("/api/chat/sessions/${seg(sessionId)}/messages", chat)
        self.assertIn("body.focus = chatState.focus", chat)
        self.assertIn('"data:"', chat)
        for event_type in ("token", "status", "done", "error"):
            self.assertIn(f'event.type === "{event_type}"', chat)

    def test_chat_handles_limit_statuses(self):
        chat = self.sources["chat.js"] + self.sources["app.js"]
        for status in ("400", "402", "429"):
            self.assertIn(status, chat)
        self.assertIn("/settings?upgrade=pro", self.sources["app.js"])

    def test_all_fetches_same_origin_credentials(self):
        self.assertIn('credentials: "same-origin"', self.sources["api.js"])
        self.assertIn('credentials: "same-origin"', self.sources["chat.js"])
        for name, src in self.sources.items():
            for match in re.finditer(r"\bfetch\(", src):
                if name in ("api.js", "chat.js"):
                    continue
                self.fail(f"{name} must call the api.js helpers, not fetch() directly")

    def test_cytoscape_graph_uses_contract_fields(self):
        graph = self.sources["graph.js"]
        self.assertIn("window.cytoscape(", graph)
        for field in ("health", "findingsTotal", "metrics"):
            self.assertIn(field, graph)
        for edge_type in ("contains", "imports", "calls"):
            self.assertIn(f'edge[type = "{edge_type}"]', graph)

    def test_node_panel_renders_contract_sections(self):
        panel = self.sources["node-panel.js"]
        for key in ("neighbors", "incoming", "outgoing", "source", "start_line", "recommendation", "cwe"):
            self.assertIn(key, panel)
        self.assertIn("Ask about this", panel)

    def test_no_inline_handlers_built_by_js(self):
        self.assertNotIn("onclick", self.all_js)
        self.assertNotIn("onerror", self.all_js)

    def test_innerhtml_only_in_report_renderer(self):
        for name, src in self.sources.items():
            if name == "report.js":
                continue
            self.assertNotIn("innerHTML", src, f"{name} must build DOM with el()/textContent")
        self.assertIn("esc(", self.sources["report.js"], "report renderer must escape server strings")

    def test_mermaid_initialized_in_js_not_html(self):
        self.assertIn("mermaid.initialize(", self.sources["report.js"])
        self.assertIn('securityLevel: "strict"', self.sources["report.js"])

    def test_module_sizes_within_budget(self):
        for name, src in self.sources.items():
            with self.subTest(file=name):
                self.assertLess(len(src.splitlines()), MAX_JS_LINES, f"{name} exceeds {MAX_JS_LINES} lines")


class TestWorkspaceJsParse(unittest.TestCase):
    def test_workspace_js_parses_with_node(self):
        if shutil.which("node") is None:
            self.skipTest("Node.js not available for JS syntax check")
        for name in WORKSPACE_JS:
            with self.subTest(file=name):
                result = subprocess.run(
                    ["node", "--check", str(FRONTEND / name)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
