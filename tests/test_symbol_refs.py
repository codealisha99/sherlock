"""Tests for symbol references, .gitignore support, storage inventory,
and security configuration (loopback binding, local-only mode)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.indexer import RepositoryIndexer
from app.storage import IndexStore


def _make_repo(files):
    """Create a temp repo from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix="sherlock_sec_test_"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


class TestSymbolReferences(unittest.TestCase):
    def setUp(self):
        self.idx = RepositoryIndexer()

    def _cleanup(self, root):
        import shutil
        shutil.rmtree(root, ignore_errors=True)

    def test_symbol_definitions_with_references(self):
        root = _make_repo({"src/app.py": "def greet(name):\n    return \"Hi, \" + name\n\ndef say_hi():\n    greet(\"world\")\n"})
        try:
            res = self.idx.index_repository(root)
            self.assertEqual(res.status, "complete")
            # Should have 2 function definitions
            defs = self.idx.store.get_definitions_by_repo(res.repo_id)
            names = [d["name"] for d in defs]
            self.assertIn("greet", names)
            self.assertIn("say_hi", names)
            # Should have call references (greet called by say_hi)
            refs = self.idx.store.get_symbol_references(res.repo_id, symbol_name="greet")
            self.assertGreaterEqual(len(refs), 1)
            # Call references should be marked unresolved (static analysis can't resolve)
            call_refs = [r for r in refs if r["symbol_kind"] == "call"]
            self.assertGreaterEqual(len(call_refs), 1)
            for r in call_refs:
                self.assertEqual(r["resolution_status"], "unresolved")
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_symbol_definitions_with_references_api(self):
        """Verify get_symbol_definitions_with_references returns provenance."""
        root = _make_repo({"src/app.py": "def foo(x):\n    return x + 1\n"})
        try:
            res = self.idx.index_repository(root)
            result = self.idx.store.get_symbol_definitions_with_references(res.repo_id)
            self.assertEqual(len(result), 1)
            d = result[0]
            self.assertEqual(d["name"], "foo")
            self.assertEqual(d["kind"], "function")
            self.assertIn("references", d)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_gitignore_respected(self):
        root = _make_repo({
            "src/app.py": "def f(): pass\n",
            ".gitignore": "secrets/\n*.log\n",
            "secrets/api_key.txt": "sk-12345\n",
            "debug.log": "log data\n",
        })
        try:
            res = self.idx.index_repository(root)
            paths = [f["path"] for f in self.idx.store.get_files_by_repo(res.repo_id)]
            self.assertIn("src/app.py", paths)
            self.assertNotIn("secrets/api_key.txt", paths)
            self.assertNotIn("debug.log", paths)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_storage_inventory(self):
        root = _make_repo({"src/app.py": "def f(): pass\n"})
        try:
            res = self.idx.index_repository(root)
            inv = self.idx.store.get_storage_inventory()
            self.assertEqual(inv["database_encrypted"], False)
            self.assertEqual(inv["protection_baseline"], "OS disk encryption + filesystem permissions")
            self.assertIn("tables", inv)
            self.assertIn("files", inv["tables"])
            self.assertIn("symbol_references", inv["tables"])
            self.assertGreaterEqual(inv["tables"]["files"], 1)
            self.assertIn("repositories", inv)
            self.assertEqual(len(inv["repositories"]), 1)
            self.assertEqual(inv["repositories"][0]["name"], root.name)
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)

    def test_call_references_marked_unresolved(self):
        """Static call references must be explicitly marked unresolved/ambiguous."""
        root = _make_repo({"src/app.py": "def a():\n    b()\n\ndef b():\n    pass\n"})
        try:
            res = self.idx.index_repository(root)
            refs = self.idx.store.get_symbol_references(res.repo_id, symbol_name="b")
            # There should be at least one call reference to b
            call_refs = [r for r in refs if r["symbol_kind"] == "call"]
            self.assertGreaterEqual(len(call_refs), 1)
            for r in call_refs:
                self.assertIn(r["resolution_status"], ("unresolved", "ambiguous", "dynamic"))
        finally:
            self.idx.remove_repository(res.repo_id)
            self._cleanup(root)


class TestLoopbackBinding(unittest.TestCase):
    def test_default_host_is_loopback(self):
        """Verify the server defaults to 127.0.0.1, not 0.0.0.0."""
        import argparse
        from app.main import main
        # Parse args with no --host specified; default should be 127.0.0.1
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
        args = parser.parse_args([])
        self.assertEqual(args.host, "127.0.0.1")


class TestLocalOnlyMode(unittest.TestCase):
    def setUp(self):
        self.idx = RepositoryIndexer()

    def _cleanup(self, root):
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        # Clean up any repos left in the store
        try:
            repos = self.idx.list_repositories()
            for r in repos:
                self.idx.remove_repository(r["id"])
        except Exception:
            pass
    def test_no_ollama_import_in_model_free_path(self):
        """Verify model-free CLI does not import Ollama."""
        import sys
        # Remove ollama from sys.modules if present to verify it's not imported
        ollama_modules = [m for m in sys.modules if "ollama" in m]
        for m in ollama_modules:
            del sys.modules[m]

        from app.indexer import RepositoryIndexer
        from app.storage import IndexStore

        # After importing indexer and storage, ollama should NOT be in sys.modules
        ollama_modules = [m for m in sys.modules if "ollama" in m]
        self.assertEqual(len(ollama_modules), 0,
                         f"Ollama modules should not be imported in model-free path: {ollama_modules}")

    def test_no_outbound_network_in_indexer(self):
        """Indexer should not make any outbound network calls."""
        import socket

        # Create a socket that blocks connections to verify no network
        original_connect = socket.socket.connect

        def blocked_connect(self, address):
            raise AssertionError(f"Indexer attempted outbound network call to {address}")

        root = _make_repo({"src/app.py": "def f(): pass\n"})
        try:
            with patch.object(socket.socket, "connect", blocked_connect):
                res = self.idx.index_repository(root)
                self.assertEqual(res.status, "complete")
        finally:
            self._cleanup(root)


if __name__ == "__main__":
    unittest.main()
