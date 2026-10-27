"""Integration tests for LangSmith observability (app/rag/tracing.py, app/langsmith_config.py).

Tests cover:
- Tracing disabled (default)
- Missing credentials
- Tracing enabled with mocked SDK
- Safe metadata propagation
- Secret and source-content redaction
- Tracing failures not breaking RAG responses
- Correct instrumentation of each pipeline stage

No LangSmith credentials or network access required.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from app.langsmith_config import (
    LangSmithConfig,
    get_langsmith_config,
    safe_repo_id,
    sanitize_for_trace,
)
from app.rag.tracing import (
    RAGTracer,
    RAGTraceMetadata,
    get_tracer,
    reset_tracer,
    tracing_diagnostics,
)


class TestLangSmithConfig(unittest.TestCase):
    """Tests for LangSmith configuration."""

    def test_default_disabled(self):
        """Tracing is disabled by default."""
        config = LangSmithConfig()
        self.assertFalse(config.tracing_enabled)
        self.assertFalse(config.is_ready())
        self.assertFalse(config.should_capture_io())

    def test_from_env_enabled(self):
        """Tracing enabled via environment variable."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            config = LangSmithConfig.from_env()
            self.assertTrue(config.tracing_enabled)
            self.assertFalse(config.is_ready())  # no API key

    def test_from_env_with_api_key(self):
        """Tracing enabled with API key → ready."""
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "test-key-123",
                "LANGSMITH_PROJECT": "sherlock-test",
            },
        ):
            config = LangSmithConfig.from_env()
            self.assertTrue(config.tracing_enabled)
            self.assertTrue(config.is_ready())
            self.assertEqual(config.project, "sherlock-test")

    def test_from_env_capture_io(self):
        """Capture I/O is opt-in via environment variable."""
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "SHERLOCK_LANGSMITH_CAPTURE_IO": "true",
            },
        ):
            config = LangSmithConfig.from_env()
            self.assertTrue(config.should_capture_io())

    def test_from_env_capture_io_default_false(self):
        """Capture I/O defaults to False."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            config = LangSmithConfig.from_env()
            self.assertFalse(config.should_capture_io())

    def test_safe_repo_id_truncation(self):
        """Repository IDs are truncated for pseudonymization."""
        long_id = "a" * 50
        result = safe_repo_id(long_id)
        self.assertEqual(len(result), 12)

    def test_safe_repo_id_empty(self):
        """Empty repo ID returns 'unknown-repo'."""
        self.assertEqual(safe_repo_id(""), "unknown-repo")
        self.assertEqual(safe_repo_id(None), "unknown-repo")

    def test_sanitize_for_trace_redacts_paths(self):
        """File paths are redacted in trace text."""
        result = sanitize_for_trace("/Users/alisha/secret/file.py")
        self.assertIn("[PATH]", result)
        self.assertNotIn("/Users/alisha", result)

    def test_sanitize_for_trace_redacts_secrets(self):
        """Secret-like patterns are redacted."""
        result = sanitize_for_trace("api_key=sk-1234567890abcdef")
        self.assertIn("[REDACTED]", result)
        self.assertNotIn("sk-1234567890abcdef", result)

    def test_sanitize_for_trace_none(self):
        """None input returns None."""
        self.assertIsNone(sanitize_for_trace(None))

    def test_sanitize_for_trace_truncation(self):
        """Long text is truncated to 200 chars."""
        long_text = "x" * 500
        result = sanitize_for_trace(long_text)
        self.assertEqual(len(result), 200)

    def test_get_langsmith_config_default(self):
        """get_langsmith_config returns default (disabled) config."""
        config = get_langsmith_config()
        self.assertIsInstance(config, LangSmithConfig)
        self.assertFalse(config.tracing_enabled)


class TestTracingDiagnostics(unittest.TestCase):
    """Tests for tracing diagnostics."""

    def test_diagnostics_disabled(self):
        """Diagnostics when tracing is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            # Need to clear LANGSMITH_TRACING if set
            os.environ.pop("LANGSMITH_TRACING", None)
            diag = tracing_diagnostics()
            self.assertFalse(diag["tracing_enabled"])
            self.assertEqual(diag["tracing_status"], "disabled")
            self.assertTrue(diag["langsmith_installed"])

    def test_diagnostics_enabled_no_key(self):
        """Diagnostics when tracing enabled but no API key."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            os.environ.pop("LANGSMITH_API_KEY", None)
            diag = tracing_diagnostics()
            self.assertTrue(diag["tracing_enabled"])
            self.assertEqual(diag["tracing_status"], "credentials_missing")
            self.assertFalse(diag["api_key_configured"])

    def test_diagnostics_enabled_with_key(self):
        """Diagnostics when tracing enabled with API key."""
        with patch.dict(
            os.environ,
            {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-key"},
        ):
            diag = tracing_diagnostics()
            self.assertTrue(diag["tracing_enabled"])
            self.assertEqual(diag["tracing_status"], "enabled")
            self.assertTrue(diag["api_key_configured"])


class TestTracerNoop(unittest.TestCase):
    """Tests that tracer is a no-op when disabled."""

    def setUp(self):
        reset_tracer()
        self.tracer = RAGTracer(LangSmithConfig())  # disabled

    def test_start_trace_noop(self):
        """start_trace returns a request ID without creating a run."""
        request_id = self.tracer.start_trace("repo-1", "hybrid", 10)
        self.assertIsInstance(request_id, str)
        self.assertTrue(request_id.startswith("rag-"))

    def test_end_trace_noop(self):
        """end_trace does nothing when disabled."""
        metadata = RAGTraceMetadata(
            request_id="req-1",
            repo_id="repo-1",
            retrieval_method="hybrid",
            limit=10,
        )
        # Should not raise
        self.tracer.end_trace(metadata)

    def test_trace_stage_noop(self):
        """trace_stage is a no-op context manager when disabled."""
        with self.tracer.trace_stage("validation"):
            pass  # Should not raise

    def test_enabled_false_when_disabled(self):
        """enabled is False when tracing is disabled."""
        self.assertFalse(self.tracer.enabled)

    def test_trace_function_returns_original(self):
        """trace_function returns original function when disabled."""
        def my_func():
            return 42
        wrapped = self.tracer.trace_function(my_func)
        self.assertIs(wrapped, my_func)
        self.assertEqual(wrapped(), 42)


class TestTracerWithMockedSDK(unittest.TestCase):
    """Tests tracer behavior with a mocked LangSmith SDK."""

    def setUp(self):
        reset_tracer()
        self.config = LangSmithConfig(
            tracing_enabled=True,
            api_key="test-key-123",
            project="sherlock-test",
        )

    def test_tracer_enabled_with_mock(self):
        """tracer.enabled is True when SDK available and tracing enabled."""
        with patch("app.rag.tracing._LANGSMITH_AVAILABLE", True):
            tracer = RAGTracer(self.config)
            self.assertTrue(tracer.enabled)

    def test_start_trace_creates_run(self):
        """start_trace calls client.create_run when enabled."""
        mock_client = MagicMock()
        with patch("app.rag.tracing._LANGSMITH_AVAILABLE", True):
            tracer = RAGTracer(self.config)
            tracer._client = mock_client
            request_id = tracer.start_trace("repo-1", "hybrid", 10)
            self.assertIsInstance(request_id, str)
            mock_client.create_run.assert_called_once()

    def test_end_trace_updates_run(self):
        """end_trace calls client.update_run when enabled."""
        mock_client = MagicMock()
        with patch("app.rag.tracing._LANGSMITH_AVAILABLE", True):
            tracer = RAGTracer(self.config)
            tracer._client = mock_client
            tracer._trace_id = "run-123"
            tracer._start_time = 0.0
            metadata = RAGTraceMetadata(
                request_id="req-1",
                repo_id="repo-1",
                retrieval_method="hybrid",
                limit=10,
                abstained=False,
            )
            tracer.end_trace(metadata)
            mock_client.update_run.assert_called_once()

    def test_trace_failure_does_not_break_rag(self):
        """Tracing failures are caught and do not propagate."""
        mock_client = MagicMock()
        mock_client.create_run.side_effect = Exception("LangSmith down")
        with patch("app.rag.tracing._LANGSMITH_AVAILABLE", True):
            tracer = RAGTracer(self.config)
            tracer._client = mock_client
            # Should not raise
            request_id = tracer.start_trace("repo-1", "hybrid", 10)
            self.assertIsInstance(request_id, str)

    def test_safe_metadata_no_source_code(self):
        """Trace metadata does not include source code or full chunks."""
        metadata = RAGTraceMetadata(
            request_id="req-1",
            repo_id="my-repo",
            retrieval_method="hybrid",
            limit=10,
            num_chunks_retrieved=5,
            num_chunks_used=3,
            context_tokens=450,
            retrieval_latency_ms=12.5,
            total_latency_ms=150.0,
        )
        d = metadata.to_dict()
        # No source code, no file contents, no full paths
        self.assertNotIn("content", str(d))
        self.assertNotIn("def ", str(d))
        self.assertNotIn("class ", str(d))
        # Safe fields present
        self.assertEqual(d["num_chunks_retrieved"], 5)
        self.assertEqual(d["context_tokens"], 450)


class TestGlobalTracer(unittest.TestCase):
    """Tests for the global tracer instance."""

    def setUp(self):
        reset_tracer()

    def test_get_tracer_returns_instance(self):
        """get_tracer returns a RAGTracer instance."""
        tracer = get_tracer()
        self.assertIsInstance(tracer, RAGTracer)

    def test_get_tracer_singleton(self):
        """get_tracer returns the same instance."""
        t1 = get_tracer()
        t2 = get_tracer()
        self.assertIs(t1, t2)

    def test_reset_tracer(self):
        """reset_tracer clears the global instance."""
        get_tracer()
        reset_tracer()
        # After reset, get_tracer should create a new instance
        tracer = get_tracer()
        self.assertIsInstance(tracer, RAGTracer)


class TestTracerMetadata(unittest.TestCase):
    """Tests for RAGTraceMetadata."""

    def test_to_dict_flat(self):
        """to_dict produces a flat dict."""
        metadata = RAGTraceMetadata(
            request_id="req-1",
            repo_id="my-repo",
            retrieval_method="semantic",
            limit=5,
            num_chunks_retrieved=3,
            num_chunks_used=2,
            context_tokens=200,
            retrieval_latency_ms=10.0,
            assembly_latency_ms=5.0,
            generation_latency_ms=100.0,
            citation_validation_latency_ms=2.0,
            total_latency_ms=117.0,
            model_name="qwen3-coder:30b",
            citations_valid=1,
            citations_total=1,
            abstained=False,
        )
        d = metadata.to_dict()
        self.assertEqual(d["request_id"], "req-1")
        self.assertEqual(d["repo_id"], "my-repo")
        self.assertEqual(d["retrieval_method"], "semantic")
        self.assertEqual(d["num_chunks_retrieved"], 3)
        self.assertEqual(d["total_latency_ms"], 117.0)
        self.assertEqual(d["model_name"], "qwen3-coder:30b")
        self.assertFalse(d["abstained"])

    def test_to_dict_redacts_abstention_reason(self):
        """abstention_reason is sanitized in to_dict."""
        metadata = RAGTraceMetadata(
            request_id="req-1",
            repo_id="my-repo",
            retrieval_method="hybrid",
            limit=10,
            abstained=True,
            abstention_reason="/Users/alisha/secret/path not found",
        )
        d = metadata.to_dict()
        self.assertIn("[PATH]", d["abstention_reason"])
        self.assertNotIn("/Users/alisha", d["abstention_reason"])


if __name__ == "__main__":
    unittest.main()
