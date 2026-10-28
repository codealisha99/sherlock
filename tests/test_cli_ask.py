"""CLI tests for the `ask` RAG command.

Uses deterministic mocks for the RAG pipeline and indexer. No model downloads,
no Ollama server, no network access.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from unittest.mock import patch

from app.main import cmd_ask, main as cli_main


class _Args:
    """Simple namespace for CLI arguments."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Default stream to False for backward compatibility
        self.stream = self.__dict__.get("stream", False)


class _FakeRepo:
    """Fake repository row from the store."""

    def __init__(self, repo_id="repo-test"):
        self.id = repo_id
        self.name = "test-repo"
        self.path = "/tmp/test-repo"
        self.status = "complete"
        self.file_count = 2
        self.total_lines = 100


class _FakeStore:
    """Fake store that returns a repo and delegates to a fake indexer."""

    def __init__(self, repo_id="repo-test"):
        self.repo = _FakeRepo(repo_id)

    def connect(self):
        pass

    def close(self):
        pass

    def get_repository(self, repo_id):
        if repo_id == self.repo.id:
            return {
                "id": self.repo.id,
                "name": self.repo.name,
                "path": self.repo.path,
                "status": self.repo.status,
                "file_count": self.repo.file_count,
                "total_lines": self.repo.total_lines,
            }
        return None


class _FakeIndexer:
    """Fake indexer with a fake store."""

    def __init__(self, repo_id="repo-test"):
        self.store = _FakeStore(repo_id)


class _FakeRAGResult:
    """Fake RAG result for testing CLI output."""

    def __init__(
        self,
        abstained=False,
        abstention_reason=None,
        generation_error=None,
        retrieval_error=None,
        answer="The add function is defined in calculator.py [S1].",
        citations=None,
        retrieval_method="hybrid",
        num_chunks_retrieved=2,
        num_chunks_used=2,
    ):
        self.abstained = abstained
        self.abstention_reason = abstention_reason
        self.generation_error = generation_error
        self.retrieval_error = retrieval_error
        self.answer = answer
        self.citations = citations or []
        self.retrieval_method = retrieval_method
        self.num_chunks_retrieved = num_chunks_retrieved
        self.num_chunks_used = num_chunks_used


class _FakeCitation:
    """Fake citation for testing CLI output."""

    def __init__(
        self,
        source_id="S1",
        file_path="src/calculator.py",
        start_line=1,
        end_line=2,
        chunk_type="code",
    ):
        self.source_id = source_id
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type


class TestCLIAsk(unittest.TestCase):
    """Tests for the `ask` CLI command."""

    def _make_pipeline_mock(self, fake_result):
        """Create a mocked RAGPipeline class that returns fake_result."""
        mock_pipeline_class = patch("app.rag.pipeline.RAGPipeline")
        mock_pipeline = mock_pipeline_class.start()
        mock_pipeline.return_value.ask.return_value = fake_result
        return mock_pipeline_class, mock_pipeline

    def test_ask_grounded_answer(self):
        """A grounded answer prints the answer and citations with exit code 0."""
        fake_result = _FakeRAGResult(
            abstained=False,
            answer="Use the add function [S1].",
            citations=[_FakeCitation()],
            retrieval_method="hybrid",
            num_chunks_retrieved=2,
            num_chunks_used=2,
        )

        with patch("app.main._get_indexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(
                        _Args(
                            repo_id="repo-test",
                            question="How do I add?",
                            retrieval_method="hybrid",
                            limit=10,
                        )
                    )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Answer", output)
        self.assertIn("Use the add function [S1].", output)
        self.assertIn("Citations", output)
        self.assertIn("[S1]", output)
        self.assertIn("src/calculator.py:1-2", output)

    def test_ask_abstention(self):
        """Abstention prints a clear reason and returns exit code 2."""
        fake_result = _FakeRAGResult(
            abstained=True,
            abstention_reason="No relevant content found in the indexed repository.",
            answer="",
            citations=[],
        )

        with patch("app.main._get_indexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cmd_ask(
                        _Args(
                            repo_id="repo-test",
                            question="Anything?",
                            retrieval_method="hybrid",
                            limit=10,
                        )
                    )

        self.assertEqual(exit_code, 2)
        output = stdout.getvalue()
        self.assertIn("Abstained", output)
        self.assertIn("No relevant content found", output)

    def test_ask_nonexistent_repo(self):
        """Nonexistent repository returns exit code 1."""
        with patch("app.main._get_indexer", return_value=_FakeIndexer()):
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cmd_ask(_Args(repo_id="bad-repo", question="test", retrieval_method="hybrid", limit=10))

        self.assertEqual(exit_code, 1)
        stderr_output = stderr.getvalue()
        self.assertIn("Repository not found", stderr_output)

    def test_ask_no_citations(self):
        """An answer without validated citations prints a notice."""
        fake_result = _FakeRAGResult(
            abstained=False,
            answer="I cannot answer from the evidence.",
            citations=[],
            retrieval_method="lexical",
            num_chunks_retrieved=1,
            num_chunks_used=1,
        )

        with patch("app.main._get_indexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                stdout = StringIO()
                with redirect_stdout(stdout):
                    exit_code = cmd_ask(
                        _Args(
                            repo_id="repo-test",
                            question="obscure question",
                            retrieval_method="lexical",
                            limit=5,
                        )
                    )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("No citations", output)
        self.assertIn("Chunks retrieved: 1, used: 1", output)

    def test_main_dispatches_to_ask(self):
        """The `main` entrypoint dispatches `ask` to cmd_ask."""
        fake_result = _FakeRAGResult(abstained=False, answer="Answer text.", citations=[])

        with patch("app.main._get_indexer", return_value=_FakeIndexer()):
            with patch("app.rag.pipeline.RAGPipeline") as mock_pipeline_class:
                mock_pipeline = mock_pipeline_class.return_value
                mock_pipeline.ask.return_value = fake_result

                old_argv = sys.argv
                sys.argv = ["app.main", "ask", "--repo-id", "repo-test", "--question", "hello"]
                try:
                    stdout = StringIO()
                    with redirect_stdout(stdout):
                        exit_code = cli_main()
                finally:
                    sys.argv = old_argv

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Answer text", output)


if __name__ == "__main__":
    unittest.main()
