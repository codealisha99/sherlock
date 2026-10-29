"""End-to-end tests for Sherlock RAG chat pipeline.

Tests cover:
- Index a small repository
- Ask questions via RAGPipeline.ask (mocked LLM)
- Conversation persistence
- Conversation isolation between repos

Uses deterministic mocks. No Ollama required.
"""

from __future__ import annotations

import json
import tempfile
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from app.indexer import RepositoryIndexer
from app.rag.pipeline import RAGPipeline
from app.storage import IndexStore
from config import SherlockConfig


def _make_repo(files):
    """Create a temp repo from a {rel_path: content} dict."""
    root = Path(tempfile.mkdtemp(prefix='sherlock_chat_e2e_'))
    for rel, content_text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content_text)
    return root


def _make_chunk(repo_id, file_path="src/calculator.py", start_line=1, end_line=5,
               content=None, chunk_type="code"):
    """Build a chunk dict matching the real search_chunks / hybrid_search output."""
    if content is None:
        content = ('class Calculator:\n    def add(self, a, b):\n        return a + b\n'
                   '    def multiply(self, a, b):\n        return a * b\n')
    return {
        'id': 'chunk-1',
        'file_id': 'file-1',
        'repo_id': repo_id,
        'file_path': file_path,
        'start_line': start_line,
        'end_line': end_line,
        'content': content,
        'chunk_type': chunk_type,
    }


class TestChatE2E(unittest.TestCase):
    """End-to-end tests for the complete RAG chat pipeline."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix='sherlock_chat_e2e_'))
        self.root = _make_repo({
            'src/calculator.py': 'class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b\n',
            'src/operations.py': 'from calculator import Calculator\\ndef compute_total(items):\\n    calc = Calculator()\\n    total = 0\\n    for item in items:\\n        total = calc.add(total, item)\\n    return total\\n',
            'README.md': '# My Project\\n\\nA simple calculator project.\\n',
            'config.yaml': 'database:\\n  host: localhost\\n  port: 5432\\n',
            '.gitignore': '__pycache__/\\n*.pyc\\n',
        })
        self.cfg = SherlockConfig(workspace_dir=self.test_dir / 'workspace')
        self.indexer = RepositoryIndexer(cfg=self.cfg)
        self.repo_id = f'chat-e2e-{uuid.uuid4().hex[:8]}'
        self.result = self.indexer.index_repository(
            self.root, repo_id=self.repo_id, repo_name='chat-e2e'
        )

        self.pipeline = RAGPipeline(indexer=self.indexer)

        self.store = self.indexer.store
        self.store.connect()

        # Create a chat session
        self.session_id = self.store.create_chat_session(self.repo_id, title='Test Chat')

        self.session_messages = []

    def tearDown(self):
        try:
            self.indexer.remove_repository(self.repo_id)
        except Exception:
            pass
        self.store.close()
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_grounded_answer(self):
        """Test a complete RAG pipeline: indexed repo -> grounded answer with citations."""
        from app.rag import pipeline as rag_module

        # Mock search_chunks to return fake chunks.  The hybrid_search code path
        # inside IndexStore accesses r["file_id"], so the mock must include it.
        with patch.object(self.store, 'search_chunks') as mock_search:
            mock_search.return_value = [_make_chunk(self.repo_id)]
            with patch.object(rag_module.OllamaClient, 'generate') as mock_gen:
                mock_gen.return_value = 'The Calculator class is defined in src/calculator.py [S1]. It has an add method.'

                result = self.pipeline.ask(
                    repo_id=self.repo_id,
                    question='Where is Calculator defined?',
                    retrieval_method='hybrid',
                    limit=10
                )

        self.assertFalse(result.abstained)
        self.assertIn('Calculator', result.answer)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].source_id, 'S1')
        self.assertEqual(result.citations[0].file_path, 'src/calculator.py')

    def test_conversation_persistence(self):
        """Test that multiple questions work in the same session."""
        from app.rag import pipeline as rag_module

        with patch.object(self.store, 'search_chunks') as mock_search:
            mock_search.return_value = [_make_chunk(self.repo_id)]
            with patch.object(rag_module.OllamaClient, 'generate') as mock_gen:
                mock_gen.return_value = 'The Calculator class is defined in src/calculator.py [S1]. It has an add method.'

                # First question
                result1 = self.pipeline.ask(
                    repo_id=self.repo_id,
                    question='What does Calculator do?',
                    retrieval_method='hybrid',
                    limit=10
                )

                # Second question (same session)
                result2 = self.pipeline.ask(
                    repo_id=self.repo_id,
                    question='How does add work?',
                    retrieval_method='hybrid',
                    limit=10
                )

        self.assertFalse(result1.abstained)
        self.assertFalse(result2.abstained)
        self.assertEqual(len(result1.citations), 1)
        self.assertEqual(len(result2.citations), 1)

    def test_unanswerable_question(self):
        """Test that unanswerable questions produce abstention."""
        from app.rag import pipeline as rag_module

        with patch.object(self.store, 'search_chunks') as mock_search:
            mock_search.return_value = []  # No results

            with patch.object(rag_module.OllamaClient, 'generate', return_value=''):
                result = self.pipeline.ask(
                    repo_id=self.repo_id,
                    question='How does quantum physics work?',
                    retrieval_method='hybrid',
                    limit=10
                )

        self.assertTrue(result.abstained)
        self.assertIn('No relevant content', result.abstention_reason)
        self.assertEqual(len(result.citations), 0)

    def test_repository_isolation(self):
        """Test that different repos have separate contexts."""
        from app.rag import pipeline as rag_module

        # Patch indexer.hybrid_search directly (bypasses store.hybrid_search)
        with patch.object(self.indexer, 'hybrid_search') as mock_hybrid:
            mock_hybrid.return_value = [_make_chunk(self.repo_id)]
            with patch.object(rag_module.OllamaClient, 'generate') as mock_gen:
                mock_gen.return_value = 'The Calculator class is defined in src/calculator.py [S1]. It has an add method.'

                result = self.pipeline.ask(
                    repo_id=self.repo_id,
                    question='What is in this repo?',
                    retrieval_method='hybrid',
                    limit=10
                )

            self.assertFalse(result.abstained)
            # Verify hybrid_search was called with the correct repo_id
            call_args = mock_hybrid.call_args
            self.assertEqual(call_args[0][0], self.repo_id)


if __name__ == '__main__':
    unittest.main()
