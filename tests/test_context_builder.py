"""Unit tests for context assembly (app/rag/context_builder.py).

Uses deterministic fixtures - no model downloads, no Ollama, no network.
"""

from __future__ import annotations

import unittest

from app.rag.context_builder import (
    ChunkSource,
    ContextBundle,
    assemble_context,
    build_citation_map,
    count_tokens,
    format_context_for_prompt,
)


def _make_chunk(
    chunk_id: str,
    content: str,
    file_path: str = "src/calculator.py",
    start_line: int = 1,
    end_line: int = 5,
    rank: int = 1,
    score: float = 0.9,
    method: str = "hybrid",
) -> dict:
    return {
        "id": chunk_id,
        "repo_id": "repo-test",
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        "chunk_type": "code",
        "lexical_rank": rank,
        "semantic_rank": rank,
        "lexical_score": score,
        "semantic_score": score,
        "rrf_score": score,
    }


class TestContextBuilder(unittest.TestCase):
    """Tests for context assembly: dedup, ranking, source IDs, budget."""

    def test_assemble_basic(self):
        """Basic context assembly returns ranked chunks with source IDs."""
        results = [
            _make_chunk("c1", "def add(a, b): return a + b", rank=1, score=0.9),
            _make_chunk("c2", "def multiply(a, b): return a * b", rank=2, score=0.8),
            _make_chunk("c3", "class Calculator: pass", rank=3, score=0.7),
        ]
        bundle = assemble_context(results, "how to add", max_tokens=4000)

        self.assertEqual(len(bundle.sources), 3)
        self.assertEqual(bundle.sources[0].source_id, "S1")
        self.assertEqual(bundle.sources[1].source_id, "S2")
        self.assertEqual(bundle.sources[2].source_id, "S3")
        self.assertFalse(bundle.truncated)
        self.assertFalse(bundle.budget_exceeded)
        self.assertGreater(bundle.total_tokens, 0)

    def test_deduplication(self):
        """Duplicate chunk IDs are removed, first occurrence kept."""
        results = [
            _make_chunk("c1", "def add(a, b): return a + b", rank=1, score=0.9),
            _make_chunk("c1", "def add(a, b): return a + b", rank=1, score=0.9),
            _make_chunk("c2", "def multiply(a, b): return a * b", rank=2, score=0.8),
        ]
        bundle = assemble_context(results, "add", max_tokens=4000)
        self.assertEqual(len(bundle.sources), 2)
        source_ids = [s.source_id for s in bundle.sources]
        self.assertEqual(source_ids, ["S1", "S2"])

    def test_empty_results(self):
        """Empty results produce an empty bundle."""
        bundle = assemble_context([], "add", max_tokens=4000)
        self.assertEqual(len(bundle.sources), 0)
        self.assertFalse(bundle.truncated)
        self.assertFalse(bundle.budget_exceeded)
        self.assertEqual(bundle.total_tokens, 0)

    def test_token_budget_truncation(self):
        """Budget truncation excludes chunks that don't fit."""
        # Each chunk is ~30 words (~22 tokens), budget of 20 fits only ~1 chunk
        results = [
            _make_chunk(
                "c1",
                "def add(a, b):\n    return a + b\n" * 3,
                rank=1,
                score=0.9,
            ),
            _make_chunk(
                "c2",
                "def multiply(a, b):\n    return a * b\n" * 3,
                rank=2,
                score=0.8,
            ),
            _make_chunk(
                "c3",
                "def subtract(a, b):\n    return a - b\n" * 3,
                rank=3,
                score=0.7,
            ),
            _make_chunk(
                "c4",
                "def divide(a, b):\n    return a / b\n" * 3,
                rank=4,
                score=0.6,
            ),
        ]
        bundle = assemble_context(results, "math", max_tokens=20)

        # Should have truncated - not all chunks fit in 20 tokens
        self.assertTrue(bundle.truncated)
        self.assertLess(len(bundle.sources), 4)
        self.assertGreaterEqual(len(bundle.sources), 1)

    def test_source_metadata_preserved(self):
        """Source metadata (path, lines) is preserved in bundle."""
        results = [
            _make_chunk(
                "c1",
                "def add(a, b): return a + b",
                file_path="src/calculator.py",
                start_line=10,
                end_line=12,
                rank=1,
                score=0.9,
            )
        ]
        bundle = assemble_context(results, "add", max_tokens=4000)
        s = bundle.sources[0]
        self.assertEqual(s.file_path, "src/calculator.py")
        self.assertEqual(s.start_line, 10)
        self.assertEqual(s.end_line, 12)

    def test_format_context_for_prompt(self):
        """Formatted context includes source IDs, paths, line ranges."""
        results = [
            _make_chunk(
                "c1",
                "def add(a, b): return a + b",
                file_path="src/calculator.py",
                start_line=10,
                end_line=12,
                rank=1,
                score=0.9,
            )
        ]
        bundle = assemble_context(results, "add", max_tokens=4000)
        formatted = format_context_for_prompt(bundle)
        self.assertIn("[S1]", formatted)
        self.assertIn("src/calculator.py:10-12:", formatted)
        self.assertIn("def add(a, b): return a + b", formatted)

    def test_build_citation_map(self):
        """Citation map maps source_id to trusted metadata."""
        results = [
            _make_chunk(
                "c1",
                "def add(a, b): return a + b",
                file_path="src/calculator.py",
                start_line=10,
                end_line=12,
                rank=1,
                score=0.9,
            )
        ]
        bundle = assemble_context(results, "add", max_tokens=4000)
        cmap = build_citation_map(bundle)
        self.assertIn("S1", cmap)
        self.assertEqual(cmap["S1"]["file_path"], "src/calculator.py")
        self.assertEqual(cmap["S1"]["start_line"], 10)
        self.assertEqual(cmap["S1"]["end_line"], 12)

    def test_count_tokens(self):
        """Token counting returns a positive integer for non-empty text."""
        count = count_tokens("hello world")
        self.assertGreater(count, 0)
        self.assertEqual(count_tokens(""), 0)


if __name__ == "__main__":
    unittest.main()
