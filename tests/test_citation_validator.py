"""Unit tests for citation validation (app/rag/citation_validator.py).

Tests deterministic citation validation without any model or network access.
"""

from __future__ import annotations

import unittest

from app.rag.citation_validator import (
    Citation,
    extract_citation_ids,
    validate_citations,
)


class TestExtractCitationIds(unittest.TestCase):
    """Tests for extract_citation_ids()."""

    def test_extracts_s1_s2_s3(self):
        """Extracts [S1], [S2], [S3] in order."""
        ids = extract_citation_ids("The answer uses [S1] and [S2] and [S3].")
        self.assertEqual(ids, ["S1", "S2", "S3"])

    def test_deduplicates(self):
        """Duplicate citation IDs are deduplicated."""
        ids = extract_citation_ids("Use [S1] and [S1] again.")
        self.assertEqual(ids, ["S1"])

    def test_ignores_malformed(self):
        """Malformed citation tokens are ignored."""
        ids = extract_citation_ids("Use [S1] and [Sabc] and [S0] and [S2].")
        # [Sabc] doesn't match \d+, [S0] matches as "0", [S1] and [S2] match
        self.assertEqual(ids, ["S1", "S0", "S2"])

    def test_empty_text(self):
        """Empty text returns no citations."""
        self.assertEqual(extract_citation_ids(""), [])

    def test_no_citations(self):
        """Text without citations returns empty list."""
        self.assertEqual(extract_citation_ids("No citations here."), [])


class TestValidateCitations(unittest.TestCase):
    """Tests for validate_citations()."""

    def _make_map(self):
        return {
            "S1": {
                "file_path": "src/calculator.py",
                "start_line": 1,
                "end_line": 5,
                "chunk_type": "code",
                "content": "class Calculator:\n    def add(self, a, b):\n        return a + b\n",
                "repo_id": "repo-test",
            },
            "S2": {
                "file_path": "src/operations.py",
                "start_line": 10,
                "end_line": 15,
                "chunk_type": "code",
                "content": "def multiply(a, b):\n    return a * b\n",
                "repo_id": "repo-test",
            },
        }

    def test_valid_citations(self):
        """Valid citations referencing real sources are accepted."""
        cmap = self._make_map()
        result = validate_citations("Use [S1] and [S2].", cmap)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].source_id, "S1")
        self.assertEqual(result[0].file_path, "src/calculator.py")
        self.assertEqual(result[1].source_id, "S2")
        self.assertEqual(result[1].file_path, "src/operations.py")

    def test_unknown_source_rejected(self):
        """Citations to non-existent sources are rejected."""
        cmap = self._make_map()
        result = validate_citations("The answer is [S99].", cmap)
        self.assertEqual(len(result), 0)

    def test_empty_text(self):
        """Empty answer returns no citations."""
        cmap = self._make_map()
        self.assertEqual(validate_citations("", cmap), [])
        self.assertEqual(validate_citations(None, cmap), [])

    def test_cross_repo_rejected(self):
        """Citations from a different repository are rejected when repo_id given."""
        cmap = self._make_map()
        result = validate_citations("Use [S1].", cmap, repo_id="other-repo")
        self.assertEqual(len(result), 0)

    def test_same_repo_accepted(self):
        """Citations from the same repository are accepted."""
        cmap = self._make_map()
        result = validate_citations("Use [S1].", cmap, repo_id="repo-test")
        self.assertEqual(len(result), 1)

    def test_unsafe_path_rejected(self):
        """Unsafe file paths (absolute, traversal) are rejected."""
        cmap = {
            "S1": {
                "file_path": "/etc/passwd",
                "start_line": 1,
                "end_line": 1,
                "content": "root:x:0:0:",
                "repo_id": "repo-test",
            },
            "S2": {
                "file_path": "../../etc/passwd",
                "start_line": 1,
                "end_line": 1,
                "content": "root:x:0:0:",
                "repo_id": "repo-test",
            },
            "S3": {
                "file_path": "src/safe.py",
                "start_line": 1,
                "end_line": 1,
                "content": "safe",
                "repo_id": "repo-test",
            },
        }
        result = validate_citations("Use [S1] and [S2] and [S3].", cmap)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_id, "S3")

    def test_invalid_line_range_rejected(self):
        """Invalid line ranges (start < 1, end < start) are rejected."""
        cmap = {
            "S1": {
                "file_path": "src/x.py",
                "start_line": 0,
                "end_line": 5,
                "content": "x",
                "repo_id": "repo-test",
            },
            "S2": {
                "file_path": "src/x.py",
                "start_line": 5,
                "end_line": 1,
                "content": "x",
                "repo_id": "repo-test",
            },
            "S3": {
                "file_path": "src/x.py",
                "start_line": 1,
                "end_line": 5,
                "content": "x",
                "repo_id": "repo-test",
            },
        }
        result = validate_citations("Use [S1] and [S2] and [S3].", cmap)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_id, "S3")

    def test_citation_object_fields(self):
        """Citation objects have all expected fields."""
        cmap = self._make_map()
        result = validate_citations("Use [S1].", cmap)
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertIsInstance(c, Citation)
        self.assertEqual(c.source_id, "S1")
        self.assertEqual(c.file_path, "src/calculator.py")
        self.assertEqual(c.start_line, 1)
        self.assertEqual(c.end_line, 5)
        self.assertEqual(c.chunk_type, "code")
        self.assertIn("class Calculator", c.content)


if __name__ == "__main__":
    unittest.main()
