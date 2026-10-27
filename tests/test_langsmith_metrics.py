"""Unit tests for LangSmith deterministic metrics (app/rag/langsmith_metrics.py).

All metrics are computed locally - no LangSmith credentials or network
access required.
"""

from __future__ import annotations

import unittest

from app.rag.langsmith_metrics import (
    RAGRequestMetrics,
    compute_abstention_metrics,
    citation_coverage,
    citation_evidence_membership,
    citation_syntax_validity,
    compute_request_metrics,
    groundedness,
    mrr,
    ndcg_at_k,
    recall_at_k,
    unsupported_claim_rate,
)


class TestRetrievalMetrics(unittest.TestCase):
    """Tests for retrieval quality metrics."""

    def test_recall_at_k_perfect(self):
        """All relevant docs retrieved at k=5."""
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 5), 1.0)

    def test_recall_at_k_partial(self):
        """Only some relevant docs in top-k."""
        self.assertEqual(recall_at_k(["a", "x", "y"], {"a", "b", "c"}, 3), 1.0 / 3.0)

    def test_recall_at_k_empty_relevant(self):
        """No relevant docs → recall 0.0."""
        self.assertEqual(recall_at_k(["a", "b"], set(), 5), 0.0)

    def test_recall_at_k_k_positive(self):
        """k must be >= 1."""
        with self.assertRaises(ValueError):
            recall_at_k(["a"], {"a"}, 0)

    def test_mrr_first_result(self):
        """First result relevant → MRR 1.0."""
        self.assertEqual(mrr(["a", "b", "c"], {"a"}), 1.0)

    def test_mrr_second_result(self):
        """Second result relevant → MRR 0.5."""
        self.assertEqual(mrr(["x", "a", "b"], {"a"}), 0.5)

    def test_mrr_no_relevant(self):
        """No relevant docs → MRR 0.0."""
        self.assertEqual(mrr(["x", "y", "z"], {"a"}), 0.0)

    def test_ndcg_at_k_perfect(self):
        """All relevant at top → nDCG 1.0."""
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3), 1.0)

    def test_ndcg_at_k_empty_relevant(self):
        """No relevant docs → nDCG 0.0."""
        self.assertEqual(ndcg_at_k(["a", "b"], set(), 3), 0.0)


class TestCitationMetrics(unittest.TestCase):
    """Tests for citation quality metrics."""

    def test_citation_syntax_validity_valid(self):
        """All [S<n>] tokens are syntactically valid."""
        result = citation_syntax_validity("Use [S1] and [S2].")
        self.assertEqual(result["total_citations"], 2)
        self.assertEqual(result["valid_syntax"], 2)
        self.assertEqual(result["invalid_syntax"], 0)
        self.assertEqual(result["validity_rate"], 1.0)

    def test_citation_syntax_validity_none(self):
        """No citations → validity_rate 1.0."""
        result = citation_syntax_validity("No citations here.")
        self.assertEqual(result["total_citations"], 0)
        self.assertEqual(result["validity_rate"], 1.0)

    def test_citation_evidence_membership_all_in(self):
        """All cited sources are in evidence map."""
        citation_map = {
            "S1": {"file_path": "src/a.py", "start_line": 1, "end_line": 5, "repo_id": "repo"},
            "S2": {"file_path": "src/b.py", "start_line": 10, "end_line": 15, "repo_id": "repo"},
        }
        result = citation_evidence_membership("Use [S1] and [S2].", citation_map)
        self.assertEqual(result["total_cited"], 2)
        self.assertEqual(result["in_evidence"], 2)
        self.assertEqual(result["membership_rate"], 1.0)

    def test_citation_evidence_membership_none_in(self):
        """No cited sources in evidence map."""
        citation_map = {"S1": {"file_path": "src/a.py", "start_line": 1, "end_line": 5, "repo_id": "repo"}}
        result = citation_evidence_membership("Use [S99].", citation_map)
        self.assertEqual(result["total_cited"], 1)
        self.assertEqual(result["in_evidence"], 0)
        self.assertEqual(result["membership_rate"], 0.0)

    def test_citation_evidence_membership_cross_repo(self):
        """Cross-repo citations are rejected when repo_id given."""
        citation_map = {
            "S1": {"file_path": "src/a.py", "start_line": 1, "end_line": 5, "repo_id": "other-repo"},
        }
        result = citation_evidence_membership("Use [S1].", citation_map, repo_id="my-repo")
        self.assertEqual(result["cross_repo_rejected"], 1)
        self.assertEqual(result["in_evidence"], 0)

    def test_citation_coverage(self):
        """Citation coverage estimates cited segments."""
        result = citation_coverage("First sentence [S1]. Second sentence.", [])
        self.assertEqual(result["total_segments"], 2)
        self.assertEqual(result["cited_segments"], 1)

    def test_citation_coverage_empty(self):
        """Empty answer → zero segments."""
        result = citation_coverage("", [])
        self.assertEqual(result["total_segments"], 0)
        self.assertEqual(result["coverage_rate"], 0.0)


class TestAnswerQualityMetrics(unittest.TestCase):
    """Tests for answer quality metrics."""

    def test_groundedness_all_valid(self):
        """All attempted citations validated."""
        class FakeCitation:
            pass
        result = groundedness([FakeCitation(), FakeCitation()], 2)
        self.assertEqual(result["valid_citations"], 2)
        self.assertEqual(result["total_attempted"], 2)
        self.assertEqual(result["groundedness_rate"], 1.0)

    def test_groundedness_none_valid(self):
        """No citations validated."""
        result = groundedness([], 3)
        self.assertEqual(result["valid_citations"], 0)
        self.assertEqual(result["total_attempted"], 3)
        self.assertEqual(result["groundedness_rate"], 0.0)

    def test_groundedness_no_attempts(self):
        """No citation attempts → groundedness 1.0."""
        result = groundedness([], 0)
        self.assertEqual(result["groundedness_rate"], 1.0)

    def test_unsupported_claim_rate(self):
        """Unsupported claim rate estimation."""
        result = unsupported_claim_rate("First [S1]. Second.", [])
        self.assertEqual(result["total_segments"], 2)
        self.assertEqual(result["unsupported_segments"], 1)
        self.assertEqual(result["unsupported_rate"], 0.5)


class TestAbstentionMetrics(unittest.TestCase):
    """Tests for abstention metrics."""

    def test_abstention_correct(self):
        """Abstained and not answerable → correct abstention."""
        result = compute_abstention_metrics(True, False, "No evidence")
        self.assertTrue(result["correct_abstention"])
        self.assertFalse(result["incorrect_abstention"])
        self.assertEqual(result["abstention_precision"], 1.0)
        self.assertEqual(result["abstention_recall"], 1.0)

    def test_abstention_incorrect(self):
        """Abstained but answerable → incorrect abstention."""
        result = compute_abstention_metrics(True, True, "No evidence")
        self.assertTrue(result["incorrect_abstention"])
        self.assertFalse(result["correct_abstention"])

    def test_abstention_no_labels(self):
        """No answerability label → no precision/recall."""
        result = compute_abstention_metrics(True, None, "No evidence")
        self.assertTrue(result["abstained"])
        self.assertNotIn("abstention_precision", result)


class TestComputeRequestMetrics(unittest.TestCase):
    """Tests for compute_request_metrics aggregation."""

    class _FakeResult:
        def __init__(self):
            self.query = "test"
            self.repo_id = "repo-test"
            self.answer = "The answer is [S1]."
            self.abstained = False
            self.abstention_reason = None
            self.retrieval_method = "hybrid"
            self.num_chunks_retrieved = 2
            self.num_chunks_used = 1
            self.citations = []
            self.sources = []
            self.retrieval_error = None
            self.generation_error = None

    class _FakeSource:
        def __init__(self, source_id="S1", repo_id="repo-test", file_path="src/a.py", start_line=1, end_line=5):
            self.source_id = source_id
            self.repo_id = repo_id
            self.file_path = file_path
            self.start_line = start_line
            self.end_line = end_line

    class _FakeLatency:
        def __init__(self):
            self.retrieval_ms = 10.0
            self.assembly_ms = 5.0
            self.generation_ms = 100.0
            self.citation_validation_ms = 2.0
            self.total_ms = 117.0

        def to_dict(self):
            return {
                "retrieval_latency_ms": self.retrieval_ms,
                "assembly_latency_ms": self.assembly_ms,
                "generation_latency_ms": self.generation_ms,
                "citation_validation_latency_ms": self.citation_validation_ms,
                "total_latency_ms": self.total_ms,
            }

    def test_compute_request_metrics_basic(self):
        """Basic metric aggregation without relevance labels."""
        result = self._FakeResult()
        latency = self._FakeLatency()
        metrics = compute_request_metrics(
            request_id="req-123",
            repo_id="repo-test",
            retrieval_method="hybrid",
            result=result,
            latency=latency,
        )
        self.assertEqual(metrics.request_id, "req-123")
        self.assertEqual(metrics.repo_id, "repo-test")
        self.assertEqual(metrics.retrieval_method, "hybrid")
        self.assertFalse(metrics.abstained)
        self.assertEqual(metrics.num_chunks_retrieved, 2)
        self.assertEqual(metrics.num_chunks_used, 1)
        self.assertEqual(metrics.latency.total_ms, 117.0)

    def test_compute_request_metrics_with_relevance(self):
        """Metric aggregation with relevance labels."""
        result = self._FakeResult()
        result.sources = [
            self._FakeSource(source_id="S1", repo_id="repo-test", file_path="src/a.py"),
        ]
        latency = self._FakeLatency()
        metrics = compute_request_metrics(
            request_id="req-123",
            repo_id="repo-test",
            retrieval_method="hybrid",
            result=result,
            latency=latency,
            relevance_labels={"S1"},
        )
        self.assertEqual(metrics.retrieval_metrics["recall@1"], 1.0)
        self.assertEqual(metrics.retrieval_metrics["mrr"], 1.0)

    def test_compute_request_metrics_to_dict(self):
        """to_dict produces flat dict for LangSmith metadata."""
        result = self._FakeResult()
        latency = self._FakeLatency()
        metrics = compute_request_metrics(
            request_id="req-123",
            repo_id="repo-test",
            retrieval_method="lexical",
            result=result,
            latency=latency,
        )
        d = metrics.to_dict()
        self.assertEqual(d["request_id"], "req-123")
        self.assertEqual(d["retrieval_method"], "lexical")
        self.assertEqual(d["total_latency_ms"], 117.0)
        self.assertIn("citation_validity_rate", d)
        self.assertIn("citation_membership_rate", d)
        self.assertIn("answer_groundedness_rate", d)


if __name__ == "__main__":
    unittest.main()
