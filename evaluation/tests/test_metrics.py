"""Unit tests for retrieval metrics in evaluation/metrics.py.

Each metric is tested against small, manually verifiable examples to
ensure correctness of the implementation.
"""

import unittest

from evaluation.metrics import (
    average_precision,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    evaluate_query,
    aggregate_metrics,
)


class TestRecallAtK(unittest.TestCase):
    """Tests for recall_at_k."""

    def test_perfect_ranking_all_relevant_retrieved(self):
        """All relevant docs retrieved within top k -> recall 1.0."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 1.0)

    def test_relevant_at_rank_one_only(self):
        """Only first result relevant -> recall = 1/3."""
        retrieved = ["a", "x", "y"]
        relevant = ["a", "b", "c"]
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 1 / 3)

    def test_no_relevant_retrieved(self):
        """No relevant docs in top k -> recall 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 0.0)

    def test_empty_retrieval_list(self):
        """Empty retrieved list -> recall 0.0."""
        retrieved = []
        relevant = ["a", "b"]
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 0.0)

    def test_no_relevant_documents(self):
        """No relevant docs in gold labels -> recall 0.0."""
        retrieved = ["a", "b", "c"]
        relevant = []
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 0.0)

    def test_duplicate_retrieved_ids(self):
        """Duplicate IDs in retrieved list don't inflate recall."""
        retrieved = ["a", "a", "a", "b", "c"]
        relevant = ["a", "b"]
        # a appears once as relevant, b appears once -> 2/2 = 1.0
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 1.0)

    def test_fewer_than_k_retrieved(self):
        """Fewer than k results retrieved, some relevant."""
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c", "d"]
        self.assertEqual(recall_at_k(retrieved, relevant, 10), 0.5)

    def test_k_larger_than_retrieved_list(self):
        """k larger than number of retrieved results."""
        retrieved = ["a"]
        relevant = ["a", "b"]
        self.assertEqual(recall_at_k(retrieved, relevant, 10), 0.5)

    def test_k_must_be_positive(self):
        """k=0 raises ValueError."""
        with self.assertRaises(ValueError):
            recall_at_k(["a"], ["a"], 0)


class TestPrecisionAtK(unittest.TestCase):
    """Tests for precision_at_k."""

    def test_perfect_precision(self):
        """All top k results relevant -> precision 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        self.assertEqual(precision_at_k(retrieved, relevant, 3), 1.0)

    def test_relevant_at_rank_one_only(self):
        """Only first result relevant -> precision 1/3."""
        retrieved = ["a", "x", "y"]
        relevant = ["a", "b"]
        self.assertEqual(precision_at_k(retrieved, relevant, 3), 1 / 3)

    def test_no_relevant_retrieved(self):
        """No relevant docs in top k -> precision 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertEqual(precision_at_k(retrieved, relevant, 3), 0.0)

    def test_empty_retrieval_list(self):
        """Empty retrieved list -> precision 0.0."""
        retrieved = []
        relevant = ["a", "b"]
        self.assertEqual(precision_at_k(retrieved, relevant, 5), 0.0)

    def test_no_relevant_documents_but_retrieved(self):
        """No relevant docs in gold, some retrieved -> precision 0.0."""
        retrieved = ["a", "b"]
        relevant = []
        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.0)

    def test_fewer_than_k_retrieved_denominator_is_k(self):
        """Fewer than k retrieved -> denominator is k (missing count as non-relevant)."""
        retrieved = ["a"]
        relevant = ["a", "b"]
        # 1 relevant out of k=3 positions -> 1/3
        self.assertEqual(precision_at_k(retrieved, relevant, 3), 1 / 3)

    def test_k_must_be_positive(self):
        """k=0 raises ValueError."""
        with self.assertRaises(ValueError):
            precision_at_k(["a"], ["a"], 0)


class TestMrr(unittest.TestCase):
    """Tests for MRR."""

    def test_first_result_relevant(self):
        """First result relevant -> MRR 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b"]
        self.assertEqual(mrr(retrieved, relevant), 1.0)

    def test_second_result_relevant(self):
        """Second result relevant -> MRR 1/2."""
        retrieved = ["x", "a", "b"]
        relevant = ["a", "b"]
        self.assertEqual(mrr(retrieved, relevant), 1 / 2)

    def test_third_result_relevant(self):
        """Third result relevant -> MRR 1/3."""
        retrieved = ["x", "y", "a"]
        relevant = ["a", "b"]
        self.assertEqual(mrr(retrieved, relevant), 1 / 3)

    def test_no_relevant_retrieved(self):
        """No relevant docs retrieved -> MRR 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertEqual(mrr(retrieved, relevant), 0.0)

    def test_empty_retrieved_list(self):
        """Empty retrieved list -> MRR 0.0."""
        retrieved = []
        relevant = ["a"]
        self.assertEqual(mrr(retrieved, relevant), 0.0)

    def test_no_relevant_documents(self):
        """No relevant docs in gold -> MRR 0.0."""
        retrieved = ["a", "b"]
        relevant = []
        self.assertEqual(mrr(retrieved, relevant), 0.0)


class TestAveragePrecision(unittest.TestCase):
    """Tests for average_precision."""

    def test_perfect_ranking(self):
        """All relevant docs at top -> AP 1.0."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]
        # Precision@1=1/1, @2=2/2, @3=3/3 -> avg = 1.0
        self.assertEqual(average_precision(retrieved, relevant), 1.0)

    def test_relevant_at_ranks_1_and_3(self):
        """Relevant at ranks 1 and 3 -> AP = (1/1 + 2/3)/2."""
        retrieved = ["a", "x", "b"]
        relevant = ["a", "b"]
        # (1 + 2/3) / 2 = 5/6
        self.assertAlmostEqual(average_precision(retrieved, relevant), 5 / 6)

    def test_no_relevant_retrieved(self):
        """No relevant docs retrieved -> AP 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertEqual(average_precision(retrieved, relevant), 0.0)

    def test_empty_retrieved_list(self):
        """Empty retrieved list -> AP 0.0."""
        retrieved = []
        relevant = ["a"]
        self.assertEqual(average_precision(retrieved, relevant), 0.0)

    def test_no_relevant_documents(self):
        """No relevant docs in gold -> AP 0.0."""
        retrieved = ["a", "b"]
        relevant = []
        self.assertEqual(average_precision(retrieved, relevant), 0.0)


class TestNdcgAtK(unittest.TestCase):
    """Tests for nDCG@k."""

    def test_perfect_ranking_binary(self):
        """All relevant docs at top -> nDCG 1.0."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]
        self.assertEqual(ndcg_at_k(retrieved, relevant, 5), 1.0)

    def test_single_relevant_at_rank_one(self):
        """Single relevant at rank 1 -> nDCG 1.0."""
        retrieved = ["a", "x", "y"]
        relevant = ["a"]
        self.assertEqual(ndcg_at_k(retrieved, relevant, 3), 1.0)

    def test_single_relevant_at_rank_two(self):
        """Single relevant at rank 2 -> nDCG = 1/log2(3) / 1 = 1/log2(3)."""
        retrieved = ["x", "a", "y"]
        relevant = ["a"]
        import math
        expected = 1.0 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 3), expected)

    def test_no_relevant_retrieved(self):
        """No relevant docs retrieved -> nDCG 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertEqual(ndcg_at_k(retrieved, relevant, 3), 0.0)

    def test_no_relevant_documents(self):
        """No relevant docs in gold -> nDCG 0.0."""
        retrieved = ["a", "b"]
        relevant = []
        self.assertEqual(ndcg_at_k(retrieved, relevant, 3), 0.0)

    def test_k_must_be_positive(self):
        """k=0 raises ValueError."""
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], ["a"], 0)


class TestEvaluateQuery(unittest.TestCase):
    """Tests for evaluate_query (combined metrics)."""

    def test_perfect_query(self):
        """Perfect ranking produces all 1.0 metrics."""
        result = evaluate_query(["a", "b", "c"], ["a", "b", "c"], k_values=(1, 3))
        self.assertEqual(result["recall@1"], 1 / 3)
        self.assertEqual(result["recall@3"], 1.0)
        self.assertEqual(result["precision@3"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["average_precision"], 1.0)
        self.assertEqual(result["ndcg@3"], 1.0)

    def test_no_relevant_results(self):
        """No relevant results -> recall/MRR/AP/nDCG all 0.0."""
        result = evaluate_query(["x", "y"], ["a", "b"], k_values=(5,))
        self.assertEqual(result["recall@5"], 0.0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["average_precision"], 0.0)
        self.assertEqual(result["ndcg@5"], 0.0)

    def test_empty_retrieval(self):
        """Empty retrieval -> all metrics 0.0."""
        result = evaluate_query([], ["a"], k_values=(5,))
        self.assertEqual(result["recall@5"], 0.0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["average_precision"], 0.0)


class TestAggregateMetrics(unittest.TestCase):
    """Tests for aggregate_metrics."""

    def test_aggregate_two_queries(self):
        """Two queries averaged macro."""
        results = [
            {"recall@5": 1.0, "mrr": 1.0, "num_retrieved": 3, "num_relevant": 2},
            {"recall@5": 0.0, "mrr": 0.0, "num_retrieved": 2, "num_relevant": 2},
        ]
        agg = aggregate_metrics(results)
        self.assertEqual(agg["recall@5"], 0.5)
        self.assertEqual(agg["mrr"], 0.5)
        self.assertEqual(agg["num_queries"], 2.0)

    def test_empty_list(self):
        """Empty list -> empty dict."""
        self.assertEqual(aggregate_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
