"""Unit tests for LangSmith evaluation runner (evaluation/langsmith_runner.py).

Tests local evaluation without LangSmith credentials or network access.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.langsmith_dataset import (
    DATASET_VERSION,
    filter_by_category,
    get_answerable_examples,
    get_unanswerable_examples,
    load_dataset,
)
from evaluation.langsmith_runner import LangSMEvaluationRunner, run_langsmith_evaluation


class TestLangSmithDataset(unittest.TestCase):
    """Tests for the LangSmith evaluation dataset."""

    def test_load_dataset(self):
        """Dataset loads with correct structure."""
        ds = load_dataset()
        self.assertEqual(ds["dataset_version"], DATASET_VERSION)
        self.assertIn("examples", ds)
        self.assertGreater(len(ds["examples"]), 0)

    def test_examples_have_required_fields(self):
        """Each example has required evaluation fields."""
        ds = load_dataset()
        required = ["example_id", "question", "repo", "answerability", "category"]
        for ex in ds["examples"]:
            for field in required:
                self.assertIn(field, ex, f"Missing field {field} in {ex['example_id']}")

    def test_example_ids_unique(self):
        """Example IDs are unique."""
        ds = load_dataset()
        ids = [ex["example_id"] for ex in ds["examples"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_answerable_examples(self):
        """Returns examples marked as answerable."""
        examples = get_answerable_examples()
        for ex in examples:
            self.assertTrue(ex["answerability"])

    def test_get_unanswerable_examples(self):
        """Returns examples marked as unanswerable."""
        examples = get_unanswerable_examples()
        for ex in examples:
            self.assertFalse(ex["answerability"])

    def test_filter_by_category(self):
        """Filters examples by category."""
        examples = filter_by_category("exact_identifier")
        self.assertGreater(len(examples), 0)
        for ex in examples:
            self.assertEqual(ex["category"], "exact_identifier")

    def test_has_unanswerable_examples(self):
        """Dataset contains unanswerable examples for abstention testing."""
        unanswerable = get_unanswerable_examples()
        self.assertGreater(len(unanswerable), 0)


class TestLangSMEvaluationRunner(unittest.TestCase):
    """Tests for LangSMEvaluationRunner without LangSmith."""

    def setUp(self):
        self.runner = LangSMEvaluationRunner()

    def test_runner_init_without_langsmith(self):
        """Runner initializes without LangSmith (default)."""
        self.assertFalse(self.runner.use_langsmith)
        self.assertIsNone(self.runner.tracer)

    def test_run_example_basic(self):
        """Running a single example produces a result dict."""
        example = {
            "example_id": "ex-test-1",
            "question": "How does add work?",
            "repo": "calc",
            "expected_files": ["src/calculator.py"],
            "expected_symbols": ["add"],
            "answerability": True,
            "category": "exact_identifier",
            "retrieval_relevance": {"src/calculator.py"},
        }
        result = self.runner.run_example(example, retrieval_method="hybrid")
        self.assertEqual(result["example_id"], "ex-test-1")
        self.assertEqual(result["question"], "How does add work?")
        self.assertEqual(result["repo"], "calc")
        self.assertIn("answer", result)
        self.assertIn("abstained", result)
        self.assertIn("metrics", result)
        self.assertIn("total_latency_ms", result)

    def test_run_example_unanswerable(self):
        """Unanswerable example should produce abstention."""
        example = {
            "example_id": "ex-test-2",
            "question": "What is quantum entanglement?",
            "repo": "calc",
            "expected_files": [],
            "expected_symbols": [],
            "answerability": False,
            "category": "no_result",
            "retrieval_relevance": set(),
        }
        result = self.runner.run_example(example, retrieval_method="hybrid")
        self.assertTrue(result["abstained"])

    def test_run_benchmark_all_methods(self):
        """Running benchmark with all methods produces results."""
        examples = [
            {
                "example_id": "ex-bench-1",
                "question": "How does add work?",
                "repo": "calc",
                "expected_files": ["src/calculator.py"],
                "answerability": True,
                "category": "exact_identifier",
                "retrieval_relevance": {"src/calculator.py"},
            }
        ]
        benchmark = self.runner.run_benchmark(
            examples=examples, retrieval_methods=["hybrid"]
        )
        self.assertEqual(benchmark["num_examples"], 1)
        self.assertEqual(benchmark["retrieval_methods"], ["hybrid"])
        self.assertIn("hybrid", benchmark["results_by_method"])
        self.assertIn("hybrid", benchmark["aggregates"])

    def test_run_benchmark_aggregates(self):
        """Benchmark aggregates have expected structure."""
        examples = [
            {
                "example_id": "ex-bench-2",
                "question": "How does add work?",
                "repo": "calc",
                "expected_files": ["src/calculator.py"],
                "answerability": True,
                "category": "exact_identifier",
                "retrieval_relevance": {"src/calculator.py"},
            },
            {
                "example_id": "ex-bench-3",
                "question": "Quantum physics?",
                "repo": "calc",
                "expected_files": [],
                "answerability": False,
                "category": "no_result",
                "retrieval_relevance": set(),
            },
        ]
        benchmark = self.runner.run_benchmark(
            examples=examples, retrieval_methods=["hybrid"]
        )
        agg = benchmark["aggregates"]["hybrid"]
        self.assertEqual(agg["total_examples"], 2)
        self.assertIn("abstention_rate", agg)
        self.assertIn("avg_latency_ms", agg)
        self.assertIn("unanswerable_abstention_rate", agg)
        self.assertIn("answerable_abstention_rate", agg)

    def test_generate_report_creates_files(self):
        """generate_report writes JSON and Markdown reports."""
        examples = [
            {
                "example_id": "ex-report-1",
                "question": "How does add work?",
                "repo": "calc",
                "expected_files": ["src/calculator.py"],
                "answerability": True,
                "category": "exact_identifier",
                "retrieval_relevance": {"src/calculator.py"},
            }
        ]
        benchmark = self.runner.run_benchmark(
            examples=examples, retrieval_methods=["hybrid"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report = self.runner.generate_report(benchmark, output_dir=output_dir)
            self.assertEqual(report["num_examples"], 1)
            self.assertFalse(report["use_langsmith"])
            self.assertTrue(report["local_evaluation"])

            json_path = output_dir / "langsmith_eval_latest.json"
            md_path = output_dir / "langsmith_eval_latest.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

    def test_use_langsmith_false_without_credentials(self):
        """use_langsmith is False when no credentials configured."""
        runner = LangSMEvaluationRunner()
        self.assertFalse(runner.use_langsmith)


class TestRunLangSmithEvaluation(unittest.TestCase):
    """Tests for the convenience run function."""

    def test_run_langsmith_evaluation_local(self):
        """run_langsmith_evaluation runs locally without LangSmith."""
        report = run_langsmith_evaluation(retrieval_methods=["hybrid"])
        self.assertEqual(report["num_examples"], 8)
        self.assertFalse(report["use_langsmith"])
        self.assertTrue(report["local_evaluation"])
        self.assertIn("hybrid", report["aggregates"])

    def test_run_langsmith_evaluation_output_dir(self):
        """Report is written to the specified output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report = run_langsmith_evaluation(
                output_dir=output_dir, retrieval_methods=["hybrid"]
            )
            self.assertTrue((output_dir / "langsmith_eval_latest.json").exists())
            self.assertTrue((output_dir / "langsmith_eval_latest.md").exists())


if __name__ == "__main__":
    unittest.main()
