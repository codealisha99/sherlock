"""Unit tests for dataset validation in evaluation/datasets/validate.py."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.datasets.validate import (
    DatasetValidationError,
    load_and_validate_dataset,
    validate_dataset,
)


def make_valid_dataset(num_queries: int = 2) -> dict:
    """Build a minimal valid dataset dict."""
    return {
        "dataset_version": "0.1.0",
        "description": "Test dataset",
        "num_queries": num_queries,
        "queries": [
            {
                "query_id": "q001",
                "query": "def add",
                "description": "Exact function lookup",
                "category": "exact_identifier",
                "repo": "calc",
                "expected_files": ["calc/calculator.py"],
                "expected_symbols": ["add"],
                "should_have_results": True,
                "why_relevant": "Exact function name match",
            },
            {
                "query_id": "q002",
                "query": "nonexistent_xyz",
                "description": "Query with no results",
                "category": "no_result",
                "repo": "calc",
                "expected_files": [],
                "expected_symbols": [],
                "should_have_results": False,
                "why_relevant": "Function does not exist",
            },
        ],
    }


class TestValidateDataset(unittest.TestCase):
    """Tests for the validate_dataset function."""

    def test_valid_dataset(self):
        """A well-formed dataset passes validation."""
        data = make_valid_dataset()
        errors = validate_dataset(data)
        self.assertEqual(errors, [], f"Expected no errors, got: {errors}")

    def test_missing_top_level_key(self):
        """Missing top-level key produces an error."""
        data = {"dataset_version": "0.1.0", "description": "x", "num_queries": 0}
        errors = validate_dataset(data)
        self.assertTrue(any("queries" in e for e in errors))

    def test_duplicate_query_ids(self):
        """Duplicate query IDs produce an error."""
        data = make_valid_dataset(num_queries=2)
        data["queries"][1]["query_id"] = "q001"  # Duplicate
        errors = validate_dataset(data)
        self.assertTrue(any("Duplicate query ID" in e for e in errors))

    def test_empty_query_text(self):
        """Empty query text produces an error."""
        data = make_valid_dataset(num_queries=1)
        data["queries"][0]["query"] = ""
        errors = validate_dataset(data)
        self.assertTrue(any("empty 'query' field" in e for e in errors))

    def test_duplicate_query_text(self):
        """Duplicate query text produces an error."""
        data = make_valid_dataset(num_queries=2)
        data["queries"][1]["query"] = data["queries"][0]["query"]
        errors = validate_dataset(data)
        self.assertTrue(any("duplicate query text" in e for e in errors))

    def test_missing_required_field(self):
        """Missing required field produces an error."""
        data = make_valid_dataset(num_queries=1)
        del data["queries"][0]["expected_files"]
        errors = validate_dataset(data)
        self.assertTrue(any("expected_files" in e for e in errors))

    def test_absolute_path(self):
        """Absolute path in expected_files produces an error."""
        data = make_valid_dataset(num_queries=1)
        data["queries"][0]["expected_files"] = ["/abs/path/to/file.py"]
        errors = validate_dataset(data)
        self.assertTrue(any("relative path" in e for e in errors))

    def test_should_have_true_but_no_files(self):
        """should_have_results=True with no expected_files produces an error."""
        data = make_valid_dataset(num_queries=1)
        data["queries"][0]["should_have_results"] = True
        data["queries"][0]["expected_files"] = []
        errors = validate_dataset(data)
        self.assertTrue(
            any("should_have_results" in e and "expected_files" in e for e in errors)
        )

    def test_should_have_false_but_has_files(self):
        """should_have_results=False with expected_files produces an error."""
        data = make_valid_dataset(num_queries=1)
        data["queries"][0]["should_have_results"] = False
        errors = validate_dataset(data)
        self.assertTrue(
            any("should_have_results" in e and "expected_files" in e for e in errors)
        )


class TestLoadAndValidateDataset(unittest.TestCase):
    """Tests for load_and_validate_dataset function."""

    def test_valid_dataset(self):
        """load_and_validate_dataset works with a valid file."""
        data = make_valid_dataset()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            result = load_and_validate_dataset(tmp_path)
            self.assertEqual(result["dataset_version"], "0.1.0")
            self.assertEqual(result["num_queries"], 2)
        finally:
            Path(tmp_path).unlink()

    def test_nonexistent_file(self):
        """load_and_validate_dataset raises on missing file."""
        with self.assertRaises(DatasetValidationError):
            load_and_validate_dataset(Path("/nonexistent/dataset.json"))

    def test_invalid_json(self):
        """load_and_validate_dataset raises on invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json")
            tmp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError):
                load_and_validate_dataset(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_invalid_dataset(self):
        """load_and_validate_dataset raises on invalid dataset."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"dataset_version": "0.1.0"}, f)  # Missing queries
            tmp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError):
                load_and_validate_dataset(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_real_dataset(self):
        """The real gold_standard.json passes validation."""
        ds_path = (
            Path(__file__).resolve().parent.parent / "datasets" / "gold_standard.json"
        )
        self.assertTrue(ds_path.exists(), f"Dataset file not found: {ds_path}")
        data = json.loads(ds_path.read_text())
        errors = validate_dataset(data)
        self.assertEqual(
            errors, [], f"Real dataset has validation errors: {errors}"
        )
        self.assertGreaterEqual(
            data["num_queries"],
            30,
            f"Expected >= 30 queries, got {data['num_queries']}",
        )


if __name__ == "__main__":
    unittest.main()