"""Validation for the Sherlock model-free benchmark dataset.

Checks that the gold-standard dataset is well-formed and consistent before
it is used in benchmark runs. All checks are deterministic and independent
of the Sherlock indexing/search code.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetValidationError(Exception):
    """Raised when the dataset fails validation."""
    pass


def validate_dataset(data: Dict[str, Any]) -> List[str]:
    """Validate a dataset dictionary and return a list of error messages.

    An empty list means the dataset is valid.
    """
    errors: List[str] = []

    # Top-level required keys
    for key in ("dataset_version", "description", "num_queries", "queries"):
        if key not in data:
            errors.append(f"Missing required top-level key: {key}")
            return errors  # Can't proceed without top-level structure

    queries = data.get("queries", [])
    if not isinstance(queries, list):
        errors.append("'queries' must be a list")
        return errors

    if len(queries) == 0:
        errors.append("Dataset contains no queries")
        return errors

    seen_ids: set = set()
    seen_query_texts: set = set()

    required_query_fields = (
        "query_id", "query", "description", "category", "repo",
        "expected_files", "expected_symbols", "should_have_results",
        "why_relevant",
    )

    for idx, q in enumerate(queries):
        qid = q.get("query_id", f"<index {idx}>")

        # Check for duplicate query IDs
        if qid in seen_ids:
            errors.append(f"Duplicate query ID: {qid}")
        seen_ids.add(qid)

        # Check for required fields
        for field in required_query_fields:
            if field not in q:
                errors.append(
                    f"Query {qid}: missing required field '{field}'"
                )

        # Check query text is non-empty
        query_text = q.get("query", "")
        if not query_text or not query_text.strip():
            errors.append(f"Query {qid}: empty 'query' field")

        # Check for duplicate query text
        if query_text and query_text in seen_query_texts:
            errors.append(f"Query {qid}: duplicate query text: {query_text!r}")
        seen_query_texts.add(query_text)

        # Check repo is non-empty
        repo = q.get("repo", "")
        if not repo or not repo.strip():
            errors.append(f"Query {qid}: empty 'repo' field")

        # Check expected_files is a list of non-empty strings
        expected_files = q.get("expected_files", [])
        if not isinstance(expected_files, list):
            errors.append(f"Query {qid}: 'expected_files' must be a list")
        else:
            for i, ef in enumerate(expected_files):
                if not isinstance(ef, str) or not ef.strip():
                    errors.append(
                        f"Query {qid}: 'expected_files[{i}]' must be a non-empty string"
                    )
                elif ef.startswith("/"):
                    errors.append(
                        f"Query {qid}: 'expected_files[{i}]' must be a relative path, "
                        f"got absolute path: {ef}"
                    )

        # Check expected_symbols is a list of strings
        expected_symbols = q.get("expected_symbols", [])
        if not isinstance(expected_symbols, list):
            errors.append(f"Query {qid}: 'expected_symbols' must be a list")
        else:
            for i, es in enumerate(expected_symbols):
                if not isinstance(es, str) or not es.strip():
                    errors.append(
                        f"Query {qid}: 'expected_symbols[{i}]' must be a non-empty string"
                    )

        # Check should_have_results is boolean
        should_have = q.get("should_have_results")
        if not isinstance(should_have, bool):
            errors.append(
                f"Query {qid}: 'should_have_results' must be a boolean, "
                f"got {type(should_have).__name__}"
            )

        # Check category is non-empty
        category = q.get("category", "")
        if not category or not category.strip():
            errors.append(f"Query {qid}: empty 'category' field")

        # Contradiction check: should_have_results=False but expected_files non-empty
        if not should_have and expected_files:
            errors.append(
                f"Query {qid}: 'should_have_results' is False but "
                f"'expected_files' is non-empty: {expected_files}"
            )

        # Contradiction check: should_have_results=True but no expected_files
        if should_have and not expected_files:
            errors.append(
                f"Query {qid}: 'should_have_results' is True but "
                f"'expected_files' is empty"
            )

    return errors


def load_and_validate_dataset(path: Path) -> Dict[str, Any]:
    """Load a dataset JSON file and validate it.

    Returns the parsed dataset dict. Raises DatasetValidationError if
    the file cannot be read or the dataset fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetValidationError(f"Dataset file not found: {path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    errors = validate_dataset(data)
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise DatasetValidationError(
            f"Dataset validation failed ({len(errors)} error(s)):\n{error_msg}"
        )

    return data


__all__ = ["validate_dataset", "load_and_validate_dataset", "DatasetValidationError"]