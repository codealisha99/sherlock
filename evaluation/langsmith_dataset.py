"""LangSmith evaluation dataset for Sherlock RAG.

Version-controlled benchmark that runs without LangSmith.
Contains representative questions about code structure, symbols, imports,
call relationships, multi-file questions, and unanswerable questions.

Each example contains only reference data needed for evaluation:
- question: the natural-language question
- repo: the repository to query
- expected_files: files expected to contain relevant evidence
- expected_symbols: symbols expected to be relevant
- expected_chunks: chunk IDs expected to be relevant (when known)
- answerability: whether the question is answerable from indexed content
- category: query category for metrics breakdown
- retrieval_relevance: set of relevant chunk IDs for recall/MRR/nDCG
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASET_VERSION = "0.1.0"

# Default benchmark: synthetic examples covering key categories.
# These run against any indexed repository with matching structure.
DEFAULT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "example_id": "ex-001",
        "question": "Where is the Calculator class defined?",
        "repo": "calc",
        "expected_files": ["src/calculator.py"],
        "expected_symbols": ["Calculator"],
        "expected_chunks": [],
        "answerability": True,
        "category": "exact_identifier",
        "retrieval_relevance": {"src/calculator.py"},
        "description": "Exact class name lookup",
    },
    {
        "example_id": "ex-002",
        "question": "How does the add function work?",
        "repo": "calc",
        "expected_files": ["src/calculator.py", "src/operations.py"],
        "expected_symbols": ["add"],
        "expected_chunks": [],
        "answerability": True,
        "category": "exact_identifier",
        "retrieval_relevance": {"src/calculator.py", "src/operations.py"},
        "description": "Exact function name lookup across files",
    },
    {
        "example_id": "ex-003",
        "question": "What does the subtract function do?",
        "repo": "calc",
        "expected_files": ["src/operations.py"],
        "expected_symbols": ["subtract"],
        "expected_chunks": [],
        "answerability": True,
        "category": "exact_identifier",
        "retrieval_relevance": {"src/operations.py"},
        "description": "Exact function name lookup",
    },
    {
        "example_id": "ex-004",
        "question": "How are exceptions handled in the reader?",
        "repo": "dataproc",
        "expected_files": ["dataproc/reader.py"],
        "expected_symbols": ["read_csv"],
        "expected_chunks": [],
        "answerability": True,
        "category": "error_handling",
        "retrieval_relevance": {"dataproc/reader.py"},
        "description": "Error handling question",
    },
    {
        "example_id": "ex-005",
        "question": "How do calculator and operations work together?",
        "repo": "calc",
        "expected_files": ["src/calculator.py", "src/operations.py"],
        "expected_symbols": ["add", "multiply"],
        "expected_chunks": [],
        "answerability": True,
        "category": "cross_file",
        "retrieval_relevance": {"src/calculator.py", "src/operations.py"},
        "description": "Cross-file relationship question",
    },
    {
        "example_id": "ex-006",
        "question": "What is the quantum entanglement implementation?",
        "repo": "calc",
        "expected_files": [],
        "expected_symbols": [],
        "expected_chunks": [],
        "answerability": False,
        "category": "no_result",
        "retrieval_relevance": set(),
        "description": "Unanswerable question (no relevant content)",
    },
    {
        "example_id": "ex-007",
        "question": "How do I import the string utilities?",
        "repo": "utils",
        "expected_files": ["utils/string_utils.py"],
        "expected_symbols": ["slugify"],
        "expected_chunks": [],
        "answerability": True,
        "category": "import_discovery",
        "retrieval_relevance": {"utils/string_utils.py"},
        "description": "Import discovery question",
    },
    {
        "example_id": "ex-008",
        "question": "What does the slugify function do?",
        "repo": "utils",
        "expected_files": ["utils/string_utils.py"],
        "expected_symbols": ["slugify"],
        "expected_chunks": [],
        "answerability": True,
        "category": "partial_identifier",
        "retrieval_relevance": {"utils/string_utils.py"},
        "description": "Partial identifier lookup",
    },
]

DATASET: Dict[str, Any] = {
    "dataset_version": DATASET_VERSION,
    "description": "Sherlock RAG evaluation benchmark",
    "num_examples": len(DEFAULT_EXAMPLES),
    "examples": DEFAULT_EXAMPLES,
}


def load_dataset() -> Dict[str, Any]:
    """Load the LangSmith evaluation dataset.

    Returns the dataset dict with examples. Runs without LangSmith.
    """
    return DATASET


def save_dataset(path: Path) -> None:
    """Save the dataset to a JSON file (for version control)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(DATASET, f, indent=2)


def get_example_by_id(example_id: str) -> Optional[Dict[str, Any]]:
    """Get a single example by its stable ID."""
    for ex in DEFAULT_EXAMPLES:
        if ex["example_id"] == example_id:
            return ex
    return None


def filter_by_category(category: str) -> List[Dict[str, Any]]:
    """Filter examples by category."""
    return [ex for ex in DEFAULT_EXAMPLES if ex["category"] == category]


def get_unanswerable_examples() -> List[Dict[str, Any]]:
    """Get examples that should produce abstention."""
    return [ex for ex in DEFAULT_EXAMPLES if ex["answerability"] is False]


def get_answerable_examples() -> List[Dict[str, Any]]:
    """Get examples that should produce answers."""
    return [ex for ex in DEFAULT_EXAMPLES if ex["answerability"] is True]
