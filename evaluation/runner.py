"""Evaluation runner for Sherlock Milestone A.

Uses the real Sherlock indexing and search engine (RepositoryIndexer,
IndexStore) to evaluate retrieval quality against a gold-standard dataset.
No Ollama, no embeddings, no vector database, no functional RAG.
"""

import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on the path so `app` and `evaluation` packages
# can be imported regardless of how this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.indexer import RepositoryIndexer
from app.storage import IndexStore
from config import SherlockConfig

from evaluation.datasets.generate_gold_standard import build_dataset
from evaluation.datasets.validate import DatasetValidationError, validate_dataset

# The application's default index database path. Evaluation MUST NOT use this.
# Evaluating against the application's database would pollute it with
# evaluation-only repositories and break existing tests.
APPLICATION_DB_PATH = Path(__file__).resolve().parent.parent / "index" / "sherlock.db"


def _get_git_commit() -> Optional[str]:
    """Get the current Git commit hash, or None if not a git repository."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return None


def _get_git_branch() -> Optional[str]:
    """Get the current Git branch name, or None if not a git repository."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return None

class EvaluationRunner:
    """Runs the Sherlock retrieval benchmark against synthetic fixtures.

    Uses an isolated SQLite database that is separate from the application's
    default database (index/sherlock.db). The evaluation database is created
    in a temporary directory by default, ensuring that benchmarking never
    pollutes the application's production index.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
    ):
        if db_path is not None:
            # Explicitly provided database path. Reject the application's
            # default database path to prevent accidental pollution.
            app_db = APPLICATION_DB_PATH.resolve()
            requested = Path(db_path).resolve()
            if requested == app_db:
                raise ValueError(
                    f"Evaluation database path must not be the application's "
                    f"default database ({app_db}). Use a separate path for "
                    "evaluation data to avoid polluting the application index."
                )
            self.db_path = requested
        else:
            # Default: use a temporary directory so evaluation is fully
            # isolated from the application's database.
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="sherlock_eval_"))
            self.db_path = self.workspace_dir / "eval.db"

        self.workspace_dir = self.db_path.parent
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.indexer: Optional[RepositoryIndexer] = None
        self.results: List[Dict[str, Any]] = []

    def _get_indexer(self) -> RepositoryIndexer:
        """Get or create a RepositoryIndexer backed by the evaluation database."""
        if self.indexer is None:
            # Remove old eval database for a clean run
            if self.db_path.exists():
                self.db_path.unlink()
            # Use a config pointing to the evaluation database
            eval_cfg = SherlockConfig(index_db_path=self.db_path)
            self.indexer = RepositoryIndexer(cfg=eval_cfg)
        return self.indexer

    def close(self) -> None:
        """Close the indexer and clean up temporary resources."""
        if self.indexer is not None:
            self.indexer.store.close()
            self.indexer = None
        # Clean up temporary workspace if we created one
        if hasattr(self, "_temp_workspace") and self._temp_workspace:
            import shutil
            shutil.rmtree(self._temp_workspace, ignore_errors=True)

    def __enter__(self) -> "EvaluationRunner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def index_fixture(
        self, repo_name: str, fixture_dir: Path, subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Index a single synthetic fixture repository.

        Returns indexing metrics: file count, chunk count, symbol count,
        import count, errors, skipped files, duration.
        """
        indexer = self._get_indexer()
        repo_id = f"eval_{repo_name}"

        start = time.monotonic()
        result = indexer.index_repository(
            fixture_dir / subdirectory if subdirectory else fixture_dir,
            repo_id=repo_id,
            repo_name=repo_name,
            subdirectory=subdirectory,
            root_path=fixture_dir,
        )
        duration = time.monotonic() - start

        indexing_record = {
            "repo_id": result.repo_id,
            "repo_name": result.repo_name,
            "status": result.status,
            "file_count": result.file_count,
            "total_lines": result.total_lines,
            "chunk_count": result.chunk_count,
            "definition_count": result.definition_count,
            "import_count": result.import_count,
            "skipped_files": result.skipped_files,
            "errors": result.errors,
            "duration_seconds": round(duration, 4),
        }
        return indexing_record

    def search_repository(
        self, repo_id: str, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Run a search query against the indexed repository."""
        indexer = self._get_indexer()
        indexer.store.connect()
        try:
            results = indexer.store.search_chunks(repo_id, query, limit=limit)
        finally:
            indexer.store.close()
        return results

    def run_benchmark(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the full benchmark against all queries.

        Parameters
        ----------
        queries:
            List of query dicts from the gold standard dataset.

        Returns
        -------
        dict with per-query results and aggregate metrics.
        """
        from evaluation.metrics import evaluate_query, aggregate_metrics

        query_results = []

        for query_entry in queries:
            repo_id = f"eval_{query_entry['repo']}"
            query_text = query_entry["query"]
            query_id = query_entry["query_id"]

            # Index the fixture repository if not already indexed
            self._ensure_indexed(query_entry["repo"], query_entry.get("fixture_dir"))

            # Run the search
            search_start = time.monotonic()
            try:
                raw_results = self.search_repository(repo_id, query_text)
                search_duration = time.monotonic() - search_start
                search_error = None
            except Exception as exc:
                search_duration = time.monotonic() - search_start
                raw_results = []
                search_error = str(exc)

            # Normalize results to document identifiers
            retrieved_ids = [r["file_path"] for r in raw_results]

            # Determine relevant documents from gold standard
            expected_files = set(query_entry.get("expected_files", []))
            expected_symbols = set(query_entry.get("expected_symbols", []))

            # Compute metrics
            metrics = evaluate_query(
                retrieved_ids, expected_files, k_values=(1, 5, 10)
            )

            # Build diagnostic record
            query_result = {
                "query_id": query_id,
                "query": query_text,
                "category": query_entry.get("category", "unknown"),
                "repo": query_entry.get("repo", ""),
                "should_have_results": query_entry.get(
                    "should_have_results", True
                ),
                "expected_files": sorted(expected_files),
                "expected_symbols": sorted(expected_symbols),
                "retrieved_files": retrieved_ids,
                "num_retrieved": len(retrieved_ids),
                "search_duration_seconds": round(search_duration, 4),
                "search_error": search_error,
                "metrics": metrics,
                "raw_results": raw_results[:10],  # Keep first 10 for diagnosis
            }

            # Add failure diagnosis if applicable
            if query_entry.get("should_have_results", True) and not metrics.get(
                "recall@10", 0
            ):
                query_result["failure_category"] = "no_relevant_retrieved"
            elif not query_entry.get("should_have_results", True) and len(
                retrieved_ids
            ) > 0:
                query_result["failure_category"] = "false_positive"
            else:
                query_result["failure_category"] = None

            query_results.append(query_result)
            self.results.append(query_result)

        aggregate = aggregate_metrics([q["metrics"] for q in query_results])
        aggregate["num_queries"] = len(query_results)

        return {
            "query_results": query_results,
            "aggregate_metrics": aggregate,
        }

    def _ensure_indexed(self, repo_name: str, fixture_dir: Optional[Path] = None):
        """Ensure the fixture repository is indexed."""
        indexer = self._get_indexer()
        repo_id = f"eval_{repo_name}"
        existing = indexer.get_repository(repo_id)
        if existing is not None:
            return

        if fixture_dir is None:
            # Find the fixture directory
            fixtures_dir = Path(__file__).resolve().parent / "fixtures"
            fixture_dir = fixtures_dir / repo_name

        # Index with fixture_dir.parent as root so that stored paths are
        # relative to the fixtures root (e.g., "calc/calculator.py") rather
        # than relative to the fixture subdirectory (e.g., "calculator.py").
        # This matches the gold-standard path format in the dataset.
        self.index_fixture(
            repo_name, fixture_dir.parent, subdirectory=fixture_dir.name
        )

    def generate_report(
        self, output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate JSON and Markdown reports from evaluation results."""
        from evaluation.metrics import aggregate_metrics

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        query_results = self.results
        aggregate = aggregate_metrics([q["metrics"] for q in query_results])

        # Build category-level metrics
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for qr in query_results:
            cat = qr["category"]
            categories.setdefault(cat, []).append(qr)

        category_metrics = {}
        for cat, qrs in categories.items():
            cat_agg = aggregate_metrics([q["metrics"] for q in qrs])
            cat_agg["num_queries"] = len(qrs)
            category_metrics[cat] = cat_agg

        # Build failure records
        failures = []
        for qr in query_results:
            if qr["failure_category"] is not None:
                failures.append(
                    {
                        "query_id": qr["query_id"],
                        "query": qr["query"],
                        "category": qr["category"],
                        "failure_category": qr["failure_category"],
                        "expected_files": qr["expected_files"],
                        "expected_symbols": qr["expected_symbols"],
                        "retrieved_files": qr["retrieved_files"],
                        "metrics": qr["metrics"],
                        "search_error": qr["search_error"],
                    }
                )

        report = {
            "dataset_version": "0.1.0",
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "num_queries": len(query_results),
            "aggregate_metrics": aggregate,
            "category_metrics": category_metrics,
            "failures": failures,
            "query_results": query_results,
            # Reproducibility metadata
            "metadata": {
                "git_commit": _get_git_commit(),
                "git_branch": _get_git_branch(),
                "python_version": platform.python_version(),
                "sqlite_version": sqlite3.sqlite_version,
                "os": platform.platform(),
            },
        }

        # Write JSON report
        json_path = output_dir / "latest.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Write Markdown report
        md_path = output_dir / "latest.md"
        self._write_markdown_report(report, md_path)

        return report

    def _write_markdown_report(self, report: Dict[str, Any], path: Path):
        """Write a human-readable Markdown report."""
        lines = [
            "# Sherlock Retrieval Benchmark Report",
            "",
            f"- **Dataset version:** {report['dataset_version']}",
            f"- **Benchmark timestamp:** {report['benchmark_timestamp']}",
            f"- **Number of queries:** {report['num_queries']}",
            "",
            "## Aggregate Metrics",
            "",
        ]

        am = report["aggregate_metrics"]
        for key in sorted(am.keys()):
            if key == "num_queries":
                continue
            lines.append(f"- **{key}:** {am[key]:.4f}")

        lines.extend(
            [
                "",
                "## Category-Level Metrics",
                "",
            ]
        )

        for cat, metrics in sorted(report["category_metrics"].items()):
            lines.append(f"### {cat}")
            lines.append(f"- Queries: {metrics.get('num_queries', 0)}")
            for key in sorted(metrics.keys()):
                if key == "num_queries":
                    continue
                lines.append(f"  - {key}: {metrics[key]:.4f}")
            lines.append("")

        lines.extend(
            [
                "## Failures",
                "",
            ]
        )

        if not report["failures"]:
            lines.append("- No failures detected.")
        else:
            for f in report["failures"]:
                lines.append(f"- **{f['query_id']}:** {f['query']}")
                lines.append(f"  - Category: {f['category']}")
                lines.append(f"  - Failure: {f['failure_category']}")
                lines.append(f"  - Expected files: {f['expected_files']}")
                lines.append(f"  - Retrieved files: {f['retrieved_files']}")
                lines.append("")

        path.write_text("\n".join(lines))


def run_full_benchmark(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to run the full benchmark.

    Parameters
    ----------
    output_dir:
        Directory to write reports. Defaults to evaluation/results/.

    Returns
    -------
    dict with the full benchmark report (from generate_report).
    """
    from evaluation.datasets.generate_gold_standard import build_dataset
    from evaluation.datasets.validate import validate_dataset

    queries = build_dataset()

    # Build full dataset dict for validation
    dataset = {
        "dataset_version": "0.1.0",
        "description": "Gold-standard benchmark dataset for Sherlock retrieval evaluation",
        "num_queries": len(queries),
        "queries": queries,
    }

    # Validate the dataset before running the benchmark
    errors = validate_dataset(dataset)
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise DatasetValidationError(
            f"Dataset validation failed ({len(errors)} error(s)):\n{error_msg}"
        )

    runner = EvaluationRunner()
    runner.run_benchmark(queries)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"
    return runner.generate_report(output_dir)


if __name__ == "__main__":
    import sys

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = run_full_benchmark(output_dir)
    print(
        f"Benchmark complete: {report['num_queries']} queries, "
        f"recall@5={report['aggregate_metrics'].get('recall@5', 0):.4f}, "
        f"mrr={report['aggregate_metrics'].get('mrr', 0):.4f}"
    )
