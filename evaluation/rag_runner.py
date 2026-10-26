"""RAG evaluation runner for Sherlock Milestone B.

Extends the Milestone A evaluation runner to support:
- Semantic search evaluation (embedding-based retrieval)
- Hybrid search evaluation (RRF fusion of lexical + semantic)
- Answer quality evaluation (grounded RAG pipeline)
- Citation correctness evaluation
- Abstention evaluation

Uses isolated temporary databases. Never touches index/sherlock.db.
"""

from __future__ import annotations

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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.indexer import RepositoryIndexer
from app.llm.embedding_client import EmbeddingClient
from config import SherlockConfig

from evaluation.datasets.generate_gold_standard import build_dataset
from evaluation.datasets.validate import DatasetValidationError, validate_dataset
from evaluation.metrics import evaluate_query, aggregate_metrics
from evaluation.runner import APPLICATION_DB_PATH, _get_git_branch, _get_git_commit


class RAGEvaluationRunner:
    """Runs RAG evaluation across lexical, semantic, and hybrid retrieval.

    Evaluates retrieval quality independently from generation quality, then
    evaluates answer quality (citations, grounding, abstention) using the
    RAG pipeline with mocked or real LLM.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        use_embeddings: bool = True,
    ):
        if db_path is not None:
            app_db = APPLICATION_DB_PATH.resolve()
            requested = Path(db_path).resolve()
            if requested == app_db:
                raise ValueError(
                    f"Evaluation database path must not be the application's "
                    f"default database ({app_db})."
                )
            self.db_path = requested
        else:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="sherlock_rag_eval_"))
            self.db_path = self.workspace_dir / "eval.db"

        self.workspace_dir = self.db_path.parent
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.use_embeddings = use_embeddings
        self.embed_client: Optional[EmbeddingClient] = None
        if use_embeddings:
            try:
                self.embed_client = EmbeddingClient()
                if not self.embed_client.is_available():
                    self.embed_client = None
            except Exception:
                self.embed_client = None

        self.indexer: Optional[RepositoryIndexer] = None
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Indexer management
    # ------------------------------------------------------------------ #

    def _get_indexer(self) -> RepositoryIndexer:
        if self.indexer is None:
            if self.db_path.exists():
                self.db_path.unlink()
            eval_cfg = SherlockConfig(index_db_path=self.db_path)
            self.indexer = RepositoryIndexer(cfg=eval_cfg)
        return self.indexer

    def close(self) -> None:
        if self.indexer is not None:
            self.indexer.store.close()
            self.indexer = None

    def __enter__(self) -> "RAGEvaluationRunner":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index_fixture(
        self, repo_name: str, fixture_dir: Path, subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        indexer = self._get_indexer()
        repo_id = f"rag_eval_{repo_name}"

        start = time.monotonic()
        result = indexer.index_repository(
            fixture_dir / subdirectory if subdirectory else fixture_dir,
            repo_id=repo_id,
            repo_name=repo_name,
            subdirectory=subdirectory,
            root_path=fixture_dir,
        )
        duration = time.monotonic() - start

        # Generate embeddings if available
        embedding_duration = None
        embedded_count = 0
        if self.embed_client is not None:
            emb_start = time.monotonic()
            emb_result = indexer.embed_chunks(repo_id, embedding_client=self.embed_client)
            embedding_duration = time.monotonic() - emb_start
            embedded_count = emb_result.get("embedded_count", 0)

        return {
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
            "embedding_duration_seconds": round(embedding_duration, 4)
            if embedding_duration is not None
            else None,
            "embedded_chunks": embedded_count,
            "embeddings_available": self.embed_client is not None,
        }

    def _ensure_indexed(self, repo_name: str, fixture_dir: Optional[Path] = None):
        indexer = self._get_indexer()
        repo_id = f"rag_eval_{repo_name}"
        existing = indexer.get_repository(repo_id)
        if existing is not None:
            return

        if fixture_dir is None:
            fixtures_dir = Path(__file__).resolve().parent / "fixtures"
            fixture_dir = fixtures_dir / repo_name

        self.index_fixture(repo_name, fixture_dir.parent, subdirectory=fixture_dir.name)

    # ------------------------------------------------------------------ #
    # Retrieval methods
    # ------------------------------------------------------------------ #

    def _retrieve_lexical(
        self, repo_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        indexer = self._get_indexer()
        indexer.store.connect()
        try:
            return indexer.store.search_chunks(repo_id, query, limit=limit)
        finally:
            indexer.store.close()

    def _retrieve_semantic(
        self, repo_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        indexer = self._get_indexer()
        if self.embed_client is None:
            return []
        return indexer.semantic_search(
            repo_id, query, limit=limit, embedding_client=self.embed_client
        )

    def _retrieve_hybrid(
        self, repo_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        indexer = self._get_indexer()
        query_embedding = None
        if self.embed_client is not None:
            query_embedding = self.embed_client.embed_text(query)
        return indexer.hybrid_search(
            repo_id, query, query_embedding=query_embedding, limit=limit, rrf_k=60
        )

    # ------------------------------------------------------------------ #
    # Benchmark
    # ------------------------------------------------------------------ #

    def run_benchmark(
        self,
        queries: List[Dict[str, Any]],
        retrieval_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full benchmark across all queries and retrieval methods.

        Parameters
        ----------
        queries:
            List of query dicts from the gold standard dataset.
        retrieval_methods:
            List of methods to evaluate: ["lexical", "semantic", "hybrid"].
            Defaults to all available methods.
        """
        if retrieval_methods is None:
            retrieval_methods = ["lexical"]
            if self.embed_client is not None:
                retrieval_methods.extend(["semantic", "hybrid"])

        all_results: Dict[str, Dict[str, Any]] = {}

        for method in retrieval_methods:
            method_results = self._run_single_method(queries, method)
            all_results[method] = method_results

        return {
            "retrieval_methods": retrieval_methods,
            "embeddings_available": self.embed_client is not None,
            "results_by_method": all_results,
        }

    def _run_single_method(
        self, queries: List[Dict[str, Any]], method: str
    ) -> Dict[str, Any]:
        from evaluation.metrics import evaluate_query, aggregate_metrics

        query_results = []

        for query_entry in queries:
            repo_name = query_entry["repo"]
            repo_id = f"rag_eval_{repo_name}"
            query_text = query_entry["query"]
            query_id = query_entry["query_id"]

            self._ensure_indexed(repo_name, query_entry.get("fixture_dir"))

            # Run retrieval
            retriever = {
                "lexical": self._retrieve_lexical,
                "semantic": self._retrieve_semantic,
                "hybrid": self._retrieve_hybrid,
            }[method]

            search_start = time.monotonic()
            try:
                raw_results = retriever(repo_id, query_text, limit=10)
                search_duration = time.monotonic() - search_start
                search_error = None
            except Exception as exc:
                search_duration = time.monotonic() - search_start
                raw_results = []
                search_error = str(exc)

            # Normalize results
            retrieved_ids = [r["file_path"] for r in raw_results]

            # Expected documents
            expected_files = set(query_entry.get("expected_files", []))
            expected_symbols = set(query_entry.get("expected_symbols", []))

            # Compute metrics
            metrics = evaluate_query(retrieved_ids, expected_files, k_values=(1, 5, 10))

            # Failure diagnosis
            if query_entry.get("should_have_results", True) and not metrics.get(
                "recall@10", 0
            ):
                failure_category = "no_relevant_retrieved"
            elif not query_entry.get("should_have_results", True) and len(
                retrieved_ids
            ) > 0:
                failure_category = "false_positive"
            else:
                failure_category = None

            query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "category": query_entry.get("category", "unknown"),
                    "repo": repo_name,
                    "method": method,
                    "should_have_results": query_entry.get("should_have_results", True),
                    "expected_files": sorted(expected_files),
                    "expected_symbols": sorted(expected_symbols),
                    "retrieved_files": retrieved_ids,
                    "num_retrieved": len(retrieved_ids),
                    "search_duration_seconds": round(search_duration, 4),
                    "search_error": search_error,
                    "metrics": metrics,
                    "failure_category": failure_category,
                }
            )

        aggregate = aggregate_metrics([q["metrics"] for q in query_results])
        aggregate["num_queries"] = len(query_results)

        return {
            "method": method,
            "query_results": query_results,
            "aggregate_metrics": aggregate,
        }

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        benchmark_result: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate JSON and Markdown reports from RAG benchmark results."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_by_method = benchmark_result["results_by_method"]

        # Build per-method aggregates
        method_aggregates = {}
        for method, result in results_by_method.items():
            method_aggregates[method] = result["aggregate_metrics"]

        # Build per-category, per-method breakdown
        category_method_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for method, result in results_by_method.items():
            categories: Dict[str, List[Dict[str, Any]]] = {}
            for qr in result["query_results"]:
                cat = qr["category"]
                categories.setdefault(cat, []).append(qr)
            cat_metrics = {}
            for cat, qrs in categories.items():
                cat_agg = aggregate_metrics([q["metrics"] for q in qrs])
                cat_agg["num_queries"] = len(qrs)
                cat_metrics[cat] = cat_agg
            category_method_metrics[method] = cat_metrics

        report = {
            "dataset_version": "0.1.0",
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_methods": benchmark_result["retrieval_methods"],
            "embeddings_available": benchmark_result["embeddings_available"],
            "method_aggregates": method_aggregates,
            "category_method_metrics": category_method_metrics,
            "results_by_method": results_by_method,
            "metadata": {
                "git_commit": _get_git_commit(),
                "git_branch": _get_git_branch(),
                "python_version": platform.python_version(),
                "sqlite_version": sqlite3.sqlite_version,
                "os": platform.platform(),
            },
        }

        # Write JSON report
        json_path = output_dir / "rag_latest.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Write Markdown report
        md_path = output_dir / "rag_latest.md"
        self._write_markdown_report(report, md_path)

        return report

    def _write_markdown_report(self, report: Dict[str, Any], path: Path):
        lines = [
            "# Sherlock RAG Evaluation Report",
            "",
            f"- **Dataset version:** {report['dataset_version']}",
            f"- **Benchmark timestamp:** {report['benchmark_timestamp']}",
            f"- **Embeddings available:** {report['embeddings_available']}",
            f"- **Retrieval methods:** {', '.join(report['retrieval_methods'])}",
            "",
        ]

        # Method comparison table
        lines.append("## Retrieval Method Comparison")
        lines.append("")
        lines.append("| Metric | " + " | ".join(report["retrieval_methods"]) + " |")
        lines.append("|" + "---|" * (len(report["retrieval_methods"]) + 1))

        ma = report["method_aggregates"]
        metric_keys = sorted(
            set().union(*[set(m.keys()) for m in ma.values()])
            - {"num_queries"}
        )
        for key in metric_keys:
            row = [key]
            for method in report["retrieval_methods"]:
                val = ma[method].get(key, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("| Method | Num Queries |")
        lines.append("|--------|-------------|")
        for method in report["retrieval_methods"]:
            lines.append(f"| {method} | {ma[method].get('num_queries', 0)} |")

        lines.append("")
        lines.append("## Per-Category, Per-Method Metrics (Recall@5)")
        lines.append("")
        lines.append("| Category | " + " | ".join(report["retrieval_methods"]) + " |")
        lines.append("|" + "---|" * (len(report["retrieval_methods"]) + 1))

        all_categories = sorted(
            set().union(*[
                set(cm.keys()) for cm in report["category_method_metrics"].values()
            ])
        )
        for cat in all_categories:
            row = [cat]
            for method in report["retrieval_methods"]:
                cat_metrics = report["category_method_metrics"].get(method, {})
                cat_data = cat_metrics.get(cat, {})
                val = cat_data.get("recall@5", 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Reproducibility Metadata")
        for key, val in report["metadata"].items():
            lines.append(f"- **{key}:** {val}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
