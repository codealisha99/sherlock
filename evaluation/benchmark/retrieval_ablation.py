"""Sherlock Retrieval Ablation Benchmark.

Benchmarks BM25-only, dense-only, and hybrid (RRF) retrieval configurations
independently against the 35-query gold-standard dataset.

Reuses existing Sherlock infrastructure (RepositoryIndexer, IndexStore,
EmbeddingClient) without modifying production code.

Run with:
    .venv/bin/python evaluation/benchmark/retrieval_ablation.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.indexer import RepositoryIndexer
from app.llm.embedding_client import EmbeddingClient
from app.storage import IndexStore
from config import SherlockConfig

from evaluation.metrics import evaluate_query, aggregate_metrics

# Gold standard dataset
GOLD_STANDARD_PATH = (
    _PROJECT_ROOT / "evaluation" / "datasets" / "gold_standard.json"
)


def _percentile(data: List[float], p: int) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + (sorted_data[f + 1] - sorted_data[f]) * c
    return sorted_data[f]


class RetrievalAblationBenchmark:
    """Benchmark retrieval configurations independently.

    Supports:
      - BM25-only (FTS5)
      - Dense-only (sqlite-vec semantic)
      - Hybrid (RRF fusion of BM25 + dense)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        use_embeddings: bool = True,
    ):
        if db_path is not None:
            self.db_path = db_path
        else:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="sherlock_ablation_"))
            self.db_path = self.workspace_dir / "eval.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

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

    def __enter__(self) -> "RetrievalAblationBenchmark":
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
        repo_id = f"ablation_{repo_name}"

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
            "duration_seconds": round(duration, 4),
            "embedding_duration_seconds": round(embedding_duration, 4)
            if embedding_duration is not None
            else None,
            "embedded_chunks": embedded_count,
            "embeddings_available": self.embed_client is not None,
        }

    def _ensure_indexed(
        self, repo_name: str, fixture_dir: Optional[Path] = None
    ) -> None:
        indexer = self._get_indexer()
        repo_id = f"ablation_{repo_name}"
        existing = indexer.get_repository(repo_id)
        if existing is not None:
            return

        if fixture_dir is None:
            fixtures_dir = _PROJECT_ROOT / "evaluation" / "fixtures"
            fixture_dir = fixtures_dir / repo_name

        self.index_fixture(repo_name, fixture_dir.parent, subdirectory=fixture_dir.name)

    # ------------------------------------------------------------------ #
    # Retrieval methods
    # ------------------------------------------------------------------ #

    def _retrieve_bm25(
        self, repo_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        indexer = self._get_indexer()
        indexer.store.connect()
        try:
            return indexer.store.search_chunks(repo_id, query, limit=limit)
        finally:
            indexer.store.close()

    def _retrieve_dense(
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
        return indexer.hybrid_search(
            repo_id,
            query,
            limit=limit,
            rrf_k=60,
            embedding_client=self.embed_client,
        )

    # ------------------------------------------------------------------ #
    # Benchmark runner
    # ------------------------------------------------------------------ #

    def run_single_method(
        self,
        queries: List[Dict[str, Any]],
        method: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Run retrieval benchmark for a single method."""
        retriever = {
            "bm25": self._retrieve_bm25,
            "dense": self._retrieve_dense,
            "hybrid": self._retrieve_hybrid,
        }[method]

        query_results = []
        retrieval_latencies: List[float] = []

        for query_entry in queries:
            repo_name = query_entry["repo"]
            repo_id = f"ablation_{repo_name}"
            query_text = query_entry["query"]
            query_id = query_entry["query_id"]

            self._ensure_indexed(repo_name)

            # Run retrieval with timing
            search_start = time.perf_counter()
            try:
                raw_results = retriever(repo_id, query_text, limit=limit)
                search_duration = time.perf_counter() - search_start
                search_error = None
            except Exception as exc:
                search_duration = time.perf_counter() - search_start
                raw_results = []
                search_error = str(exc)

            retrieval_latencies.append(search_duration)

            # Normalize results to document identifiers (file paths), deduplicated
            seen_ids = set()
            retrieved_ids = []
            for r in raw_results:
                fid = r["file_path"]
                if fid not in seen_ids:
                    seen_ids.add(fid)
                    retrieved_ids.append(fid)

            # Expected documents from gold standard
            expected_files = set(query_entry.get("expected_files", []))
            expected_symbols = set(query_entry.get("expected_symbols", []))

            # Compute metrics
            metrics = evaluate_query(retrieved_ids, expected_files, k_values=(1, 5, 10, 20))

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

            # Duplicate detection
            duplicate_rate = 0.0
            if len(raw_results) > 0:
                unique_ids = set(r["id"] for r in raw_results)
                duplicate_rate = 1.0 - (len(unique_ids) / len(raw_results))

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
                    "retrieval_duration_seconds": round(search_duration, 4),
                    "search_error": search_error,
                    "metrics": metrics,
                    "failure_category": failure_category,
                    "duplicate_rate": duplicate_rate,
                }
            )

        aggregate = aggregate_metrics([q["metrics"] for q in query_results])
        aggregate["num_queries"] = len(query_results)

        # Latency statistics
        latencies_ms = [l * 1000 for l in retrieval_latencies]
        aggregate["retrieval_latency_p50_ms"] = round(
            statistics.median(latencies_ms), 2
        )
        aggregate["retrieval_latency_p90_ms"] = round(
            _percentile(latencies_ms, 90), 2
        )
        aggregate["retrieval_latency_p95_ms"] = round(
            _percentile(latencies_ms, 95), 2
        )
        aggregate["retrieval_latency_mean_ms"] = round(
            statistics.mean(latencies_ms), 2
        )
        aggregate["retrieval_latency_min_ms"] = round(min(latencies_ms), 2)
        aggregate["retrieval_latency_max_ms"] = round(max(latencies_ms), 2)

        # Average duplicate rate
        aggregate["avg_duplicate_rate"] = round(
            statistics.mean([q["duplicate_rate"] for q in query_results]), 4
        )

        return {
            "method": method,
            "embeddings_available": self.embed_client is not None,
            "limit": limit,
            "query_results": query_results,
            "aggregate_metrics": aggregate,
        }

    def run_ablation(
        self,
        queries: List[Dict[str, Any]],
        methods: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Run ablation across all retrieval methods."""
        if methods is None:
            methods = ["bm25"]
            if self.embed_client is not None:
                methods.extend(["dense", "hybrid"])

        all_results: Dict[str, Dict[str, Any]] = {}

        for method in methods:
            result = self.run_single_method(queries, method, limit=limit)
            all_results[method] = result

        return {
            "retrieval_methods": methods,
            "embeddings_available": self.embed_client is not None,
            "limit": limit,
            "results_by_method": all_results,
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": "gold_standard",
            "num_queries": len(queries),
        }

    def generate_report(
        self,
        ablation_result: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate JSON and Markdown reports."""
        if output_dir is None:
            output_dir = (
                _PROJECT_ROOT / "evaluation" / "benchmark" / "results"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Write JSON
        json_path = output_dir / f"retrieval_ablation_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(ablation_result, f, indent=2)

        # Write Markdown
        md_path = output_dir / f"retrieval_ablation_{timestamp}.md"
        self._write_markdown_report(ablation_result, md_path)

        return {
            "json_path": str(json_path),
            "md_path": str(md_path),
            "report": ablation_result,
        }

    def _write_markdown_report(
        self, report: Dict[str, Any], path: Path
    ) -> None:
        lines = [
            "# Sherlock Retrieval Ablation Benchmark Report",
            "",
            f"- **Benchmark timestamp:** {report['benchmark_timestamp']}",
            f"- **Dataset:** {report['dataset']}",
            f"- **Number of queries:** {report['num_queries']}",
            f"- **Embeddings available:** {report['embeddings_available']}",
            f"- **Retrieval methods:** {', '.join(report['retrieval_methods'])}",
            f"- **Limit:** {report['limit']}",
            "",
            "## Method Comparison",
            "",
        ]

        # Build comparison table
        methods = report["retrieval_methods"]
        ma = report["results_by_method"]

        header = "| Metric | " + " | ".join(methods) + " |"
        separator = "|" + "---|" * (len(methods) + 1)
        lines.append(header)
        lines.append(separator)

        metric_keys = [
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@20",
            "precision@1",
            "precision@5",
            "precision@10",
            "mrr",
            "ndcg@5",
            "ndcg@10",
            "retrieval_latency_p50_ms",
            "retrieval_latency_p95_ms",
            "avg_duplicate_rate",
        ]

        for key in metric_keys:
            row = [key]
            for method in methods:
                val = ma[method]["aggregate_metrics"].get(key, 0.0)
                if "ms" in key:
                    row.append(f"{val:.2f} ms")
                elif key == "avg_duplicate_rate":
                    row.append(f"{val:.4f}")
                else:
                    row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Per-Category Recall@5")
        lines.append("")
        lines.append("| Category | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))

        all_categories = sorted(
            set().union(*[
                set(
                    qr["category"]
                    for qr in ma[method]["query_results"]
                )
                for method in methods
            ])
        )
        for cat in all_categories:
            row = [cat]
            for method in methods:
                cat_results = [
                    q for q in ma[method]["query_results"] if q["category"] == cat
                ]
                if cat_results:
                    cat_agg = aggregate_metrics([q["metrics"] for q in cat_results])
                    val = cat_agg.get("recall@5", 0.0)
                    row.append(f"{val:.4f}")
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")

        with open(path, "w") as f:
            f.write("\n".join(lines))


def main():
    queries = json.load(open(GOLD_STANDARD_PATH))["queries"]

    print("Sherlock Retrieval Ablation Benchmark")
    print("=" * 60)
    print(f"Queries: {len(queries)}")
    print()

    with RetrievalAblationBenchmark() as bench:
        result = bench.run_ablation(queries)

        # Print comparison
        methods = result["retrieval_methods"]
        ma = result["results_by_method"]

        print("\n## Method Comparison")
        print(
            f"{'Metric':<25} " + " ".join(f"{m:>12}" for m in methods)
        )
        print("-" * (25 + 13 * len(methods)))

        metric_keys = [
            ("recall@1", "recall@1"),
            ("recall@5", "recall@5"),
            ("recall@10", "recall@10"),
            ("mrr", "MRR"),
            ("ndcg@5", "nDCG@5"),
            ("retrieval_latency_p50_ms", "Latency P50 (ms)"),
        ]

        for key, label in metric_keys:
            row = f"{label:<25}"
            for method in methods:
                val = ma[method]["aggregate_metrics"].get(key, 0.0)
                row += f" {val:>12.4f}"
            print(row)

        # Save report
        output = bench.generate_report(result)
        print(f"\nReport saved to:")
        print(f"  JSON: {output['json_path']}")
        print(f"  MD:   {output['md_path']}")


if __name__ == "__main__":
    main()
