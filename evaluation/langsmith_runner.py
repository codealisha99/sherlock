"""LangSmith evaluation runner for Sherlock RAG.

Runs a local benchmark using Sherlock's RAG pipeline and optionally
synchronizes results to LangSmith (opt-in, requires credentials).

Local evaluation always works without LangSmith. Hosted evaluation is
explicitly opted-in via LANGSMITH_TRACING=true and LANGSMITH_API_KEY.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.indexer import RepositoryIndexer
from app.llm.embedding_client import EmbeddingClient
from app.llm.ollama_client import OllamaClient
from app.rag.langsmith_metrics import (
    LatencyMetrics,
    compute_request_metrics,
)
from app.rag.pipeline import RAGPipeline
from app.rag.tracing import RAGTracer, get_langsmith_config, reset_tracer
from config import SherlockConfig
from evaluation.langsmith_dataset import (
    DATASET_VERSION,
    get_answerable_examples,
    get_unanswerable_examples,
    load_dataset,
)

logger = __import__("logging").getLogger(__name__)


class LangSMEvaluationRunner:
    """Runs Sherlock RAG evaluation against the LangSmith dataset.

    Local mode: runs the benchmark using Sherlock's RAG pipeline with
    deterministic metrics. No LangSmith credentials required.

    Hosted mode: when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set,
    results are also submitted to LangSmith as traces and optionally as
    an evaluation experiment.
    """

    def __init__(
        self,
        indexer: Optional[RepositoryIndexer] = None,
        ollama_client: Optional[OllamaClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        use_langsmith: Optional[bool] = None,
    ):
        config = get_langsmith_config()
        self.use_langsmith = use_langsmith if use_langsmith is not None else config.is_ready()
        self.config = config

        # Use isolated temp database for evaluation (never pollute default DB)
        eval_cfg = SherlockConfig(index_db_path=Path(tempfile.mkdtemp(prefix="sherlock_langsmith_eval_db_")) / "index" / "sherlock.db")
        self.indexer = indexer or RepositoryIndexer(cfg=eval_cfg)
        self.ollama_client = ollama_client or OllamaClient()
        self.embedding_client = embedding_client or EmbeddingClient()

        self.tracer = RAGTracer(self.config) if self.use_langsmith else None
        self.pipeline = RAGPipeline(
            indexer=self.indexer,
            ollama_client=self.ollama_client,
            embedding_client=self.embedding_client,
            tracer=self.tracer,
        )

        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def run_example(
        self,
        example: Dict[str, Any],
        retrieval_method: str = "hybrid",
    ) -> Dict[str, Any]:
        """Run a single evaluation example.

        Returns a dict with the question, answer, metrics, and diagnostics.
        """
        repo_id = f"eval_{example['repo']}"
        question = example["question"]
        relevance_labels = example.get("retrieval_relevance", set())
        answerable = example.get("answerability", True)

        # Index the repository if not already indexed
        self._ensure_indexed(example["repo"])

        # Run the RAG pipeline with latency tracking
        start = time.monotonic()
        result = self.pipeline.ask(
            repo_id=repo_id,
            question=question,
            retrieval_method=retrieval_method,
            limit=10,
        )
        total_ms = (time.monotonic() - start) * 1000

        # Build latency metrics
        latency = LatencyMetrics(
            retrieval_ms=result.retrieval_error is None and 0.0 or None,
            assembly_ms=0.0,
            generation_ms=0.0,
            citation_validation_ms=0.0,
            total_ms=round(total_ms, 2),
        )
        # Try to get actual latencies from tracer metadata
        if self.tracer and self.tracer._trace_id:
            pass  # Latencies are recorded in trace metadata, not result

        # Compute deterministic metrics
        metrics = compute_request_metrics(
            request_id=f"eval-{example['example_id']}",
            repo_id=repo_id,
            retrieval_method=retrieval_method,
            result=result,
            latency=latency,
            relevance_labels=relevance_labels,
            answerable=answerable,
        )

        example_result = {
            "example_id": example["example_id"],
            "question": question,
            "repo": example["repo"],
            "category": example.get("category", "unknown"),
            "answerability": answerable,
            "retrieval_method": retrieval_method,
            "answer": result.answer,
            "abstained": result.abstained,
            "abstention_reason": result.abstention_reason,
            "num_chunks_retrieved": result.num_chunks_retrieved,
            "num_chunks_used": result.num_chunks_used,
            "citations_valid": len(result.citations),
            "retrieval_error": result.retrieval_error,
            "generation_error": result.generation_error,
            "metrics": metrics.to_dict(),
            "total_latency_ms": round(total_ms, 2),
            "llm_used": self.ollama_client is not None and not result.abstained,
        }

        self.results.append(example_result)
        return example_result

    def run_benchmark(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        retrieval_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full benchmark.

        Parameters
        ----------
        examples:
            List of examples to run. Defaults to all examples in the dataset.
        retrieval_methods:
            List of retrieval methods to compare. Defaults to ["hybrid"].
            When LangSmith embeddings are available, includes "semantic" too.

        Returns
        -------
        dict with per-example results and aggregate metrics by method.
        """
        if examples is None:
            examples = load_dataset()["examples"]

        if retrieval_methods is None:
            retrieval_methods = ["hybrid"]
            # Check if semantic retrieval is available
            try:
                ec = EmbeddingClient()
                if ec.is_available():
                    retrieval_methods.append("semantic")
            except Exception:
                pass

        all_results: Dict[str, Any] = {}

        for method in retrieval_methods:
            method_results = []
            for example in examples:
                result = self.run_example(example, retrieval_method=method)
                method_results.append(result)
            all_results[method] = method_results

        # Compute aggregates
        aggregates = {}
        for method, results in all_results.items():
            aggregates[method] = self._aggregate_results(results)

        return {
            "dataset_version": DATASET_VERSION,
            "num_examples": len(examples),
            "retrieval_methods": retrieval_methods,
            "use_langsmith": self.use_langsmith,
            "results_by_method": all_results,
            "aggregates": aggregates,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate metrics across examples for a single method."""
        if not results:
            return {}

        total = len(results)
        abstained = sum(1 for r in results if r["abstained"])
        answered = total - abstained
        total_latency = sum(r["total_latency_ms"] for r in results)
        avg_latency = total_latency / total if total > 0 else 0.0

        # Aggregate metric fields
        metric_keys = set()
        for r in results:
            metric_keys.update(r["metrics"].keys())

        agg_metrics: Dict[str, Any] = {}
        for key in metric_keys:
            values = [r["metrics"].get(key, 0.0) for r in results]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                agg_metrics[key] = round(sum(numeric_values) / len(numeric_values), 4)

        # Answerability-specific metrics
        answerable_examples = [r for r in results if r["answerability"] is True]
        unanswerable_examples = [r for r in results if r["answerability"] is False]

        abstention_on_unanswerable = sum(
            1 for r in unanswerable_examples if r["abstained"]
        )
        abstention_on_answerable = sum(
            1 for r in answerable_examples if r["abstained"]
        )

        return {
            "total_examples": total,
            "abstained": abstained,
            "answered": answered,
            "abstention_rate": round(abstained / total, 4) if total > 0 else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "abstention_on_unanswerable": abstention_on_unanswerable,
            "abstention_on_answerable": abstention_on_answerable,
            "unanswerable_abstention_rate": round(
                abstention_on_unanswerable / len(unanswerable_examples), 4
            ) if unanswerable_examples else 0.0,
            "answerable_abstention_rate": round(
                abstention_on_answerable / len(answerable_examples), 4
            ) if answerable_examples else 0.0,
            "metrics": agg_metrics,
        }

    # ------------------------------------------------------------------ #
    # Repository indexing
    # ------------------------------------------------------------------ #

    def _ensure_indexed(self, repo_name: str) -> None:
        """Ensure the repository is indexed (creates a fixture if needed)."""
        repo_id = f"eval_{repo_name}"
        existing = self.indexer.get_repository(repo_id)
        if existing is not None:
            return

        # Create a minimal fixture repository for evaluation
        import tempfile
        from pathlib import Path as _Path

        root = _Path(tempfile.mkdtemp(prefix="sherlock_langsmith_eval_"))
        (root / "src").mkdir(parents=True, exist_ok=True)

        # Create files based on repo name
        if repo_name == "calc":
            (root / "src" / "calculator.py").write_text(
                "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b\n"
            )
            (root / "src" / "operations.py").write_text(
                "def subtract(a, b):\n    return a - b\n"
            )
        elif repo_name == "dataproc":
            (root / "dataproc").mkdir(parents=True, exist_ok=True)
            (root / "dataproc" / "reader.py").write_text(
                "def read_csv(path):\n    import csv\n    with open(path) as f:\n        return list(csv.reader(f))\n"
            )
        elif repo_name == "utils":
            (root / "utils").mkdir(parents=True, exist_ok=True)
            (root / "utils" / "string_utils.py").write_text(
                "def slugify(text):\n    return text.lower().replace(' ', '-')\n"
            )

        self.indexer.index_repository(root, repo_id=repo_id, repo_name=repo_name)

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        benchmark_result: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a JSON and Markdown evaluation report."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "dataset_version": benchmark_result["dataset_version"],
            "num_examples": benchmark_result["num_examples"],
            "retrieval_methods": benchmark_result["retrieval_methods"],
            "use_langsmith": benchmark_result["use_langsmith"],
            "aggregates": benchmark_result["aggregates"],
            "timestamp": benchmark_result["timestamp"],
            "llm_used": any(
                r["llm_used"] for results in benchmark_result["results_by_method"].values()
                for r in results
            ),
            "local_evaluation": not benchmark_result["use_langsmith"],
        }

        # Write JSON
        json_path = output_dir / "langsmith_eval_latest.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Write Markdown
        md_path = output_dir / "langsmith_eval_latest.md"
        self._write_markdown_report(report, md_path)

        return report

    def _write_markdown_report(self, report: Dict[str, Any], path: Path) -> None:
        lines = [
            "# Sherlock LangSmith Evaluation Report",
            "",
            f"- **Dataset version:** {report['dataset_version']}",
            f"- **Number of examples:** {report['num_examples']}",
            f"- **LangSmith enabled:** {report['use_langsmith']}",
            f"- **Local evaluation:** {report['local_evaluation']}",
            f"- **LLM used:** {report['llm_used']}",
            f"- **Timestamp:** {report['timestamp']}",
            "",
            "## Retrieval Method Comparison",
            "",
            "| Method | Examples | Abstained | Answered | Abstention Rate | Avg Latency (ms) |",
            "|--------|----------|-----------|----------|-----------------|------------------|",
        ]

        for method in report["retrieval_methods"]:
            agg = report["aggregates"].get(method, {})
            lines.append(
                f"| {method} | {agg.get('total_examples', 0)} | "
                f"{agg.get('abstained', 0)} | {agg.get('answered', 0)} | "
                f"{agg.get('abstention_rate', 0):.4f} | {agg.get('avg_latency_ms', 0)} |"
            )

        lines.append("")
        lines.append("## Answerability Metrics")
        lines.append("")
        lines.append(
            "| Method | Unanswerable Abstention Rate | Answerable Abstention Rate |"
        )
        lines.append(
            "|--------|------------------------------|----------------------------|"
        )
        for method in report["retrieval_methods"]:
            agg = report["aggregates"].get(method, {})
            lines.append(
                f"| {method} | {agg.get('unanswerable_abstention_rate', 0):.4f} | "
                f"{agg.get('answerable_abstention_rate', 0):.4f} |"
            )

        lines.append("")
        lines.append("## Key Metrics by Method")
        for method in report["retrieval_methods"]:
            agg = report["aggregates"].get(method, {})
            metrics = agg.get("metrics", {})
            lines.append(f"### {method}")
            for key in sorted(metrics.keys()):
                if key not in ("num_results",):
                    lines.append(f"- {key}: {metrics[key]:.4f}")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))


def run_langsmith_evaluation(
    output_dir: Optional[Path] = None,
    retrieval_methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function to run the LangSmith evaluation.

    Parameters
    ----------
    output_dir:
        Directory for report output. Defaults to results/.
    retrieval_methods:
        List of retrieval methods to compare.

    Returns
    -------
    dict with the full evaluation report.
    """
    runner = LangSMEvaluationRunner()
    benchmark = runner.run_benchmark(retrieval_methods=retrieval_methods)
    report = runner.generate_report(benchmark, output_dir=output_dir)
    return report
