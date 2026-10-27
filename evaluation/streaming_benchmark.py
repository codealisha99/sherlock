"""Streaming benchmark for Sherlock RAG - measures TTFT, TTFP, and total latency.

Compares streaming vs non-streaming (baseline) performance.
Uses time.perf_counter() for high-resolution timing.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.indexer import RepositoryIndexer
from app.rag.pipeline import RAGPipeline


# --------------------------------------------------------------------------- #
# Benchmark questions (same as baseline for fair comparison)
# --------------------------------------------------------------------------- #

BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "question": "Where is the Calculator class defined?",
        "category": "simple_symbol_lookup",
        "repo": "bench-repo",
    },
    {
        "id": "q2",
        "question": "How does the add function work?",
        "category": "function_explanation",
        "repo": "bench-repo",
    },
    {
        "id": "q3",
        "question": "How do calculator and operations work together?",
        "category": "cross_file_reasoning",
        "repo": "bench-repo",
    },
    {
        "id": "q4",
        "question": "What is the quantum entanglement implementation?",
        "category": "unanswerable",
        "repo": "bench-repo",
    },
    {
        "id": "q5",
        "question": "How does authentication work in this codebase?",
        "category": "multi_file_concept",
        "repo": "bench-repo",
    },
]


def _percentile(data: List[float], p: int) -> float:
    """Compute the p-th percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + (sorted_data[f + 1] - sorted_data[f]) * c
    return sorted_data[f]


def run_single_query_streaming(
    pipeline: RAGPipeline,
    repo_id: str,
    question: str,
    retrieval_method: str = "hybrid",
) -> Dict[str, Any]:
    """Run a single streaming RAG query with detailed timing.

    Measures:
    - TTFP: Time to first progress event
    - TTFT: Time to first token chunk
    - Total latency: Time to done event
    - Per-stage latencies
    """
    t0 = time.perf_counter()

    ttfp_ms = None  # Time to first progress
    ttft_ms = None  # Time to first token
    first_progress_event = None

    num_chunks_retrieved = 0
    num_chunks_used = 0
    token_count = 0
    accumulated_answer = ""
    citations = []
    abstained = False
    abstention_reason = None
    stage_latencies: Dict[str, float] = {}
    error_message = None
    total_ms = 0.0

    async def _run():
        nonlocal ttfp_ms, ttft_ms, first_progress_event
        nonlocal num_chunks_retrieved, num_chunks_used, token_count
        nonlocal accumulated_answer, citations, abstained, abstention_reason
        nonlocal error_message, total_ms

        async for event in pipeline.ask_stream(
            repo_id=repo_id,
            question=question,
            retrieval_method=retrieval_method,
            limit=10,
        ):
            etype = event.get("type")
            elapsed = (time.perf_counter() - t0) * 1000

            if etype == "progress":
                stage = event.get("stage", "unknown")
                if ttfp_ms is None:
                    ttfp_ms = elapsed
                    first_progress_event = stage

                # Record stage latencies
                if "latency_ms" in event:
                    stage_latencies[stage] = event["latency_ms"]

                if stage == "retrieval_complete":
                    num_chunks_retrieved = event.get("num_chunks", 0)
                elif stage == "context_assembly_complete":
                    num_chunks_used = event.get("num_chunks", 0)

            elif etype == "token":
                if ttft_ms is None:
                    ttft_ms = elapsed
                token_count += 1
                accumulated_answer += event.get("text", "")

            elif etype == "done":
                abstained = event.get("abstained", False)
                abstention_reason = event.get("abstention_reason")
                citations = event.get("citations", [])
                num_chunks_retrieved = event.get("num_chunks_retrieved", num_chunks_retrieved)
                num_chunks_used = event.get("num_chunks_used", num_chunks_used)
                total_ms = event.get("total_ms", elapsed)
                break

            elif etype == "error":
                error_message = event.get("message", "Unknown error")
                total_ms = elapsed
                break

    asyncio.run(_run())

    return {
        "ttfp_ms": ttfp_ms,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "first_progress_event": first_progress_event,
        "num_chunks_retrieved": num_chunks_retrieved,
        "num_chunks_used": num_chunks_used,
        "token_count": token_count,
        "stage_latencies": stage_latencies,
        "abstained": abstained,
        "abstention_reason": abstention_reason,
        "error": error_message,
        "answer_length": len(accumulated_answer),
        "citations_valid": len(citations),
        "retrieval_method": retrieval_method,
    }


def run_streaming_benchmark(repetitions: int = 5) -> Dict[str, Any]:
    """Run the streaming benchmark."""
    indexer = RepositoryIndexer()
    pipeline = RAGPipeline(indexer=indexer)

    results: List[Dict[str, Any]] = []

    for q in BENCHMARK_QUESTIONS:
        repo_id = f"eval_{q['repo']}"
        question_results = []

        for rep in range(repetitions):
            result = run_single_query_streaming(pipeline, repo_id, q["question"])
            result["query_id"] = q["id"]
            result["question"] = q["question"]
            result["category"] = q["category"]
            result["repetition"] = rep
            question_results.append(result)

        results.append(question_results)

    indexer.store.close()

    # Compute aggregates per question
    aggregates = []
    for q_results in results:
        successful = [r for r in q_results if not r.get("abstained") and not r.get("error")]
        if not successful:
            continue

        total_times = [r["total_ms"] for r in successful]
        ttft_times = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
        ttfp_times = [r["ttfp_ms"] for r in successful if r["ttfp_ms"] is not None]

        aggregates.append({
            "query_id": q_results[0]["query_id"],
            "question": q_results[0]["question"],
            "category": q_results[0]["category"],
            "repetitions": len(successful),
            "total_p50_ms": statistics.median(total_times),
            "total_min_ms": min(total_times),
            "total_max_ms": max(total_times),
            "ttft_p50_ms": statistics.median(ttft_times) if ttft_times else 0,
            "ttft_min_ms": min(ttft_times) if ttft_times else 0,
            "ttft_max_ms": max(ttft_times) if ttft_times else 0,
            "ttfp_p50_ms": statistics.median(ttfp_times) if ttfp_times else 0,
            "ttfp_min_ms": min(ttfp_times) if ttfp_times else 0,
            "ttfp_max_ms": max(ttfp_times) if ttfp_times else 0,
            "avg_tokens": statistics.mean([r["token_count"] for r in successful]),
            "avg_chunks_used": statistics.mean([r["num_chunks_used"] for r in successful]),
        })

    # Overall aggregates
    all_results_flat = [r for qr in results for r in qr if not r.get("abstained") and not r.get("error")]
    if all_results_flat:
        all_total = [r["total_ms"] for r in all_results_flat]
        all_ttft = [r["ttft_ms"] for r in all_results_flat if r["ttft_ms"] is not None]
        all_ttfp = [r["ttfp_ms"] for r in all_results_flat if r["ttfp_ms"] is not None]

        overall = {
            "total_queries": len(aggregates),
            "total_successful": len(all_results_flat),
            "overall_total_p50_ms": statistics.median(all_total),
            "overall_total_p95_ms": _percentile(all_total, 95),
            "overall_total_min_ms": min(all_total),
            "overall_total_max_ms": max(all_total),
            "overall_ttft_p50_ms": statistics.median(all_ttft) if all_ttft else 0,
            "overall_ttft_p95_ms": _percentile(all_ttft, 95) if all_ttft else 0,
            "overall_ttfp_p50_ms": statistics.median(all_ttfp) if all_ttfp else 0,
            "overall_ttfp_p95_ms": _percentile(all_ttfp, 95) if all_ttfp else 0,
            "overall_avg_tokens": statistics.mean([r["token_count"] for r in all_results_flat]),
        }
    else:
        overall = {
            "total_queries": len(aggregates),
            "total_successful": 0,
        }

    return {
        "timestamp": time.time(),
        "mode": "streaming",
        "repetitions_per_query": repetitions,
        "aggregates": aggregates,
        "overall": overall,
    }


def main():
    repetitions = 5
    print(f"Sherlock Streaming Benchmark ({repetitions} repetitions per query)")
    print("=" * 70)

    result = run_streaming_benchmark(repetitions=repetitions)

    print("\n--- Per-Query Results ---")
    for agg in result["aggregates"]:
        print(f"\n[{agg['query_id']}] {agg['category']}")
        print(f"  Question: {agg['question']}")
        print(f"  Reps: {agg['repetitions']}")
        print(f"  Total p50: {agg['total_p50_ms']:.1f}ms (min={agg['total_min_ms']:.1f}, max={agg['total_max_ms']:.1f})")
        print(f"  TTFT p50:  {agg['ttft_p50_ms']:.1f}ms (min={agg['ttft_min_ms']:.1f}, max={agg['ttft_max_ms']:.1f})")
        print(f"  TTFP p50:  {agg['ttfp_p50_ms']:.1f}ms (min={agg['ttfp_min_ms']:.1f}, max={agg['ttfp_max_ms']:.1f})")
        print(f"  Avg tokens: {agg['avg_tokens']:.1f}")
        print(f"  Avg chunks used: {agg['avg_chunks_used']:.1f}")

    print("\n--- Overall (Streaming) ---")
    o = result["overall"]
    print(f"  Total queries: {o['total_queries']}")
    print(f"  Successful: {o['total_successful']}")
    print(f"  Total p50:  {o['overall_total_p50_ms']:.1f}ms")
    print(f"  Total p95:  {o['overall_total_p95_ms']:.1f}ms")
    print(f"  Total min:  {o['overall_total_min_ms']:.1f}ms")
    print(f"  Total max:  {o['overall_total_max_ms']:.1f}ms")
    print(f"  TTFT p50:   {o['overall_ttft_p50_ms']:.1f}ms")
    print(f"  TTFT p95:   {o['overall_ttft_p95_ms']:.1f}ms")
    print(f"  TTFP p50:   {o['overall_ttfp_p50_ms']:.1f}ms")
    print(f"  TTFP p95:   {o['overall_ttfp_p95_ms']:.1f}ms")
    print(f"  Avg tokens: {o['overall_avg_tokens']:.1f}")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "streaming_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
