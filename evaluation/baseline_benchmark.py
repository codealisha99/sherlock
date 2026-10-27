"""Baseline benchmark for Sherlock RAG - measures current (non-streaming) performance.

Uses time.perf_counter() for high-resolution timing. Runs multiple repetitions
to capture variability. Does NOT require LangSmith or any external service.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.indexer import RepositoryIndexer
from app.rag.context_builder import assemble_context, build_citation_map, format_context_for_prompt
from app.rag.pipeline import RAGPipeline


# --------------------------------------------------------------------------- #
# Benchmark questions (representative categories)
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


def run_single_query(
    pipeline: RAGPipeline,
    repo_id: str,
    question: str,
    retrieval_method: str = "hybrid",
) -> Dict[str, Any]:
    """Run a single RAG query with detailed timing.

    Since the current pipeline is fully buffered (no streaming), we measure:
    - request_start: before pipeline.ask()
    - retrieval_done: when retrieval completes (instrumented via tracer)
    - generation_done: when generation completes
    - request_end: when pipeline.ask() returns
    """
    # We instrument by wrapping with our own timing around each stage
    # by calling the pipeline's internal methods individually with timing.

    store = pipeline.store
    indexer = pipeline.indexer

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Stage 1: Validation
    t_val_start = time.perf_counter()
    repo = store.get_repository(repo_id)
    t_val_end = time.perf_counter()
    timings["validation_ms"] = (t_val_end - t_val_start) * 1000

    if repo is None:
        return {"error": "Repository not found", "timings": timings}

    # Stage 2: Retrieval
    t_ret_start = time.perf_counter()
    try:
        raw_results = pipeline._retrieve(repo_id, question, retrieval_method, limit=10)
    except Exception as exc:
        return {
            "error": f"Retrieval failed: {exc}",
            "timings": timings,
        }
    t_ret_end = time.perf_counter()
    timings["retrieval_ms"] = (t_ret_end - t_ret_start) * 1000
    num_retrieved = len(raw_results)

    if not raw_results:
        t_end = time.perf_counter()
        return {
            "abstained": True,
            "abstention_reason": "No relevant content found",
            "total_ms": (t_end - t0) * 1000,
            "timings": timings,
            "num_chunks_retrieved": 0,
            "num_chunks_used": 0,
        }

    # Stage 3: Context assembly
    t_asm_start = time.perf_counter()
    bundle = assemble_context(raw_results, question, max_tokens=pipeline.max_context_tokens)
    t_asm_end = time.perf_counter()
    timings["context_assembly_ms"] = (t_asm_end - t_asm_start) * 1000

    if bundle.budget_exceeded or not bundle.sources:
        t_end = time.perf_counter()
        return {
            "abstained": True,
            "abstention_reason": "Context assembly failed",
            "total_ms": (t_end - t0) * 1000,
            "timings": timings,
            "num_chunks_retrieved": num_retrieved,
            "num_chunks_used": len(bundle.sources),
        }

    # Stage 4: Generation (fully buffered - no streaming)
    context_text = format_context_for_prompt(bundle)
    citation_map = build_citation_map(bundle)

    t_gen_start = time.perf_counter()
    try:
        answer_text = pipeline._generate_answer(question, context_text)
    except Exception as exc:
        return {
            "error": f"Generation failed: {exc}",
            "timings": timings,
        }
    t_gen_end = time.perf_counter()
    timings["generation_ms"] = (t_gen_end - t_gen_start) * 1000

    if not answer_text or not answer_text.strip():
        t_end = time.perf_counter()
        return {
            "abstained": True,
            "abstention_reason": "Empty answer",
            "total_ms": (t_end - t0) * 1000,
            "timings": timings,
            "num_chunks_retrieved": num_retrieved,
            "num_chunks_used": len(bundle.sources),
        }

    # Stage 5: Citation validation
    t_cit_start = time.perf_counter()
    citations = pipeline._validate_citations(answer_text, citation_map)
    t_cit_end = time.perf_counter()
    timings["citation_validation_ms"] = (t_cit_end - t_cit_start) * 1000

    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000

    # TTFT = time to first token = generation start + time to get full response
    # Since current pipeline is buffered, TTFT == generation end
    ttft_ms = (t_gen_end - t0) * 1000

    return {
        "abstained": False,
        "answer": answer_text.strip(),
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "timings": timings,
        "num_chunks_retrieved": num_retrieved,
        "num_chunks_used": len(bundle.sources),
        "citations_valid": len(citations),
        "retrieval_method": retrieval_method,
    }


def run_benchmark(repetitions: int = 5) -> Dict[str, Any]:
    """Run the full benchmark with multiple repetitions."""
    indexer = RepositoryIndexer()
    pipeline = RAGPipeline(indexer=indexer)

    results: List[Dict[str, Any]] = []

    for q in BENCHMARK_QUESTIONS:
        repo_id = f"eval_{q['repo']}"
        question_results = []

        for rep in range(repetitions):
            result = run_single_query(pipeline, repo_id, q["question"])
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
        successful = [r for r in q_results if not r.get("abstained") and "error" not in r]
        if not successful:
            continue

        total_times = [r["total_ms"] for r in successful]
        ttft_times = [r["ttft_ms"] for r in successful]
        retrieval_times = [r["timings"].get("retrieval_ms", 0) for r in successful]
        generation_times = [r["timings"].get("generation_ms", 0) for r in successful]
        assembly_times = [r["timings"].get("context_assembly_ms", 0) for r in successful]

        aggregates.append({
            "query_id": q_results[0]["query_id"],
            "question": q_results[0]["question"],
            "category": q_results[0]["category"],
            "repetitions": len(successful),
            "total_p50_ms": statistics.median(total_times),
            "total_min_ms": min(total_times),
            "total_max_ms": max(total_times),
            "ttft_p50_ms": statistics.median(ttft_times),
            "ttft_min_ms": min(ttft_times),
            "ttft_max_ms": max(ttft_times),
            "retrieval_p50_ms": statistics.median(retrieval_times),
            "generation_p50_ms": statistics.median(generation_times),
            "assembly_p50_ms": statistics.median(assembly_times),
            "avg_chunks_used": statistics.mean([r["num_chunks_used"] for r in successful]),
            "avg_citations": statistics.mean([r["citations_valid"] for r in successful]),
        })

    # Overall aggregates
    all_total = []
    all_ttft = []
    all_retrieval = []
    all_generation = []
    for agg in aggregates:
        # Re-run to get all individual times
        pass

    # Re-collect all individual times
    all_results_flat = [r for qr in results for r in qr if not r.get("abstained") and "error" not in r]
    if all_results_flat:
        all_total = [r["total_ms"] for r in all_results_flat]
        all_ttft = [r["ttft_ms"] for r in all_results_flat]
        all_retrieval = [r["timings"].get("retrieval_ms", 0) for r in all_results_flat]
        all_generation = [r["timings"].get("generation_ms", 0) for r in all_results_flat]

    overall = {
        "total_queries": len(aggregates),
        "total_successful": len(all_results_flat),
        "overall_total_p50_ms": statistics.median(all_total) if all_total else 0,
        "overall_total_p95_ms": _percentile(all_total, 95) if all_total else 0,
        "overall_total_min_ms": min(all_total) if all_total else 0,
        "overall_total_max_ms": max(all_total) if all_total else 0,
        "overall_ttft_p50_ms": statistics.median(all_ttft) if all_ttft else 0,
        "overall_ttft_p95_ms": _percentile(all_ttft, 95) if all_ttft else 0,
        "overall_retrieval_p50_ms": statistics.median(all_retrieval) if all_retrieval else 0,
        "overall_generation_p50_ms": statistics.median(all_generation) if all_generation else 0,
    }

    return {
        "timestamp": time.time(),
        "repetitions_per_query": repetitions,
        "aggregates": aggregates,
        "overall": overall,
    }


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


def main():
    repetitions = 5
    print(f"Sherlock Baseline Benchmark ({repetitions} repetitions per query)")
    print("=" * 70)

    result = run_benchmark(repetitions=repetitions)

    print("\n--- Per-Query Results ---")
    for agg in result["aggregates"]:
        print(f"\n[{agg['query_id']}] {agg['category']}")
        print(f"  Question: {agg['question']}")
        print(f"  Reps: {agg['repetitions']}")
        print(f"  Total p50: {agg['total_p50_ms']:.1f}ms (min={agg['total_min_ms']:.1f}, max={agg['total_max_ms']:.1f})")
        print(f"  TTFT p50:  {agg['ttft_p50_ms']:.1f}ms (min={agg['ttft_min_ms']:.1f}, max={agg['ttft_max_ms']:.1f})")
        print(f"  Retrieval p50: {agg['retrieval_p50_ms']:.1f}ms")
        print(f"  Generation p50: {agg['generation_p50_ms']:.1f}ms")
        print(f"  Assembly p50:  {agg['assembly_p50_ms']:.1f}ms")
        print(f"  Avg chunks used: {agg['avg_chunks_used']:.1f}")
        print(f"  Avg citations:   {agg['avg_citations']:.1f}")

    print("\n--- Overall ---")
    o = result["overall"]
    print(f"  Total queries: {o['total_queries']}")
    print(f"  Successful: {o['total_successful']}")
    print(f"  Total p50:  {o['overall_total_p50_ms']:.1f}ms")
    print(f"  Total p95:  {o['overall_total_p95_ms']:.1f}ms")
    print(f"  Total min:  {o['overall_total_min_ms']:.1f}ms")
    print(f"  Total max:  {o['overall_total_max_ms']:.1f}ms")
    print(f"  TTFT p50:   {o['overall_ttft_p50_ms']:.1f}ms")
    print(f"  TTFT p95:   {o['overall_ttft_p95_ms']:.1f}ms")
    print(f"  Retrieval p50: {o['overall_retrieval_p50_ms']:.1f}ms")
    print(f"  Generation p50: {o['overall_generation_p50_ms']:.1f}ms")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "baseline_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
