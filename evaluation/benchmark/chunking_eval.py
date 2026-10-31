"""Sherlock Chunking Strategy Evaluation Benchmark.

Compares Sherlock's current production chunker (fixed 60-line blocks) against
alternative chunking strategies by directly inserting chunks into isolated
SQLite databases. Does NOT modify production code.

Strategies evaluated:
1. Current production chunker (fixed 60-line, non-overlapping) - PRODUCTION
2. Fixed-size token chunking (~300 tokens, no overlap)
3. Recursive chunking (paragraph-aware splits on blank lines)

AST-aware strategies (4, 5) require access to file content and path which
the production _chunk_lines() signature does not provide. They are documented
but not benchmarked without modifying production code.

Run with:
    .venv/bin/python evaluation/benchmark/chunking_eval.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.storage import IndexStore
from app.tools.filesys import SafeFilesystem
from app.tools.repository_scanner import RepositoryScanner
from config import SherlockConfig

from evaluation.metrics import evaluate_query, aggregate_metrics

GOLD_STANDARD_PATH = (
    _PROJECT_ROOT / "evaluation" / "datasets" / "gold_standard.json"
)

# Token estimation heuristic (same as context_builder)
def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 0.75) + text.count(".") + text.count(","))


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + (sorted_data[f + 1] - sorted_data[f]) * c
    return sorted_data[f]


# --------------------------------------------------------------------------- #
# Chunking Strategies (all receive lines only, matching production signature)
# --------------------------------------------------------------------------- #

def chunk_current_production(lines: List[str], max_lines: int = 60) -> List[Tuple[int, int, str]]:
    """Strategy 1: Current production - fixed 60-line blocks, no overlap."""
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        block = "\n".join(lines[start:end])
        chunks.append((start + 1, end, block))
        start = end
    return chunks


def chunk_fixed_token(lines: List[str], max_tokens: int = 300) -> List[Tuple[int, int, str]]:
    """Strategy 2: Fixed-size token chunking (~300 tokens, no overlap)."""
    chunks = []
    start = 0
    while start < len(lines):
        end = start
        token_count = 0
        while end < len(lines):
            line_tokens = _estimate_tokens(lines[end])
            if token_count + line_tokens > max_tokens and end > start:
                break
            token_count += line_tokens
            end += 1
        if end == start:
            end = start + 1
        block = "\n".join(lines[start:end])
        chunks.append((start + 1, end, block))
        start = end
    return chunks


def chunk_recursive(lines: List[str]) -> List[Tuple[int, int, str]]:
    """Strategy 3: Recursive chunking - split on blank lines (paragraphs)."""
    paragraphs: List[Tuple[int, int, str]] = []
    para_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and i > para_start:
            block = "\n".join(lines[para_start:i])
            if block.strip():
                paragraphs.append((para_start + 1, i, block))
            para_start = i + 1
    if para_start < len(lines):
        block = "\n".join(lines[para_start:])
        if block.strip():
            paragraphs.append((para_start + 1, len(lines), block))

    if not paragraphs:
        return chunk_current_production(lines)
    return paragraphs


# Note: AST-aware strategies (4, 5) need content + file_path which
# production _chunk_lines() doesn't provide. They're listed for reference
# but would require production code changes to test.

CHUNKING_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "current_production": {
        "display": "Current Production (60-line blocks)",
        "func": chunk_current_production,
    },
    "fixed_token": {
        "display": "Fixed Token (~300 tokens)",
        "func": chunk_fixed_token,
    },
    "recursive": {
        "display": "Recursive (paragraph-aware)",
        "func": chunk_recursive,
    },
}


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


class ChunkingBenchmark:
    """Evaluate chunking strategies against the gold-standard dataset."""

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.embed_client: Optional[EmbeddingClient] = None
        if use_embeddings:
            try:
                from app.llm.embedding_client import EmbeddingClient as EC
                self.embed_client = EC()
                if not self.embed_client.is_available():
                    self.embed_client = None
            except Exception:
                self.embed_client = None

        self.workspace_dir = Path(tempfile.mkdtemp(prefix="sherlock_chunk_"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def _index_repo_with_strategy(
        self,
        strategy_name: str,
        repo_name: str,
        fixture_dir: Path,
        subdirectory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a repository using a specific chunking strategy.

        Reads files directly, applies the chunking strategy, inserts chunks
        into a fresh SQLite database, and optionally embeds them.
        """
        db_path = self.workspace_dir / f"chunk_{strategy_name}_{repo_name}.db"
        if db_path.exists():
            db_path.unlink()

        eval_cfg = SherlockConfig(index_db_path=db_path)
        store = IndexStore(cfg=eval_cfg)
        store.connect()

        start = time.monotonic()

        # Use fixture_dir.parent (the fixtures dir) as root so paths match
        # gold standard format (e.g., "calc/calculator.py").
        root = fixture_dir.parent
        fs = SafeFilesystem(root, eval_cfg)

        # Register repository with the subdirectory path
        repo_path = str(fixture_dir)
        repo_id = f"chunk_{strategy_name}_{repo_name}"
        store.register_repository(repo_id, repo_path, repo_name)

        chunk_count = 0
        file_count = 0
        total_lines = 0
        all_chunk_sizes: List[int] = []

        for p in sorted(root.rglob("*")):
            # Only include files under this repo's subdirectory
            try:
                p.relative_to(fixture_dir)
            except ValueError:
                continue
            if not p.is_file():
                continue

            rel_path = str(p.relative_to(root))
            is_binary = fs.is_binary(p)
            is_secret = fs.is_secret_file(p)
            size_bytes = p.stat().st_size

            if is_binary or is_secret or size_bytes > eval_cfg.max_file_size_bytes:
                continue

            content, truncated = fs.read_file(p)
            if truncated:
                continue

            lines = content.splitlines()
            line_count = len(lines)
            total_lines += line_count
            language = "Python" if p.suffix == ".py" else "Text"
            file_id = f"{repo_id}:{rel_path}"
            mtime = p.stat().st_mtime
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            store.upsert_file(
                repo_id=repo_id,
                file_id=file_id,
                path=rel_path,
                language=language,
                size_bytes=size_bytes,
                line_count=line_count,
                is_binary=is_binary,
                is_secret=is_secret,
                mtime=mtime,
                content_hash=content_hash,
            )

            # Apply chunking strategy
            chunk_func = CHUNKING_STRATEGIES[strategy_name]["func"]
            chunks = chunk_func(lines)

            for start_line, end_line, block in chunks:
                store.add_chunk(
                    repo_id=repo_id,
                    file_id=file_id,
                    content=block,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="code",
                )
                all_chunk_sizes.append(_estimate_tokens(block))
                chunk_count += 1

            file_count += 1

        # Embed chunks if available
        embedding_duration = None
        embedded_count = 0
        if self.embed_client is not None:
            emb_start = time.monotonic()
            chunk_rows = store.connect().execute(
                "SELECT id, content FROM chunks WHERE repo_id = ? ORDER BY id",
                (repo_id,),
            ).fetchall()
            for crow in chunk_rows:
                try:
                    vector = self.embed_client.embed_text(crow["content"])
                    store.upsert_chunk_embedding(crow["id"], vector)
                    embedded_count += 1
                except Exception:
                    pass
            embedding_duration = time.monotonic() - emb_start

        duration = time.monotonic() - start
        store.close()

        return {
            "repo_id": repo_id,
            "repo_name": repo_name,
            "strategy": strategy_name,
            "status": "complete",
            "file_count": file_count,
            "total_lines": total_lines,
            "chunk_count": chunk_count,
            "duration_seconds": round(duration, 4),
            "embedding_duration_seconds": round(embedding_duration, 4)
            if embedding_duration is not None else None,
            "embedded_chunks": embedded_count,
            "chunk_sizes_tokens": all_chunk_sizes,
            "embeddings_available": self.embed_client is not None,
        }

    def _retrieve_from_db(
        self,
        strategy_name: str,
        repo_name: str,
        query: str,
        limit: int,
        retrieval_method: str = "bm25",
    ) -> List[Dict[str, Any]]:
        """Retrieve from the strategy-specific database."""
        db_path = self.workspace_dir / f"chunk_{strategy_name}_{repo_name}.db"
        if not db_path.exists():
            return []

        eval_cfg = SherlockConfig(index_db_path=db_path)
        store = IndexStore(cfg=eval_cfg)
        store.connect()

        try:
            repo_id = f"chunk_{strategy_name}_{repo_name}"
            if retrieval_method == "bm25":
                results = store.search_chunks(repo_id, query, limit=limit)
            elif retrieval_method == "dense" and self.embed_client is not None:
                results = store.search_chunk_embeddings(
                    repo_id, self.embed_client.embed_text(query), limit=limit
                )
            else:
                results = store.search_chunks(repo_id, query, limit=limit)
            return results
        finally:
            store.close()

    def _evaluate_strategy(
        self,
        strategy_name: str,
        queries: List[Dict[str, Any]],
        limit: int = 10,
        retrieval_method: str = "bm25",
    ) -> Dict[str, Any]:
        """Evaluate a single chunking strategy across all queries."""
        fixtures_dir = _PROJECT_ROOT / "evaluation" / "fixtures"

        # Index all repos with this strategy first
        repo_index_info: Dict[str, Dict[str, Any]] = {}
        for query_entry in queries:
            repo_name = query_entry["repo"]
            if repo_name in repo_index_info:
                continue
            fixture_dir = fixtures_dir / repo_name
            info = self._index_repo_with_strategy(
                strategy_name, repo_name, fixture_dir
            )
            repo_index_info[repo_name] = info

        # Run all queries
        query_results = []
        retrieval_latencies: List[float] = []

        for query_entry in queries:
            repo_name = query_entry["repo"]
            query_text = query_entry["query"]
            query_id = query_entry["query_id"]

            search_start = time.perf_counter()
            try:
                raw_results = self._retrieve_from_db(
                    strategy_name, repo_name, query_text, limit, retrieval_method
                )
                search_duration = time.perf_counter() - search_start
                search_error = None
            except Exception as exc:
                search_duration = time.perf_counter() - search_start
                raw_results = []
                search_error = str(exc)

            retrieval_latencies.append(search_duration)

            retrieved_ids = [r["file_path"] for r in raw_results]
            # Deduplicate while preserving order - multiple chunks from same file
            seen = set()
            retrieved_ids_deduped = []
            for fid in retrieved_ids:
                if fid not in seen:
                    seen.add(fid)
                    retrieved_ids_deduped.append(fid)
            expected_files = set(query_entry.get("expected_files", []))
            metrics = evaluate_query(retrieved_ids_deduped, expected_files, k_values=(1, 5, 10))

            query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "category": query_entry.get("category", "unknown"),
                    "repo": repo_name,
                    "strategy": strategy_name,
                    "retrieved_files": retrieved_ids,
                    "num_retrieved": len(retrieved_ids),
                    "retrieval_duration_seconds": round(search_duration, 4),
                    "search_error": search_error,
                    "metrics": metrics,
                }
            )

        aggregate = aggregate_metrics([q["metrics"] for q in query_results])
        aggregate["num_queries"] = len(query_results)

        # Latency stats
        latencies_ms = [l * 1000 for l in retrieval_latencies]
        aggregate["retrieval_latency_p50_ms"] = round(statistics.median(latencies_ms), 2)
        aggregate["retrieval_latency_p95_ms"] = round(_percentile(latencies_ms, 95), 2)
        aggregate["retrieval_latency_mean_ms"] = round(statistics.mean(latencies_ms), 2)

        # Chunk size stats from indexed repos
        all_chunk_sizes: List[int] = []
        total_chunks = 0
        total_files = 0
        for info in repo_index_info.values():
            all_chunk_sizes.extend(info.get("chunk_sizes_tokens", []))
            total_chunks += info.get("chunk_count", 0)
            total_files += info.get("file_count", 0)

        if all_chunk_sizes:
            aggregate["chunk_count"] = total_chunks
            aggregate["file_count"] = total_files
            aggregate["avg_chunk_tokens"] = round(statistics.mean(all_chunk_sizes), 1)
            aggregate["median_chunk_tokens"] = round(statistics.median(all_chunk_sizes), 1)
            aggregate["p95_chunk_tokens"] = round(_percentile(all_chunk_sizes, 95), 1)
            aggregate["max_chunk_tokens"] = max(all_chunk_sizes)

        return {
            "strategy": strategy_name,
            "retrieval_method": retrieval_method,
            "query_results": query_results,
            "aggregate_metrics": aggregate,
            "index_info": repo_index_info,
        }

    def cleanup(self):
        """Remove temporary databases."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)


def main():
    queries = json.load(open(GOLD_STANDARD_PATH))["queries"]

    print("Sherlock Chunking Strategy Evaluation")
    print("=" * 60)
    print(f"Queries: {len(queries)}")
    try:
        from app.llm.embedding_client import EmbeddingClient as EC
        ec = EC()
        print(f"Embeddings available: {ec.is_available()}")
    except Exception:
        print("Embeddings available: False")
    print()

    results = []

    for strategy_name, strategy_info in CHUNKING_STRATEGIES.items():
        print(f"\n--- Evaluating: {strategy_info['display']} ---")

        bench = ChunkingBenchmark()
        try:
            result = bench._evaluate_strategy(
                strategy_name, queries, limit=10, retrieval_method="bm25"
            )
            results.append(result)

            am = result["aggregate_metrics"]
            print(f"  recall@1:  {am.get('recall@1', 0):.4f}")
            print(f"  recall@5:  {am.get('recall@5', 0):.4f}")
            print(f"  recall@10: {am.get('recall@10', 0):.4f}")
            print(f"  MRR:       {am.get('mrr', 0):.4f}")
            print(f"  nDCG@5:    {am.get('ndcg@5', 0):.4f}")
            print(f"  Latency P50: {am.get('retrieval_latency_p50_ms', 'N/A')} ms")
            print(f"  Chunks: {am.get('chunk_count', 'N/A')}, avg tokens: {am.get('avg_chunk_tokens', 'N/A')}")
        finally:
            bench.cleanup()

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<32} {'recall@1':>10} {'recall@5':>10} {'recall@10':>11} {'MRR':>8} {'nDCG@5':>8}")
    print("-" * 75)
    for r in results:
        am = r["aggregate_metrics"]
        name = r["strategy"]
        print(
            f"{name:<32} "
            f"{am.get('recall@1', 0):>10.4f} "
            f"{am.get('recall@5', 0):>10.4f} "
            f"{am.get('recall@10', 0):>11.4f} "
            f"{am.get('mrr', 0):>8.4f} "
            f"{am.get('ndcg@5', 0):>8.4f}"
        )


if __name__ == "__main__":
    main()
