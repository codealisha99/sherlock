"""Standard retrieval metrics for Sherlock Milestone A.

All functions are pure and deterministic. They take ranked retrieved
document identifiers and gold-standard relevance labels and return a
float score. No dependence on SQLite, Ollama, or the indexing engine.

Conventions documented per function.
"""

from typing import Dict, Iterable, List, Sequence, Set


def _as_set(items: Iterable[str]) -> Set[str]:
    """Normalize an iterable of document identifiers into a set."""
    return set(items)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Recall@k.

    Fraction of all known relevant documents that appear within the top k
    retrieved results.

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved document identifiers (best first).
    relevant:
        Set of document identifiers judged relevant for the query.
    k:
        Cutoff rank. Must be >= 1.

    Returns
    -------
    float in [0, 1].
        0.0 if there are no relevant documents.
        1.0 if all relevant documents are within the top k.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    top_k = retrieved[:k]
    retrieved_relevant = rel_set.intersection(top_k)
    return len(retrieved_relevant) / len(rel_set)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Precision@k.

    Fraction of the top k retrieved results that are relevant.

    Convention
    ----------
    If fewer than k documents are retrieved, the denominator remains k
    (missing results count as non-relevant). This penalizes systems that
    return too few results and is the standard IR convention for fixed-cutoff
    precision.

    Returns
    -------
    float in [0, 1].
        0.0 if k < 1 or if there are no relevant documents AND no retrieved
        documents; otherwise the precision fraction.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rel_set = _as_set(relevant)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    retrieved_relevant = rel_set.intersection(top_k)
    return len(retrieved_relevant) / k


def mrr(
    retrieved: Sequence[str],
    relevant: Iterable[str],
) -> float:
    """Mean Reciprocal Rank for a single query.

    Reciprocal rank of the first relevant document in the ranked list.
    Returns 0.0 if no relevant document is retrieved.

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved document identifiers (best first).
    relevant:
        Set of document identifiers judged relevant for the query.

    Returns
    -------
    float in [0, 1].
        1.0 / rank of first relevant doc, or 0.0 if none retrieved.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved: Sequence[str],
    relevant: Iterable[str],
) -> float:
    """Average Precision (AP) for a single query.

    Average of precision@k computed at every rank k where a relevant
    document is retrieved. If there are no relevant documents, returns 0.0.
    If there are relevant documents but none are retrieved, returns 0.0.

    This is the standard "uninterpolated" average precision used in MAP.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0

    relevant_found = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            relevant_found += 1
            precision_sum += relevant_found / rank

    if relevant_found == 0:
        return 0.0
    return precision_sum / len(rel_set)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """nDCG@k with binary relevance.

    Normalized Discounted Cumulative Gain using binary relevance gains
    (relevant = 1, non-relevant = 0) and log2 discounting:
        DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
        IDCG@k = ideal DCG@k (all relevant docs first)
        nDCG@k = DCG@k / IDCG@k

    Returns 0.0 if there are no relevant documents.

    Parameters
    ----------
    retrieved:
        Ranked list of retrieved document identifiers (best first).
    relevant:
        Set of document identifiers judged relevant for the query.
    k:
        Cutoff rank. Must be >= 1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0

    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            dcg += 1.0 / _log2(i + 1)

    # Ideal DCG: min(len(rel_set), k) relevant docs at the top.
    n_relevant = min(len(rel_set), k)
    idcg = sum(1.0 / _log2(i + 1) for i in range(1, n_relevant + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _log2(x: int) -> float:
    """log2 of a positive integer."""
    import math

    return math.log2(x)


def evaluate_query(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute all standard metrics for a single query.

    Returns a flat dict keyed by metric name, e.g.:
        "recall@1", "recall@5", "recall@10",
        "precision@5", "mrr", "average_precision", "ndcg@5".

    MRR, AP, and nDCG@k are reported at their natural cutoffs.
    """
    rel_set = _as_set(relevant)
    results: Dict[str, float] = {}

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved, rel_set, k)
        results[f"precision@{k}"] = precision_at_k(retrieved, rel_set, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, rel_set, k)

    results["mrr"] = mrr(retrieved, rel_set)
    results["average_precision"] = average_precision(retrieved, rel_set)
    results["num_retrieved"] = float(len(retrieved))
    results["num_relevant"] = float(len(rel_set))
    results["num_relevant_retrieved"] = float(
        len(rel_set.intersection(retrieved))
    )
    return results


def aggregate_metrics(
    per_query_results: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-query metric dicts into overall averages.

    Averages are macro-averages (simple mean across queries). Queries with
    no relevant documents contribute 0.0 to recall/MRR/AP/nDCG per the
    metric definitions, so they are included in the mean rather than
    silently dropped.

    Parameters
    ----------
    per_query_results:
        List of dicts as produced by evaluate_query.

    Returns
    -------
    dict of metric name -> macro-averaged float.
    """
    if not per_query_results:
        return {}

    keys = per_query_results[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [r.get(key, 0.0) for r in per_query_results]
        aggregated[key] = sum(values) / len(values)
    aggregated["num_queries"] = float(len(per_query_results))
    return aggregated
