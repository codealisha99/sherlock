# Sherlock Development Ledger

Records each delivery increment: feature/fix, tests, commit hash, push result, remaining work.

## Milestone: Phase 1 - Hybrid Retrieval (RRF) [COMPLETE]

| Increment | Feature/Fix | Tests | Commit Hash | Push Result | Remaining |
|---|---|---|---|---|---|
| 1 | Baseline audit + architecture map | Existing test suite (61 tests pass) | 832d647 (pre-existing) | Pushed (pre-existing) | Normalize lexical results |
| 2 | Record reproducible test commands | unittest discover -s tests -s evaluation/tests | — | — | Implement RRF |
| 3 | Normalize lexical + semantic result structures; RRF fusion; dedup; repo scoping; result limits; CLI hybrid-search | tests/test_hybrid_search.py (10 tests pass) | 419e2ec | Pushed to origin/main | CLI help verification |
| 4 | CLI help verification; invalid limit/rrf_k validation at storage + indexer + CLI layers | tests/test_cli_embedding.py (9 new CLI tests) + tests/test_hybrid_search.py (6 new storage validation tests) | d161fa6 | Pushed to origin/main | Retrieval diagnostics |

## Milestone: Phase 2 - Grounded RAG (Context Assembly, Generation, Citations, Abstention)

| Increment | Feature/Fix | Tests | Commit Hash | Push Result | Remaining |
|---|---|---|---|---|---|
| 5 | Context assembly: ContextChunk/ContextBundle dataclasses, dedup, ranking preservation, token budget, metadata preservation | tests/test_context_builder.py (new) | 5d65da6 | Pushed to origin/main | LLM generation interface |
| 6 | RAG orchestration pipeline: retrieve → context → generate → validate → abstain | tests/test_rag_pipeline.py (new) | 66a21da | Pushed to origin/main | Citation validation & abstention tests |
| 7 | Citation validation: validate source IDs against evidence map, path/line range validation | tests/test_citation_validator.py (new) | — | — | Abstention logic |
| 8 | Abstention logic: no evidence, low relevance, conflicting evidence → abstain | tests/test_abstention.py (new) | — | — | CLI `ask` command |
| 9 | CLI `ask` command: natural-language Q&A with citations | tests/test_cli_ask.py (new) | — | — | API endpoint |
| 10 | RAG API endpoint: `/api/ask` using same orchestration service as CLI | tests/test_api_ask.py (new) | — | — | Frontend integration / eval |
| 11 | RAG evaluation: extend evaluation runner for semantic/hybrid + answer quality | tests/test_rag_evaluation.py (new) | — | — | Documentation |
