# Sherlock Development Ledger

Records each delivery increment: feature/fix, tests, commit hash, push result, remaining work.

## Milestone: Phase 1 - Hybrid Retrieval (RRF)

| Increment | Feature/Fix | Tests | Commit Hash | Push Result | Remaining |
|-----------|-------------|-------|-------------|-------------|-----------|
| 1 | Baseline audit + architecture map | Existing test suite (61 tests pass) | 832d647 (pre-existing) | Pushed (pre-existing) | Normalize lexical results |
| 2 | Record reproducible test commands | unittest discover -s tests -s evaluation/tests | — | — | Implement RRF |
| 3 | Normalize lexical + semantic result structures; RRF fusion; dedup; repo scoping; result limits; CLI hybrid-search | tests/test_hybrid_search.py (10 tests pass) | 419e2ec | Pushed to origin/main | CLI help verification |
| 4 | CLI help verification; invalid limit/rrf_k validation at storage + indexer + CLI layers | tests/test_cli_embedding.py (9 new CLI tests) + tests/test_hybrid_search.py (6 new storage validation tests) | d161fa6 | Pushed to origin/main | Retrieval diagnostics |
