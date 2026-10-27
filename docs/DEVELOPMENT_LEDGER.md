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
| 5 | Context assembly: ContextChunk/ContextBundle dataclasses, dedup, ranking preservation, token budget, metadata preservation | tests/test_context_builder.py (14 tests pass) | 5d65da6 | Pushed to origin/main | LLM generation interface |
| 6 | RAG orchestration pipeline: retrieve → context → generate → validate → abstain | tests/test_context_builder.py (14) + tests/test_rag_pipeline.py (16) | 5b569d5 | Pushed to origin/main | CLI `ask` command |
| 7 | Citation validation: validate source IDs against evidence map (integrated into pipeline) | tests/test_rag_pipeline.py (citation tests) | 5b569d5 | Pushed to origin/main | Abstention logic |
| 8 | Abstention logic: no evidence, low relevance, conflicting evidence → abstain | tests/test_rag_pipeline.py (abstention tests) | 5b569d5 | Pushed to origin/main | CLI `ask` command |
| 9 | CLI `ask` command: natural-language Q&A with citations | tests/test_cli_ask.py (5 tests pass) | 953bc51 | Pushed to origin/main | API endpoint |
| 10 | RAG API endpoint: `/api/ask` using same orchestration service as CLI | tests/test_api_ask.py (7 tests pass) | 3dadedb | Pushed to origin/main | Frontend integration / eval |
| 11 | RAG evaluation: extend evaluation runner for semantic/hybrid + answer quality | tests/test_rag_evaluation.py (6 tests pass) | 6983aed | Pushed to origin/main | Acceptance tests |
| 12 | End-to-end acceptance tests for all 7 mandatory acceptance criteria | tests/test_acceptance_e2e.py (34 tests pass) | fd9f242 | Pushed to origin/main | LangSmith integration |

## Milestone: Phase 3 - LangSmith Observability & Evaluation

| Increment | Feature/Fix | Tests | Commit Hash | Push Result | Remaining |
|---|---|---|---|---|---|
| 13 | LangSmith dependency + configuration module (opt-in tracing, safe defaults) | — | 1c0c00e | Pushed to origin/main | Tracing instrumentation |
| 14 | LangSmith tracing instrumentation in RAG pipeline (7 stages, safe metadata) | tests/test_rag_pipeline.py (16) + tests/test_cli_ask.py (5) + tests/test_api_ask.py (7) | 4807a76 | Pushed to origin/main | Deterministic metrics module |
| 15 | LangSmith deterministic metrics module (retrieval, citation, grounding, abstention, latency) | tests/test_langsmith_metrics.py (26 tests pass) | a5e5ecd | Pushed to origin/main | CLI diagnostics command |
| 16 | CLI `langsmith` diagnostic command + integration tests | tests/test_langsmith_integration.py (30 tests pass) | 6acceb6 | Pushed to origin/main | LangSmith evaluation workflow |
| 17 | LangSmith evaluation dataset + hosted evaluation workflow | tests/test_langsmith_evaluation.py (16 tests pass) | 8bb049e | Pushed to origin/main | CLI eval command |
| 18 | CLI `langsmith-eval` command for running evaluation benchmark | tests/test_langsmith_evaluation.py (16 tests pass) | 912071d | Pushed to origin/main | Documentation |

## Milestone: Phase 4 - Streaming & Latency Optimization

| Increment | Feature/Fix | Tests | Commit Hash | Push Result | Remaining |
|---|---|---|---|---|---|
| 19 | Ollama client: async generate_stream() with stream: True | — | 903a552 | Pushed to origin/main | RAG pipeline streaming |
| 20 | RAG pipeline: ask_stream() generator with stage events + token streaming | tests/test_rag_streaming.py (new) | — | — | API streaming endpoint |
| 21 | API: /api/ask/stream SSE endpoint | tests/test_api_stream.py (new) | — | — | CLI streaming |
| 22 | CLI: ask --stream flag for incremental output | tests/test_cli_stream.py (new) | — | — | Frontend streaming UI |
| 23 | Frontend: streaming RAG Q&A UI with SSE | tests/test_frontend_stream.py (new) | — | — | Benchmarking |
| 24 | Baseline + post-change benchmark with before/after report | evaluation/streaming_benchmark.py (new) | — | — | Documentation |
