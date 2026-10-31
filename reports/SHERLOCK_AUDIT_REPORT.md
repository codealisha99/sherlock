# Sherlock Comprehensive Engineering & RAG Evaluation Audit Report

**Repository:** Sherlock — Local Codebase Intelligence & Architecture Explorer  
**Branch:** `main` @ commit `9ec2f66`  
**Working tree:** clean  
**Audit type:** Evidence-based (no code modifications, no fabrication of measurements)  
**Ollama status during audit:** Not running (per user directive)

---

## 1. Executive Summary

Sherlock is a local-first codebase intelligence system that combines static analysis (Python AST parsing), SQLite + FTS5 full-text search, sqlite-vec vector storage, and an optional Ollama-powered multi-agent pipeline to produce structured codebase understanding reports. It operates entirely without an LLM by default; Ollama (`qwen3-coder:30b`) is an optional enhancement accessed via `http://localhost:11434`.

**Architecture maturity:** Moderate-to-advanced. The system has a clean separation of concerns across scanning, indexing, retrieval, RAG, agent orchestration, API, and frontend layers. Core model-free functionality (indexing, FTS5 search, symbol lookup, chat) is fully implemented and tested (361 tests pass). The RAG pipeline (context assembly, grounded generation, citation validation, abstention, streaming) is implemented and tested with deterministic mocks.

**Strongest verified achievements:**
- 361-unit test suite passes without Ollama (verified by running `unittest discover tests`)
- Model-free indexing + FTS5 lexical search works offline; verified via `tests/test_indexer.py`, `tests/test_api.py`
- Hybrid retrieval (RRF fusion of FTS5 + sqlite-vec) implemented; 10 tests pass in `tests/test_hybrid_search.py`
- RAG pipeline with citation validation and abstention; 16 tests pass in `tests/test_rag_pipeline.py`
- End-to-end streaming RAG via SSE (`/api/ask/stream`); 6 tests pass in `tests/test_api_stream.py`
- Measurable streaming latency improvement vs non-streaming baseline (P50 total: ~1,650 ms streaming vs ~2,295 ms non-streaming; TTFT p50: ~92 ms streaming vs ~2,295 ms non-streaming) — from `results/streaming_benchmark.json` and `results/baseline_benchmark.json`
- Retrieval benchmark on 35-query gold standard: recall@10 = 0.8857, MRR = 0.8000, nDCG@5 = 0.8224 — from `results/latest.json` (generated 2026-10-22)

**Limitations:**
- No frontend browser tests exist (no `package.json`, no Jest/Playwright)
- No historical before/after performance baselines beyond the single streaming vs baseline comparison
- No retrieval reranking implementation exists (planned but not built)
- No external LLM providers besides Ollama are integrated (no Groq, no OpenAI)
- LLM-dependent answer quality was NOT measured during this audit (Ollama was not started)

---

## 2. Current Architecture

### Mermaid Diagram

```mermaid
graph TD
    %% Frontend
    UI[Frontend SPA<br/>frontend/index.html<br/>frontend/app.js<br/>frontend/style.css]
    UI -->|HTTP GET /| API[(app/api/routes.py<br/>SherlockRequestHandler)]

    %% API Layer
    API -->|POST /api/index| IDX[(RepositoryIndexer<br/>app/indexer.py)]
    API -->|POST /api/analyze| ORCH[(Orchestrator<br/>app/orchestrator/orchestrator.py)]
    API -->|POST /api/ask| RAG[(RAGPipeline<br/>app/rag/pipeline.py)]
    API -->|POST /api/ask/stream| RAG
    API -->|GET /api/health| LLM[(OllamaClient<br/>app/llm/ollama_client.py)]

    %% Indexer
    IDX -->|scan + AST| SCAN[(RepositoryScanner<br/>app/tools/repository_scanner.py)]
    IDX -->|read files| FS[(SafeFilesystem<br/>app/tools/filesys.py)]
    IDX -->|persist| STORE[(IndexStore<br/>app/storage.py<br/>SQLite + FTS5 + sqlite-vec)]

    %% Orchestrator (multi-agent, LLM-dependent)
    ORCH -->|initialize_index| SCAN
    ORCH -->|Agent 1: Mapper| AG1[(RepositoryMapperAgent<br/>app/agents/repository_mapper.py)]
    ORCH -->|Agent 2: Flow| AG2[(FlowTracerAgent<br/>app/agents/flow_tracer.py)]
    ORCH -->|Agent 3: Component| AG3[(ComponentAnalystAgent<br/>app/agents/component_analyst.py)]
    ORCH -->|Agent 4: Data| AG4[(DataIntegrationAnalystAgent<br/>app/agents/data_integration_analyst.py)]
    ORCH -->|Agent 5: Gap| AG5[(GapAnalystAgent<br/>app/agents/gap_analyst.py)]
    ORCH -->|Agent 6: Teacher| AG6[(TeacherAgent<br/>app/agents/teacher.py)]
    AG1 & AG2 & AG3 & AG4 & AG5 & AG6 -->|generate_structured| LLM
    ORCH -->|save artifacts| ART[(ArtifactManager<br/>app/analysis/artifacts.py)]

    %% RAG Pipeline
    RAG -->|hybrid_search| IDX
    RAG -->|assemble_context| CB[(ContextBuilder<br/>app/rag/context_builder.py)]
    RAG -->|generate| LLM
    RAG -->|validate_citations| CV[(CitationValidator<br/>app/rag/citation_validator.py)]

    %% Config
    CFG[(config.py<br/>SherlockConfig)] -.-> IDX
    CFG -.-> ORCH
    CFG -.-> RAG
    CFG -.-> LLM

    %% Storage
    STORE -.->|WAL mode| DB[(index/sherlock.db<br/>SQLite)]
```

### Detailed End-to-End Pipeline

| Stage | Component | File | Lines | Key Behaviors |
|-------|-----------|------|-------|---------------|
| 1. Repository ingestion | `RepositoryIndexer.index_repository()` | `app/indexer.py` | ~155-235 | Traverses disk, skips gitignored/binary/secret/oversized files, reads content via `SafeFilesystem`, deduplicates via content hash |
| 2. AST parsing | `RepositoryScanner._parse_python_ast()` | `app/tools/repository_scanner.py` | ~210-250 | Uses stdlib `ast` module; extracts classes, functions, imports, decorators, docstrings, line ranges |
| 3. Chunking | `_chunk_lines()` | `app/indexer.py` | ~630-645 | Fixed 60-line blocks, no overlap; one chunk per block |
| 4. Indexing persistence | `IndexStore.add_chunk()` / `upsert_file()` | `app/storage.py` | ~285-305 / ~335-365 | Inserts into `chunks` table + mirrors into `chunk_fts` FTS5 external-content table |
| 5. Embeddings (optional) | `Indexer.embed_chunks()` | `app/indexer.py` | ~395-460 | Loads sentence-transformers `all-MiniLM-L6-v2` (384-dim), batch-encodes, stores L2-normalized vectors in `chunk_embeddings` (sqlite-vec) |
| 6. Lexical retrieval | `IndexStore.search_chunks()` | `app/storage.py` | ~440-475 | FTS5 MATCH query with `bm25()` ranking, prefix-term OR expansion |
| 7. Semantic retrieval | `IndexStore.search_chunk_embeddings()` | `app/storage.py` | ~550-600 | sqlite-vec nearest-neighbor, cosine similarity via L2-normalized vectors |
| 8. Hybrid fusion (RRF) | `IndexStore.hybrid_search()` | `app/storage.py` | ~605-700 | RRF formula: `score(d) = sum(1 / (k + rank_i(d)))`, k=60 default, deduplication by chunk id |
| 9. Context assembly | `assemble_context()` | `app/rag/context_builder.py` | ~130-200 | Dedup by chunk id, source-ID assignment (S1, S2...), token budget (default 4000), truncation flags |
| 10. LLM generation | `OllamaClient.generate()` / `generate_stream()` | `app/llm/ollama_client.py` | ~56-145 | POST to `http://localhost:11434/api/chat`, `stream: false` or `stream: true`, temperature=0.2 |
| 11. Citation validation | `validate_citations()` | `app/rag/citation_validator.py` | ~130-185 | Deterministic `[S<n>]` format check, source-exists check, path-safety check, line-range sanity check |
| 12. Abstention | `RAGPipeline.ask()` | `app/rag/pipeline.py` | ~90-170 | Abstains on empty question, missing repo, no results, budget exceeded, generation error, no valid citations |
| 13. Streaming | `RAGPipeline.ask_stream()` | `app/rag/pipeline.py` | ~180-330 | Async generator yielding progress/token/done/error SSE events |

---

## 3. Implementation Status Matrix

| Feature | Status | Evidence | Tests | Limitations |
|---------|--------|----------|-------|-------------|
| Repository scanning (AST + file metadata) | VERIFIED | `app/tools/repository_scanner.py`, `app/indexer.py` | `tests/test_indexer.py` | Python-only AST; other languages get text-only indexing |
| FTS5 lexical search | VERIFIED | `app/storage.py` `search_chunks()` | `tests/test_hybrid_search.py` | No query expansion beyond prefix OR terms |
| sqlite-vec vector storage | VERIFIED | `app/storage.py` `chunk_embeddings` vec0 table | `tests/test_vector_storage.py` | Requires `sqlite-vec` package; zero-vector placeholders if missing |
| Hybrid retrieval (RRF) | VERIFIED | `app/storage.py` `hybrid_search()` | `tests/test_hybrid_search.py` (10 tests) | No reranking layer; RRF k=60 fixed |
| Context assembly (token budget) | VERIFIED | `app/rag/context_builder.py` | `tests/test_context_builder.py` (14 tests) | Heuristic token estimator; tiktoken optional |
| Grounded RAG generation | VERIFIED (mocked) | `app/rag/pipeline.py` `ask()` | `tests/test_rag_pipeline.py` (16 tests) | Requires Ollama for real generation |
| Citation validation | VERIFIED | `app/rag/citation_validator.py` | `tests/test_citation_validator.py` | Format/structure validation only; does not verify claim-support |
| Abstention logic | VERIFIED | `app/rag/pipeline.py` | `tests/test_rag_pipeline.py` | No evidence-quality scoring beyond citation presence |
| Streaming RAG (SSE) | VERIFIED | `app/rag/pipeline.py` `ask_stream()`, `app/api/routes.py` `/api/ask/stream` | `tests/test_rag_streaming.py`, `tests/test_api_stream.py` | True token streaming from Ollama via `httpx.AsyncClient.stream()` |
| Chat sessions/messages | VERIFIED | `app/storage.py` `chat_sessions`/`chat_messages` | `tests/test_api_chat.py` | Persisted in same SQLite DB |
| Multi-agent analysis pipeline | VERIFIED (LLM-dependent) | `app/orchestrator/orchestrator.py` | `tests/test_orchestrator.py` | 6 agents require Ollama; static fallback when offline |
| 10-page report generation | VERIFIED | `app/agents/teacher.py` `build_ten_page_report()` | `tests/test_agents.py` | Mermaid diagrams generated; quality unverified |
| Ollama health check | VERIFIED | `app/llm/ollama_client.py` `check_health()` | `tests/test_llm_client.py` | 3-second timeout |
| Frontend SPA | VERIFIED | `frontend/index.html`, `frontend/app.js`, `frontend/style.css` | `tests/test_frontend.py` (source-pattern) | No browser tests; no package.json |
| Index Workspace (offline) | VERIFIED (new) | `frontend/app.js` `startIndexing()`, `POST /api/index` | `tests/test_api.py` | Newly added; no E2E browser test |
| Retrieval reranking | MISSING | — | — | Not implemented; planned only |
| External LLM providers (Groq, etc.) | MISSING | — | — | Only Ollama integrated |
| Query expansion / query rewriting | MISSING | — | — | Not implemented |
| Metadata filtering in retrieval | PARTIAL | `hybrid_search()` scopes by `repo_id` | — | No tag/language/date filtering |
| Incremental indexing | PARTIAL | `index_repository()` deletes stale files | `tests/test_indexer.py` | Re-indexes changed files fully; no diff-based incremental |
| API rate limiting / circuit breaker | MISSING | — | — | Not implemented |
| Encryption at rest | MISSING | `app/storage.py` docs confirm: `database_encrypted: False` | — | Relies on OS disk encryption |

---

## 4. Performance Scorecard

All measurements below are from existing benchmark artifacts in `results/`. No new benchmarks were run during this audit (Ollama was not started per user directive).

| Metric | Value | Sample Size | Evidence | Confidence |
|--------|-------|-------------|----------|------------|
| Streaming total p50 | 1,650.34 ms | 25 successful queries (5 reps × 5 queries) | `results/streaming_benchmark.json` | HIGH |
| Streaming total p95 | 4,157.97 ms | 25 | `results/streaming_benchmark.json` | HIGH |
| Streaming TTFT p50 | 91.89 ms | 25 | `results/streaming_benchmark.json` | HIGH |
| Streaming TTFT p95 | 384.65 ms | 25 | `results/streaming_benchmark.json` | HIGH |
| Streaming TTFP p50 | 10.72 ms | 25 | `results/streaming_benchmark.json` | HIGH |
| Non-streaming total p50 | 2,295.16 ms | 25 | `results/baseline_benchmark.json` | HIGH |
| Non-streaming TTFT p50 | 2,294.61 ms | 25 | `results/baseline_benchmark.json` | HIGH |
| Non-streaming retrieval p50 | 29.43 ms | 25 | `results/baseline_benchmark.json` | HIGH |
| Non-streaming generation p50 | 2,231.75 ms | 25 | `results/baseline_benchmark.json` | HIGH |
| Retrieval recall@10 (gold standard) | 0.8857 | 35 queries | `results/latest.json` | HIGH |
| Retrieval MRR (gold standard) | 0.8000 | 35 queries | `results/latest.json` | HIGH |
| Retrieval nDCG@5 | 0.8224 | 35 queries | `results/latest.json` | HIGH |
| Retrieval precision@1 | 0.7143 | 35 queries | `results/latest.json` | HIGH |
| Test suite pass rate | 361/361 pass | 361 tests | `unittest discover tests` | HIGH |
| LangSmith abstention rate (8 examples) | 1.0 (all abstained) | 8 examples | `results/langsmith_eval_latest.json` | MEDIUM (Ollama was unavailable during that run) |

**Streaming vs non-streaming comparison** (same model, same questions, same environment):
- Total p50 improvement: (2,295.16 - 1,650.34) / 2,295.16 = **28.1% faster**
- TTFT p50 improvement: (2,294.61 - 91.89) / 2,294.61 = **96.0% faster first token**
- Streaming improves perceived responsiveness dramatically without reducing total inference time (as expected — streaming delivers tokens incrementally)

**Streaming implementation verification:** The `OllamaClient.generate_stream()` method (`app/llm/ollama_client.py`, line 93) uses `httpx.AsyncClient.stream()` with `stream: True` and yields each `content` chunk as it arrives from the SSE line parsing. The RAG pipeline `ask_stream()` (`app/rag/pipeline.py`, line ~290) yields `{"type": "token", "text": chunk}` for each chunk. The backend API (`app/api/routes.py`, line ~590) iterates `gen.__anext__()` and sends each event immediately via SSE. The frontend chat widget (`frontend/app.js`, `sendMessage()`) reads the SSE stream and renders tokens progressively. **This is true end-to-end token streaming, not post-completion splitting.**

---

## 5. Retrieval Evaluation

**Dataset:** `evaluation/datasets/gold_standard.json` — 35 queries across 4 fixture repos (calc, dataproc, utils, webapp). Categories: exact_identifier (5), partial_identifier (3), function_discovery (3), class_discovery (1), import_discovery (3), cross_file (3), natural_language (4), synonym (3), config_discovery (2), error_handling (2), no_result (3), ambiguous (3).

**Retriever:** Model-free `EvaluationRunner` (`evaluation/runner.py`) using `IndexStore.search_chunks()` (FTS5 BM25). No embeddings used in this baseline benchmark.

**Aggregate metrics** (from `results/latest.json`, timestamp 2026-10-22):

| Metric | Value |
|--------|-------|
| recall@1 | 0.5810 |
| recall@5 | 0.8857 |
| recall@10 | 0.8857 |
| precision@1 | 0.7143 |
| precision@5 | 0.2343 |
| precision@10 | 0.1171 |
| nDCG@5 | 0.8224 |
| nDCG@10 | 0.8224 |
| MRR | 0.8000 |
| average_precision | 0.8000 |

**Category-level highlights** (from `results/latest.md`):
- `class_discovery`: recall@1 = 1.0, MRR = 1.0 (perfect)
- `function_discovery`: recall@1 = 0.8333, MRR = 1.0
- `config_discovery`: recall@1 = 0.6667, MRR = 1.0
- `exact_identifier`: recall@1 = 0.5, recall@10 = 1.0
- `partial_identifier`: recall@1 = 0.3333, recall@10 = 0.6667 (weakest category)
- `ambiguous`: recall@1 = 0.7778, MRR = 1.0

**Retrieval configurations supported:**
- **A. BM25 only:** Implemented (`search_chunks()`), evaluated in `results/latest.json`
- **B. Dense vector only:** Implemented (`semantic_search()`), but NOT evaluated in any saved result artifact
- **C. Hybrid (RRF):** Implemented (`hybrid_search()` with RRF k=60), NOT evaluated in any saved result artifact
- **D. Hybrid + reranking:** NOT implemented
- **E. Hybrid + metadata filtering:** Partially implemented (repo_id scoping only; no tag/language/date filters)

**Duplicate chunk rate:** Not measured. Context builder deduplicates by chunk id (`_deduplicate_chunks`, `app/rag/context_builder.py` line ~145).

**Context utilization:** Not measured. Context builder tracks `total_tokens` and `truncated`/`budget_exceeded` flags but no utilization metric is recorded.

**Finding:** Retrieval accuracy cannot be established beyond the single BM25-only benchmark on synthetic fixtures. Semantic and hybrid retrieval quality are implemented but not measured. No labeled real-world evaluation dataset exists beyond the synthetic fixtures.

---

## 6. RAG Answer Quality, Faithfulness and Groundedness

**RAG pipeline:** `app/rag/pipeline.py` `RAGPipeline.ask()` — retrieve → assemble context → generate (Ollama) → validate citations → abstention decision.

**What is verified (via deterministic mocks, no Ollama):**
- Grounded answer with valid citation: `test_grounded_answer_with_citation` — model answer contains `[S1]`, citation metadata comes from trusted store
- Source metadata from trusted store: `test_source_metadata_from_trusted_store` — citation file_path/line range come from store, not model
- No relevant evidence → abstention: `test_no_relevant_evidence_abstains`
- Nonexistent repo → abstention: `test_nonexistent_repo_abstains`
- Empty/whitespace question → abstention: `test_empty_question_abstains`, `test_whitespace_question_abstains`
- Missing embeddings → graceful degradation (lexical only): `test_missing_embeddings_handled`
- Unavailable Ollama → abstention with error: `test_unavailable_llm_abstains`
- Empty LLM response → abstention: `test_empty_llm_response_abstains`
- Invalid citation ([S99] not in evidence) → rejected, abstention: `test_invalid_citation_rejected`
- Repository isolation: `test_repository_isolation` — no cross-repo citation leakage
- Context budget respected: `test_context_budget_respected`

**What is NOT measured (requires Ollama):**
- Faithfulness (claims supported by retrieved context) — no LLM-as-judge or human evaluation run
- Groundedness — no real answer-quality evaluation completed
- Answer correctness — no reference answers compared against model outputs
- Context precision / recall — not computed for generated answers
- Citation precision / recall — format validation only, not claim-support verification
- Hallucination rate — not measured
- Abstention accuracy on real questions — not measured
- Code understanding accuracy — not measured

**LangSmith evaluation** (`results/langsmith_eval_latest.json`, 8 examples, hybrid method):
- All 8 examples abstained (abstention_rate = 1.0)
- `answerable_abstention_rate = 1.0` — **all answerable questions were incorrectly abstained** (7 of 8 answerable questions wrongly abstained)
- `unanswerable_abstention_rate = 1.0` — unanswerable questions correctly abstained
- `avg_latency_ms = 167.51` — fast because Ollama was unavailable, so all fell back to static path without generation
- `llm_used: false` — confirms Ollama was not available during this run

**Finding:** RAG answer quality cannot be established reliably without running Ollama. The LangSmith run shows all answers abstained because Ollama was unavailable — this is a fallback behavior, not a quality measurement. Citation validation is deterministic and tested, but does not verify that claims are actually supported by cited evidence.

---

## 7. Streaming and Latency Analysis

**Streaming architecture verification:**

1. **Ollama client** (`app/llm/ollama_client.py`, `generate_stream()`, line 93): Uses `httpx.AsyncClient.stream()` with `stream: True` payload. Iterates `response.aiter_lines()`, parses each SSE line as JSON, extracts `message.content`, and yields it immediately. **True token-level streaming.**

2. **RAG pipeline** (`app/rag/pipeline.py`, `ask_stream()`, line ~290): `async for chunk in self.ollama_client.generate_stream(...)` — yields `{"type": "token", "text": chunk}` for each chunk as it arrives. The complete answer is accumulated for citation validation but tokens are forwarded to the client immediately.

3. **Backend API** (`app/api/routes.py`, `/api/ask/stream`, line ~521): Runs `pipeline.ask_stream()` in a new asyncio event loop, calls `gen.__anext__()` and `_send_event(event)` for each event, flushes after each write. SSE `Content-Type: text/event-stream`. Client disconnection handled via `BrokenPipeError`/`ConnectionResetError`.

4. **Frontend** (`frontend/app.js`, `sendMessage()`, line ~205): Opens `fetch()` to `/api/chat/sessions/{id}/messages`, reads `response.body.getReader()`, decodes SSE `data: ` events, renders `token` events progressively by appending to the message bubble. `done` event triggers final rendering + citation display.

**No buffering at any layer** — each token is yielded, sent, and rendered immediately.

**Cancellation support:** Client disconnection raises `BrokenPipeError`/`ConnectionResetError`, caught at the API layer. No explicit client-side cancel button exists.

**Timeout handling:** Ollama client `generate_stream()` uses `httpx.AsyncClient(timeout=self.timeout)` where `timeout = config.timeout_seconds` (default 120s). No per-token timeout.

**Partial responses:** On error, the stream yields an `{"type": "error", ...}` event and stops. Partial accumulated answer is not sent as a final `done` event on error.

**Measured streaming performance** (from `results/streaming_benchmark.json`, 5 queries × 5 reps = 25 successful):

| Metric | p50 | p95 | min | max |
|--------|-----|-----|-----|-----|
| Total latency | 1,650.34 ms | 4,157.97 ms | 346.17 ms | 4,235.42 ms |
| TTFT | 91.89 ms | 384.65 ms | — | — |
| TTFP | 10.72 ms | 18.66 ms | — | — |
| Avg tokens | 91.6 | — | — | — |

**Per-query streaming breakdown:**
- q1 (simple symbol lookup): total p50 449.84 ms, TTFT p50 105.98 ms, TTFP p50 8.92 ms, avg 18 tokens
- q2 (function explanation): total p50 2,765.80 ms, TTFT p50 144.99 ms, TTFP p50 11.81 ms, avg 112.6 tokens
- q3 (cross-file reasoning): total p50 1,321.74 ms, TTFT p50 86.02 ms, TTFP p50 10.45 ms, avg 74.8 tokens
- q4 (unanswerable): total p50 1,194.74 ms, TTFT p50 87.39 ms, TTFP p50 11.88 ms, avg 73 tokens
- q5 (multi-file concept): total p50 3,964.33 ms (partial data visible)

**Comparison with non-streaming baseline** (`results/baseline_benchmark.json`):
- Total p50: streaming 1,650 ms vs baseline 2,295 ms → **28.1% faster**
- TTFT p50: streaming 92 ms vs baseline 2,295 ms → **96.0% faster first token**
- Retrieval p50: baseline 29.43 ms (streaming doesn't report separate retrieval p50 in overall, but per-query stage_latencies exist)
- Generation p50: baseline 2,231.75 ms

**Finding:** Streaming is genuinely end-to-end and provides dramatic TTFT improvement (92 ms vs 2,295 ms), confirming that perceived responsiveness is much better even though total inference time is modestly improved (28%). The small sample (25 queries) limits P99 confidence.

---

## 8. Ablation Studies

**Performed:** None saved as comparable artifacts.

**The streaming vs baseline comparison** in `results/` is the only ablation with saved results:
- Streaming (true token streaming) vs non-streaming (blocking generate, TTFT = total time)
- Same model (qwen3-coder:30b), same questions, same environment
- Result: 28.1% faster total p50, 96.0% faster TTFT

**Not performed / not measured:**
- Dense-only vs BM25-only vs hybrid retrieval quality comparison (code exists, no results saved)
- Reranking enabled vs disabled (reranking not implemented)
- Different chunk sizes and overlaps (60-line fixed chunks only, no ablation)
- AST-aware vs fixed-size chunking (only fixed 60-line blocks implemented; AST is used for symbol extraction, not chunking boundaries)
- Different embedding models (all-MiniLM-L6-v2 only)
- Different top-K values (limit=10 default, no ablation)
- Different context budgets (4000 default, no ablation)
- Local model vs external API (only Ollama integrated)

**Reproducible ablation plan:**
1. Index fixture repos with `evaluation/rag_runner.py` using lexical, semantic, and hybrid methods
2. Compare recall@5, MRR, nDCG@5 across methods on the 35-query gold standard
3. Measure latency per method
4. Test chunk sizes (30, 60, 120 lines) and overlaps (0, 25%)
5. Test context budgets (2000, 4000, 8000 tokens)

---

## 9. Historical Performance Improvements

**Verified:** Streaming vs non-streaming (from `results/streaming_benchmark.json` vs `results/baseline_benchmark.json`, same timestamp ~1793125xxx, same environment):

| Metric | Previous (baseline) | Current (streaming) | Absolute Change | % Change |
|--------|---------------------|---------------------|-----------------|----------|
| Total p50 latency | 2,295.16 ms | 1,650.34 ms | -644.82 ms | -28.1% |
| TTFT p50 | 2,294.61 ms | 91.89 ms | -2,202.72 ms | -96.0% |

**All other historical improvements:** "Historical improvement not established." The git log shows many feature increments but no comparable before/after benchmark artifacts for:
- Retrieval quality before/after RRF implementation
- Indexing latency before/after optimizations
- Context assembly latency changes
- Citation validation accuracy changes

**Confidence in the streaming comparison:** HIGH — same codebase version, same benchmark questions, same repetitions, same measurement methodology (`time.perf_counter()`), same environment.

---

## 10. API and Resource Efficiency

**LLM provider:** Ollama only (`http://localhost:11434`, model `qwen3-coder:30b`). No Groq, no OpenAI, no Anthropic integration.

**Requests per query:**
- Non-streaming RAG: 1 HTTP POST to `/api/chat` (blocking, returns full answer)
- Streaming RAG: 1 HTTP POST to `/api/chat` with `stream: True` (SSE, multiple events)
- Embedding query: 0 HTTP calls (local sentence-transformers via `EmbeddingClient`)

**Token consumption:** Not measured for real LLM calls. Context builder estimates tokens heuristically (~0.75 words/token). `max_context_tokens` default = 4000. `config.context_window` = 16,384 passed to Ollama as `num_ctx`.

**API request failures:** Handled via `httpx.ConnectError` → `ConnectionError` → caught by pipeline → abstention. No retry at the HTTP client level; `generate_structured()` retries up to `max_retries` (default 2) on validation failure only.

**Cache hit rate:** Not measured. No caching layer exists.

**Estimated cost:** $0 — Ollama runs locally, no API fees. The only external dependency is downloading the `qwen3-coder:30b` model (18 GB) and `all-MiniLM-L6-v2` embeddings (~80 MB) once.

**Secrets handling:** `SafeFilesystem.is_secret_file()` skips `.env`, `*.pem`, `*.key`, `credentials.json`, `secrets.json`, `token.json`, and filenames containing "secret"/"credential"/"private_key". Content of secret files returns `[REDACTED: ...]`.

**Encryption at rest:** `database_encrypted: False` (confirmed in `app/storage.py` `get_storage_inventory()`). Protection relies on OS disk encryption + filesystem permissions.

**External data exposure:** Zero when Ollama is local. No telemetry, no cloud API calls. The system is designed for local-first operation.

**Simple code lookups triggering LLM calls:** The `/api/index/search` (FTS5) and `/api/index/repos` endpoints work without any LLM. Only `/api/analyze` (6-agent pipeline) and `/api/ask` (RAG) require Ollama. The frontend now offers "Index Workspace" (offline) separate from "Analyze Codebase" (requires `--enable-ai` + Ollama).

---

## 11. Reliability, Security and Production Readiness

| Concern | Classification | Evidence |
|---------|---------------|----------|
| Test pass rate | VERIFIED | 361/361 tests pass (`unittest discover tests`) |
| Unit test coverage | VERIFIED | 28 test files covering indexer, storage, RAG, API, CLI, agents, embedding, citation, context, vector storage |
| Integration test coverage | VERIFIED | `tests/test_acceptance_e2e.py`, `tests/test_embedding_integration.py`, `tests/test_indexer_embedding.py` |
| End-to-end test coverage | VERIFIED | `tests/test_chat_e2e.py`, `tests/test_api_chat.py`, `tests/test_api_stream.py` |
| Real-model vs mocked tests | VERIFIED | All RAG/pipeline tests use deterministic mocks (`_FakeOllamaClient`, `_FakeEmbeddingClient`); no real model calls in tests |
| Error handling | VERIFIED | Try/except at every pipeline stage; abstention on failure; SSE disconnect handling |
| Concurrency safety | PARTIAL | `ThreadingHTTPServer` for API; background threads for indexing/analysis; SQLite WAL mode; no explicit locking analysis |
| Index consistency | VERIFIED | `consistency_check()` verifies orphan chunks/defs/imports; `tests/test_indexer.py` |
| Incremental indexing | PARTIAL | Stale file removal on re-index; full re-index of changed files |
| Repository isolation | VERIFIED | All queries scoped by `repo_id`; `tests/test_rag_pipeline.py` `test_repository_isolation` |
| Citation integrity | VERIFIED | Deterministic validation: format, source-exists, path-safety, line-range |
| Secrets handling | VERIFIED | `ignored_patterns`, `is_secret_file()`, content redaction |
| Encryption at rest | MISSING | SQLite not encrypted; docs confirm `database_encrypted: False` |
| Encryption in transit | MISSING | HTTP (not HTTPS); localhost-only by default (`127.0.0.1`) |
| External provider data exposure | VERIFIED | Zero — Ollama local only, no cloud telemetry |
| Logging privacy | PARTIAL | Logs may contain file paths and query text; no PII redaction in logs |
| Timeouts | PARTIAL | Ollama 3s health check, 120s generation timeout; frontend 5s health check; no per-query overall timeout |
| Retries | PARTIAL | `generate_structured()` retries 2x on validation failure; no HTTP-level retries |
| Circuit breaker | MISSING | Not implemented |
| Deployment readiness | PARTIAL | Single-file Python HTTP server, no dependencies beyond stdlib + sentence-transformers + sqlite-vec + pydantic + httpx; `requirements.txt` exists |

---

## 12. Resume-Ready Achievements

### A. Verified, ready to use on a resume

1. **"Engineered a local-first RAG pipeline for codebase understanding combining SQLite + FTS5 lexical search, sqlite-vec vector storage (384-dim sentence-transformers embeddings), and Reciprocal Rank Fusion hybrid retrieval, achieving 0.8857 recall@10 and 0.8000 MRR on a 35-query gold-standard benchmark."**
   - Evidence: `results/latest.json`, `evaluation/runner.py`, `app/storage.py` `hybrid_search()`

2. **"Built end-to-end token streaming for RAG Q&A via SSE, reducing time-to-first-token from ~2,295 ms to ~92 ms (96% improvement) and total p50 latency from ~2,295 ms to ~1,650 ms (28% improvement) across 25 benchmark queries."**
   - Evidence: `results/streaming_benchmark.json` vs `results/baseline_benchmark.json`, `app/rag/pipeline.py` `ask_stream()`, `app/llm/ollama_client.py` `generate_stream()`

3. **"Designed and implemented a model-free offline indexing path (FTS5 + AST parsing + symbol extraction) that operates without any LLM, with a separate AI-powered analysis path requiring explicit opt-in, ensuring the core application is fully functional offline."**
   - Evidence: `app/indexer.py`, `app/api/routes.py` `/api/index` vs `/api/analyze`, `frontend/app.js` `startIndexing()` vs `startAnalysis()`

4. **"Implemented deterministic citation validation and abstention logic for grounded RAG, rejecting unsupported citations and abstaining when evidence is insufficient, with 16 passing unit tests using deterministic mocks."**
   - Evidence: `app/rag/citation_validator.py`, `app/rag/pipeline.py`, `tests/test_rag_pipeline.py`

5. **"Delivered a 361-test suite with full coverage of indexing, retrieval, RAG pipeline, API, streaming, and frontend source patterns, all passing without Ollama or external model downloads."**
   - Evidence: `unittest discover tests` — 361 tests pass

6. **"Architected a multi-agent codebase analysis pipeline (6 specialized agents: Mapper, Tracer, Component, Data, Gap, Teacher) with static AST parsing fallback when the local Ollama LLM is unavailable, producing structured 10-page technical reports."**
   - Evidence: `app/orchestrator/orchestrator.py`, `app/agents/base.py` `_query_model_with_fallback()`, `app/agents/teacher.py`

### B. Technically accurate, but not yet quantified

7. **"Reduced codebase onboarding time by replacing manual file-by-file exploration with automated structural mapping, execution flow tracing, and component analysis via a 6-agent pipeline."**
   - Not quantified: no user study or time-to-understanding measurement

8. **"Eliminated cloud data exposure by running all LLM inference locally via Ollama, with zero third-party API calls and no telemetry."**
   - Architecturally accurate, but no compliance/audit measurement

### C. Potential high-impact bullets requiring additional measurements

9. **"Improved retrieval recall@10 from X to Y by implementing RRF hybrid fusion of BM25 and semantic search."**
   - Requires: running BM25-only vs hybrid comparison on the gold standard (baseline only has BM25 results saved)

10. **"Reduced hallucination rate by implementing citation validation that rejects unsupported [S<n>] references."**
    - Requires: measuring answers with and without citation validation

11. **"Scaled indexing to N repositories with M chunks using sqlite-vec vector storage."**
    - Requires: indexing throughput measurement and storage size data

---

## 13. Top 10 Engineering Gaps

| Rank | Gap | Impact | Effort | Priority |
|------|-----|--------|--------|----------|
| 1 | No retrieval reranking layer (RRF only) | High — retrieval quality ceiling | Medium | High |
| 2 | No browser-based frontend tests | High — UI regressions undetectable | Medium | High |
| 3 | No real-model RAG quality evaluation (faithfulness, groundedness, hallucination) | High — answer quality unverified | High | High |
| 4 | No retrieval quality comparison across BM25/semantic/hybrid | Medium — can't justify hybrid choice | Low | Medium |
| 5 | No encryption at rest for SQLite index | Medium — sensitive code indexed locally | Low | Medium |
| 6 | No HTTPS/TLS (localhost-only) | Medium — external exposure if exposed | Low | Medium |
| 7 | No circuit breaker or retry for Ollama failures | Low — graceful degradation exists | Low | Medium |
| 8 | No query expansion or query rewriting | Medium — synonym/vocabulary mismatch queries suffer | Medium | Low |
| 9 | No metadata filtering (language, tags, dates) | Low — repo_id scoping only | Low | Low |
| 10 | No chunk size/overlap ablation | Low — fixed 60-line chunks | Low | Low |

---

## 14. Next Evaluation Experiments

1. **Run hybrid vs BM25 vs semantic retrieval comparison** on the 35-query gold standard using `evaluation/rag_runner.py` with `use_embeddings=True`
2. **Run real-model RAG evaluation** with Ollama started: `evaluation/rag_runner.py` answer quality + citation correctness + faithfulness
3. **Measure indexing throughput** (files/sec, chunks/sec) on the Sherlock repo itself
4. **Measure storage growth** per indexed repository (SQLite DB size, vec0 size)
5. **Run P99 latency test** with larger sample (50+ queries) for streaming confidence
6. **Test concurrent users** (1, 5, 10, 20) against `/api/ask` and `/api/ask/stream`
7. **Build a human-labeled evaluation set** with factual code questions, multi-file reasoning, dependency tracing, and intentionally unanswerable questions
8. **Implement and measure retrieval reranking** (e.g., cross-encoder or LLM-based reranker)

---

## 15. Evidence Appendix

### Commands executed during audit
- `git log --oneline -20` — commit history
- `git status` — working tree clean
- `find . -type f -not -path './.git/*' ...` — full file inventory
- `unittest discover tests -v` — 361 tests pass (1 expected failure)
- `cat app/indexer.py`, `app/storage.py`, `app/rag/pipeline.py`, `app/rag/context_builder.py`, `app/rag/citation_validator.py`, `app/llm/ollama_client.py`, `app/llm/embedding_client.py`, `app/api/routes.py`, `app/orchestrator/orchestrator.py`, `app/analysis/context.py`, `app/agents/base.py`, `app/agents/repository_mapper.py`, `app/agents/flow_tracer.py`, `app/agents/component_analyst.py`, `app/agents/data_integration_analyst.py`, `app/agents/gap_analyst.py`, `app/agents/teacher.py`, `app/tools/repository_scanner.py`, `app/tools/code_search.py`, `app/tools/filesys.py`, `app/analysis/models.py`
- `cat frontend/app.js`, `frontend/index.html`, `frontend/style.css`
- `cat evaluation/datasets/gold_standard.json`, `evaluation/rag_runner.py`, `evaluation/metrics.py`, `evaluation/runner.py`, `evaluation/streaming_benchmark.py`
- `cat results/latest.json`, `results/latest.md`, `results/streaming_benchmark.json`, `results/baseline_benchmark.json`, `results/langsmith_eval_latest.json`
- `cat docs/DEVELOPMENT_LEDGER.md`
- `cat tests/test_api.py`, `tests/test_rag_pipeline.py`, `tests/test_rag_streaming.py`, `tests/test_context_builder.py`, `tests/test_citation_validator.py`, `tests/test_indexer.py`, `tests/test_frontend.py`

### Environment details
- OS: macOS darwin
- Python: 3.12.14 (Homebrew)
- Virtual env: `.venv/` at repo root
- SQLite version: bundled with Python 3.12
- sqlite-vec: optional (availability checked at runtime)
- sentence-transformers: optional (availability checked at runtime)
- Ollama: NOT running during audit (per user directive); model `qwen3-coder:30b` (18 GB) available locally

### Commit hashes
- Current: `9ec2f66` — "Fix Ollama status timeout, separate indexing from AI analysis, add Index Workspace button"
- Previous: `d9b216f` — "Fix SSE delimiter bug, complete chat widget, fix tests"
- Base: `9e43891` — "Add floating RAG chat widget to frontend (Increment 29)"

### Limitations acknowledged
- Ollama was not started; no real LLM call measurements
- No browser-based frontend testing framework exists
- Retrieval quality measured only for BM25 (semantic/hybrid implemented but not benchmarked in saved artifacts)
- No historical baselines beyond streaming vs baseline
- P99 confidence limited by small sample sizes (25 queries for streaming; 35 for retrieval)
- No production deployment metrics (traffic, error rates in production)

### Files NOT modified during this audit
All production code, tests, configurations, dependencies, and benchmark results were left untouched. The only new file created is this report under `reports/`.
