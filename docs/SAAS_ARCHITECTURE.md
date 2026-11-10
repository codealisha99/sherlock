# Sherlock SaaS Layer — Architecture & Contracts

Sherlock takes a GitHub URL, clones it, builds an AST graph + vector embeddings,
runs a security/quality scan, and serves an interactive graph UI with a chat
scoped to any node. This document is the contract between backend modules and
the frontend. **Do not change a contract without updating this file.**

## Runtime shape

- Server: stdlib `ThreadingHTTPServer` + `app/api/routes.py::SherlockRequestHandler` (unchanged style).
- `app/saas/router.py::process(handler, method)` runs first in every `do_GET/do_POST/do_DELETE`:
  body cache (1 MB cap) → rate limit → CSRF same-origin check → session user → SaaS route table → legacy-route policy (auth, ownership, local-path gate, prompt-injection screen).
- SaaS handlers have signature `def handler(req: app.saas.http.Request) -> app.saas.http.Response` and raise `HTTPError(status, message)` for errors. Error body: `{"status": "error", "error": "<message>"}`.
- Persistence: `app/saas/db.py::get_store()` → `SaasStore` (SQLite at `SHERLOCK_SAAS_DB`, default `index/saas.db`). Code index stays in `index/sherlock.db` (`app/storage.py::IndexStore`).
- Plans/quotas: `app/saas/plans.py` (`PLANS`, `plan_limits`, `EVENT_ANALYSIS`, `EVENT_CHAT`).

## Environment

| Var | Purpose | Default |
|---|---|---|
| `SHERLOCK_AUTH_REQUIRED` | SaaS mode (auth + ownership). `false` = single-user local dev | `true` |
| `SHERLOCK_ALLOW_LOCAL_PATHS` | Allow `/api/analyze` + `/api/index` with server paths | `true` only when auth off |
| `SHERLOCK_SECRET_KEY` | Master key for BYOK encryption (Fernet key derivation). Required in production | dev: generated to `index/.secret_key` (0600) with a warning |
| `SHERLOCK_SAAS_DB` | SaaS SQLite path | `index/saas.db` |
| `SHERLOCK_COOKIE_SECURE` | Set `Secure` on session cookie | `false` |
| `SHERLOCK_TRUST_PROXY` | Honour `X-Forwarded-For` | `false` |
| `SHERLOCK_ALLOWED_ORIGINS` | Extra comma-separated origins for CSRF check | — |
| `SHERLOCK_PUBLIC_URL` | Base URL for Stripe redirects | `http://localhost:8000` |
| `SHERLOCK_REPOS_DIR` | Where GitHub repos are cloned | `workspace/repos` |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_PRO` | Billing. Unset → billing endpoints report `billing_enabled: false` / 503 | — |

## Module ownership

| Module | Owns |
|---|---|
| `app/saas/security.py` | `SECURITY_HEADERS: tuple[tuple[str,str],...]`, `hash_password(pw)->str`, `verify_password(pw, stored)->bool` (scrypt, constant-time), `new_token()->str`, `hash_token(tok)->str` (sha256 hex), `encrypt_secret(plain)->str`, `decrypt_secret(token)->str` (Fernet), `validate_email(s)->str` (normalized, raises `ValueError`), `validate_password(s)->None` (raises `ValueError`) |
| `app/saas/auth.py` | `SESSION_COOKIE = "sherlock_session"`, `resolve_user(headers)->Optional[dict]` (cookie **or** `Authorization: Bearer <session token>`), `issue_session(user_id)->str` (returns `Set-Cookie` header value), `clear_cookie()->str` |
| `app/saas/auth_routes.py` | signup, login (lockout after 5 fails / 15 min), logout, me |
| `app/saas/ratelimit.py` | `check(bucket, key)->Optional[float]` (None = allowed, else retry-after seconds). Buckets: `auth` 10/min, `api` 120/min, `ingest` 5/min, `webhook` 300/min, `chat` 20/min. Thread-safe sliding window, bounded memory |
| `app/saas/prompt_guard.py` | `inspect_user_input(text)->GuardVerdict(blocked: bool, score: float, reasons: list[str])`, `sanitize_retrieved_context(text)->str`, `SYSTEM_GUARDRAILS: str` |
| `app/saas/byok.py` | Provider registry, key validation, SSRF-safe base URL validation, `gateway_for_user(user_id)->Optional[LLMGateway]` |
| `app/saas/byok_routes.py` | list_keys, save_key, delete_key |
| `app/saas/billing.py` + `billing_routes.py` | Stripe Checkout (subscription), webhook signature verification + idempotency, plan changes |
| `app/saas/github_ingest.py` | GitHub URL validation, safe shallow clone, size limits |
| `app/saas/vuln_scanner.py` | Static security rules (Python AST + multi-language regex), secret detection, dependency hygiene |
| `app/saas/quality.py` | Complexity/size/test-gap metrics ("where things are lagging") |
| `app/saas/graph.py` | Graph builder from index (files, symbols, imports, calls) + JS/TS import extraction |
| `app/saas/ingest_routes.py` | Job runner + start_ingest, list_jobs, get_job, list_repos, get_graph, get_findings, get_node |

## HTTP API (all JSON)

### Auth
- `POST /api/auth/signup` `{email, password}` → `201 {user}` + `Set-Cookie`. 409 if email exists. Password ≥ 10 chars, letters + digits.
- `POST /api/auth/login` `{email, password}` → `200 {user}` + `Set-Cookie`. 401 generic "Invalid email or password". 423 when locked.
- `POST /api/auth/logout` → `200 {status:"ok"}` + clearing cookie.
- `GET /api/auth/me` → `{user, usage: {analyses_this_month, chat_messages_today}, limits}`.
- `user` = `{id, email, plan, created_at}` (never password hash / customer id).

### BYOK (bring your own key)
- `GET /api/keys` → `{keys: [{provider, key_hint, base_url, model, created_at}], providers: [{id, name, default_base_url, default_model}]}`
- `POST /api/keys` `{provider, api_key, model?, base_url?}` → `201 {key}`. Key encrypted at rest; only last-4 hint ever returned.
- `DELETE /api/keys/{provider}` → `{status:"deleted"}` / 404.
- Providers: `openai`, `anthropic`, `groq`, `openrouter`, `custom` (OpenAI-compatible; `base_url` required, https only, public hosts only).

### Billing
- `GET /api/billing/plans` → `{plans: [{id, name, price_cents, analyses_per_month, chat_messages_per_day, max_repo_mb}], billing_enabled}`
- `GET /api/billing/status` → `{plan, billing_enabled, payments: [...]}`
- `POST /api/billing/checkout` `{plan: "pro"}` → `{checkout_url}`; 503 if billing not configured.
- `POST /api/billing/webhook` (Stripe) — verifies `Stripe-Signature`, idempotent per event id.

### Repos / analysis
- `POST /api/repos/github` `{url}` → `202 {job}`. Enforces plan `analyses_per_month`, max 2 concurrent jobs per user.
- `GET /api/repos/jobs` → `{jobs: [...]}`; `GET /api/repos/jobs/{id}` → `{job}`
  - `job` = `{id, github_url, repo_id, analysis_id, status: queued|running|done|failed, stage: queued|cloning|indexing|embedding|scanning|graphing|done|failed, progress: 0-100, error, created_at, updated_at, result?}`
  - `result` = `{repo_id, repo_name, stats: {files, lines, symbols, chunks, embeddings}, summary: <findings summary>}`
- `GET /api/repos` (also shadows `GET /api/index/repos` in SaaS mode) → `{repositories: [...], count}` — only the caller's.
- `GET /api/repos/{repo_id}/graph` →
  ```json
  {"repo_id": "...",
   "nodes": [{"id": "file:app/x.py", "type": "file|class|function|method|external",
              "label": "x.py", "file_path": "app/x.py", "line": 1, "language": "Python",
              "metrics": {"loc": 120, "complexity": 7},
              "findings": {"critical": 0, "high": 1, "medium": 0, "low": 2, "info": 0},
              "health": "good|warn|bad"}],
   "edges": [{"source": "file:app/x.py", "target": "sym:app/x.py::Foo", "type": "contains|imports|calls"}],
   "stats": {"nodes": 0, "edges": 0, "truncated": false}}
  ```
  Node ids: `file:<path>`, `sym:<path>::<QualifiedName>`, `ext:<module>`.
- `GET /api/repos/{repo_id}/findings` →
  ```json
  {"repo_id": "...",
   "summary": {"score": 0-100, "by_severity": {...}, "by_category": {...}, "total": 0},
   "findings": [{"id", "category": "security|secrets|quality|maintainability|testing|dependency",
                 "rule_id", "title", "severity": "critical|high|medium|low|info",
                 "file_path", "line", "snippet", "description", "recommendation",
                 "cwe": "CWE-78"|null, "node_id"}]}
  ```
- `GET /api/repos/{repo_id}/node?id=<node_id>` → `{node, findings: [...], neighbors: {incoming: [{id,label,type,edge}], outgoing: [...]}, source: {path, start_line, end_line, lines: [{line, content}]}}`

### Chat (existing endpoints, now owner-scoped)
- `POST /api/chat/sessions` `{repo_id, title?}` → `201 {session_id, ...}`
- `POST /api/chat/sessions/{id}/messages` `{question, focus?: {node_id?, file_path?, symbol?}}` → SSE stream (`data: {type: token|done|error|status, ...}`).
  Uses the caller's BYOK key when present, else the platform provider. Counts against `chat_messages_per_day`. Questions are screened by `prompt_guard`.

## Frontend routes (static)
- `/` → `frontend/landing.html` (marketing + pricing)
- `/login`, `/signup` → `frontend/auth.html`
- `/settings` → `frontend/settings.html` (account, BYOK keys, billing)
- `/app` → `frontend/index.html` (workspace: GitHub URL → progress → graph explorer + findings + report + chat)

## Security model summary
- Passwords: `hashlib.scrypt` (n=2**14, r=8, p=1, 16-byte salt), constant-time compare.
- Sessions: 256-bit random token in `HttpOnly; SameSite=Lax` cookie; only SHA-256 hash stored; 7-day TTL; all sessions revoked on password change.
- BYOK: Fernet (AES-128-CBC + HMAC-SHA256) with key derived from `SHERLOCK_SECRET_KEY`; decrypted only in-process at call time; never logged or returned.
- Rate limiting: per-IP and per-user sliding windows by bucket.
- CSRF: `Origin` must match `Host` (or allow-list) on non-GET; SameSite=Lax cookie.
- Prompt injection: user questions screened (block on high score); retrieved repository content sanitized + fenced as untrusted data; system prompt guardrails.
- Clone safety: https://github.com only, strict owner/repo regex, `git clone --depth 1 --no-tags --single-branch` with hooks disabled, `GIT_TERMINAL_PROMPT=0`, timeout, size cap, symlinks not followed.
- Webhooks: HMAC-SHA256 `Stripe-Signature` verification with 5-min tolerance, event-id idempotency.
- Security headers on API: CSP, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`.

## v2 additions (findings quality, multi-language graph, AI)

### Dependency CVEs (`app/saas/dep_audit.py`)
New job stage `auditing` (between `scanning` and `graphing`). Parses `requirements*.txt`, `pyproject.toml` (PEP 621 + Poetry), `Pipfile.lock`, `poetry.lock`, `package.json`, `package-lock.json` (v2/v3), `yarn.lock` (v1), `go.mod`, `Cargo.lock`, and queries the OSV.dev `querybatch` API. Network failure → job continues; `summary.dependencies.error` explains.
Dependency findings use the normal finding shape plus:
```json
{"category": "dependency", "rule_id": "dep-vuln", "package": "requests", "version": "2.19.0",
 "ecosystem": "PyPI", "advisory_ids": ["GHSA-...", "CVE-..."], "fixed_versions": ["2.20.0"],
 "references": ["https://osv.dev/vulnerability/GHSA-..."]}
```
`summary.dependencies` = `{"scanned": 42, "vulnerable": 3, "ecosystems": ["PyPI"], "manifests": ["requirements.txt"], "error": null}`.

### Noise reduction
Low-value per-symbol rules (`q-docstring`, `q-no-tests`) are aggregated: at most one finding per file (docstrings) and one repo-level `q-no-tests` finding listing untested modules. Aggregated findings carry `"occurrences": [{"file_path", "line"}]` and `"count"`. Findings get `"confidence": "high"|"medium"|"low"`. Score only penalizes medium+ confidence.

### Multi-language graph (`app/saas/lang_parse.py`)
tree-sitter parses JS/TS/TSX/Go/Java/Rust/Ruby (Python stays on stdlib `ast`): classes/functions/methods → graph symbol nodes (`sym:<path>::<Qual.Name>`), imports → edges, simple call resolution → `calls` edges. Falls back to regex if a grammar is missing.

### AI triage
- `GET /api/capabilities` (public) → `{"platform_model": bool, "platform_model_name": str|null}`. With a user cookie also returns `{"byok": bool}`.
- `POST /api/repos/{repo_id}/triage` `{finding_ids?: [...]}` (default: up to 15 untriaged security/secrets/dependency findings with severity ≥ medium) → `{findings: [...updated...], triaged: n, skipped: n}`. Uses the caller's BYOK key, else the platform model; 409 `{"error": "No AI model available..."}` if neither. Each triage counts 1 chat-quota unit per finding (402 when exhausted). Rate bucket `triage` (5/min).
- Finding gains `"triage": {"verdict": "confirmed"|"likely"|"false_positive"|"uncertain", "confidence": 0.0-1.0, "explanation": str, "fix": str|null, "model": str, "triaged_at": iso}`. Summary gains `"triaged": n, "false_positives": n`; false positives are excluded from the score.

### Hosted default model
Env `SHERLOCK_LLM_PROVIDER` (openai|anthropic|groq|openrouter|custom) + `SHERLOCK_LLM_API_KEY` (+ optional `SHERLOCK_LLM_MODEL`, `SHERLOCK_LLM_BASE_URL`). When set, chat and triage work without BYOK; usage is bounded by plan quotas. Legacy `GROQ_API_KEY`/`LLM_PROVIDER` still work.
Chat with no model available returns `409 {"error": "No AI model is configured. Add your own API key in Settings.", "code": "no_model"}` **before** streaming.

### Graph-aware chat
With `focus`, the server builds extra context from the graph artifact: the node's source window, its incoming/outgoing neighbours (callers/callees/imports) with their source snippets (bounded), and the node's findings. Passed to `RAGPipeline.ask_stream(..., extra_context=[chunks])` and given priority in the context budget; cited like normal chunks.
