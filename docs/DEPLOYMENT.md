# Deployment: Vercel + Railway + Neon

```
browser ──► Vercel (static frontend, CDN)
              │  /api/*  (rewrite, same-origin for the browser)
              ▼
            Railway  "api" service (Docker, Singapore)  ──► Neon Postgres (Singapore)
              │  volume /data: code index (SQLite + sqlite-vec), clones, graphs
```

| Piece | Where | Holds |
|---|---|---|
| Frontend | Vercel project `sherlock` | `frontend/` bundled by `scripts/build_vercel.py` |
| Backend | Railway project `sherlock`, service `api` | `Dockerfile`, `railway.json` |
| Account data | Neon project `sherlock` (org `alisha`), db `sherlock`, role `sherlock_app` | users, sessions, encrypted BYOK keys, jobs, usage, payments |
| Code index | Railway volume `/data` | derived data; rebuildable by re-analyzing a repo |

## Railway variables (service `api`)

| Var | Value |
|---|---|
| `DATABASE_URL` | Neon **pooled** connection string (`…-pooler…`) |
| `SHERLOCK_SECRET_KEY` | random 48+ bytes. **Never rotate casually**: it encrypts stored BYOK keys; changing it makes them undecryptable (users must re-enter keys) |
| `SHERLOCK_AUTH_REQUIRED` | `true` |
| `SHERLOCK_COOKIE_SECURE` | `true` |
| `SHERLOCK_EDGE_SECRET` | random 32+ URL-safe chars (`python -c 'import secrets;print(secrets.token_urlsafe(32))'`). Vercel's `/api` rewrite targets `/_edge/<secret>/api/…`; with it set, `/api/*` (except `/api/health`) is 404 when hit directly, and the client IP is taken from `X-Vercel-Proxied-For` (overwritten by Vercel, so not spoofable) |
| `SHERLOCK_TRUST_PROXY` | `true` (fallback client-IP source for non-edge requests) |
| `SHERLOCK_DATA_DIR` | `/data` |
| `SHERLOCK_DB_POOL_MAX` | `5` |
| `SHERLOCK_ALLOWED_ORIGINS` | the Vercel production origin (CSRF check: browser Origin is Vercel, Host is Railway) |
| `SHERLOCK_PUBLIC_URL` | the Vercel production origin (Stripe redirect URLs) |
| optional `SHERLOCK_LLM_PROVIDER` / `SHERLOCK_LLM_API_KEY` / `SHERLOCK_LLM_MODEL` | hosted default model; without it users must add their own key |
| optional `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_PRO` | billing; without it Pro shows "Coming soon". Webhook URL: `<vercel origin>/api/billing/webhook` |

## Deploy

```bash
# backend
railway up --ci --service api

# frontend (BACKEND = the Railway public origin; the edge secret is baked into vercel.json)
SHERLOCK_EDGE_SECRET=$(railway variables --service api --kv | sed -n 's/^SHERLOCK_EDGE_SECRET=//p') \
  python scripts/build_vercel.py https://api-production-cb158.up.railway.app
cd dist/vercel && vercel link --yes --project sherlock && vercel deploy --prod --yes
```

## Live URLs
- App: https://sherlock-blush.vercel.app
- API origin (direct access serves only `/api/health`; everything else goes through Vercel): https://api-production-cb158.up.railway.app
- Railway region: `asia-southeast1-eqsg3a` (Singapore), same region as Neon `aws-ap-southeast-1`

## Gotchas hit during the first deploy
- **Region:** `deploy.multiRegionConfig` in `railway.json` was *not* applied to the existing
  service (it ran in `sfo`, ~2.5 s per authenticated request because every query crossed the
  Pacific). Set it on the service instance (dashboard → Settings → Regions, or the
  `serviceInstanceUpdate` GraphQL mutation), redeploy, and only then attach the volume:
  a volume is created in the service's current region. After the move: ~0.2 s per request.
- **HOME after privilege drop:** `setpriv` keeps the environment, so `HOME=/root`; libpq then
  probes `/root/.postgresql/postgresql.crt`, gets EACCES (fatal, unlike ENOENT) and every
  Postgres connection fails. The entrypoint resets `HOME=/home/app`.
- **Variables set with `--skip-deploys`** are not seen by the running container; redeploy.
- **Vercel `cleanUrls`:** rewrite destinations must not end in `.html` (they 404).

Rotating `SHERLOCK_EDGE_SECRET`: set it on Railway, redeploy the backend, then rebuild and
redeploy Vercel. Between the two steps `/api` returns 404 through Vercel.

## Notes / limits
- Single Railway replica: rate limits and job threads are in-process, and the SQLite index lives on one volume. Scale-out needs a shared rate-limit store, a real job queue and moving the index to Postgres + pgvector.
- Analysis jobs run in-process. On startup, any job still `queued`/`running` is marked failed ("Interrupted by a server restart"), because its thread died with the previous container. Correct only with one replica.
- Analysis quota is charged when a job **succeeds**; in-flight jobs count toward the monthly limit at submit, so concurrent submits can't overshoot it.
- `/api/health` runs `SELECT 1` against the account DB and returns 503 if it fails. The server also refuses to start (exits after the 30 s pool timeout) when the DB is unreachable, so a deploy that can't reach Neon is never promoted.
- Volume: ~5-6 MB per analysis. Watch `/data` usage in the Railway dashboard; no automatic cleanup yet.
- Image is ~2.3 GB (CPU torch + baked embedding model); the model loads offline (`HF_HUB_OFFLINE=1`).
- Postgres integration tests: create a throwaway Neon branch and run
  `SHERLOCK_TEST_DATABASE_URL=<branch url> pytest tests/test_saas_db_postgres.py`.
