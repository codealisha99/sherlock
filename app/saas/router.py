"""SaaS routing + middleware for the stdlib HTTP handler.

`process(handler, method)` is called first thing in do_GET/do_POST/do_DELETE.
It returns True when it fully handled the request (SaaS route, or a rejection
such as 401/403/429), False when the legacy handler should continue.

Middleware order for every /api request:
  1. read + cache body (bounded), so legacy handlers can still read rfile
  2. rate limit (per IP, and per user once authenticated)
  3. CSRF: state-changing requests must be same-origin (webhook exempt)
  4. resolve session user
  5. SaaS route table dispatch, else legacy-route policy:
     auth required, local-path endpoints disabled, repo/session/analysis
     ownership enforced, prompt-injection screening on chat questions.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple
from urllib.parse import parse_qs, urlparse

from app.saas.http import HTTPError, Request, Response

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_000_000

# (method, regex, "module:function", auth_required, rate_bucket)
ROUTE_SPECS: List[Tuple[str, str, str, bool, str]] = [
    # --- auth
    ("POST", r"/api/auth/signup", "app.saas.auth_routes:signup", False, "auth"),
    ("POST", r"/api/auth/login", "app.saas.auth_routes:login", False, "auth"),
    ("POST", r"/api/auth/logout", "app.saas.auth_routes:logout", False, "api"),
    ("GET", r"/api/auth/me", "app.saas.auth_routes:me", True, "api"),
    # --- BYOK
    ("GET", r"/api/keys", "app.saas.byok_routes:list_keys", True, "api"),
    ("POST", r"/api/keys", "app.saas.byok_routes:save_key", True, "api"),
    ("DELETE", r"/api/keys/(?P<provider>[a-z0-9_-]{1,32})", "app.saas.byok_routes:delete_key", True, "api"),
    # --- billing
    ("GET", r"/api/billing/plans", "app.saas.billing_routes:plans", False, "api"),
    ("GET", r"/api/billing/status", "app.saas.billing_routes:status", True, "api"),
    ("POST", r"/api/billing/checkout", "app.saas.billing_routes:checkout", True, "api"),
    ("POST", r"/api/billing/webhook", "app.saas.billing_routes:webhook", False, "webhook"),
    # --- GitHub ingest, graph, findings
    ("POST", r"/api/repos/github", "app.saas.ingest_routes:start_ingest", True, "ingest"),
    ("GET", r"/api/repos/jobs", "app.saas.ingest_routes:list_jobs", True, "api"),
    ("GET", r"/api/repos/jobs/(?P<job_id>[a-f0-9]{32})", "app.saas.ingest_routes:get_job", True, "api"),
    ("GET", r"/api/repos", "app.saas.ingest_routes:list_repos", True, "api"),
    ("GET", r"/api/index/repos", "app.saas.ingest_routes:list_repos", True, "api"),
    ("GET", r"/api/repos/(?P<repo_id>[A-Za-z0-9_.:-]{1,128})/graph", "app.saas.ingest_routes:get_graph", True, "api"),
    ("GET", r"/api/repos/(?P<repo_id>[A-Za-z0-9_.:-]{1,128})/findings", "app.saas.ingest_routes:get_findings", True, "api"),
    ("GET", r"/api/repos/(?P<repo_id>[A-Za-z0-9_.:-]{1,128})/node", "app.saas.ingest_routes:get_node", True, "api"),
    ("POST", r"/api/repos/(?P<repo_id>[A-Za-z0-9_.:-]{1,128})/triage", "app.saas.triage_routes:triage", True, "triage"),
    # --- platform capabilities (public)
    ("GET", r"/api/capabilities", "app.saas.triage_routes:capabilities", False, "api"),
]

CSRF_EXEMPT = {"/api/billing/webhook"}
PUBLIC_LEGACY = {"/api/health"}
LOCAL_PATH_ENDPOINTS = {"/api/analyze", "/api/index"}
# Legacy Q&A endpoints; the chat-session endpoint counts quota itself after validation.
CHAT_PATHS = {"/api/ask", "/api/ask/stream"}

_compiled: Optional[List[Tuple[str, Pattern[str], str, bool, str]]] = None


def auth_required() -> bool:
    """SaaS mode. Set SHERLOCK_AUTH_REQUIRED=false for single-user local dev."""
    return os.getenv("SHERLOCK_AUTH_REQUIRED", "true").lower() != "false"


def allow_local_paths() -> bool:
    """Local-path analysis reads arbitrary server dirs: off whenever auth is on."""
    if os.getenv("SHERLOCK_ALLOW_LOCAL_PATHS", "").lower() == "true":
        return True
    return not auth_required()


def _routes() -> List[Tuple[str, Pattern[str], str, bool, str]]:
    global _compiled
    if _compiled is None:
        _compiled = [(m, re.compile(f"^{p}$"), t, a, b) for m, p, t, a, b in ROUTE_SPECS]
    return _compiled


def _resolve(target: str) -> Callable[[Request], Response]:
    module_name, func_name = target.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def match_route(method: str, path: str) -> Optional[Tuple[str, Dict[str, str], bool, str]]:
    for m, pattern, target, auth, bucket in _routes():
        if m != method:
            continue
        hit = pattern.match(path)
        if hit:
            return target, hit.groupdict(), auth, bucket
    return None


# ----------------------------------------------------------------- transport
def _client_ip(handler: Any) -> str:
    if os.getenv("SHERLOCK_TRUST_PROXY", "").lower() == "true":
        fwd = handler.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return handler.client_address[0] if handler.client_address else "unknown"


def _read_body(handler: Any) -> bytes:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        raise HTTPError(400, "Invalid Content-Length")
    if length < 0:
        raise HTTPError(400, "Invalid Content-Length")
    if length > MAX_BODY_BYTES:
        raise HTTPError(413, "Request body too large")
    body = handler.rfile.read(length) if length else b""
    # Re-expose to legacy handlers that read rfile themselves.
    handler.rfile = io.BytesIO(body)
    return body


def send_response(handler: Any, resp: Response) -> None:
    from app.saas.security import SECURITY_HEADERS

    payload = resp.encoded()
    handler.send_response(resp.status)
    handler.send_header("Content-Type", resp.content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    for k, v in SECURITY_HEADERS:
        handler.send_header(k, v)
    for k, v in resp.headers:
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(payload)


def _error(handler: Any, err: HTTPError) -> None:
    send_response(
        handler,
        Response(status=err.status, body={"status": "error", "error": err.message}, headers=err.headers),
    )


def _same_origin(handler: Any) -> bool:
    origin = handler.headers.get("Origin")
    if not origin:
        # Non-browser clients (curl, tests) send no Origin; browsers always do on
        # cross-site POSTs. SameSite=Lax cookies cover the remaining form case.
        return True
    host = handler.headers.get("Host", "")
    allowed = {f"http://{host}", f"https://{host}"}
    extra = os.getenv("SHERLOCK_ALLOWED_ORIGINS", "")
    allowed |= {o.strip() for o in extra.split(",") if o.strip()}
    return origin in allowed


# ------------------------------------------------------------- legacy policy
def _extract_scope(path: str, query: Dict[str, List[str]], body: bytes) -> Dict[str, Optional[str]]:
    data: Dict[str, Any] = {}
    if body:
        try:
            parsed = json.loads(body.decode("utf-8"))
            data = parsed if isinstance(parsed, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {}
    repo_id = query.get("repo_id", [None])[0] or data.get("repo_id")
    m = re.match(r"^/api/index/repos/([^/]+)", path)
    if m:
        repo_id = m.group(1)
    session_id = None
    m = re.match(r"^/api/chat/sessions/([^/]+)", path)
    if m:
        session_id = m.group(1)
    analysis_id = None
    m = re.match(r"^/api/analysis/([^/]+)", path)
    if m:
        analysis_id = m.group(1)
    question = data.get("question") if isinstance(data.get("question"), str) else None
    return {"repo_id": repo_id, "session_id": session_id, "analysis_id": analysis_id, "question": question}


def _session_repo(session_id: str) -> Optional[str]:
    from app.storage import IndexStore

    store = IndexStore()
    try:
        session = store.get_chat_session(session_id)
    finally:
        store.close()
    return session["repo_id"] if session else None


def _enforce_legacy_policy(path: str, user: Optional[Dict[str, Any]], scope: Dict[str, Optional[str]]) -> None:
    from app.saas.db import get_store

    if path in LOCAL_PATH_ENDPOINTS and not allow_local_paths():
        raise HTTPError(403, "Local-path analysis is disabled. Submit a GitHub URL via POST /api/repos/github.")
    if not auth_required():
        return
    if user is None:
        raise HTTPError(401, "Authentication required")
    store = get_store()
    repo_id = scope["repo_id"]
    if scope["session_id"]:
        session_repo = _session_repo(scope["session_id"])
        if session_repo is None:
            raise HTTPError(404, "Chat session not found")
        repo_id = session_repo
    if repo_id and not store.user_owns_repo(user["id"], repo_id):
        raise HTTPError(404, "Repository not found")
    if scope["analysis_id"] and not store.user_owns_analysis(user["id"], scope["analysis_id"]):
        raise HTTPError(404, "Analysis not found")


def _screen_question(question: Optional[str]) -> None:
    if not question:
        return
    from app.saas.prompt_guard import inspect_user_input

    verdict = inspect_user_input(question)
    if verdict.blocked:
        raise HTTPError(400, "Your message was blocked by the prompt-injection filter: " + "; ".join(verdict.reasons))


def consume_chat_quota(user: Dict[str, Any]) -> Optional[str]:
    """Record one chat message; return an error message when over the plan limit."""
    from datetime import datetime, timezone

    from app.saas.db import get_store
    from app.saas.plans import EVENT_CHAT, plan_limits

    store = get_store()
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    allowed = int(plan_limits(user.get("plan", "free"))["chat_messages_per_day"])
    if store.count_usage_since(user["id"], EVENT_CHAT, day_start) >= allowed:
        return f"Daily chat limit reached ({allowed} messages). Upgrade your plan for more."
    store.record_usage(user["id"], EVENT_CHAT)
    return None


def _require_model(handler: Any, user: Dict[str, Any]) -> None:
    """Refuse before spending quota when no model can answer (same as chat route)."""
    from app.saas.llm_platform import resolve_gateway

    gateway, _ = resolve_gateway(
        user, getattr(handler, "gateway", None), allow_ollama=bool(getattr(handler, "enable_ai", False))
    )
    if gateway is None:
        raise HTTPError(409, "No AI model is configured. Add your own API key in Settings.")


def _rate_limit(bucket: str, key: str) -> None:
    from app.saas.ratelimit import check

    retry_after = check(bucket, key)
    if retry_after is not None:
        secs = str(max(1, int(retry_after + 0.999)))
        raise HTTPError(429, "Too many requests. Please slow down.", headers=(("Retry-After", secs),))


# ------------------------------------------------------------------- entry
def process(handler: Any, method: str) -> bool:
    """Run middleware + SaaS dispatch. True => response already sent."""
    parsed = urlparse(handler.path)
    path = parsed.path
    if not path.startswith("/api/"):
        return False  # static assets: legacy _serve_static handles them
    query = parse_qs(parsed.query)
    handler.saas_user = None
    try:
        body = _read_body(handler) if method in ("POST", "DELETE") else b""
        ip = _client_ip(handler)

        route = match_route(method, path)
        bucket = route[3] if route else ("chat" if path in CHAT_PATHS or path.endswith("/messages") else "api")
        _rate_limit(bucket, f"ip:{ip}")

        if method != "GET" and path not in CSRF_EXEMPT and not _same_origin(handler):
            raise HTTPError(403, "Cross-origin request rejected")

        from app.saas.auth import resolve_user

        user = resolve_user(handler.headers)
        handler.saas_user = user
        if user is not None:
            _rate_limit(bucket, f"user:{user['id']}")

        if route is not None:
            target, params, needs_auth, _ = route
            # In local dev mode (auth off) the legacy list route stays legacy.
            if path == "/api/index/repos" and not auth_required():
                return False
            if needs_auth and user is None:
                raise HTTPError(401, "Authentication required")
            req = Request(
                method=method, path=path, query=query, headers=handler.headers,
                body=body, client_ip=ip, user=user, path_params=params,
            )
            send_response(handler, _resolve(target)(req))
            return True

        if path in PUBLIC_LEGACY:
            return False
        scope = _extract_scope(path, query, body)
        _enforce_legacy_policy(path, user, scope)
        _screen_question(scope["question"])
        if path in CHAT_PATHS and user is not None:
            _require_model(handler, user)
            limit_error = consume_chat_quota(user)
            if limit_error:
                raise HTTPError(402, limit_error)
        return False
    except HTTPError as err:
        _error(handler, err)
        return True
    except Exception:  # never leak internals
        logger.exception("Unhandled error in SaaS middleware for %s %s", method, path)
        _error(handler, HTTPError(500, "Internal server error"))
        return True
