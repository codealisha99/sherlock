"""Auth tests: cookie parsing, session resolution, auth handlers, and router integration.

Handlers are exercised directly with app.saas.http.Request objects. The router
integration section drives app.saas.router.process with a minimal fake handler.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from email.message import Message
from typing import Any, Dict, List, Optional, Tuple

import pytest

from app.saas import auth, auth_routes, ratelimit, router
from app.saas.db import SaasStore, iso, set_store
from app.saas.http import HTTPError, Request
from app.saas.security import hash_token

PASSWORD = "correct-horse-42"
EMAIL = "alice@example.com"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_AUTH_REQUIRED", "true")
    monkeypatch.delenv("SHERLOCK_RATE_LIMIT_DISABLED", raising=False)
    monkeypatch.delenv("SHERLOCK_COOKIE_SECURE", raising=False)
    db = SaasStore(tmp_path / "s.db")
    set_store(db)
    ratelimit.reset()
    yield db
    set_store(None)
    ratelimit.reset()


def make_request(method: str, path: str, body: Optional[Dict[str, Any]] = None, cookie: str = "", user=None) -> Request:
    headers = {"Cookie": cookie} if cookie else {}
    raw = json.dumps(body).encode() if body is not None else b""
    return Request(
        method=method, path=path, query={}, headers=headers, body=raw,
        client_ip="127.0.0.1", user=user,
    )


def cookie_from(set_cookie_value: str) -> str:
    """Turn a Set-Cookie value into the Cookie request header value."""
    return set_cookie_value.split(";")[0]


def signup(store_obj, email: str = EMAIL, password: str = PASSWORD) -> Tuple[Dict[str, Any], str]:
    resp = auth_routes.signup(make_request("POST", "/api/auth/signup", {"email": email, "password": password}))
    return resp.body["user"], dict(resp.headers)["Set-Cookie"]


# ------------------------------------------------------------ cookie + tokens
def test_issue_session_sets_hardened_cookie(store):
    user, set_cookie = signup(store, "cookie@example.com")
    assert set_cookie.startswith("sherlock_session=")
    for attr in ("Path=/", "HttpOnly", "SameSite=Lax", "Max-Age=604800"):
        assert attr in set_cookie
    assert "Secure" not in set_cookie


def test_cookie_secure_flag_follows_env(store, monkeypatch):
    monkeypatch.setenv("SHERLOCK_COOKIE_SECURE", "true")
    _, set_cookie = signup(store, "secure@example.com")
    assert "; Secure" in set_cookie


def test_clear_cookie_expires_immediately():
    value = auth.clear_cookie()
    assert value.startswith("sherlock_session=;")
    assert "Max-Age=0" in value


def test_only_token_hash_is_stored(store):
    _, set_cookie = signup(store, "hash@example.com")
    token = cookie_from(set_cookie).split("=", 1)[1]
    with store._conn() as c:
        hashes = [r[0] for r in c.execute("SELECT token_hash FROM sessions").fetchall()]
    assert hash_token(token) in hashes
    assert token not in hashes


@pytest.mark.parametrize(
    "cookie",
    [
        "sherlock_session",
        "sherlock_session=",
        "=====",
        "sherlock_session=\x00\x01",
        "sherlock_session=" + "a" * 5000,
        "garbage; ;; ==; sherlock_session",
        "other=1",
        "ÿþ=bad",
    ],
)
def test_malformed_cookie_resolves_to_none(store, cookie):
    assert auth.resolve_user({"Cookie": cookie}) is None


def test_resolve_user_with_email_message_headers(store):
    user, set_cookie = signup(store, "msg@example.com")
    msg = Message()
    msg["Cookie"] = cookie_from(set_cookie)
    resolved = auth.resolve_user(msg)
    assert resolved is not None and resolved["id"] == user["id"]


def test_resolve_user_via_bearer_token(store):
    user, set_cookie = signup(store, "bearer@example.com")
    token = cookie_from(set_cookie).split("=", 1)[1]
    resolved = auth.resolve_user({"Authorization": f"Bearer {token}"})
    assert resolved is not None and resolved["email"] == "bearer@example.com"


@pytest.mark.parametrize("header", ["Bearer", "Bearer ", "Basic abc", "Bearer x" + "y" * 400, "Bearer nope"])
def test_bad_authorization_header_resolves_to_none(store, header):
    signup(store, "badauth@example.com")
    assert auth.resolve_user({"Authorization": header}) is None


def test_resolve_user_none_headers_do_not_raise(store):
    assert auth.resolve_user({}) is None
    assert auth.resolve_user(None) is None  # type: ignore[arg-type]


def test_expired_session_is_rejected(store):
    user = store.create_user("expired@example.com", "x")
    token = "expired-token-value"
    store.create_session(hash_token(token), user["id"], timedelta(seconds=-1))
    assert auth.resolve_user({"Cookie": f"sherlock_session={token}"}) is None


def test_public_user_hides_secrets(store):
    user, _ = signup(store, "public@example.com")
    assert set(user) == {"id", "email", "plan", "created_at"}


def test_end_session_revokes_token(store):
    _, set_cookie = signup(store, "logout-unit@example.com")
    cookie = cookie_from(set_cookie)
    assert auth.resolve_user({"Cookie": cookie}) is not None
    auth.end_session({"Cookie": cookie})
    assert auth.resolve_user({"Cookie": cookie}) is None


# ------------------------------------------------------------- signup / login
def test_signup_returns_201_user_and_cookie(store):
    resp = auth_routes.signup(make_request("POST", "/api/auth/signup", {"email": " Bob@Example.com ", "password": PASSWORD}))
    assert resp.status == 201
    assert resp.body["user"]["email"] == "bob@example.com"
    assert "password_hash" not in resp.body["user"]
    assert dict(resp.headers)["Set-Cookie"].startswith("sherlock_session=")


def test_signup_duplicate_email_is_409(store):
    signup(store, "dupe@example.com")
    with pytest.raises(HTTPError) as exc:
        signup(store, "DUPE@example.com")
    assert exc.value.status == 409


def test_signup_validates_input(store):
    with pytest.raises(HTTPError) as exc:
        signup(store, "not-an-email")
    assert exc.value.status == 400
    with pytest.raises(HTTPError) as exc:
        signup(store, "weak@example.com", "password")  # no digit
    assert exc.value.status == 400


def test_signup_rejects_non_object_json(store):
    req = Request(method="POST", path="/api/auth/signup", query={}, headers={}, body=b"[1,2]", client_ip="x")
    with pytest.raises(HTTPError) as exc:
        auth_routes.signup(req)
    assert exc.value.status == 400


def test_login_success_resets_failures(store):
    signup(store, "login@example.com")
    store.update_user(store.get_user_by_email("login@example.com")["id"], failed_logins=3)
    resp = auth_routes.login(make_request("POST", "/api/auth/login", {"email": "login@example.com", "password": PASSWORD}))
    assert resp.status == 200
    row = store.get_user_by_email("login@example.com")
    assert row["failed_logins"] == 0 and row["locked_until"] is None


def _login_error(email: str, password: str) -> HTTPError:
    with pytest.raises(HTTPError) as exc:
        auth_routes.login(make_request("POST", "/api/auth/login", {"email": email, "password": password}))
    return exc.value


def test_login_generic_error_for_unknown_email_and_wrong_password(store):
    signup(store, "generic@example.com")
    wrong_pw = _login_error("generic@example.com", "wrong-password-1")
    unknown = _login_error("nobody@example.com", PASSWORD)
    assert wrong_pw.status == unknown.status == 401
    assert wrong_pw.message == unknown.message == "Invalid email or password"


def test_login_malformed_email_is_generic_401(store):
    err = _login_error("not an email", PASSWORD)
    assert err.status == 401 and err.message == "Invalid email or password"


def test_login_locks_after_five_failures(store):
    signup(store, "lock@example.com")
    statuses = [_login_error("lock@example.com", "wrong-password-1").status for _ in range(4)]
    assert statuses == [401, 401, 401, 401]
    fifth = _login_error("lock@example.com", "wrong-password-1")
    assert fifth.status == 423
    row = store.get_user_by_email("lock@example.com")
    assert row["locked_until"] is not None


def test_locked_account_rejects_even_correct_password(store):
    signup(store, "locked2@example.com")
    for _ in range(5):
        _login_error("locked2@example.com", "wrong-password-1")
    assert _login_error("locked2@example.com", PASSWORD).status == 423


def test_lock_expires_and_login_succeeds(store):
    user, _ = signup(store, "expiry@example.com")
    past = iso(datetime.now(timezone.utc) - timedelta(minutes=1))
    store.update_user(user["id"], failed_logins=5, locked_until=past)
    resp = auth_routes.login(make_request("POST", "/api/auth/login", {"email": "expiry@example.com", "password": PASSWORD}))
    assert resp.status == 200
    assert store.get_user(user["id"])["failed_logins"] == 0


def test_lock_window_is_fifteen_minutes(store):
    signup(store, "window@example.com")
    for _ in range(5):
        _login_error("window@example.com", "wrong-password-1")
    locked_until = datetime.fromisoformat(store.get_user_by_email("window@example.com")["locked_until"])
    remaining = locked_until - datetime.now(timezone.utc)
    assert timedelta(minutes=14) < remaining <= timedelta(minutes=15)


# ------------------------------------------------------------- logout / me
def test_logout_revokes_session_and_clears_cookie(store):
    _, set_cookie = signup(store, "logout@example.com")
    cookie = cookie_from(set_cookie)
    resp = auth_routes.logout(make_request("POST", "/api/auth/logout", cookie=cookie))
    assert resp.status == 200 and resp.body == {"status": "ok"}
    assert dict(resp.headers)["Set-Cookie"] == auth.clear_cookie()
    assert auth.resolve_user({"Cookie": cookie}) is None


def test_me_returns_user_usage_and_limits(store):
    user, set_cookie = signup(store, "me@example.com")
    store.record_usage(user["id"], "analysis")
    store.record_usage(user["id"], "chat_message")
    store.record_usage(user["id"], "chat_message")
    req = make_request("GET", "/api/auth/me", user=auth.resolve_user({"Cookie": cookie_from(set_cookie)}))
    resp = auth_routes.me(req)
    assert resp.body["user"]["email"] == "me@example.com"
    assert resp.body["usage"] == {"analyses_this_month": 1, "chat_messages_today": 2}
    assert resp.body["limits"]["analyses_per_month"] == 3
    assert resp.body["limits"]["chat_messages_per_day"] == 30


def test_me_without_user_is_401(store):
    with pytest.raises(HTTPError) as exc:
        auth_routes.me(make_request("GET", "/api/auth/me"))
    assert exc.value.status == 401


def test_me_usage_excludes_previous_month(store):
    user, _ = signup(store, "oldusage@example.com")
    old = (datetime.now(timezone.utc).replace(day=1) - timedelta(days=2)).replace(hour=12)
    with store._conn() as c:
        c.execute(
            "INSERT INTO usage_events (user_id, kind, created_at) VALUES (?, 'analysis', ?)",
            (user["id"], iso(old)),
        )
    resp = auth_routes.me(make_request("GET", "/api/auth/me", user=auth.public_user(user)))
    assert resp.body["usage"]["analyses_this_month"] == 0


# ------------------------------------------------------- router integration
class FakeHandler:
    """Minimal stand-in for SherlockRequestHandler: records what router.process sends."""

    def __init__(self, method: str, path: str, headers: Dict[str, str], body: bytes = b"") -> None:
        self.command = method
        self.path = path
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.client_address = ("127.0.0.1", 5000)
        self.status: Optional[int] = None
        self.sent: List[Tuple[str, str]] = []

    def send_response(self, code: int, message: Optional[str] = None) -> None:
        self.status = code

    def send_header(self, keyword: str, value: str) -> None:
        self.sent.append((keyword, value))

    def end_headers(self) -> None:
        pass

    def json(self) -> Dict[str, Any]:
        return json.loads(self.wfile.getvalue().decode("utf-8"))

    def header(self, name: str) -> Optional[str]:
        return next((v for k, v in self.sent if k.lower() == name.lower()), None)


def call(method: str, path: str, body: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> FakeHandler:
    raw = json.dumps(body).encode() if body is not None else b""
    hdrs = {"Host": "localhost:8000", **(headers or {})}
    if raw:
        hdrs["Content-Type"] = "application/json"
        hdrs["Content-Length"] = str(len(raw))
    handler = FakeHandler(method, path, hdrs, raw)
    assert router.process(handler, method) is True
    return handler


def test_router_signup_then_me_with_cookie(store):
    signup_h = call("POST", "/api/auth/signup", {"email": "e2e@example.com", "password": PASSWORD})
    assert signup_h.status == 201
    cookie = cookie_from(signup_h.header("Set-Cookie"))
    assert signup_h.header("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'none'"

    me_h = call("GET", "/api/auth/me", headers={"Cookie": cookie})
    assert me_h.status == 200
    assert me_h.json()["user"]["email"] == "e2e@example.com"


def test_router_me_without_cookie_is_401(store):
    h = call("GET", "/api/auth/me")
    assert h.status == 401
    assert h.json() == {"status": "error", "error": "Authentication required"}


def test_router_cross_origin_post_is_403(store):
    h = call("POST", "/api/auth/login", {"email": EMAIL, "password": PASSWORD}, headers={"Origin": "https://evil.example"})
    assert h.status == 403
    assert h.json()["error"] == "Cross-origin request rejected"


def test_router_same_origin_post_allowed(store):
    h = call("POST", "/api/auth/login", {"email": EMAIL, "password": PASSWORD}, headers={"Origin": "http://localhost:8000"})
    assert h.status == 401  # reached the handler (no such user yet)


def test_router_rate_limit_returns_429_with_retry_after(store, monkeypatch):
    monkeypatch.setattr(ratelimit, "_clock", lambda: 1000.0)
    statuses = [call("POST", "/api/auth/signup", {}).status for _ in range(10)]
    assert statuses == [400] * 10
    limited = call("POST", "/api/auth/signup", {})
    assert limited.status == 429
    retry = int(limited.header("Retry-After"))
    assert 1 <= retry <= 60


def test_router_rejects_oversized_body(store):
    handler = FakeHandler("POST", "/api/auth/signup", {"Host": "localhost", "Content-Length": "2000000"}, b"")
    assert router.process(handler, "POST") is True
    assert handler.status == 413


def test_router_logout_then_me_is_401(store):
    cookie = cookie_from(call("POST", "/api/auth/signup", {"email": "out@example.com", "password": PASSWORD}).header("Set-Cookie"))
    out = call("POST", "/api/auth/logout", headers={"Cookie": cookie})
    assert out.status == 200
    assert call("GET", "/api/auth/me", headers={"Cookie": cookie}).status == 401
