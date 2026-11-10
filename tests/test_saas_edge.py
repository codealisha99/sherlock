"""Edge gate + client identity behind Vercel (SHERLOCK_EDGE_SECRET).

Encodes what was measured in production: through Vercel every user arrived from
a few shared egress IPs, so per-IP limits were both bypassable and able to lock
out unrelated users. X-Vercel-Proxied-For is overwritten by Vercel (trusted);
X-Vercel-Forwarded-For / X-Forwarded-For pass client values through (untrusted).
"""

from __future__ import annotations

import json

import pytest

from app.saas import ratelimit
from tests.test_api import simulate_http_request

SECRET = "edge-secret-for-tests-0123456789abcdef"
EDGE = f"/_edge/{SECRET}"


@pytest.fixture()
def edge(monkeypatch):
    monkeypatch.setenv("SHERLOCK_EDGE_SECRET", SECRET)
    ratelimit.reset()
    yield
    ratelimit.reset()


def _status(method: str, path: str, headers: dict | None = None, body: dict | None = None) -> int:
    raw = json.dumps(body).encode() if body is not None else b""
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    return simulate_http_request(method, path, raw, hdrs)[0]


def _login(headers: dict) -> int:
    return _status("POST", f"{EDGE}/api/auth/login", headers,
                   {"email": "nobody@example.com", "password": "wrongpassword1"})


def test_direct_api_is_hidden_but_health_stays_reachable(edge):
    assert _status("GET", "/api/billing/plans") == 404
    assert _status("GET", "/api/health") == 200
    assert _status("GET", f"{EDGE}/api/billing/plans") == 200


def test_wrong_or_malformed_secret_is_404(edge):
    assert _status("GET", "/_edge/wrong-secret/api/billing/plans") == 404
    assert _status("GET", f"/_edge/{SECRET}") == 404  # no path after the secret


def test_rate_limit_keys_on_proxied_for_and_ignores_spoofable_headers(edge):
    # Same real client, attacker-controlled headers varying on every request.
    codes = [
        _login({"X-Vercel-Proxied-For": "198.51.100.7",
                "X-Vercel-Forwarded-For": f"203.0.113.{i}",
                "X-Forwarded-For": f"192.0.2.{i}"})
        for i in range(12)
    ]
    assert codes[:10] == [401] * 10
    assert codes[10:] == [429, 429]


def test_one_noisy_client_does_not_lock_out_others(edge):
    for _ in range(11):
        _login({"X-Vercel-Proxied-For": "198.51.100.7"})
    assert _login({"X-Vercel-Proxied-For": "198.51.100.8"}) == 401  # different user: not limited


def test_garbage_proxied_for_falls_back_to_socket_ip(edge):
    assert _login({"X-Vercel-Proxied-For": "not-an-ip"}) == 401


def test_without_secret_behaviour_is_unchanged(monkeypatch):
    monkeypatch.delenv("SHERLOCK_EDGE_SECRET", raising=False)
    assert _status("GET", "/api/billing/plans") == 200


def test_health_reports_503_when_database_is_down(monkeypatch, tmp_path):
    # Measured in prod: health said 200 while every DB query failed, so Railway
    # promoted a broken deploy. Health must reflect the account DB.
    from app.saas.db import SaasStore, set_store

    store = SaasStore(tmp_path / "saas.db")
    set_store(store)
    try:
        assert _status("GET", "/api/health") == 200

        def down(*_a, **_k):
            raise ConnectionError("db down")

        monkeypatch.setattr(store, "_one", down)
        status, _, body = simulate_http_request("GET", "/api/health", b"", {})
        assert status == 503
        assert b"db down" not in body  # no internals leaked
    finally:
        set_store(None)
