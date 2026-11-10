"""Tests for Stripe billing: signatures, checkout form, webhooks, plan changes."""

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs

import httpx
import pytest

from app.saas import billing, billing_routes
from app.saas.billing import (
    create_checkout_session,
    handle_event,
    verify_webhook_signature,
)
from app.saas.db import SaasStore, set_store
from app.saas.http import HTTPError, Request

WEBHOOK_SECRET = "whsec_test_0123456789"
NOW = 1_700_000_000
PUBLIC = "https://sherlock.example"


@pytest.fixture
def store(tmp_path):
    db = SaasStore(tmp_path / "s.db")
    set_store(db)
    yield db
    set_store(None)


@pytest.fixture
def stripe_env(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_" + "x" * 24)
    monkeypatch.setenv("STRIPE_PRICE_PRO", "price_123")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("SHERLOCK_PUBLIC_URL", PUBLIC)


@pytest.fixture
def user(store):
    return store.create_user("pay@example.com", "hash")


def _sign(payload: bytes, t: int, secret: str = WEBHOOK_SECRET) -> str:
    sig = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={t},v1={sig}"


def _req(user, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    return Request(
        method="POST", path="/api/billing", query={}, headers=headers or {}, body=data,
        client_ip="127.0.0.1", user=user,
    )


def _install_stripe(monkeypatch, handler):
    monkeypatch.setattr(billing, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def _event(event_id, etype, obj):
    return {"id": event_id, "type": etype, "data": {"object": obj}}


def _checkout_completed(event_id, user_id, session_id="cs_1", customer="cus_1", plan="pro", paid=True):
    return _event(event_id, "checkout.session.completed", {
        "id": session_id,
        "payment_status": "paid" if paid else "unpaid",
        "status": "complete" if paid else "open",
        "client_reference_id": user_id,
        "customer": customer,
        "metadata": {"user_id": user_id, "plan": plan},
    })


# ------------------------------------------------------------ enabled flag
def test_billing_enabled_requires_secret_and_price(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PRICE_PRO", raising=False)
    assert billing.billing_enabled() is False
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    assert billing.billing_enabled() is False
    monkeypatch.setenv("STRIPE_PRICE_PRO", "price_123")
    assert billing.billing_enabled() is True


# ------------------------------------------------------------ signatures
def test_valid_signature_accepted():
    payload = b'{"id":"evt_1"}'
    verify_webhook_signature(payload, _sign(payload, NOW), WEBHOOK_SECRET, now=NOW)


def test_wrong_secret_rejected():
    payload = b"{}"
    with pytest.raises(ValueError):
        verify_webhook_signature(payload, _sign(payload, NOW, "whsec_other"), WEBHOOK_SECRET, now=NOW)


def test_tampered_payload_rejected():
    header = _sign(b'{"amount":1}', NOW)
    with pytest.raises(ValueError):
        verify_webhook_signature(b'{"amount":999}', header, WEBHOOK_SECRET, now=NOW)


def test_expired_timestamp_rejected_and_boundary_accepted():
    payload = b"{}"
    verify_webhook_signature(payload, _sign(payload, NOW - 300), WEBHOOK_SECRET, now=NOW)
    with pytest.raises(ValueError, match="tolerance"):
        verify_webhook_signature(payload, _sign(payload, NOW - 301), WEBHOOK_SECRET, now=NOW)


def test_future_timestamp_outside_tolerance_rejected():
    payload = b"{}"
    with pytest.raises(ValueError, match="tolerance"):
        verify_webhook_signature(payload, _sign(payload, NOW + 400), WEBHOOK_SECRET, now=NOW)


def test_multiple_v1_signatures_any_match_accepted():
    payload = b"{}"
    valid = _sign(payload, NOW).split("v1=")[1]
    header = f"t={NOW},v1={'0' * 64},v1={valid}"
    verify_webhook_signature(payload, header, WEBHOOK_SECRET, now=NOW)


def test_no_matching_v1_rejected_and_v0_ignored():
    payload = b"{}"
    with pytest.raises(ValueError):
        verify_webhook_signature(payload, f"t={NOW},v1={'0' * 64},v0=abc", WEBHOOK_SECRET, now=NOW)


@pytest.mark.parametrize("header", ["", "garbage", "t=abc,v1=00", f"t={NOW}", "v1=00"])
def test_malformed_headers_rejected(header):
    with pytest.raises(ValueError):
        verify_webhook_signature(b"{}", header, WEBHOOK_SECRET, now=NOW)


def test_empty_secret_rejected():
    with pytest.raises(ValueError):
        verify_webhook_signature(b"{}", _sign(b"{}", NOW), "", now=NOW)


# ------------------------------------------------------------ checkout
def test_checkout_form_fields_and_pending_payment(stripe_env, monkeypatch, user, store):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["idem"] = request.headers.get("idempotency-key")
        captured["form"] = parse_qs(request.content.decode(), keep_blank_values=True)
        return httpx.Response(200, json={"id": "cs_test_1", "url": "https://checkout.stripe.com/c/pay/cs_test_1"})

    _install_stripe(monkeypatch, handler)
    url = create_checkout_session(user, "pro")

    form = captured["form"]
    assert url == "https://checkout.stripe.com/c/pay/cs_test_1"
    assert captured["url"] == "https://api.stripe.com/v1/checkout/sessions"
    assert captured["auth"].startswith("Basic ")  # secret key via basic auth, not in URL
    assert captured["idem"]
    assert form["mode"] == ["subscription"]
    assert form["line_items[0][price]"] == ["price_123"]
    assert form["line_items[0][quantity]"] == ["1"]
    assert form["client_reference_id"] == [user["id"]]
    assert form["metadata[user_id]"] == [user["id"]]
    assert form["metadata[plan]"] == ["pro"]
    assert form["customer_email"] == [user["email"]]
    assert "customer" not in form
    assert form["success_url"] == [
        f"{PUBLIC}/settings?checkout=success&session_id={{CHECKOUT_SESSION_ID}}"
    ]
    assert form["cancel_url"] == [f"{PUBLIC}/settings?checkout=cancel"]
    assert store.get_payment_by_session("cs_test_1")["status"] == "pending"


def test_checkout_uses_existing_customer_when_present(stripe_env, monkeypatch, user):
    captured = {}

    def handler(request):
        captured["form"] = parse_qs(request.content.decode())
        return httpx.Response(200, json={"id": "cs_2", "url": "https://checkout.stripe.com/x"})

    _install_stripe(monkeypatch, handler)
    create_checkout_session({**user, "stripe_customer_id": "cus_existing"}, "pro")
    assert captured["form"]["customer"] == ["cus_existing"]
    assert "customer_email" not in captured["form"]


def test_checkout_stripe_failure_is_stripe_error(stripe_env, monkeypatch, user):
    _install_stripe(monkeypatch, lambda r: httpx.Response(500, json={"error": {"message": "boom"}}))
    with pytest.raises(billing.StripeError):
        create_checkout_session(user, "pro")


def test_checkout_network_failure_is_stripe_error(stripe_env, monkeypatch, user):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _install_stripe(monkeypatch, handler)
    with pytest.raises(billing.StripeError, match="Could not reach"):
        create_checkout_session(user, "pro")


def test_checkout_route_success(stripe_env, monkeypatch, user):
    _install_stripe(monkeypatch, lambda r: httpx.Response(200, json={"id": "cs_r", "url": "https://c.example/r"}))
    resp = billing_routes.checkout(_req(user, {"plan": "pro"}))
    assert resp.status == 200
    assert resp.body == {"checkout_url": "https://c.example/r"}


def test_checkout_disabled_is_503(monkeypatch, user):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PRICE_PRO", raising=False)
    with pytest.raises(HTTPError) as info:
        billing_routes.checkout(_req(user, {"plan": "pro"}))
    assert info.value.status == 503
    assert info.value.message == "Billing is not configured"


@pytest.mark.parametrize("body", [{"plan": "free"}, {"plan": "enterprise"}, {}, {"plan": 3}])
def test_checkout_rejects_non_paid_or_unknown_plan(stripe_env, user, body):
    with pytest.raises(HTTPError) as info:
        billing_routes.checkout(_req(user, body))
    assert info.value.status == 400


def test_checkout_when_already_on_plan_is_400(stripe_env, store, user):
    store.update_user(user["id"], plan="pro")
    with pytest.raises(HTTPError) as info:
        billing_routes.checkout(_req(user, {"plan": "pro"}))
    assert info.value.status == 400


def test_checkout_stripe_error_maps_to_502(stripe_env, monkeypatch, user):
    _install_stripe(monkeypatch, lambda r: httpx.Response(402, json={}))
    with pytest.raises(HTTPError) as info:
        billing_routes.checkout(_req(user, {"plan": "pro"}))
    assert info.value.status == 502


def test_status_and_plans_routes(stripe_env, store, user, monkeypatch):
    body = billing_routes.plans(_req(None)).body
    assert body["billing_enabled"] is True
    assert {p["id"] for p in body["plans"]} == {"free", "pro"}
    assert "stripe_customer_id" not in json.dumps(body)

    monkeypatch.delenv("STRIPE_PRICE_PRO")
    assert billing_routes.plans(_req(None)).body["billing_enabled"] is False

    status = billing_routes.status(_req(user)).body
    assert status["plan"] == "free" and status["payments"] == []


# ------------------------------------------------------------ webhook route
def test_webhook_requires_secret(monkeypatch, user):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    with pytest.raises(HTTPError) as info:
        billing_routes.webhook(_req(None, raw=b"{}"))
    assert info.value.status == 503


def test_webhook_bad_signature_is_400(stripe_env, user):
    req = _req(None, raw=b"{}", headers={"Stripe-Signature": _sign(b"{}", int(time.time()), "whsec_bad")})
    with pytest.raises(HTTPError) as info:
        billing_routes.webhook(req)
    assert info.value.status == 400


def test_webhook_valid_event_applies_and_is_idempotent(stripe_env, store, user):
    store.create_payment(user["id"], "cs_1", "pro", 1900, "usd")
    payload = json.dumps(_checkout_completed("evt_wh_1", user["id"])).encode()
    req = _req(None, raw=payload, headers={"Stripe-Signature": _sign(payload, int(time.time()))})

    first = billing_routes.webhook(req)
    assert first.status == 200 and first.body == {"received": True}
    assert store.get_user(user["id"])["plan"] == "pro"

    second = billing_routes.webhook(req)
    assert second.status == 200 and second.body == {"received": True}
    assert store.get_user(user["id"])["plan"] == "pro"


def test_webhook_non_json_with_valid_signature_is_400(stripe_env):
    raw = b"not json"
    req = _req(None, raw=raw, headers={"Stripe-Signature": _sign(raw, int(time.time()))})
    with pytest.raises(HTTPError) as info:
        billing_routes.webhook(req)
    assert info.value.status == 400


# ------------------------------------------------------------ event handling
def test_checkout_completed_upgrades_and_marks_payment_paid(store, user):
    store.create_payment(user["id"], "cs_1", "pro", 1900, "usd")
    assert handle_event(_checkout_completed("evt_a", user["id"])) == "processed"

    fresh = store.get_user(user["id"])
    assert fresh["plan"] == "pro"
    assert fresh["stripe_customer_id"] == "cus_1"
    assert store.get_payment_by_session("cs_1")["status"] == "paid"


def test_duplicate_event_is_not_reapplied(store, user):
    handle_event(_checkout_completed("evt_dup", user["id"]))
    store.update_user(user["id"], plan="free")  # later state change
    assert handle_event(_checkout_completed("evt_dup", user["id"])) == "duplicate"
    assert store.get_user(user["id"])["plan"] == "free"


def test_unpaid_checkout_does_not_upgrade(store, user):
    handle_event(_checkout_completed("evt_unpaid", user["id"], paid=False))
    assert store.get_user(user["id"])["plan"] == "free"


def test_checkout_plan_must_be_known_paid_plan(store, user):
    handle_event(_checkout_completed("evt_bad_plan", user["id"], plan="enterprise"))
    handle_event(_checkout_completed("evt_free_plan", user["id"], plan="free"))
    assert store.get_user(user["id"])["plan"] == "free"


def test_checkout_matches_user_by_customer_fallback(store, user):
    store.update_user(user["id"], stripe_customer_id="cus_fallback")
    event = _event("evt_fb", "checkout.session.completed", {
        "id": "cs_fb", "payment_status": "paid", "customer": "cus_fallback",
        "metadata": {"plan": "pro"},
    })
    handle_event(event)
    assert store.get_user(user["id"])["plan"] == "pro"


def test_subscription_deleted_downgrades_by_customer(store, user):
    handle_event(_checkout_completed("evt_up", user["id"]))
    assert store.get_user(user["id"])["plan"] == "pro"
    handle_event(_event("evt_del", "customer.subscription.deleted", {"id": "sub_1", "customer": "cus_1"}))
    assert store.get_user(user["id"])["plan"] == "free"


@pytest.mark.parametrize("status", ["canceled", "unpaid", "incomplete_expired"])
def test_subscription_updated_to_ended_status_downgrades(store, user, status):
    handle_event(_checkout_completed("evt_up2", user["id"]))
    handle_event(_event(f"evt_upd_{status}", "customer.subscription.updated",
                        {"id": "sub_1", "customer": "cus_1", "status": status}))
    assert store.get_user(user["id"])["plan"] == "free"


def test_subscription_updated_active_upgrades_and_past_due_is_ignored(store, user):
    store.update_user(user["id"], stripe_customer_id="cus_1")
    handle_event(_event("evt_act", "customer.subscription.updated",
                        {"id": "sub_1", "customer": "cus_1", "status": "active"}))
    assert store.get_user(user["id"])["plan"] == "pro"
    handle_event(_event("evt_pd", "customer.subscription.updated",
                        {"id": "sub_1", "customer": "cus_1", "status": "past_due"}))
    assert store.get_user(user["id"])["plan"] == "pro"


def test_full_upgrade_downgrade_upgrade_flow(store, user):
    store.create_payment(user["id"], "cs_1", "pro", 1900, "usd")
    handle_event(_checkout_completed("evt_f1", user["id"]))
    handle_event(_event("evt_f2", "customer.subscription.deleted", {"customer": "cus_1"}))
    assert store.get_user(user["id"])["plan"] == "free"
    handle_event(_checkout_completed("evt_f3", user["id"], session_id="cs_2"))
    assert store.get_user(user["id"])["plan"] == "pro"


def test_invoice_payment_failed_logs_only(store, user):
    handle_event(_checkout_completed("evt_pre", user["id"]))
    assert handle_event(_event("evt_inv", "invoice.payment_failed", {"customer": "cus_1"})) == "processed"
    assert store.get_user(user["id"])["plan"] == "pro"


def test_unknown_event_type_ignored(store, user):
    assert handle_event(_event("evt_x", "charge.refunded", {})) == "ignored"


def test_event_without_id_rejected():
    with pytest.raises(ValueError):
        handle_event({"type": "checkout.session.completed", "data": {"object": {}}})


def test_unknown_user_does_not_crash(store):
    assert handle_event(_checkout_completed("evt_nobody", "no-such-user")) == "processed"


def test_failed_handler_releases_event_for_stripe_retry(store, user, monkeypatch):
    import app.saas.billing as billing

    real = billing._HANDLERS["checkout.session.completed"]

    def boom(_obj):
        raise RuntimeError("db down")

    monkeypatch.setitem(billing._HANDLERS, "checkout.session.completed", boom)
    with pytest.raises(RuntimeError):
        handle_event(_checkout_completed("evt_retry", user["id"]))
    monkeypatch.setitem(billing._HANDLERS, "checkout.session.completed", real)
    assert handle_event(_checkout_completed("evt_retry", user["id"])) == "processed"
    assert store.get_user(user["id"])["plan"] == "pro"
