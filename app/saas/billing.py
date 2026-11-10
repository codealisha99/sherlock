"""Stripe billing: Checkout (subscriptions), webhook verification, plan changes.

Calls the Stripe REST API with httpx (form-encoded), no SDK. Plan changes happen
only from verified webhook events. Plan names from Stripe metadata are accepted
only if they are paid plans in PLANS.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from app.saas.db import get_store
from app.saas.plans import PLANS

logger = logging.getLogger(__name__)

STRIPE_API_BASE = "https://api.stripe.com/v1"
STRIPE_TIMEOUT_SECONDS = 15.0
DEFAULT_TOLERANCE_SECONDS = 300
FREE_PLAN = "free"
PRO_PLAN = "pro"
PRICE_ENV_BY_PLAN: Dict[str, str] = {PRO_PLAN: "STRIPE_PRICE_PRO"}
_ACTIVE_STATUSES = frozenset({"active", "trialing"})
_ENDED_STATUSES = frozenset({"canceled", "unpaid", "incomplete_expired"})


class StripeError(Exception):
    """Stripe call failed. The message is safe to show to users."""


def billing_enabled() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY") and os.getenv("STRIPE_PRICE_PRO"))


def public_url() -> str:
    return os.getenv("SHERLOCK_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(timeout=STRIPE_TIMEOUT_SECONDS)


# ----------------------------------------------------------------- checkout
def _stripe_post(path: str, fields: List[Tuple[str, str]]) -> Dict[str, Any]:
    secret = os.getenv("STRIPE_SECRET_KEY") or ""
    try:
        with _client() as client:
            response = client.post(
                f"{STRIPE_API_BASE}{path}",
                data=dict(fields),  # keys are unique, so a dict encodes the same form
                auth=(secret, ""),  # secret key as basic-auth user; never in URL or logs
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
    except httpx.HTTPError:
        logger.warning("Stripe request to %s failed on the network", path)
        raise StripeError("Could not reach the payment provider. Please try again.") from None
    if response.status_code != 200:
        logger.warning("Stripe %s returned HTTP %s", path, response.status_code)
        raise StripeError("The payment provider rejected the request. Please try again later.")
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise StripeError("The payment provider returned an unexpected response.")
    return data


def create_checkout_session(user: Dict[str, Any], plan: str) -> str:
    """Create a Stripe subscription Checkout session and return its hosted URL."""
    price_env = PRICE_ENV_BY_PLAN.get(plan)
    if price_env is None or plan not in PLANS:
        raise ValueError("Unknown plan.")
    price = os.getenv(price_env)
    if not billing_enabled() or not price:
        raise ValueError("Billing is not configured.")

    base = public_url()
    fields: List[Tuple[str, str]] = [
        ("mode", "subscription"),
        ("line_items[0][price]", price),
        ("line_items[0][quantity]", "1"),
        ("success_url", f"{base}/settings?checkout=success&session_id={{CHECKOUT_SESSION_ID}}"),
        ("cancel_url", f"{base}/settings?checkout=cancel"),
        ("client_reference_id", user["id"]),
        ("metadata[user_id]", user["id"]),
        ("metadata[plan]", plan),
    ]
    if user.get("stripe_customer_id"):
        fields.append(("customer", user["stripe_customer_id"]))
    else:
        fields.append(("customer_email", user["email"]))

    session = _stripe_post("/checkout/sessions", fields)
    url, session_id = session.get("url"), session.get("id")
    if not isinstance(url, str) or not isinstance(session_id, str):
        raise StripeError("The payment provider returned an unexpected response.")
    get_store().create_payment(user["id"], session_id, plan, PLANS[plan]["price_cents"], "usd")
    return url


# ----------------------------------------------------------------- webhooks
def verify_webhook_signature(
    payload: bytes,
    sig_header: str,
    secret: str,
    tolerance: int = DEFAULT_TOLERANCE_SECONDS,
    now: Optional[float] = None,
) -> None:
    """Verify a Stripe-Signature header (v1 HMAC-SHA256 over f"{t}.{payload}").

    Accepts when any v1 signature matches. Rejects timestamps more than `tolerance`
    seconds away from `now`. Raises ValueError on any failure.
    """
    if not secret:
        raise ValueError("Webhook secret is not configured.")
    if not sig_header:
        raise ValueError("Missing Stripe-Signature header.")

    timestamp: Optional[int] = None
    signatures: List[str] = []
    for item in sig_header.split(","):
        key, _, value = item.strip().partition("=")
        if key == "t":
            try:
                timestamp = int(value)
            except ValueError:
                raise ValueError("Malformed Stripe-Signature header.") from None
        elif key == "v1":
            signatures.append(value)
    if timestamp is None or not signatures:
        raise ValueError("Malformed Stripe-Signature header.")

    current = time.time() if now is None else now
    if abs(current - timestamp) > tolerance:
        raise ValueError("Webhook timestamp is outside the tolerance window.")

    signed = f"{timestamp}.".encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest().encode("ascii")
    matched = False
    for sig in signatures:
        if hmac.compare_digest(expected, sig.encode("utf-8", errors="replace")):
            matched = True
    if not matched:
        raise ValueError("No matching webhook signature.")


def _event_object(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("data")
    obj = data.get("object") if isinstance(data, dict) else None
    return obj if isinstance(obj, dict) else {}


def _metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _find_user(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Match by client_reference_id / metadata.user_id first, then by Stripe customer id."""
    store = get_store()
    user_id = obj.get("client_reference_id") or _metadata(obj).get("user_id")
    if isinstance(user_id, str) and user_id:
        user = store.get_user(user_id)
        if user is not None:
            return user
    customer = obj.get("customer")
    if isinstance(customer, str) and customer:
        return store.get_user_by_customer(customer)
    return None


def _on_checkout_completed(obj: Dict[str, Any]) -> None:
    store = get_store()
    paid = obj.get("payment_status") == "paid" or obj.get("status") == "complete"
    user = _find_user(obj)
    if user is None:
        logger.warning("checkout.session.completed: no matching user")
        return
    if not paid:
        return
    plan = _metadata(obj).get("plan")
    if plan not in PRICE_ENV_BY_PLAN:
        logger.warning("checkout.session.completed: ignoring unknown plan")
        return
    fields: Dict[str, Any] = {"plan": plan}
    customer = obj.get("customer")
    if isinstance(customer, str) and customer:
        fields["stripe_customer_id"] = customer
    store.update_user(user["id"], **fields)
    session_id = obj.get("id")
    if isinstance(session_id, str):
        store.update_payment_status(session_id, "paid")


def _set_plan_from_subscription(obj: Dict[str, Any], plan: str) -> None:
    user = _find_user(obj)
    if user is None:
        logger.warning("subscription event: no matching user")
        return
    get_store().update_user(user["id"], plan=plan)


def _on_subscription_deleted(obj: Dict[str, Any]) -> None:
    _set_plan_from_subscription(obj, FREE_PLAN)


def _on_subscription_updated(obj: Dict[str, Any]) -> None:
    status = obj.get("status")
    if status in _ACTIVE_STATUSES:
        _set_plan_from_subscription(obj, PRO_PLAN)
    elif status in _ENDED_STATUSES:
        _set_plan_from_subscription(obj, FREE_PLAN)
    # past_due / incomplete: leave the plan alone; Stripe keeps retrying.


def _on_invoice_payment_failed(obj: Dict[str, Any]) -> None:
    logger.warning("Stripe invoice payment failed; plan unchanged, Stripe will retry")


_HANDLERS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "checkout.session.completed": _on_checkout_completed,
    "customer.subscription.deleted": _on_subscription_deleted,
    "customer.subscription.updated": _on_subscription_updated,
    "invoice.payment_failed": _on_invoice_payment_failed,
}


def handle_event(event: Dict[str, Any]) -> str:
    """Apply a verified Stripe event. Returns "processed", "duplicate", or "ignored".

    Idempotent per event id. The id is claimed before the handler runs (so
    concurrent deliveries can't double-apply) and released if the handler
    fails, so Stripe's retry is processed instead of dropped as a duplicate.
    """
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Event has no id.")
    handler = _HANDLERS.get(str(event.get("type")))
    if handler is None:
        logger.info("Ignoring Stripe event type=%s", event.get("type"))
        return "ignored"
    store = get_store()
    if not store.mark_webhook_processed(event_id):
        return "duplicate"
    try:
        handler(_event_object(event))
    except Exception:
        store.unmark_webhook(event_id)
        raise
    return "processed"
