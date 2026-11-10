"""Billing HTTP handlers: plan list, status, checkout, Stripe webhook."""

from __future__ import annotations

import json
import logging
import os

from app.saas.billing import (
    PRICE_ENV_BY_PLAN,
    StripeError,
    billing_enabled,
    create_checkout_session,
    handle_event,
    verify_webhook_signature,
)
from app.saas.db import get_store
from app.saas.http import HTTPError, Request, Response
from app.saas.plans import PLANS

logger = logging.getLogger(__name__)


def plans(req: Request) -> Response:
    catalog = [
        {
            "id": plan_id,
            "name": p["name"],
            "price_cents": p["price_cents"],
            "analyses_per_month": p["analyses_per_month"],
            "chat_messages_per_day": p["chat_messages_per_day"],
            "max_repo_mb": p["max_repo_mb"],
        }
        for plan_id, p in PLANS.items()
    ]
    return Response(200, {"plans": catalog, "billing_enabled": billing_enabled()})


def status(req: Request) -> Response:
    user = req.require_user()
    store = get_store()
    fresh = store.get_user(user["id"])
    if fresh is None:
        raise HTTPError(401, "Authentication required")
    return Response(
        200,
        {
            "plan": fresh["plan"],
            "billing_enabled": billing_enabled(),
            "payments": store.list_payments(user["id"]),
        },
    )


def checkout(req: Request) -> Response:
    user = req.require_user()
    if not billing_enabled():
        raise HTTPError(503, "Billing is not configured")
    plan = req.json().get("plan")
    if not isinstance(plan, str) or plan not in PRICE_ENV_BY_PLAN:
        raise HTTPError(400, "Unknown plan.")
    fresh = get_store().get_user(user["id"])
    if fresh is None:
        raise HTTPError(401, "Authentication required")
    if fresh["plan"] == plan:
        raise HTTPError(400, "You are already on this plan.")
    try:
        url = create_checkout_session(fresh, plan)
    except StripeError as exc:
        raise HTTPError(502, str(exc)) from None
    return Response(200, {"checkout_url": url})


def webhook(req: Request) -> Response:
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPError(503, "Billing webhooks are not configured")
    try:
        verify_webhook_signature(req.body, req.headers.get("Stripe-Signature", ""), secret)
    except ValueError as exc:
        logger.warning("Rejected Stripe webhook: %s", exc)
        raise HTTPError(400, "Invalid signature") from None
    try:
        event = json.loads(req.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPError(400, "Invalid payload") from None
    if not isinstance(event, dict):
        raise HTTPError(400, "Invalid payload")
    try:
        outcome = handle_event(event)
    except ValueError:
        raise HTTPError(400, "Invalid event") from None
    logger.info("Stripe webhook %s: %s", event.get("type"), outcome)
    return Response(200, {"received": True})
