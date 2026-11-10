"""Subscription plan limits. Single source of truth for quotas and pricing."""

from __future__ import annotations

from typing import Any, Dict

PLANS: Dict[str, Dict[str, Any]] = {
    "free": {
        "name": "Free",
        "price_cents": 0,
        "analyses_per_month": 3,
        "chat_messages_per_day": 30,
        "max_repo_mb": 50,
    },
    "pro": {
        "name": "Pro",
        "price_cents": 1900,
        "analyses_per_month": 100,
        "chat_messages_per_day": 1000,
        "max_repo_mb": 500,
    },
}

# Usage event kinds counted against plan limits.
EVENT_ANALYSIS = "analysis"
EVENT_CHAT = "chat_message"


def plan_limits(plan: str) -> Dict[str, Any]:
    return PLANS.get(plan, PLANS["free"])
