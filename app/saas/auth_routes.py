"""Auth endpoints: signup, login (with lockout), logout, me."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

from app.saas import auth
from app.saas.db import get_store, iso, utc_now
from app.saas.http import HTTPError, Request, Response
from app.saas.plans import EVENT_ANALYSIS, EVENT_CHAT, plan_limits
from app.saas.security import hash_password, validate_email, validate_password, verify_password

logger = logging.getLogger(__name__)

MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=15)
INVALID_CREDENTIALS = "Invalid email or password"


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """Hash used to spend the same time on unknown emails as on real ones."""
    return hash_password("timing-equalizer-not-a-real-password")


def _credentials(req: Request) -> tuple[Any, str]:
    body = req.json()
    password = body.get("password")
    return body.get("email"), password if isinstance(password, str) else ""


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _is_locked(user: Dict[str, Any]) -> bool:
    until = _parse_dt(user.get("locked_until"))
    return until is not None and until > utc_now()


def _user_response(status: int, user: Dict[str, Any]) -> Response:
    cookie = auth.issue_session(user["id"])
    return Response(status=status, body={"user": auth.public_user(user)}, headers=(("Set-Cookie", cookie),))


def signup(req: Request) -> Response:
    email_raw, password = _credentials(req)
    try:
        email = validate_email(email_raw)
        validate_password(password)
    except ValueError as exc:
        raise HTTPError(400, str(exc))

    store = get_store()
    if store.get_user_by_email(email) is not None:
        raise HTTPError(409, "An account with this email already exists")
    try:
        user = store.create_user(email, hash_password(password))
    except ValueError:  # lost a race with a concurrent signup
        raise HTTPError(409, "An account with this email already exists")
    logger.info("New account created: user_id=%s", user["id"])
    return _user_response(201, user)


def login(req: Request) -> Response:
    email_raw, password = _credentials(req)
    store = get_store()
    try:
        email = validate_email(email_raw)
    except ValueError:
        verify_password(password, _dummy_hash())
        raise HTTPError(401, INVALID_CREDENTIALS)

    user = store.get_user_by_email(email)
    if user is None:
        verify_password(password, _dummy_hash())
        raise HTTPError(401, INVALID_CREDENTIALS)
    if _is_locked(user):
        raise HTTPError(423, "Account temporarily locked after too many failed attempts. Try again later.")

    if not verify_password(password, user["password_hash"]):
        failures = int(user["failed_logins"]) + 1
        if failures >= MAX_FAILED_LOGINS:
            store.update_user(user["id"], failed_logins=failures, locked_until=iso(utc_now() + LOCKOUT))
            logger.warning("Account locked after %d failed logins: user_id=%s", failures, user["id"])
            raise HTTPError(423, "Account temporarily locked after too many failed attempts. Try again later.")
        store.update_user(user["id"], failed_logins=failures)
        raise HTTPError(401, INVALID_CREDENTIALS)

    store.update_user(user["id"], failed_logins=0, locked_until=None)
    return _user_response(200, user)


def logout(req: Request) -> Response:
    auth.end_session(req.headers)
    return Response(
        status=200,
        body={"status": "ok"},
        headers=(("Set-Cookie", auth.clear_cookie()),),
    )


def me(req: Request) -> Response:
    user = req.require_user()
    store = get_store()
    now = utc_now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    usage = {
        "analyses_this_month": store.count_usage_since(user["id"], EVENT_ANALYSIS, month_start),
        "chat_messages_today": store.count_usage_since(user["id"], EVENT_CHAT, day_start),
    }
    return Response(
        body={"user": user, "usage": usage, "limits": dict(plan_limits(user["plan"]))},
    )
