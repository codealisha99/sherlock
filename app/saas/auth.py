"""Session authentication: cookie or Bearer token -> user.

Tokens are random 256-bit values handed to the client; only their SHA-256 is
stored. Resolution never raises on malformed input: it returns None.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from http.cookies import CookieError, SimpleCookie
from typing import Any, Dict, Mapping, Optional

from app.saas.db import get_store
from app.saas.security import hash_token, new_token

logger = logging.getLogger(__name__)

SESSION_COOKIE = "sherlock_session"
SESSION_TTL = timedelta(days=7)
MAX_TOKEN_CHARS = 256  # reject absurdly long inputs before hashing


def _header(headers: Any, name: str) -> str:
    """Case-insensitive header lookup for email.message.Message or plain dicts."""
    try:
        value = headers.get(name) or headers.get(name.lower())
    except AttributeError:
        return ""
    return value if isinstance(value, str) else ""


def _cookie_secure() -> bool:
    return os.getenv("SHERLOCK_COOKIE_SECURE", "false").lower() == "true"


def _cookie_token(headers: Any) -> Optional[str]:
    raw = _header(headers, "Cookie")
    if not raw:
        return None
    try:
        jar = SimpleCookie()
        jar.load(raw)
    except CookieError:
        return None
    morsel = jar.get(SESSION_COOKIE)
    return morsel.value if morsel else None


def _bearer_token(headers: Any) -> Optional[str]:
    auth = _header(headers, "Authorization").strip()
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _session_token(headers: Any) -> Optional[str]:
    """Bearer header wins over the cookie; both are length-capped."""
    token = _bearer_token(headers) or _cookie_token(headers)
    if not token or len(token) > MAX_TOKEN_CHARS:
        return None
    return token


def public_user(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Only fields safe to return to clients (never password hash or customer id)."""
    return {
        "id": row["id"],
        "email": row["email"],
        "plan": row["plan"],
        "created_at": row["created_at"],
    }


def resolve_user(headers: Any) -> Optional[Dict[str, Any]]:
    """Return the public user for the session in cookie or Bearer header, else None."""
    try:
        token = _session_token(headers)
        if token is None:
            return None
        row = get_store().get_session_user(hash_token(token))
        return public_user(row) if row else None
    except Exception:  # malformed input must never break the request pipeline
        logger.debug("Session resolution failed", exc_info=True)
        return None


def issue_session(user_id: str) -> str:
    """Create a session and return the Set-Cookie header value for it."""
    token = new_token()
    get_store().create_session(hash_token(token), user_id, SESSION_TTL)
    return _cookie_header(token, int(SESSION_TTL.total_seconds()))


def clear_cookie() -> str:
    return _cookie_header("", 0)


def end_session(headers: Any) -> None:
    """Revoke the session presented in the request, if any."""
    token = _session_token(headers)
    if token is not None:
        get_store().delete_session(hash_token(token))


def _cookie_header(value: str, max_age: int) -> str:
    parts = [f"{SESSION_COOKIE}={value}", "Path=/", "HttpOnly", "SameSite=Lax", f"Max-Age={max_age}"]
    if _cookie_secure():
        parts.append("Secure")
    return "; ".join(parts)
