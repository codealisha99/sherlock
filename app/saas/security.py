"""Cryptographic primitives and security headers for the SaaS layer.

- Passwords: hashlib.scrypt with a random salt, constant-time verification.
- Session tokens: 256-bit random values; only their SHA-256 is stored.
- Secrets at rest (BYOK keys): Fernet, keyed from SHERLOCK_SECRET_KEY.
- Input validation for email and password.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import re
import secrets
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from config import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Security headers
# --------------------------------------------------------------------------- #

_PERMISSIONS_POLICY = "camera=(), microphone=(), geolocation=(), usb=(), payment=()"

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Permissions-Policy", _PERMISSIONS_POLICY),
)

STATIC_SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    (
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self' https://checkout.stripe.com",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Permissions-Policy", _PERMISSIONS_POLICY),
)

# --------------------------------------------------------------------------- #
# Password hashing (scrypt)
# --------------------------------------------------------------------------- #

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_SALT_BYTES = 16
_SCRYPT_DKLEN = 32
_SCRYPT_PREFIX = "scrypt"

# Upper bounds for parameters read from storage: keeps verify() from being
# turned into a memory/CPU exhaustion vector by a tampered hash string.
_MAX_N = 2**16
_MAX_R = 16
_MAX_P = 4
_MAX_DKLEN = 64


def _scrypt(password: bytes, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    maxmem = 128 * r * (n + p) + 1024 * 1024
    return hashlib.scrypt(password, salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=maxmem)


def hash_password(password: str) -> str:
    """Return "scrypt$n$r$p$salt_b64$hash_b64"."""
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    digest = _scrypt(password.encode("utf-8"), salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN)
    return "$".join(
        [
            _SCRYPT_PREFIX,
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def _parse_scrypt(stored: str) -> tuple[int, int, int, bytes, bytes]:
    """Parse a stored hash. Raises ValueError on anything malformed or out of bounds."""
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != _SCRYPT_PREFIX:
        raise ValueError("unrecognised hash format")
    n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    if not (2 <= n <= _MAX_N and n & (n - 1) == 0 and 1 <= r <= _MAX_R and 1 <= p <= _MAX_P):
        raise ValueError("scrypt parameters out of bounds")
    salt = base64.b64decode(parts[4], validate=True)
    digest = base64.b64decode(parts[5], validate=True)
    if not salt or not 16 <= len(digest) <= _MAX_DKLEN:
        raise ValueError("invalid salt or digest length")
    return n, r, p, salt, digest


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Returns False for any malformed input instead of raising."""
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    try:
        n, r, p, salt, expected = _parse_scrypt(stored)
        actual = _scrypt(password.encode("utf-8"), salt, n, r, p, len(expected))
    except (ValueError, TypeError, binascii.Error):
        return False
    return hmac.compare_digest(actual, expected)


# --------------------------------------------------------------------------- #
# Session tokens
# --------------------------------------------------------------------------- #


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Secret encryption (Fernet)
# --------------------------------------------------------------------------- #

_DEV_KEY_FILENAME = ".secret_key"
_FERNET_INFO = b"sherlock-fernet-v1"


def _dev_key_path() -> Path:
    return config.base_dir / "index" / _DEV_KEY_FILENAME


def _read_or_create_dev_key() -> str:
    """Local-dev fallback: random key persisted with 0600 permissions."""
    path = _dev_key_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key)
    os.chmod(path, 0o600)
    logger.warning(
        "SHERLOCK_SECRET_KEY is not set; generated a development key at %s. "
        "Set SHERLOCK_SECRET_KEY in production.",
        path,
    )
    return key


def _master_key() -> str:
    env = os.getenv("SHERLOCK_SECRET_KEY", "").strip()
    return env if env else _read_or_create_dev_key()


def _fernet() -> Fernet:
    # Read lazily (not at import) so tests and runtime can set the env var.
    derived = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_FERNET_INFO).derive(
        _master_key().encode("utf-8")
    )
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_secret(plain: str) -> str:
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    """Raises ValueError if the token is tampered with or was made with another key."""
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid encrypted secret") from exc


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

_EMAIL_MAX = 254
_EMAIL_LOCAL_MAX = 64
_EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}")

_PASSWORD_MIN = 10
_PASSWORD_MAX = 128


def validate_email(value: str) -> str:
    """Return the normalized (stripped, lowercased) email or raise ValueError."""
    if not isinstance(value, str):
        raise ValueError("Enter a valid email address.")
    email = value.strip().lower()
    local = email.partition("@")[0]
    if (
        not email
        or len(email) > _EMAIL_MAX
        or len(local) > _EMAIL_LOCAL_MAX
        or ".." in email
        or not _EMAIL_RE.fullmatch(email)
    ):
        raise ValueError("Enter a valid email address.")
    return email


def validate_password(value: str) -> None:
    """Raise ValueError with a user-facing message if the password is too weak."""
    if not isinstance(value, str) or not _PASSWORD_MIN <= len(value) <= _PASSWORD_MAX:
        raise ValueError(f"Password must be {_PASSWORD_MIN}-{_PASSWORD_MAX} characters long.")
    if not (any(c.isalpha() for c in value) and any(c.isdigit() for c in value)):
        raise ValueError("Password must include at least one letter and one digit.")
