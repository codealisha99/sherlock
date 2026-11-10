"""Unit tests for app.saas.security: hashing, tokens, secret encryption, validation."""

from __future__ import annotations

import hashlib
import os
import stat

import pytest

from app.saas import security
from app.saas.security import (
    SECURITY_HEADERS,
    STATIC_SECURITY_HEADERS,
    decrypt_secret,
    encrypt_secret,
    hash_password,
    hash_token,
    new_token,
    validate_email,
    validate_password,
    verify_password,
)


# --------------------------------------------------------------------- hashing
def test_password_hash_roundtrip():
    stored = hash_password("correct-horse-42")
    assert stored.startswith("scrypt$16384$8$1$")
    assert verify_password("correct-horse-42", stored) is True


def test_password_wrong_password_rejected():
    stored = hash_password("correct-horse-42")
    assert verify_password("correct-horse-43", stored) is False


def test_password_salt_is_random():
    assert hash_password("abcdefghij1") != hash_password("abcdefghij1")


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "not-a-hash",
        "scrypt$16384$8$1$onlyfour$parts",
        "bcrypt$16384$8$1$AAAA$AAAA",
        "scrypt$3$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",  # n not a power of 2
        "scrypt$999999999$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",  # n too big
        "scrypt$16384$8$1$!!!notb64!!!$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "scrypt$16384$8$1$AAAAAAAAAAAAAAAAAAAAAA==$",
        "scrypt$x$8$1$AAAA$AAAA",
    ],
)
def test_verify_password_malformed_returns_false(stored):
    assert verify_password("anything-at-all", stored) is False


def test_verify_password_non_string_inputs_return_false():
    stored = hash_password("correct-horse-42")
    assert verify_password(None, stored) is False  # type: ignore[arg-type]
    assert verify_password("correct-horse-42", None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------- tokens
def test_new_token_is_random_and_long_enough():
    tokens = {new_token() for _ in range(50)}
    assert len(tokens) == 50
    assert all(len(t) >= 43 for t in tokens)  # 32 bytes urlsafe base64


def test_hash_token_is_sha256_hex():
    assert hash_token("abc") == hashlib.sha256(b"abc").hexdigest()
    assert hash_token("abc") != "abc"


# ------------------------------------------------------------ secret encryption
def test_encrypt_decrypt_roundtrip():
    secret = "sk-test-1234567890"
    cipher = encrypt_secret(secret)
    assert secret not in cipher
    assert decrypt_secret(cipher) == secret


def test_decrypt_detects_tampering():
    cipher = encrypt_secret("sk-test-1234567890")
    tampered = cipher[:-4] + ("AAAA" if cipher[-4:] != "AAAA" else "BBBB")
    with pytest.raises(ValueError):
        decrypt_secret(tampered)


def test_decrypt_rejects_garbage():
    with pytest.raises(ValueError):
        decrypt_secret("not-a-fernet-token")


def test_decrypt_fails_under_different_master_key(monkeypatch):
    cipher = encrypt_secret("sk-test-1234567890")
    monkeypatch.setenv("SHERLOCK_SECRET_KEY", "a-completely-different-master-key-value")
    with pytest.raises(ValueError):
        decrypt_secret(cipher)


def test_dev_key_generated_with_0600_when_env_unset(monkeypatch, tmp_path):
    from config import config

    monkeypatch.delenv("SHERLOCK_SECRET_KEY", raising=False)
    monkeypatch.setattr(config, "index_dir", tmp_path / "index")
    cipher = encrypt_secret("value")
    key_file = tmp_path / "index" / ".secret_key"
    assert key_file.exists()
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    assert decrypt_secret(cipher) == "value"  # same persisted key on the next call


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("User@Example.COM", "user@example.com"),
        ("  a.b+tag@sub.example.org  ", "a.b+tag@sub.example.org"),
    ],
)
def test_validate_email_normalizes(raw, expected):
    assert validate_email(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "no-at-sign", "a@b", "a..b@example.com", "@example.com", "x@exa mple.com", "a" * 65 + "@example.com", None, 42],
)
def test_validate_email_rejects(raw):
    with pytest.raises(ValueError):
        validate_email(raw)  # type: ignore[arg-type]


def test_validate_email_length_cap():
    long_local = "a" * 60
    domain = ".".join(["b" * 60] * 4) + ".com"  # > 254 chars total
    with pytest.raises(ValueError):
        validate_email(f"{long_local}@{domain}")


@pytest.mark.parametrize("pw", ["abcdefgh12", "Password123", "x" * 20 + "9"])
def test_validate_password_accepts(pw):
    validate_password(pw)  # no exception


@pytest.mark.parametrize(
    "pw",
    ["short1", "abcdefghijk", "1234567890123", "a" * 129, "", None],
)
def test_validate_password_rejects(pw):
    with pytest.raises(ValueError):
        validate_password(pw)  # type: ignore[arg-type]


def test_validate_password_message_is_user_friendly():
    with pytest.raises(ValueError, match="letter and one digit"):
        validate_password("abcdefghijklm")


# ----------------------------------------------------------------------- headers
def test_api_headers_are_locked_down():
    headers = dict(SECURITY_HEADERS)
    assert headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "Permissions-Policy" in headers


def test_static_csp_allows_only_expected_origins():
    csp = dict(STATIC_SECURITY_HEADERS)["Content-Security-Policy"]
    assert "script-src 'self' https://cdn.jsdelivr.net" in csp
    assert "form-action 'self' https://checkout.stripe.com" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-eval" not in csp


def test_module_has_no_import_time_key_read(monkeypatch):
    # Key must be read lazily so setting the env var after import takes effect.
    monkeypatch.setenv("SHERLOCK_SECRET_KEY", "lazy-read-check-key-0123456789abcdef")
    assert security._master_key() == "lazy-read-check-key-0123456789abcdef"
