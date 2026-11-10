"""Tests for BYOK: key validation, SSRF-safe base URLs, encrypted storage, routes."""

import json
import sqlite3
from contextlib import closing

import pytest

from app.llm.gateway import LLMGateway
from app.saas import byok_routes
from app.saas.byok import (
    get_user_provider,
    gateway_for_user,
    list_user_keys,
    delete_user_key,
    save_user_key,
    validate_api_key_format,
    validate_base_url,
    validate_model,
)
from app.saas.db import SaasStore, set_store
from app.saas.http import HTTPError, Request
from app.saas.security import decrypt_secret

OPENAI_KEY = "sk-Abc123Def456Ghi789Jkl0"
ANTHROPIC_KEY = "sk-ant-api03-Abc123Def456Ghi789"
GROQ_KEY = "gsk_Abc123Def456Ghi789Jkl0"
OPENROUTER_KEY = "sk-or-v1-Abc123Def456Ghi789"
CUSTOM_KEY = "custom-key-0123456789abcdef"

PARTNER_HOST = "api.partner.example"
PARTNER_IP = "93.184.216.34"


def public_resolver(host: str) -> list:
    if host == PARTNER_HOST:
        return [PARTNER_IP]
    raise OSError(f"no such host: {host}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_SECRET_KEY", "test-secret-key-not-for-production-0123456789")
    db = SaasStore(tmp_path / "s.db")
    set_store(db)
    yield db
    set_store(None)


@pytest.fixture
def user(store):
    return store.create_user("byok@example.com", "hash")


def _req(user, body=None, path_params=None):
    raw = json.dumps(body).encode() if body is not None else b""
    return Request(
        method="POST", path="/api/keys", query={}, headers={}, body=raw,
        client_ip="127.0.0.1", user=user, path_params=path_params or {},
    )


# ------------------------------------------------------------ key format
@pytest.mark.parametrize(
    "provider, key",
    [
        ("openai", OPENAI_KEY),
        ("anthropic", ANTHROPIC_KEY),
        ("groq", GROQ_KEY),
        ("openrouter", OPENROUTER_KEY),
        ("custom", CUSTOM_KEY),
    ],
)
def test_valid_key_formats_pass(provider, key):
    validate_api_key_format(provider, key)


@pytest.mark.parametrize(
    "provider, key, message",
    [
        ("openai", "sk-short", "characters"),
        ("openai", "sk-" + "x" * 300, "characters"),
        ("openai", "sk-Abc 123Def456Ghi789Jkl", "whitespace"),
        ("openai", "sk-Abc123\nDef456Ghi789Jkl", "whitespace"),
        ("openai", "sk-Abc123\x00Def456Ghi789Jkl", "whitespace"),
        ("openai", ANTHROPIC_KEY, "another provider"),
        ("anthropic", OPENAI_KEY, "start with 'sk-ant-'"),
        ("groq", "sk-Abc123Def456Ghi789Jkl0", "start with 'gsk_'"),
        ("openrouter", OPENAI_KEY, "start with 'sk-or-'"),
        ("nope", OPENAI_KEY, "Unknown provider"),
    ],
)
def test_invalid_key_formats_rejected_without_echoing_key(provider, key, message):
    with pytest.raises(ValueError) as info:
        validate_api_key_format(provider, key)
    assert message in str(info.value)
    assert key not in str(info.value)


@pytest.mark.parametrize("name", ["gpt-4o-mini", "openai/gpt-oss-120b", "openrouter/auto", "qwen3-coder:30b"])
def test_valid_model_names(name):
    assert validate_model(name) == name


@pytest.mark.parametrize("name", ["", "../../etc", "a b", "x" * 200, "model;rm"])
def test_invalid_model_names(name):
    with pytest.raises(ValueError):
        validate_model(name)


# ------------------------------------------------------------ SSRF guard
def test_public_https_host_accepted():
    url = "https://api.partner.example/v1"
    assert validate_base_url(url, resolver=public_resolver) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://api.partner.example/v1",          # not https
        "ftp://api.partner.example/v1",
        "https://user:pass@api.partner.example/v1",  # userinfo
        "https://api.partner.example/v1#frag",      # fragment
        "https://api.partner.example/v1?key=x",     # query
        "https://api.partner.example:8443/v1",      # non-443 port
        "https://localhost/v1",
        "https://app.localhost/v1",
        "https://127.0.0.1/v1",
        "https://10.0.0.5/v1",
        "https://192.168.1.10/v1",
        "https://169.254.169.254/latest",
        "https://[::1]/v1",
        "https://[::ffff:127.0.0.1]/v1",
        "https://0x7f000001/v1",                    # non-canonical IP, resolver fails
        "https:///v1",
        "",
    ],
)
def test_ssrf_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_base_url(url, resolver=public_resolver)


def test_hostname_resolving_to_private_ip_rejected():
    with pytest.raises(ValueError, match="public address"):
        validate_base_url("https://evil.example/v1", resolver=lambda h: ["10.1.2.3"])


def test_hostname_with_any_private_answer_rejected():
    with pytest.raises(ValueError, match="public address"):
        validate_base_url(
            "https://mixed.example/v1",
            resolver=lambda h: [PARTNER_IP, "127.0.0.1"],
        )


def test_unresolvable_host_rejected():
    with pytest.raises(ValueError, match="could not be resolved"):
        validate_base_url("https://nxdomain.example/v1", resolver=public_resolver)


# ------------------------------------------------------------ storage
def test_key_is_encrypted_at_rest_and_never_listed(store, user):
    saved = save_user_key(user["id"], "openai", OPENAI_KEY)

    with closing(sqlite3.connect(store.db_path)) as conn:
        (token,) = conn.execute(
            "SELECT encrypted_key FROM api_keys WHERE user_id = ?", (user["id"],)
        ).fetchone()
    assert OPENAI_KEY not in token
    assert decrypt_secret(token) == OPENAI_KEY

    listed = list_user_keys(user["id"])
    assert OPENAI_KEY not in json.dumps(listed) and OPENAI_KEY not in json.dumps(saved)
    assert saved["key_hint"] == "…" + OPENAI_KEY[-4:]
    assert set(listed[0]) == {"provider", "key_hint", "base_url", "model", "created_at"}
    assert listed[0]["base_url"] == "https://api.openai.com/v1"
    assert listed[0]["model"] == "gpt-4o-mini"


def test_known_provider_ignores_supplied_base_url(user):
    save_user_key(user["id"], "groq", GROQ_KEY, base_url="https://attacker.example/v1")
    assert list_user_keys(user["id"])[0]["base_url"] == "https://api.groq.com/openai/v1"


def test_custom_provider_requires_valid_base_url_and_model(user):
    with pytest.raises(ValueError, match="Base URL is required"):
        save_user_key(user["id"], "custom", CUSTOM_KEY, model="m1")
    with pytest.raises(ValueError, match="localhost|public"):
        save_user_key(user["id"], "custom", CUSTOM_KEY, model="m1", base_url="https://localhost/v1")
    with pytest.raises(ValueError, match="Model is required"):
        save_user_key(user["id"], "custom", CUSTOM_KEY, base_url="https://api.partner.example/v1",
                      resolver=public_resolver)


def test_provider_built_from_stored_key(user):
    save_user_key(user["id"], "groq", GROQ_KEY)
    provider = get_user_provider(user["id"])

    assert provider.name == "groq"
    assert provider.base_url == "https://api.groq.com/openai/v1"
    assert provider.model == "openai/gpt-oss-120b"
    assert provider._api_key == GROQ_KEY  # internal decrypt path only
    assert GROQ_KEY not in repr(provider)


def test_custom_provider_uses_stored_url_and_model(user):
    save_user_key(user["id"], "custom", CUSTOM_KEY, model="my-model",
                  base_url="https://api.partner.example/v1", resolver=public_resolver)
    provider = get_user_provider(user["id"], resolver=public_resolver)
    assert provider.base_url == "https://api.partner.example/v1"
    assert provider.model == "my-model"


def test_custom_url_rechecked_at_call_time(user):
    """DNS can change after save; the call-time check must catch a private answer."""
    save_user_key(user["id"], "custom", CUSTOM_KEY, model="m1",
                  base_url="https://api.partner.example/v1", resolver=public_resolver)
    with pytest.raises(ValueError, match="public address"):
        get_user_provider(user["id"], resolver=lambda h: ["169.254.169.254"])


def test_undecryptable_key_raises_safe_error(store, user):
    store.upsert_api_key(user["id"], "openai", "not-a-fernet-token", "…1234", None, "gpt-4o-mini")
    with pytest.raises(ValueError, match="could not be decrypted"):
        get_user_provider(user["id"])


def test_keys_are_scoped_per_user(store, user):
    other = store.create_user("other@example.com", "hash")
    save_user_key(user["id"], "openai", OPENAI_KEY)
    assert get_user_provider(other["id"]) is None
    assert list_user_keys(other["id"]) == []


def test_gateway_for_user_routes_to_override(user):
    assert gateway_for_user(user["id"]) is None
    save_user_key(user["id"], "openai", OPENAI_KEY)
    gateway = gateway_for_user(user["id"])
    assert isinstance(gateway, LLMGateway)
    assert gateway._quota is None
    assert set(gateway._providers) == {"openai"}


def test_delete_key_removes_provider(user):
    save_user_key(user["id"], "openai", OPENAI_KEY)
    assert delete_user_key(user["id"], "openai") is True
    assert delete_user_key(user["id"], "openai") is False
    assert get_user_provider(user["id"]) is None


def test_save_replaces_existing_key_for_same_provider(user):
    save_user_key(user["id"], "openai", OPENAI_KEY)
    save_user_key(user["id"], "openai", "sk-Zzz999Yyy888Xxx777Www6", model="gpt-4o")
    keys = list_user_keys(user["id"])
    assert len(keys) == 1 and keys[0]["model"] == "gpt-4o"
    assert keys[0]["key_hint"] == "…Www6"


# ------------------------------------------------------------ routes
def test_list_keys_returns_keys_and_provider_catalog(user):
    save_user_key(user["id"], "groq", GROQ_KEY)
    body = byok_routes.list_keys(_req(user)).body
    assert [k["provider"] for k in body["keys"]] == ["groq"]
    assert {p["id"] for p in body["providers"]} == {"openai", "anthropic", "groq", "openrouter", "custom"}
    assert GROQ_KEY not in json.dumps(body)


def test_save_key_returns_201_with_hint_only(user):
    resp = byok_routes.save_key(_req(user, {"provider": "openai", "api_key": OPENAI_KEY}))
    assert resp.status == 201
    assert resp.body["key"]["key_hint"] == "…" + OPENAI_KEY[-4:]
    assert OPENAI_KEY not in json.dumps(resp.body)


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "openai", "api_key": "sk-short"},
        {"provider": "openai"},
        {"api_key": OPENAI_KEY},
        {"provider": "openai", "api_key": OPENAI_KEY, "model": 5},
        {"provider": "custom", "api_key": CUSTOM_KEY, "model": "m", "base_url": "http://x.example/v1"},
    ],
)
def test_save_key_bad_input_is_400(user, body):
    with pytest.raises(HTTPError) as info:
        byok_routes.save_key(_req(user, body))
    assert info.value.status == 400
    assert OPENAI_KEY not in info.value.message


def test_delete_key_route(user):
    save_user_key(user["id"], "openai", OPENAI_KEY)
    assert byok_routes.delete_key(_req(user, path_params={"provider": "openai"})).status == 200
    with pytest.raises(HTTPError) as info:
        byok_routes.delete_key(_req(user, path_params={"provider": "openai"}))
    assert info.value.status == 404


def test_routes_require_user():
    anon = _req(None, {})
    with pytest.raises(HTTPError) as info:
        byok_routes.list_keys(anon)
    assert info.value.status == 401
