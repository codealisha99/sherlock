"""Gateway resolution order: BYOK, then platform model, then legacy (never unreachable Ollama)."""

from __future__ import annotations

import pytest

from app.saas import llm_platform
from app.saas.llm_platform import (
    platform_gateway,
    platform_model_name,
    reset_platform_cache,
    resolve_gateway,
)

KEY = "gsk_" + "y" * 32


class _LegacyGateway:
    """Stands in for an LLMGateway whose default provider is `name`."""

    def __init__(self, name: str):
        self._default_provider = name


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SHERLOCK_LLM_PROVIDER", "SHERLOCK_LLM_API_KEY", "SHERLOCK_LLM_MODEL",
                 "SHERLOCK_LLM_BASE_URL", "GROQ_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(name, raising=False)
    reset_platform_cache()
    yield
    reset_platform_cache()


def test_no_model_and_ollama_only_legacy_is_unavailable():
    assert platform_gateway() is None
    assert resolve_gateway(None, _LegacyGateway("ollama")) == (None, None)


def test_ollama_legacy_is_used_only_when_enabled():
    gateway, source = resolve_gateway(None, _LegacyGateway("ollama"), allow_ollama=True)
    assert source == "legacy"
    assert gateway is not None


def test_hosted_legacy_gateway_is_used_when_no_platform_model():
    legacy = _LegacyGateway("groq")
    assert resolve_gateway(None, legacy) == (legacy, "legacy")


def test_platform_model_beats_legacy(monkeypatch):
    monkeypatch.setenv("SHERLOCK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("SHERLOCK_LLM_API_KEY", KEY)
    gateway, source = resolve_gateway(None, _LegacyGateway("groq"))
    assert source == "platform"
    assert gateway is platform_gateway()
    assert platform_model_name() == "openai/gpt-oss-120b"


def test_byok_beats_platform(monkeypatch):
    monkeypatch.setenv("SHERLOCK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("SHERLOCK_LLM_API_KEY", KEY)
    sentinel = object()
    monkeypatch.setattr(llm_platform, "gateway_for_user", lambda user_id: sentinel)
    assert resolve_gateway({"id": "u1"}) == (sentinel, "byok")


def test_legacy_groq_key_is_a_platform_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    assert platform_model_name() == "openai/gpt-oss-120b"


def test_unknown_provider_disables_platform(monkeypatch):
    monkeypatch.setenv("SHERLOCK_LLM_PROVIDER", "nope")
    monkeypatch.setenv("SHERLOCK_LLM_API_KEY", KEY)
    assert platform_gateway() is None


def test_platform_repr_never_shows_key(monkeypatch):
    monkeypatch.setenv("SHERLOCK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("SHERLOCK_LLM_API_KEY", KEY)
    model = llm_platform._platform()
    assert KEY not in repr(model)
