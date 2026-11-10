"""Bring-your-own-key (BYOK): provider registry, key/URL validation, encrypted storage.

Keys are encrypted at rest (app.saas.security, Fernet) and decrypted only inside
`get_user_provider` to build a provider for a single call. No public function
here returns a plaintext key.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

from app.llm.gateway import LLMGateway
from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from app.saas.db import get_store
from app.saas.security import decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)

Resolver = Callable[[str], List[str]]

MIN_KEY_LENGTH = 20
MAX_KEY_LENGTH = 256
MAX_BASE_URL_LENGTH = 512
ALLOWED_PORT = 443
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}")
_WHITESPACE_OR_CONTROL = re.compile(r"[\s\x00-\x1f\x7f]")


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    default_base_url: Optional[str]
    default_model: str
    key_prefix: Optional[str]


PROVIDERS: Dict[str, ProviderSpec] = {
    spec.id: spec
    for spec in (
        ProviderSpec("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4o-mini", "sk-"),
        ProviderSpec("anthropic", "Anthropic", "https://api.anthropic.com/v1", "claude-sonnet-5-5", "sk-ant-"),
        ProviderSpec("groq", "Groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "gsk_"),
        ProviderSpec("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openrouter/auto", "sk-or-"),
        # Custom: any OpenAI-compatible endpoint; base_url and model are required.
        ProviderSpec("custom", "Custom (OpenAI-compatible)", None, "", None),
    )
}


def _spec(provider: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError("Unknown provider.")
    return spec


def provider_catalog() -> List[Dict[str, Any]]:
    return [
        {"id": s.id, "name": s.name, "default_base_url": s.default_base_url, "default_model": s.default_model}
        for s in PROVIDERS.values()
    ]


# ---------------------------------------------------------------- validation
def validate_api_key_format(provider: str, api_key: str) -> None:
    """Raise ValueError if the key cannot be a valid key for the provider. Never echoes the key."""
    spec = _spec(provider)
    if not isinstance(api_key, str):
        raise ValueError("API key must be a string.")
    if not MIN_KEY_LENGTH <= len(api_key) <= MAX_KEY_LENGTH:
        raise ValueError(f"API key must be {MIN_KEY_LENGTH}-{MAX_KEY_LENGTH} characters.")
    if _WHITESPACE_OR_CONTROL.search(api_key):
        raise ValueError("API key must not contain whitespace or control characters.")
    if spec.key_prefix and not api_key.startswith(spec.key_prefix):
        raise ValueError(f"{spec.name} API keys start with '{spec.key_prefix}'.")
    if provider == "openai" and api_key.startswith(("sk-ant-", "sk-or-")):
        raise ValueError("This looks like a key for another provider.")


def validate_model(name: str) -> str:
    if not name:
        raise ValueError("Model is required.")
    if not isinstance(name, str) or not _MODEL_RE.fullmatch(name):
        raise ValueError("Model name is invalid.")
    return name


def _system_resolver(host: str) -> List[str]:
    infos = socket.getaddrinfo(host, ALLOWED_PORT, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _is_public_ip(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # is_global excludes private, loopback, link-local (169.254.x), CGNAT, reserved, etc.
    return ip.is_global and not ip.is_multicast


def _addresses_for(host: str, resolver: Optional[Resolver]) -> List[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass  # not an IP literal: resolve it, and check every answer below
    try:
        addresses = (resolver or _system_resolver)(host)
    except (OSError, ValueError):
        raise ValueError("Base URL host could not be resolved.") from None
    if not addresses:
        raise ValueError("Base URL host could not be resolved.")
    return addresses


def validate_base_url(url: str, resolver: Optional[Resolver] = None) -> str:
    """Return the URL if it is https on a host that only resolves to public addresses.

    SSRF guard: rejects localhost, userinfo, fragments, query strings, non-443 ports,
    and any hostname or IP literal that is private, loopback, link-local, or reserved.
    `resolver` (host -> list of IP strings) can be injected for tests.
    """
    if not isinstance(url, str) or not url or len(url) > MAX_BASE_URL_LENGTH:
        raise ValueError(f"Base URL is required and must be under {MAX_BASE_URL_LENGTH} characters.")
    if _WHITESPACE_OR_CONTROL.search(url):
        raise ValueError("Base URL must not contain whitespace or control characters.")
    if "#" in url:
        raise ValueError("Base URL must not contain a fragment.")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError("Base URL must use https.")
    if "@" in parts.netloc:
        raise ValueError("Base URL must not contain credentials.")
    if parts.query:
        raise ValueError("Base URL must not contain a query string.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("Base URL has an invalid port.") from None
    if port not in (None, ALLOWED_PORT):
        raise ValueError("Base URL port must be 443.")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise ValueError("Base URL must include a hostname.")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("Base URL must not point to localhost.")
    for address in _addresses_for(host, resolver):
        if not _is_public_ip(address):
            raise ValueError("Base URL must resolve to a public address.")
    return url


def _resolve_base_url(spec: ProviderSpec, base_url: Optional[str], resolver: Optional[Resolver]) -> str:
    """Known providers use their fixed endpoint; custom must supply a valid one."""
    if spec.default_base_url is not None:
        return spec.default_base_url
    if not base_url:
        raise ValueError("Base URL is required for custom providers.")
    return validate_base_url(base_url, resolver)


def _key_hint(api_key: str) -> str:
    return "…" + api_key[-4:]


# ------------------------------------------------------------------ storage
def save_user_key(
    user_id: str,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    resolver: Optional[Resolver] = None,
) -> Dict[str, Any]:
    """Encrypt and store the user's key for a provider (one key per provider).

    `base_url` is only honoured for the custom provider; known providers always use
    their fixed endpoint. Returns metadata only.
    """
    spec = _spec(provider)
    validate_api_key_format(provider, api_key)
    resolved_url = _resolve_base_url(spec, base_url, resolver)
    resolved_model = validate_model(model or spec.default_model)
    get_store().upsert_api_key(
        user_id, provider, encrypt_secret(api_key), _key_hint(api_key), resolved_url, resolved_model
    )
    logger.info("Stored BYOK key for provider=%s", provider)
    return next(k for k in list_user_keys(user_id) if k["provider"] == provider)


def list_user_keys(user_id: str) -> List[Dict[str, Any]]:
    """Metadata only: never includes the encrypted or plaintext key."""
    return [
        {
            "provider": row["provider"],
            "key_hint": row["key_hint"],
            "base_url": row["base_url"],
            "model": row["model"],
            "created_at": row["created_at"],
        }
        for row in get_store().list_api_keys(user_id)
    ]


def delete_user_key(user_id: str, provider: str) -> bool:
    return get_store().delete_api_key(user_id, provider)


def get_user_provider(
    user_id: str,
    provider: Optional[str] = None,
    resolver: Optional[Resolver] = None,
) -> Optional[OpenAICompatibleProvider]:
    """Build a provider from the user's stored key (latest key if provider is None).

    Decrypts at call time. Raises ValueError if the stored key cannot be decrypted
    (for example after a SHERLOCK_SECRET_KEY rotation).
    """
    row = get_store().get_api_key(user_id, provider)
    if row is None:
        return None
    spec = _spec(row["provider"])
    # Re-validate custom URLs at call time: DNS may have changed since save.
    base_url = _resolve_base_url(spec, row["base_url"], resolver)
    try:
        api_key = decrypt_secret(row["encrypted_key"])
    except ValueError:
        raise ValueError("Stored key could not be decrypted. Re-save it in settings.") from None
    return OpenAICompatibleProvider(
        name=spec.id,
        base_url=base_url,
        api_key=api_key,
        model=row["model"] or spec.default_model,
    )


def gateway_for_user(user_id: str, resolver: Optional[Resolver] = None) -> Optional[LLMGateway]:
    """Gateway routed to the user's own provider, or None if they have no key."""
    provider = get_user_provider(user_id, resolver=resolver)
    if provider is None:
        return None
    return LLMGateway(override_provider=provider)
