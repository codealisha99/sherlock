"""Platform-hosted default model and the gateway resolution order for SaaS AI features.

Resolution order (`resolve_gateway`): the caller's BYOK key, then the platform model
(`SHERLOCK_LLM_*`, or the legacy `GROQ_API_KEY`), then the legacy handler gateway.
The legacy Ollama gateway only counts when Ollama was explicitly enabled, because
otherwise it is unreachable for SaaS users.

Secrets are read from the environment once and never logged or repr'd.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from app.llm.gateway import LLMGateway
from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from app.saas.byok import PROVIDERS, ProviderSpec, gateway_for_user, validate_api_key_format, validate_base_url

logger = logging.getLogger(__name__)

ENV_PROVIDER = "SHERLOCK_LLM_PROVIDER"
ENV_API_KEY = "SHERLOCK_LLM_API_KEY"
ENV_MODEL = "SHERLOCK_LLM_MODEL"
ENV_BASE_URL = "SHERLOCK_LLM_BASE_URL"
LEGACY_GROQ_KEY = "GROQ_API_KEY"
LEGACY_GROQ_MODEL = "GROQ_MODEL"
SOURCE_BYOK = "byok"
SOURCE_PLATFORM = "platform"
SOURCE_LEGACY = "legacy"

_lock = threading.Lock()
_built = False
_cached: Optional["PlatformModel"] = None


@dataclass(frozen=True)
class PlatformModel:
    gateway: LLMGateway
    provider: str
    model: str
    api_key: str = field(repr=False)


def _from_env() -> Optional[Tuple[ProviderSpec, str, str, str]]:
    """Return (spec, api_key, model, base_url) from the environment, or None if unset/invalid."""
    provider_id = os.getenv(ENV_PROVIDER, "").strip().lower()
    api_key = os.getenv(ENV_API_KEY, "").strip()
    if not provider_id and not api_key:
        provider_id, api_key = "groq", os.getenv(LEGACY_GROQ_KEY, "").strip()
        model_env = os.getenv(LEGACY_GROQ_MODEL, "").strip()
    else:
        model_env = os.getenv(ENV_MODEL, "").strip()
    if not api_key:
        return None
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        logger.error("Platform model disabled: %s must be one of %s", ENV_PROVIDER, sorted(PROVIDERS))
        return None
    base_url = spec.default_base_url or os.getenv(ENV_BASE_URL, "").strip()
    model = model_env or spec.default_model
    try:
        validate_api_key_format(spec.id, api_key)
        if spec.default_base_url is None:
            base_url = validate_base_url(base_url)
        if not model:
            raise ValueError(f"{ENV_MODEL} is required for the custom provider.")
    except ValueError as exc:
        logger.error("Platform model disabled: %s", exc)
        return None
    return spec, api_key, model, base_url


def _build() -> Optional[PlatformModel]:
    settings = _from_env()
    if settings is None:
        return None
    spec, api_key, model, base_url = settings
    provider = OpenAICompatibleProvider(name=spec.id, base_url=base_url, api_key=api_key, model=model)
    logger.info("Platform AI model enabled: provider=%s model=%s", spec.id, model)
    return PlatformModel(gateway=LLMGateway(override_provider=provider), provider=spec.id, model=model, api_key=api_key)


def _platform() -> Optional[PlatformModel]:
    global _built, _cached
    if not _built:
        with _lock:
            if not _built:
                _cached = _build()
                _built = True
    return _cached


def platform_gateway() -> Optional[LLMGateway]:
    """Gateway for the platform model, built once. None when no platform model is configured."""
    model = _platform()
    return model.gateway if model else None


def platform_model_name() -> Optional[str]:
    model = _platform()
    return model.model if model else None


def reset_platform_cache() -> None:
    """Forget the cached platform model so the next call re-reads the environment (tests)."""
    global _built, _cached
    with _lock:
        _built = False
        _cached = None


def _is_ollama_only(gateway: Any) -> bool:
    # LLMGateway keeps its default provider name privately; "ollama" means no hosted model.
    return getattr(gateway, "_default_provider", None) == "ollama"


def resolve_gateway(
    user: Optional[Dict[str, Any]],
    legacy_default: Any = None,
    allow_ollama: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Pick the model for a request. Returns (gateway, source) or (None, None).

    source is "byok", "platform", or "legacy". A BYOK key that cannot be decrypted
    falls through to the next source, as before.
    """
    if user is not None:
        try:
            gateway = gateway_for_user(user["id"])
        except ValueError as exc:
            logger.warning("BYOK key unusable for user; falling back: %s", exc)
            gateway = None
        if gateway is not None:
            return gateway, SOURCE_BYOK
    platform = platform_gateway()
    if platform is not None:
        return platform, SOURCE_PLATFORM
    if legacy_default is not None and (allow_ollama or not _is_ollama_only(legacy_default)):
        return legacy_default, SOURCE_LEGACY
    return None, None
