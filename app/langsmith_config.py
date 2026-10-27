"""LangSmith configuration for Sherlock RAG observability.

LangSmith tracing is opt-in and disabled by default. Sherlock works entirely
without LangSmith - no API key, account, or network connection is required
for local RAG operation.

Configuration is read from environment variables (official LangSmith SDK
conventions):

    LANGSMITH_TRACING="true"      # Enable tracing (default: "false")
    LANGSMITH_API_KEY="..."       # LangSmith API key (optional, tracing works without for local-only)
    LANGSMITH_PROJECT="sherlock"  # Project name for traces (default: "sherlock")
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # LangSmith endpoint

Sherlock-specific settings:

    SHERLOCK_LANGSMITH_CAPTURE_IO="false"  # Whether to capture question/answer text (default: "false")
    SHERLOCK_LANGSMITH_REDACT_PATHS="true"  # Redact file paths in trace metadata (default: "true")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# LangSmith environment variable names (official SDK conventions)
ENV_TRACING = "LANGSMITH_TRACING"
ENV_API_KEY = "LANGSMITH_API_KEY"
ENV_PROJECT = "LANGSMITH_PROJECT"
ENV_ENDPOINT = "LANGSMITH_ENDPOINT"

# Sherlock-specific environment variables
ENV_CAPTURE_IO = "SHERLOCK_LANGSMITH_CAPTURE_IO"
ENV_REDACT_PATHS = "SHERLOCK_LANGSMITH_REDACT_PATHS"


@dataclass
class LangSmithConfig:
    """LangSmith observability configuration.

    All fields default to disabled/safe values. Tracing is opt-in only.
    """

    tracing_enabled: bool = False
    api_key: Optional[str] = None
    project: str = "sherlock"
    endpoint: Optional[str] = None
    capture_io: bool = False
    redact_paths: bool = True

    @classmethod
    def from_env(cls) -> "LangSmithConfig":
        """Build configuration from environment variables.

        Tracing is enabled only when LANGSMITH_TRACING is explicitly "true".
        """
        tracing_enabled = os.getenv(ENV_TRACING, "false").strip().lower() == "true"
        api_key = os.getenv(ENV_API_KEY) or None
        project = os.getenv(ENV_PROJECT, "sherlock") or "sherlock"
        endpoint = os.getenv(ENV_ENDPOINT) or None
        capture_io = os.getenv(ENV_CAPTURE_IO, "false").strip().lower() == "true"
        redact_paths = os.getenv(ENV_REDACT_PATHS, "true").strip().lower() == "true"

        if tracing_enabled and not api_key:
            logger.warning(
                "LangSmith tracing is enabled but LANGSMITH_API_KEY is not set. "
                "Traces will not be submitted to LangSmith."
            )

        return cls(
            tracing_enabled=tracing_enabled,
            api_key=api_key,
            project=project,
            endpoint=endpoint,
            capture_io=capture_io,
            redact_paths=redact_paths,
        )

    def is_ready(self) -> bool:
        """Return True if tracing is enabled and an API key is configured.

        This means hosted tracing can actually submit traces.
        """
        return self.tracing_enabled and self.api_key is not None

    def should_capture_io(self) -> bool:
        """Return True if question/answer text may be included in trace metadata.

        Defaults to False for privacy. Only enable with explicit opt-in.
        """
        return self.tracing_enabled and self.capture_io


def get_langsmith_config() -> LangSmithConfig:
    """Get the current LangSmith configuration from environment."""
    return LangSmithConfig.from_env()


def safe_repo_id(repo_id: str, redact_paths: bool = True) -> str:
    """Return a pseudonymous repository identifier for trace metadata.

    The repository id is used as-is but truncated to avoid leaking long paths.
    No file contents or full filesystem paths are included.
    """
    if not repo_id:
        return "unknown-repo"
    # Truncate to first 12 chars for pseudonymization
    return repo_id[:12]


def sanitize_for_trace(text: Optional[str], redact_paths: bool = True) -> Optional[str]:
    """Sanitize text for inclusion in LangSmith trace metadata.

    Removes potential secrets and file paths. Intended for short diagnostic
    strings only - never for full source code or retrieved chunks.
    """
    if text is None:
        return None
    result = text
    if redact_paths:
        # Redact absolute paths and common secret patterns
        import re

        # Redact absolute paths like /Users/... or C:\...
        result = re.sub(r"/[^\s]+", "[PATH]", result)
        result = re.sub(r"[A-Za-z]:\\[^\s]+", "[PATH]", result)
        # Redact potential API keys/tokens
        result = re.sub(r"(?i)(api[_-]?key|token|secret|password)['\"]?\s*[:=]\s*['\"]?[^\s]+", r"\1=[REDACTED]", result)
    return result[:200] if result else None  # Truncate to 200 chars
