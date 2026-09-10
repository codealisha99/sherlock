"""Sherlock Configuration Module

Central configuration for Ollama integration, static analysis parameters,
and artifact storage.
"""

from pathlib import Path
import os
from typing import List, Set
from pydantic import BaseModel, Field


class SherlockConfig(BaseModel):
    """Configuration settings for Sherlock."""

    # Ollama settings
    ollama_host: str = Field(
        default_factory=lambda: os.getenv("SHERLOCK_OLLAMA_HOST", "http://localhost:11434")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("SHERLOCK_MODEL", "qwen3-coder:30b-16k")
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("SHERLOCK_TIMEOUT", "120"))
    )
    max_retries: int = Field(default=2)

    # Repository scanning limits & safety
    max_file_size_bytes: int = Field(default=150_000)  # 150 KB
    max_files_to_index: int = Field(default=2500)
    max_tokens_per_prompt: int = Field(default=14_000)

    # Ignored directories
    ignored_dirs: Set[str] = Field(
        default_factory=lambda: {
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            ".idea",
            ".vscode",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".gemini",
            "target",
            "vendor",
            "bin",
            "obj",
        }
    )

    # Ignored / secret files
    ignored_patterns: List[str] = Field(
        default_factory=lambda: [
            ".DS_Store",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.crt",
            "*.cert",
            "id_rsa*",
            "id_ed25519*",
            "credentials.json",
            "secrets.json",
            "token.json",
        ]
    )

    # Binary file extensions to exclude from reading
    binary_extensions: Set[str] = Field(
        default_factory=lambda: {
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".tiff", ".webp",
            ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
            ".woff", ".woff2", ".ttf", ".eot", ".otf",
            ".mp4", ".webm", ".avi", ".mov", ".mp3", ".wav", ".ogg",
            ".sqlite", ".sqlite3", ".db", ".parquet",
        }
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    workspace_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "workspace"
    )

    class Config:
        arbitrary_types_allowed = True


# Global default configuration instance
config = SherlockConfig()
