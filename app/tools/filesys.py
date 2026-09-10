"""Safe Filesystem Operations for Sherlock.

Provides bounded, read-only inspection of target repositories to ensure:
1. No path traversal outside the project folder.
2. No reading of binary files.
3. No leakage of secrets (.env, keys, credentials).
4. Safe handling of oversized files.
"""

from pathlib import Path
import fnmatch
from typing import Optional, List, Tuple
from config import config, SherlockConfig


class SafeFilesystem:
    """Safe filesystem reader that enforces project boundaries and security filters."""

    def __init__(self, root_dir: Path | str, sherlock_cfg: Optional[SherlockConfig] = None):
        self.root_dir = Path(root_dir).resolve()
        self.cfg = sherlock_cfg or config
        if not self.root_dir.exists():
            raise ValueError(f"Target directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise ValueError(f"Target path is not a directory: {self.root_dir}")

    def resolve_path(self, relative_or_abs_path: Path | str) -> Path:
        """Resolve a path and ensure it stays within the root directory."""
        path = Path(relative_or_abs_path)
        if not path.is_absolute():
            resolved = (self.root_dir / path).resolve()
        else:
            resolved = path.resolve()

        # Enforce boundary check
        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            raise PermissionError(f"Access denied: path '{resolved}' escapes project root '{self.root_dir}'")

        return resolved

    def is_secret_file(self, path: Path) -> bool:
        """Check if file matches known secret or sensitive patterns."""
        filename = path.name
        for pattern in self.cfg.ignored_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        # Additional checks
        lower_name = filename.lower()
        if any(secret_term in lower_name for secret_term in ("secret", "credential", "private_key", ".pem")):
            return True
        return False

    def is_binary(self, path: Path) -> bool:
        """Check if file has a known binary extension or contains null bytes."""
        if path.suffix.lower() in self.cfg.binary_extensions:
            return True
        # Fast byte check for null characters in the first 1024 bytes
        try:
            with open(path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
        except Exception:
            return True
        return False

    def read_file(self, relative_or_abs_path: Path | str, max_bytes: Optional[int] = None) -> Tuple[str, bool]:
        """
        Safely read a file within the project.
        Returns (content, was_truncated).
        """
        resolved = self.resolve_path(relative_or_abs_path)

        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")

        if self.is_secret_file(resolved):
            return ("[REDACTED: Sensitive or secret file excluded by security policy]", False)

        if self.is_binary(resolved):
            return (f"[BINARY FILE: {resolved.name} ({resolved.stat().st_size} bytes)]", False)

        max_limit = max_bytes or self.cfg.max_file_size_bytes
        file_size = resolved.stat().st_size

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                if file_size > max_limit:
                    content = f.read(max_limit)
                    content += f"\n\n... [TRUNCATED: File exceeds max read limit of {max_limit} bytes] ..."
                    return (content, True)
                else:
                    return (f.read(), False)
        except Exception as e:
            return (f"[ERROR reading file: {str(e)}]", False)

    def read_file_lines(
        self, relative_or_abs_path: Path | str, start_line: int = 1, end_line: Optional[int] = None
    ) -> List[str]:
        """Safely read specific line range (1-indexed) of a file."""
        resolved = self.resolve_path(relative_or_abs_path)
        if self.is_secret_file(resolved) or self.is_binary(resolved):
            return []

        lines = []
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, start=1):
                    if idx < start_line:
                        continue
                    if end_line is not None and idx > end_line:
                        break
                    lines.append(line.rstrip("\r\n"))
        except Exception:
            pass
        return lines

    def get_relative_path(self, full_path: Path | str) -> str:
        """Return relative path string from root directory."""
        path = Path(full_path).resolve()
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)
