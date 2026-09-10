"""Code Search and Grep Utility for Sherlock.

Provides targeted pattern searches across indexed files to assist agents
in finding routes, database integrations, external API calls, and symbol usages.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.tools.filesys import SafeFilesystem
from app.tools.repository_scanner import RepositoryIndex


class SearchMatch(BaseModel):
    file: str
    line_number: int
    matched_text: str
    context: str  # Surrounding lines snippet


class CodeSearch:
    """Performs search queries across the repository files."""

    def __init__(self, index: RepositoryIndex, fs: SafeFilesystem):
        self.index = index
        self.fs = fs

    def search_pattern(
        self,
        pattern: str,
        case_sensitive: bool = False,
        file_extensions: Optional[List[str]] = None,
        max_results: int = 50,
        context_lines: int = 3,
    ) -> List[SearchMatch]:
        """Search regex pattern across non-binary repository files."""
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            # Fallback to literal search
            regex = re.compile(re.escape(pattern), flags)

        matches: List[SearchMatch] = []

        for scanned in self.index.files:
            if scanned.is_binary or scanned.is_secret:
                continue

            if file_extensions and scanned.extension not in file_extensions:
                continue

            content, _ = self.fs.read_file(scanned.path)
            lines = content.splitlines()

            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    # Gather context window
                    start_ctx = max(0, idx - 1 - context_lines)
                    end_ctx = min(len(lines), idx + context_lines)
                    context_snippet = "\n".join(
                        f"{i+1:4d}: {lines[i]}" for i in range(start_ctx, end_ctx)
                    )

                    matches.append(
                        SearchMatch(
                            file=scanned.path,
                            line_number=idx,
                            matched_text=line.strip(),
                            context=context_snippet,
                        )
                    )

                    if len(matches) >= max_results:
                        return matches

        return matches

    def find_routes_and_endpoints(self) -> List[SearchMatch]:
        """Search for common route definitions across web frameworks."""
        pattern = r"(@\w+\.(get|post|put|delete|patch|options|route|api|websocket)|app\.use\(|router\.(get|post|put|delete|route)|path\(|re_path\()"
        return self.search_pattern(pattern, max_results=60)

    def find_database_usage(self) -> List[SearchMatch]:
        """Search for database, ORM, SQL, and query patterns."""
        pattern = r"(SELECT\s+.*FROM|INSERT\s+INTO|UPDATE\s+.*SET|Base\.metadata|ForeignKey|Column\(|session\.query|db\.session|cursor\.execute|mongoose\.model|PrismaClient|create_engine|MongoClient)"
        return self.search_pattern(pattern, max_results=60)

    def find_external_integrations(self) -> List[SearchMatch]:
        """Search for external API clients, HTTP requests, LLM SDKs, queues."""
        pattern = r"(requests\.(get|post|put)|httpx\.(get|post|put|AsyncClient)|aiohttp|fetch\(|axios\.|openai|ollama|anthropic|boto3|stripe|redis|pika|celery)"
        return self.search_pattern(pattern, max_results=60)

    def find_symbol_usages(self, symbol_name: str) -> List[SearchMatch]:
        """Search for usages or references to a given class/function name."""
        pattern = rf"\b{re.escape(symbol_name)}\b"
        return self.search_pattern(pattern, max_results=40)
