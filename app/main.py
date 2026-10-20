"""Sherlock: Local Codebase Intelligence System.

Main application entrypoint. Supports:

- Model-free codebase indexing, search, and inspection (no LLM required):
    python -m app.main index /path/to/repo
    python -m app.main search --repo-id <id> --query <term>
    python -m app.main inspect --repo-id <id> --file src/app.py
    python -m app.main list
    python -m app.main remove --repo-id <id>
- Existing Ollama-powered multi-agent analysis (`--analyze`).
- Existing web server (default, `--serve`).

The model-free workflow never imports, initializes, or contacts Ollama or any
other model provider. It is implemented entirely on top of the local SQLite
index store and stdlib AST parsing.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sherlock")

BANNER = r"""
==================================================================
   _____ _               _            _
  / ____| |             | |          | |
 | (___ | |__   ___ _ __| | ___   ___| | __
 \___ \| '_ \ / _ \ '__| |/ _ \ / __| |/ /
 ____) | | | |  __/ |  | | (_) | (__|   <
 |_____/|_| |\___|_|  |_|\___/ \___|_|\_\
                                               
   Local Codebase Intelligence System
==================================================================
"""


# --------------------------------------------------------------------------- #
# Model-free workflow (no LLM)
# --------------------------------------------------------------------------- #
def _get_indexer():
    """Lazy-import the indexer so the model-free path never pulls in Ollama."""
    from app.indexer import RepositoryIndexer
    return RepositoryIndexer()


def cmd_index(args) -> int:
    indexer = _get_indexer()
    repo_path = Path(args.path).resolve()
    if not repo_path.exists():
        print(f"[!] Repository path does not exist: {repo_path}", file=sys.stderr)
        return 1
    try:
        result = indexer.index_repository(
            repo_path, repo_id=args.repo_id, repo_name=args.repo_name
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[!] Indexing failed: {exc}", file=sys.stderr)
        return 1

    print(f"[+] Indexed: {result.repo_name}")
    print(f"    repo_id : {result.repo_id}")
    print(f"    status  : {result.status}")
    print(f"    files   : {result.file_count}")
    print(f"    lines   : {result.total_lines}")
    print(f"    chunks  : {result.chunk_count}")
    print(f"    defs    : {result.definition_count}")
    print(f"    imports : {result.import_count}")
    if result.skipped_files:
        print(f"    skipped : {len(result.skipped_files)} file(s)")
        for s in result.skipped_files[:10]:
            print(f"             - {s}")
    if result.errors:
        print(f"    errors  : {len(result.errors)}")
        for e in result.errors[:10]:
            print(f"             - {e}")
    if result.status != "complete":
        print(f"[!] Indexing completed with warnings/errors: {result.error}")
        return 1
    return 0


def cmd_search(args) -> int:
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        results = indexer.store.search_chunks(args.repo_id, args.query, limit=args.limit)
    finally:
        indexer.store.close()
    if not results:
        print(f"No results for '{args.query}' in repository {args.repo_id}.")
        return 0
    print(f"Search results for '{args.query}' ({len(results)}):")
    for r in results:
        print(
            f"  {r['file_path']}:{r['start_line']}-{r['end_line']} "
            f"(rank={r.get('rank', 'n/a')})"
        )
        snippet = (r.get("content") or "").strip().splitlines()[0] if r.get("content") else ""
        if snippet:
            print(f"    {snippet[:100]}")
    return 0


def cmd_inspect(args) -> int:
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        repo_row = indexer.store.get_repository(args.repo_id)
        if repo_row is None:
            print(f"[!] Repository not found: {args.repo_id}", file=sys.stderr)
            return 1
        file_row = None
        for f in indexer.store.get_files_by_repo(args.repo_id):
            if f["path"] == args.file:
                file_row = f
                break
        if file_row is None:
            print(f"[!] File not in index: {args.file}", file=sys.stderr)
            return 1
        root = Path(repo_row["path"])
    finally:
        indexer.store.close()

    from app.tools.filesys import SafeFilesystem
    fs = SafeFilesystem(root, config)
    content, truncated = fs.read_file(args.file, max_bytes=config.max_file_size_bytes)

    print(f"=== {args.file} ({file_row['language']}, {file_row['line_count']} lines) ===")
    if truncated:
        print("[... truncated ...]")
    for i, line in enumerate(content.splitlines(), start=args.start_line):
        if args.end_line is not None and i > args.end_line:
            break
        print(f"{i:6d}| {line}")
    return 0


def cmd_definitions(args) -> int:
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        if args.file:
            file_id = None
            for f in indexer.store.get_files_by_repo(args.repo_id):
                if f["path"] == args.file:
                    file_id = f["id"]
                    break
            if file_id is None:
                print(f"[!] File not in index: {args.file}", file=sys.stderr)
                return 1
            defs = indexer.store.get_definitions_by_file(file_id)
        else:
            defs = indexer.store.get_definitions_by_repo(args.repo_id)
    finally:
        indexer.store.close()
    if not defs:
        print("No definitions found.")
        return 0
    print(f"Definitions ({len(defs)}):")
    for d in defs:
        sig = f" {d['signature']}" if d.get("signature") else ""
        print(f"  {d['line']:4d}  {d['kind']:12s} {d['name']}{sig}")
    return 0


def cmd_imports(args) -> int:
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        if args.file:
            file_id = None
            for f in indexer.store.get_files_by_repo(args.repo_id):
                if f["path"] == args.file:
                    file_id = f["id"]
                    break
            if file_id is None:
                print(f"[!] File not in index: {args.file}", file=sys.stderr)
                return 1
            imps = indexer.store.get_imports_by_file(file_id)
        else:
            imps = indexer.store.get_imports_by_repo(args.repo_id)
    finally:
        indexer.store.close()
    if not imps:
        print("No imports found.")
        return 0
    print(f"Imports ({len(imps)}):")
    for imp in imps:
        print(f"  {imp['line']:4d}  {imp['kind']:10s} {imp['module']}")
    return 0


def cmd_list(args) -> int:
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        repos = indexer.store.list_repositories()
    finally:
        indexer.store.close()
    if not repos:
        print("No repositories indexed.")
        return 0
    print("Indexed repositories:")
    for r in repos:
        print(
            f"  {r['id']}  {r['status']:20s} {r['file_count']:4d} files  {r['name']}  ({r['path']})"
        )
    return 0


def cmd_status(args) -> int:
    indexer = _get_indexer()
    repo = indexer.get_repository(args.repo_id)
    if not repo:
        print(f"[!] Repository not found: {args.repo_id}", file=sys.stderr)
        return 1
    print(f"Repository: {repo['name']} ({repo['id']})")
    print(f"  path   : {repo['path']}")
    print(f"  status : {repo['status']}")
    print(f"  files  : {repo['file_count']}")
    print(f"  lines  : {repo['total_lines']}")
    if repo.get("error"):
        print(f"  error  : {repo['error']}")
    c = indexer.consistency_check(args.repo_id)
    print(f"  consistent : {c['consistent']} (chunks={c['chunk_count']})")
    return 0


def cmd_remove(args) -> int:
    indexer = _get_indexer()
    removed = indexer.remove_repository(args.repo_id)
    if removed:
        print(f"[+] Removed repository {args.repo_id} and all derived index data.")
        return 0
    print(f"[!] Repository not found: {args.repo_id}", file=sys.stderr)
    return 1


def cmd_summary(args) -> int:
    """Print a deterministic, evidence-based summary of the indexed repository."""
    indexer = _get_indexer()
    indexer.store.connect()
    try:
        repo = indexer.store.get_repository(args.repo_id)
        if not repo:
            print(f"[!] Repository not found: {args.repo_id}", file=sys.stderr)
            return 1
        files = indexer.store.get_files_by_repo(args.repo_id)
        defs = indexer.store.get_definitions_by_repo(args.repo_id)
        langs = {}
        for f in files:
            langs[f["language"]] = langs.get(f["language"], 0) + 1
    finally:
        indexer.store.close()

    print(f"Summary for {repo['name']} ({repo['id']})")
    print(f"  path       : {repo['path']}")
    print(f"  status     : {repo['status']}")
    print(f"  files      : {repo['file_count']}")
    print(f"  total lines: {repo['total_lines']}")
    print("  languages  :")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {lang}")
    print(f"  definitions: {len(defs)}")
    # Deterministic facts only - no LLM inference.
    top = sorted(defs, key=lambda d: d["line"])[:10]
    if top:
        print("  first definitions:")
        for d in top:
            print(f"    {d['line']:4d}  {d['kind']}  {d['name']}  ({d.get('file_path','?')})")
    print("  note: this summary is derived from static parsing, not an LLM.")
    return 0


# --------------------------------------------------------------------------- #
# Existing Ollama-powered paths (kept for backward compatibility)
# --------------------------------------------------------------------------- #
def _run_ollama_analyze(project_dir: str):
    from app.orchestrator.orchestrator import Orchestrator
    from app.llm.ollama_client import OllamaClient

    llm = OllamaClient()
    health = llm.check_health()
    if health["available"]:
        print(f"[*] Local Ollama: CONNECTED ({health['configured_model']})")
        if not health["model_ready"]:
            print(
                f"[!] Warning: model '{health['configured_model']}' not found. "
                f"Available: {', '.join(health['installed_models']) or 'None'}"
            )
    else:
        print(f"[!] Local Ollama: OFFLINE at {llm.host}; static fallback will be used.")

    orchestrator = Orchestrator(llm_client=llm)

    def on_progress(ctx):
        print(f"  -> [{ctx.current_agent}] Status: {ctx.status}")

    try:
        report = orchestrator.run_analysis(
            Path(project_dir).resolve(), progress_callback=on_progress
        )
    except Exception as exc:
        print(f"[!] Analysis error: {exc}", file=sys.stderr)
        return 1

    print(f"\n[+] Analysis Complete for: {report.project_name}")
    print(f"[+] Output: {orchestrator.artifact_mgr.get_analysis_dir(report.analysis_id)}")
    for page in report.pages:
        print(f"  Page {page.page_number}: {page.title}")
    return 0


def run_server(host: str, port: int, auto_open: bool = False):
    """Run the multi-threaded HTTP web server (Ollama-powered API)."""
    import webbrowser
    from http.server import ThreadingHTTPServer

    from app.api.routes import SherlockRequestHandler
    from app.orchestrator.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    SherlockRequestHandler.orchestrator = orchestrator

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SherlockRequestHandler)

    url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://localhost:{port}"
    print(f"[*] Sherlock Codebase Explorer running at: {url}")
    print("[*] Press Ctrl+C to stop the server.\n")

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Sherlock server gracefully...")
        httpd.shutdown()


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Sherlock - Local Codebase Intelligence System"
    )
    sub = parser.add_subparsers(dest="command")

    # --- Model-free subcommands ---
    p_index = sub.add_parser("index", help="Index a local repository (no LLM)")
    p_index.add_argument("path", help="Path to local repository")
    p_index.add_argument("--repo-id", help="Optional repository id (uuid generated if omitted)")
    p_index.add_argument("--repo-name", help="Optional repository name")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search indexed code chunks (no LLM)")
    p_search.add_argument("--repo-id", required=True, help="Repository id")
    p_search.add_argument("--query", required=True, help="Search term")
    p_search.add_argument("--limit", type=int, default=20, help="Max results")
    p_search.set_defaults(func=cmd_search)

    p_inspect = sub.add_parser("inspect", help="Inspect a source file with line numbers (no LLM)")
    p_inspect.add_argument("--repo-id", required=True, help="Repository id")
    p_inspect.add_argument("--file", required=True, help="File path relative to repository root")
    p_inspect.add_argument("--start-line", type=int, default=1)
    p_inspect.add_argument("--end-line", type=int, default=50)
    p_inspect.set_defaults(func=cmd_inspect)

    p_defs = sub.add_parser("definitions", help="List extracted definitions (no LLM)")
    p_defs.add_argument("--repo-id", required=True, help="Repository id")
    p_defs.add_argument("--file", help="Optional file path to scope definitions")
    p_defs.set_defaults(func=cmd_definitions)

    p_imports = sub.add_parser("imports", help="List extracted imports (no LLM)")
    p_imports.add_argument("--repo-id", required=True, help="Repository id")
    p_imports.add_argument("--file", help="Optional file path to scope imports")
    p_imports.set_defaults(func=cmd_imports)

    p_list = sub.add_parser("list", help="List indexed repositories (no LLM)")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Show repository indexing status (no LLM)")
    p_status.add_argument("--repo-id", required=True, help="Repository id")
    p_status.set_defaults(func=cmd_status)

    p_remove = sub.add_parser("remove", help="Remove a repository and its derived index data (no LLM)")
    p_remove.add_argument("--repo-id", required=True, help="Repository id")
    p_remove.set_defaults(func=cmd_remove)

    p_summary = sub.add_parser("summary", help="Deterministic repository summary (no LLM)")
    p_summary.add_argument("--repo-id", required=True, help="Repository id")
    p_summary.set_defaults(func=cmd_summary)

    # --- Existing Ollama-powered flags (backward compatible) ---
    parser.add_argument("--analyze", "-a", type=str, help="Path to local repository to analyze via Ollama multi-agent pipeline")
    parser.add_argument("--serve", "-s", action="store_true", default=True, help="Start the web server (default)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on server start")

    args = parser.parse_args()

    if args.command:
        return args.func(args)

    if args.analyze:
        return _run_ollama_analyze(args.analyze)

    run_server(args.host, args.port, auto_open=args.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
