"""Sherlock: Local Multi-Agent Codebase-Understanding System.

Main application entrypoint. Supports:
- Starting local web server and Codebase Explorer UI (`python -m app.main`)
- Direct CLI analysis of any local project (`python -m app.main --analyze /path/to/repo`)
"""

import argparse
from http.server import ThreadingHTTPServer
import logging
import os
import sys
import webbrowser
from pathlib import Path

from config import config
from app.orchestrator.orchestrator import Orchestrator
from app.api.routes import SherlockRequestHandler
from app.llm.ollama_client import OllamaClient

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
 |_____/|_| |_|\___|_|  |_|\___/ \___|_|\_\
                                           
   Local Multi-Agent Codebase-Understanding System
==================================================================
"""


def check_ollama_status(llm: OllamaClient):
    """Check and display Ollama service status."""
    health = llm.check_health()
    if health["available"]:
        print(f"[*] Local Ollama: CONNECTED ({health['configured_model']})")
        if not health["model_ready"]:
            print(f"[!] Warning: Configured model '{health['configured_model']}' was not found in 'ollama list'.")
            print(f"    Available models: {', '.join(health['installed_models']) or 'None'}")
            print(f"    Run: ollama pull {health['configured_model']}")
    else:
        print(f"[!] Local Ollama: OFFLINE or UNREACHABLE at {llm.host}")
        print(f"    Notice: Sherlock static AST synthesis fallback is active.")
        print(f"    To enable full LLM reasoning: ensure 'ollama serve' is running.")


def run_cli_analysis(project_dir: str, orchestrator: Orchestrator):
    """Run analysis synchronously on the CLI and output summary."""
    target_path = Path(project_dir).resolve()
    print(f"[*] Starting Sherlock multi-agent analysis on: {target_path}")

    def on_progress(ctx):
        print(f"  -> [{ctx.current_agent}] Status: {ctx.status}")

    try:
        report = orchestrator.run_analysis(target_path, progress_callback=on_progress)
        print("\n" + "=" * 60)
        print(f"[+] Analysis Complete for: {report.project_name}")
        print(f"[+] Output Directory: {orchestrator.artifact_mgr.get_analysis_dir(report.analysis_id)}")
        print("=" * 60)
        for page in report.pages:
            print(f"\nPage {page.page_number}: {page.title}")
            print(f"  {page.subtitle}")
        print("\n[+] View complete report in workspace/ or launch the web UI with 'python -m app.main'.")
    except Exception as e:
        print(f"[!] Analysis error: {e}", file=sys.stderr)
        sys.exit(1)


def run_server(host: str, port: int, auto_open: bool = False):
    """Run the multi-threaded HTTP web server."""
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
    parser = argparse.ArgumentParser(description="Sherlock - Local Multi-Agent Codebase-Understanding System")
    parser.add_argument("--analyze", "-a", type=str, help="Path to local repository to analyze via CLI")
    parser.add_argument("--serve", "-s", action="store_true", default=True, help="Start the web server (default)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on server start")

    args = parser.parse_args()

    llm = OllamaClient()
    check_ollama_status(llm)
    print()

    if args.analyze:
        orchestrator = Orchestrator(llm_client=llm)
        run_cli_analysis(args.analyze, orchestrator)
    else:
        run_server(args.host, args.port, auto_open=args.open)


if __name__ == "__main__":
    main()
