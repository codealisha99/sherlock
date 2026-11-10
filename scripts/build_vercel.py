"""Assemble the static frontend for Vercel into dist/vercel/.

Vercel serves files before applying rewrites, so the landing page becomes
index.html and the workspace becomes app.html (cleanUrls maps /app to it).
/api/* is proxied to the Railway backend so the browser stays same-origin
(cookies + CSRF checks work unchanged). Security headers come from
app.saas.security so the Python server and Vercel can never drift.

Usage: SHERLOCK_EDGE_SECRET=<same value as on Railway> \
       python scripts/build_vercel.py https://<backend>.up.railway.app

The edge secret becomes a path prefix on the /api rewrite so the backend can
tell proxied requests (whose client IP it can trust) from direct ones.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.saas.security import STATIC_SECURITY_HEADERS  # noqa: E402

SRC = ROOT / "frontend"
OUT = ROOT / "dist" / "vercel"
RENAMES = {"landing.html": "index.html", "index.html": "app.html"}
HSTS = ("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")


def _backend_origin(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in ("", "/"):
        raise SystemExit(f"Backend must be a bare https origin, got: {raw!r}")
    return f"https://{parsed.netloc}"


def _edge_secret() -> str:
    secret = os.getenv("SHERLOCK_EDGE_SECRET", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", secret):
        raise SystemExit("SHERLOCK_EDGE_SECRET must be set (32+ URL-safe chars, same as on Railway)")
    return secret


def vercel_config(backend: str, edge_secret: str) -> dict:
    headers = [{"key": k, "value": v} for k, v in (*STATIC_SECURITY_HEADERS, HSTS)]
    return {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "trailingSlash": False,
        "rewrites": [
            {"source": "/api/:path*", "destination": f"{backend}/_edge/{edge_secret}/api/:path*"},
            # cleanUrls: files are addressed without ".html" (a ".html" destination 404s).
            {"source": "/login", "destination": "/auth"},
            {"source": "/signup", "destination": "/auth"},
            {"source": "/pricing", "destination": "/"},
        ],
        "headers": [
            {"source": "/(.*)", "headers": headers},
            # HTML must revalidate so deploys show up immediately; assets may cache briefly.
            {"source": "/(.*)\\.(js|css)", "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]},
        ],
    }


def build(backend_url: str) -> Path:
    backend = _backend_origin(backend_url)
    edge_secret = _edge_secret()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for path in SRC.iterdir():
        if path.is_file():
            shutil.copy2(path, OUT / RENAMES.get(path.name, path.name))
    (OUT / "vercel.json").write_text(json.dumps(vercel_config(backend, edge_secret), indent=2) + "\n")
    return OUT


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(build(sys.argv[1]))
