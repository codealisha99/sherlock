"""Static checks for the public SaaS pages (landing, auth, settings).

No browser or JS runtime is needed beyond an optional `node --check`.
These guard the CSP-driven constraints (no inline scripts, no on*= handlers),
the open-redirect guard, and the API calls each page depends on.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

PAGES = {
    "landing.html": "landing.js",
    "auth.html": "auth.js",
    "settings.html": "settings.js",
}

INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc\s*=)[^>]*>", re.IGNORECASE)
INLINE_HANDLER_RE = re.compile(r"\s(on[a-z]+)\s*=", re.IGNORECASE)


def _read(name: str) -> str:
    return (FRONTEND / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("page", list(PAGES) + ["landing.js", "auth.js", "settings.js", "site.js"])
def test_asset_exists(page):
    assert (FRONTEND / page).is_file(), f"frontend/{page} must exist"


@pytest.mark.parametrize("page,script", PAGES.items())
def test_page_structure(page, script):
    html = _read(page)
    assert "<main" in html, f"{page} needs a <main> landmark"
    assert re.search(r"<title>[^<]+</title>", html), f"{page} needs a <title>"
    assert 'name="viewport"' in html and "width=device-width" in html, (
        f"{page} needs a viewport meta tag"
    )
    assert 'href="/site.css"' in html, f"{page} must link /site.css"
    assert f'src="/{script}"' in html, f"{page} must include /{script} via src"
    assert 'src="/site.js"' in html, f"{page} must include /site.js via src"
    assert html.index('src="/site.js"') < html.index(f'src="/{script}"'), (
        "site.js must load before the page script"
    )


@pytest.mark.parametrize("page", list(PAGES))
def test_no_inline_script_blocks(page):
    assert not INLINE_SCRIPT_RE.search(_read(page)), f"{page} has an inline <script> block"


@pytest.mark.parametrize("page", list(PAGES))
def test_no_inline_event_handlers(page):
    assert not INLINE_HANDLER_RE.search(_read(page)), f"{page} has an on*= attribute"


@pytest.mark.parametrize("js", ["site.js", "landing.js", "auth.js", "settings.js"])
def test_js_uses_no_innerhtml(js):
    # Server and user text must be set with textContent / DOM builders.
    assert "innerHTML" not in _read(js), f"{js} must not use innerHTML"
    assert "insertAdjacentHTML" not in _read(js), f"{js} must not use insertAdjacentHTML"


def test_site_js_api_wrapper_contract():
    src = _read("site.js")
    assert 'credentials: "same-origin"' in src
    assert "window.Sherlock" in src
    for name in ("api", "getMe", "logout", "initNav", "redirectToCheckout", "el"):
        assert re.search(rf"\b{name}\b", src), f"site.js must expose {name}"
    assert "err.status" in src, "api() must attach .status to thrown errors"


def test_auth_js_has_open_redirect_guard():
    src = _read("auth.js")
    assert "function safeNext" in src
    assert 'startsWith("/")' in src
    assert 'startsWith("//")' in src
    assert "url.origin !== window.location.origin" in src
    assert 'DEFAULT_REDIRECT = "/app"' in src


def test_auth_js_maps_error_statuses():
    src = _read("auth.js")
    for status in ("409", "401", "423", "429"):
        assert f"case {status}:" in src, f"auth.js must handle HTTP {status}"


def test_auth_js_posts_to_auth_endpoints():
    src = _read("auth.js")
    assert "/api/auth/signup" in src
    assert "/api/auth/login" in src


def test_landing_fetches_plans():
    src = _read("landing.js")
    assert '"/api/billing/plans"' in src
    assert "/api/billing/checkout" in src or "redirectToCheckout" in src
    assert 'ANALYZE_PATH = "/app"' in src
    assert "?repo=" in src
    assert "/signup?next=" in src


def test_settings_requires_login_and_calls_api():
    src = _read("settings.js")
    assert "/login?next=" in src
    for endpoint in ("/api/auth/me", "/api/keys", "/api/billing/status"):
        assert f'"{endpoint}"' in src, f"settings.js must call {endpoint}"
    assert '"/api/keys/" + encodeURIComponent(provider)' in src
    assert "checkout" in src and "upgrade" in src


def test_settings_handles_checkout_banners():
    src = _read("settings.js")
    assert '"success"' in src and '"cancel"' in src
    assert '"pro"' in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("js", ["site.js", "landing.js", "auth.js", "settings.js"])
def test_js_syntax(js):
    result = subprocess.run(
        ["node", "--check", str(FRONTEND / js)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
