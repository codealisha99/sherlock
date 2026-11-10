"""The Vercel bundle: landing at /, workspace at /app, API proxied, headers in sync."""

import json

import pytest

from scripts import build_vercel


def test_bundle_layout_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(build_vercel, "OUT", tmp_path / "out")
    out = build_vercel.build("https://api.example.up.railway.app")
    index = (out / "index.html").read_text()
    assert "landing.js" in index and "app.js" in (out / "app.html").read_text()
    cfg = json.loads((out / "vercel.json").read_text())
    rewrites = {r["source"]: r["destination"] for r in cfg["rewrites"]}
    assert rewrites["/api/:path*"] == "https://api.example.up.railway.app/api/:path*"
    assert rewrites["/login"] == "/auth" and rewrites["/signup"] == "/auth"
    assert not any(d.endswith(".html") for d in rewrites.values())  # cleanUrls would 404
    keys = {h["key"] for h in cfg["headers"][0]["headers"]}
    assert {"Content-Security-Policy", "X-Frame-Options", "Strict-Transport-Security"} <= keys


@pytest.mark.parametrize("bad", ["http://x.railway.app", "https://x.railway.app/api", "x.railway.app"])
def test_rejects_non_https_or_pathful_backend(bad):
    with pytest.raises(SystemExit):
        build_vercel._backend_origin(bad)
