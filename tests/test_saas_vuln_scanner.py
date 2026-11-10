"""Rule coverage for app/saas/vuln_scanner.py: one true positive and one negative per rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.saas.vuln_scanner import scan_repository


def _scan(tmp_path: Path, files: dict) -> list:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return scan_repository(tmp_path, list(files))


def _rules(tmp_path: Path, rel: str, content: str) -> set:
    return {f["rule_id"] for f in _scan(tmp_path, {rel: content})}


PY_CASES = [
    ("py-eval-exec", "x = eval(s)\n", "import ast\nx = ast.literal_eval(s)\n"),
    ("py-os-shell", "import os\nos.system(cmd)\n", "import os\nos.getcwd()\n"),
    ("py-shell-true", "import subprocess\nsubprocess.run(cmd, shell=True)\n",
     "import subprocess\nsubprocess.run(['ls', p])\n"),
    ("py-subprocess-string", "import subprocess\nsubprocess.run('ls ' + p)\n",
     "import subprocess\nsubprocess.run(['ls', p], check=True)\n"),
    ("py-unsafe-deserialize", "import pickle\nobj = pickle.loads(blob)\n",
     "import json\nobj = json.loads(blob)\n"),
    ("py-yaml-unsafe", "import yaml\ncfg = yaml.load(s)\n",
     "import yaml\na = yaml.safe_load(s)\nb = yaml.load(s, Loader=yaml.SafeLoader)\n"),
    ("py-sql-string", "def q(c, x):\n    c.execute(f\"SELECT * FROM t WHERE id={x}\")\n",
     "def q(c, x):\n    c.execute('SELECT * FROM t WHERE id=?', (x,))\n"),
    ("py-weak-hash", "import hashlib\nh = hashlib.md5(b'x')\n",
     "import hashlib\nh = hashlib.sha256(b'x')\ng = hashlib.md5(b'x', usedforsecurity=False)\n"),
    ("py-insecure-random", "import random\ntoken = random.choice(alphabet)\n",
     "import secrets\ntoken = secrets.token_urlsafe(16)\n"),
    ("py-verify-false", "import requests\nrequests.get(u, verify=False)\n",
     "import requests\nrequests.get(u, verify=True)\n"),
    ("py-debug-true", "app.run(debug=True)\n", "app.run(debug=False)\n"),
    ("py-mktemp", "import tempfile\np = tempfile.mktemp()\n",
     "import tempfile\nfd, p = tempfile.mkstemp()\n"),
    ("py-jwt-unverified", "import jwt\nclaims = jwt.decode(t, options={'verify_signature': False})\n",
     "import jwt\nclaims = jwt.decode(t, key, algorithms=['HS256'])\n"),
    ("py-xml-untrusted", "import xml.etree.ElementTree as ET\nroot = ET.fromstring(s)\n",
     "import defusedxml.ElementTree as ET\nroot = ET.fromstring(s)\n"),
    ("py-bind-all", "HOST = '0.0.0.0'\n", "HOST = '127.0.0.1'\n"),
    ("py-assert-auth", "def f(user):\n    assert user.is_admin\n", "def f(x):\n    assert x == 1\n"),
    ("py-bare-except", "try:\n    x()\nexcept:\n    pass\n",
     "try:\n    x()\nexcept ValueError:\n    pass\n"),
]


@pytest.mark.parametrize("rule, positive, negative", PY_CASES, ids=[c[0] for c in PY_CASES])
def test_python_rule_true_positive_and_negative(tmp_path, rule, positive, negative):
    assert rule in _rules(tmp_path / "pos", "app/mod.py", positive)
    assert rule not in _rules(tmp_path / "neg", "app/mod.py", negative)


def test_assert_auth_skipped_in_tests(tmp_path):
    assert "py-assert-auth" not in _rules(tmp_path, "tests/test_auth.py", "def t(user):\n    assert user.is_admin\n")


SECRET_CASES = [
    ("secret-aws-key", "KEY = 'AKIAABCDEFGHIJKLMNOP'\n", "KEY = 'AKIA_not_a_key'\n"),
    ("secret-github-token", "T = 'ghp_" + "a" * 36 + "'\n", "T = 'ghp_short'\n"),
    ("secret-slack-token", "S = 'xoxb-1234567890-abcdef'\n", "S = 'xoxz-nope'\n"),
    ("secret-stripe-live", "K = 'sk_live_" + "Ab3" * 8 + "'\n", "K = 'sk_test_" + "Ab3" * 8 + "'\n"),
    ("secret-google-api", "G = 'AIza" + "B" * 35 + "'\n", "G = 'AIzaShort'\n"),
    ("secret-private-key", "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n", "-----BEGIN PUBLIC KEY-----\n"),
    ("secret-generic", 'password = "Xq9vT2mLpR8zN4kW"\n', 'password = os.environ["PW"]\n'),
]


@pytest.mark.parametrize("rule, positive, negative", SECRET_CASES, ids=[c[0] for c in SECRET_CASES])
def test_secret_rule_true_positive_and_negative(tmp_path, rule, positive, negative):
    assert rule in _rules(tmp_path / "pos", "config/settings.py", positive)
    assert rule not in _rules(tmp_path / "neg", "config/settings.py", negative)


def test_generic_secret_ignores_placeholders(tmp_path):
    assert "secret-generic" not in _rules(tmp_path, "app/x.py", 'password = "changeme123"\n')


def test_secret_values_are_masked_everywhere(tmp_path):
    secret = "AKIAABCDEFGHIJKLMNOP"
    findings = _scan(tmp_path, {"app/keys.py": f"KEY = '{secret}'\n"})
    assert findings and findings[0]["severity"] == "critical"
    blob = repr(findings[0])
    assert secret not in blob
    assert "AKIA********" in findings[0]["snippet"]


JS_CASES = [
    ("js-eval", "app.js", "eval(code);\n", "JSON.parse(text);\n"),
    ("js-dom-html", "app.js", "el.innerHTML = userInput;\n", "el.textContent = userInput;\nel.innerHTML = '';\n"),
    ("js-react-html", "app.tsx", "<div dangerouslySetInnerHTML={{__html: h}} />\n", "<div>{h}</div>\n"),
    ("js-exec-template", "server.js", "exec(`ls ${dir}`);\n", "execFile('ls', [dir]);\n"),
    ("js-local-token", "auth.js", "localStorage.setItem('authToken', t);\n", "localStorage.setItem('theme', 'dark');\n"),
    ("js-http-url", "api.ts", "fetch('http://api.example.com/x');\n",
     "fetch('https://api.example.com/x');\nconst u = 'http://localhost:3000';\n"),
]


@pytest.mark.parametrize("rule, rel, positive, negative", JS_CASES, ids=[c[0] for c in JS_CASES])
def test_js_rule_true_positive_and_negative(tmp_path, rule, rel, positive, negative):
    assert rule in _rules(tmp_path / "pos", f"web/{rel}", positive)
    assert rule not in _rules(tmp_path / "neg", f"web/{rel}", negative)


def test_requirements_unpinned_flagged(tmp_path):
    findings = _scan(tmp_path, {"requirements.txt": "requests\nflask==3.0.0\n# comment\n-r other.txt\n"})
    unpinned = [f for f in findings if f["rule_id"] == "dep-unpinned-python"]
    assert [(f["line"], f["snippet"]) for f in unpinned] == [(1, "requests")]
    assert unpinned[0]["category"] == "dependency"


def test_package_json_floating_versions(tmp_path):
    pkg = '{\n  "dependencies": {\n    "left-pad": "*",\n    "lodash": "^4.17.21"\n  }\n}\n'
    findings = [f for f in _scan(tmp_path, {"package.json": pkg}) if f["rule_id"] == "dep-floating-npm"]
    assert len(findings) == 1 and findings[0]["line"] == 3


def test_test_files_downgrade_security_severity(tmp_path):
    findings = _scan(tmp_path, {"tests/test_fixture.py": "KEY = 'AKIAABCDEFGHIJKLMNOP'\n"})
    key = [f for f in findings if f["rule_id"] == "secret-aws-key"]
    assert key and key[0]["severity"] == "high"


def test_python_finding_node_id_points_at_enclosing_symbol(tmp_path):
    src = "class Svc:\n    def run(self, cmd):\n        return eval(cmd)\n"
    findings = _scan(tmp_path, {"app/svc.py": src})
    ev = [f for f in findings if f["rule_id"] == "py-eval-exec"][0]
    assert ev["node_id"] == "sym:app/svc.py::Svc.run"
    assert ev["line"] == 3


def test_module_level_finding_falls_back_to_file_node(tmp_path):
    findings = _scan(tmp_path, {"app/boot.py": "import os\nos.system('ls')\n"})
    assert findings[0]["node_id"] == "file:app/boot.py"


def test_ids_are_stable_and_results_deduplicated(tmp_path):
    files = {"app/a.py": "x = eval(s)\n"}
    first = _scan(tmp_path / "one", files)
    second = _scan(tmp_path / "two", files)
    assert [f["id"] for f in first] == [f["id"] for f in second]
    assert len({f["id"] for f in first}) == len(first)


def test_symlinks_and_binary_files_are_skipped(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = eval(s)\n")
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "link.py").symlink_to(outside)
    (tmp_path / "repo" / "blob.py").write_bytes(b"\x00eval(x)")
    assert scan_repository(tmp_path / "repo", ["link.py", "blob.py", "missing.py"]) == []


def test_path_escape_is_rejected(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "secret.py").write_text("x = eval(s)\n")
    assert scan_repository(tmp_path / "repo", ["../secret.py"]) == []


def test_every_finding_carries_confidence(tmp_path):
    findings = _scan(tmp_path, {"app/a.py": "x = eval(s)\nimport random\nt = random.choice(a)\n"})
    assert findings and all(f["confidence"] in ("high", "medium", "low") for f in findings)
    assert [f for f in findings if f["rule_id"] == "py-eval-exec"][0]["confidence"] == "medium"


def test_test_fixture_secrets_are_low_confidence(tmp_path):
    findings = _scan(tmp_path, {"tests/test_fixture.py": "KEY = 'AKIAABCDEFGHIJKLMNOP'\n"})
    key = [f for f in findings if f["rule_id"] == "secret-aws-key"][0]
    assert key["confidence"] == "low" and key["severity"] == "high"
