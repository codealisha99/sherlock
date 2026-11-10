"""Quality metrics: complexity, nesting, length, docstrings, test gaps, and the score."""

from __future__ import annotations

import ast
from pathlib import Path

from app.saas.quality import (
    analyze_quality,
    cyclomatic_complexity,
    max_nesting,
    summarize,
)

BRANCHY = '''def f(a, b, xs):
    if a and b:
        return 1
    elif a:
        return 2
    for x in xs:
        if x:
            pass
    return [y for y in xs if y]
'''


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(src).body[0]  # type: ignore[return-value]


def test_cyclomatic_complexity_known_value():
    # base 1 + if + and + elif + for + inner if + comprehension(for + if) = 8
    assert cyclomatic_complexity(_fn(BRANCHY)) == 8


def test_elif_chain_does_not_add_nesting():
    flat = _fn("def g(x):\n    if x == 1:\n        return 1\n    elif x == 2:\n        return 2\n"
               "    elif x == 3:\n        return 3\n    else:\n        return 0\n")
    assert max_nesting(flat) == 1


def test_nested_control_flow_depth():
    deep = _fn("def g(xs):\n    for x in xs:\n        if x:\n            while x:\n                pass\n")
    assert max_nesting(deep) == 3


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _long_body(n: int) -> str:
    return "    x = 0\n" + "".join(f"    x += {i}\n" for i in range(n)) + "    return x\n"


def test_quality_findings_and_metrics(tmp_path):
    complex_branches = "".join(f"    if a == {i}:\n        return {i}\n" for i in range(14))
    src = (
        '"""Module."""\n\n'
        f"def busy(a):\n{complex_branches}    return -1\n\n"
        f"def long_one():\n{_long_body(90)}\n"
        "def undocumented():\n    return 1\n\n"
        "def _private():\n    return 2\n\n"
        "# TODO: one\n# FIXME: two\n# HACK: three\n"
    )
    _write(tmp_path, "app/mod.py", src + "\n" * 5)
    _write(tmp_path, "app/other.py", '"""Doc."""\n\ndef ok():\n    """Doc."""\n    return 1\n')
    findings, metrics = analyze_quality(tmp_path, ["app/mod.py", "app/other.py"])
    rules = {(f["rule_id"], f["node_id"]) for f in findings}

    assert ("q-complexity", "sym:app/mod.py::busy") in rules
    assert ("q-function-length", "sym:app/mod.py::long_one") in rules
    docs = [f for f in findings if f["rule_id"] == "q-docstring"]
    assert len(docs) == 1 and docs[0]["file_path"] == "app/mod.py"
    assert docs[0]["count"] == 3 and docs[0]["severity"] == "info"
    undocumented_line = src.splitlines().index("def undocumented():") + 1
    assert {"file_path": "app/mod.py", "line": undocumented_line} in docs[0]["occurrences"]
    assert all("_private" not in str(o) for o in docs[0]["occurrences"])
    assert any(r == "q-todo" for r, _ in rules)
    assert metrics["sym:app/mod.py::busy"]["complexity"] > 10
    assert metrics["file:app/mod.py"]["loc"] == len((tmp_path / "app/mod.py").read_text().splitlines())


def test_untested_module_flagged_until_test_exists(tmp_path):
    body = "\n".join(f"x{i} = {i}" for i in range(40)) + "\n"
    _write(tmp_path, "app/billing_engine.py", body)
    findings, _ = analyze_quality(tmp_path, ["app/billing_engine.py"])
    untested = [f for f in findings if f["rule_id"] == "q-no-tests"]
    assert len(untested) == 1
    assert untested[0]["file_path"] == "" and untested[0]["node_id"] is None
    assert untested[0]["occurrences"] == [{"file_path": "app/billing_engine.py", "line": 1}]
    assert untested[0]["confidence"] == "low"

    _write(tmp_path, "tests/test_billing_engine.py", "def test_x():\n    pass\n")
    findings, _ = analyze_quality(tmp_path, ["app/billing_engine.py", "tests/test_billing_engine.py"])
    assert not any(f["rule_id"] == "q-no-tests" for f in findings)


def test_many_untested_modules_aggregate_into_one_finding(tmp_path):
    files = []
    for name in ("alpha_service", "beta_service", "gamma_service"):
        _write(tmp_path, f"app/{name}.py", "\n".join(f"x{i} = {i}" for i in range(40)) + "\n")
        files.append(f"app/{name}.py")
    findings, _ = analyze_quality(tmp_path, files)
    untested = [f for f in findings if f["rule_id"] == "q-no-tests"]
    assert len(untested) == 1 and untested[0]["count"] == 3
    assert {o["file_path"] for o in untested[0]["occurrences"]} == set(files)


def test_docstring_findings_one_per_file(tmp_path):
    body = "".join(f"def api_{i}():\n    return {i}\n\n" for i in range(10))
    _write(tmp_path, "app/many.py", body)
    _write(tmp_path, "app/clean.py", '"""Doc."""\n\ndef ok():\n    """Doc."""\n    return 1\n')
    findings, _ = analyze_quality(tmp_path, ["app/many.py", "app/clean.py"])
    docs = [f for f in findings if f["rule_id"] == "q-docstring"]
    assert len(docs) == 1
    assert docs[0]["count"] == 10 and len(docs[0]["occurrences"]) == 10


def test_todo_only_counts_comment_markers(tmp_path):
    text = 'def f():\n    """Lists TODO FIXME HACK markers."""\n    return 1\n'
    _write(tmp_path, "app/doc_only.py", text)
    findings, _ = analyze_quality(tmp_path, ["app/doc_only.py"])
    assert not any(f["rule_id"] == "q-todo" for f in findings)


def test_small_modules_not_flagged_as_untested(tmp_path):
    _write(tmp_path, "app/tiny.py", "x = 1\n")
    findings, _ = analyze_quality(tmp_path, ["app/tiny.py"])
    assert not any(f["rule_id"] == "q-no-tests" for f in findings)


def test_very_large_file_flagged(tmp_path):
    _write(tmp_path, "app/huge.py", "x = 1\n" * 1600)
    findings, _ = analyze_quality(tmp_path, ["app/huge.py"])
    huge = [f for f in findings if f["rule_id"] == "q-file-very-large"]
    assert huge and huge[0]["severity"] == "medium"


def test_summarize_score_and_counts():
    empty = summarize([])
    assert empty["score"] == 100 and empty["total"] == 0
    assert set(empty["by_severity"]) == {"critical", "high", "medium", "low", "info"}

    mixed = [
        {"severity": "critical", "category": "secrets"},
        {"severity": "high", "category": "security"},
        {"severity": "low", "category": "quality"},
    ]
    result = summarize(mixed)
    assert result["score"] == round(100 - 10 - 3 - 0.1)  # critical + high + low weights
    assert result["by_severity"]["critical"] == 1
    assert result["by_category"] == {"quality": 1, "secrets": 1, "security": 1}
    assert result["total"] == 3


def test_low_confidence_and_false_positives_do_not_lower_score():
    real = {"severity": "high", "category": "security", "confidence": "high"}
    noisy = {"severity": "high", "category": "security", "confidence": "low"}
    fp = {"severity": "high", "category": "security", "confidence": "high",
          "triage": {"verdict": "false_positive"}}
    confirmed = {"severity": "high", "category": "security", "confidence": "high",
                 "triage": {"verdict": "confirmed"}}
    result = summarize([real, noisy, fp, confirmed])
    assert result["score"] == 100 - 2 * 3  # only the two non-FP medium+ findings count
    assert result["total"] == 4
    assert result["triaged"] == 2 and result["false_positives"] == 1


def test_summarize_clamps_at_zero():
    many = [{"severity": "critical", "category": "secrets"}] * 20
    assert summarize(many)["score"] == 0
