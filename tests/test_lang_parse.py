"""tree-sitter symbol/import/call extraction per language, plus regex fallback."""

from __future__ import annotations

import pytest

from app.saas import lang_parse
from app.saas.lang_parse import MAX_BYTES, parse_file
from app.saas.lang_queries import rust_use_spec

TS_SRC = (
    "import { x } from './util';\n"
    "export class Svc extends Base {\n"
    "  run(v: string): number {\n"
    "    return this.helper(v) + helper();\n"
    "  }\n"
    "  private helper(v: string) { return 1; }\n"
    "}\n"
    "export const f = (a: number) => {\n"
    "  g(a);\n"
    "};\n"
    "function* gen() {}\n"
    "const lazy = await import('./lazy');\n"
    "const r = require('react');\n"
)

GO_SRC = (
    "package p\n"
    "\n"
    "import (\n"
    "\t\"fmt\"\n"
    "\t\"github.com/x/y/z\"\n"
    ")\n"
    "\n"
    "type S struct{ n int }\n"
    "\n"
    "type I interface{ M() }\n"
    "\n"
    "func (s *S) Get() int {\n"
    "\tfmt.Println(s.n)\n"
    "\treturn helper()\n"
    "}\n"
    "\n"
    "func helper() int { return 1 }\n"
)

JAVA_SRC = (
    "package a.b;\n"
    "import java.util.List;\n"
    "import java.util.*;\n"
    "import static a.b.C.m;\n"
    "public class C {\n"
    "  public C() {}\n"
    "  void run() {\n"
    "    this.go();\n"
    "    helper();\n"
    "  }\n"
    "  void go() {}\n"
    "  interface I { void x(); }\n"
    "  enum E { A }\n"
    "}\n"
)

RUST_SRC = (
    "use crate::a::{b, c};\n"
    "use std::io::Write as W;\n"
    "mod m;\n"
    "pub struct S;\n"
    "impl S {\n"
    "    pub fn new() -> Self { S }\n"
    "    fn run(&self) {\n"
    "        self.go();\n"
    "        helper();\n"
    "    }\n"
    "}\n"
    "fn helper() {}\n"
    "trait T {\n"
    "    fn t(&self) {}\n"
    "}\n"
)

RUBY_SRC = (
    "require 'json'\n"
    "require_relative 'lib/util'\n"
    "module M\n"
    "  class K < Base\n"
    "    def run\n"
    "      self.go\n"
    "      helper(1)\n"
    "    end\n"
    "    def self.make; end\n"
    "  end\n"
    "end\n"
    "def top; end\n"
)


def _syms(pf):
    return {(s.qualname, s.kind) for s in pf.symbols}


def _calls(pf):
    return {(c.caller_qualname, c.callee_name) for c in pf.calls}


def test_typescript_symbols_imports_calls():
    pf = parse_file("web/svc.ts", TS_SRC, "TypeScript")
    assert pf.parsed is True
    assert _syms(pf) >= {
        ("Svc", "class"), ("Svc.run", "method"), ("Svc.helper", "method"),
        ("f", "function"), ("gen", "function"),
    }
    assert [i.module_spec for i in pf.imports] == ["./util", "./lazy", "react"]
    assert ("Svc.run", "helper") in _calls(pf)
    assert ("f", "g") in _calls(pf)
    assert ("Svc.run", "helper") in {(c.caller_qualname, c.callee_name) for c in pf.calls if c.is_self}
    sym = next(s for s in pf.symbols if s.qualname == "Svc.run")
    assert sym.line == 3 and sym.end_line == 5


def test_tsx_grammar_selected_by_extension():
    pf = parse_file("web/View.tsx", "export function View() { return <div/>; }\n", "TypeScript")
    assert pf.parsed is True
    assert ("View", "function") in _syms(pf)


def test_javascript_export_default_and_class_expression_names():
    src = "export default class Widget {\n  render() { return 1; }\n}\n"
    pf = parse_file("ui/widget.js", src, "JavaScript")
    assert _syms(pf) == {("Widget", "class"), ("Widget.render", "method")}


def test_go_functions_methods_structs_and_imports():
    pf = parse_file("svc/s.go", GO_SRC, "Go")
    assert pf.parsed is True
    assert _syms(pf) == {("S", "class"), ("I", "class"), ("S.Get", "method"), ("helper", "function")}
    assert [i.module_spec for i in pf.imports] == ["fmt", "github.com/x/y/z"]
    assert ("S.Get", "helper") in _calls(pf)
    assert ("S.Get", "Println") in _calls(pf)


def test_java_classes_methods_nested_and_imports():
    pf = parse_file("a/b/C.java", JAVA_SRC, "Java")
    assert pf.parsed is True
    assert _syms(pf) >= {("C", "class"), ("C.run", "method"), ("C.go", "method"),
                         ("C.I", "class"), ("C.I.x", "method"), ("C.E", "class")}
    assert [i.module_spec for i in pf.imports] == ["java.util.List", "java.util", "a.b.C.m"]
    assert ("C.run", "go") in {(c.caller_qualname, c.callee_name) for c in pf.calls if c.is_self}
    assert ("C.run", "helper") in _calls(pf)


def test_rust_impl_methods_structs_traits_and_use():
    pf = parse_file("src/lib.rs", RUST_SRC, "Rust")
    assert pf.parsed is True
    assert _syms(pf) >= {("S", "class"), ("S.new", "method"), ("S.run", "method"),
                         ("helper", "function"), ("T", "class"), ("T.t", "method")}
    assert [i.module_spec for i in pf.imports] == ["crate::a", "std::io::Write", "self::m"]
    assert ("S.run", "go") in {(c.caller_qualname, c.callee_name) for c in pf.calls if c.is_self}


@pytest.mark.parametrize("decl, expected", [
    ("use crate::a::{b, c};", "crate::a"),
    ("pub use super::x::*;", "super::x"),
    ("use std::io::Write as W;", "std::io::Write"),
    ("use foo::bar;", "foo::bar"),
])
def test_rust_use_spec(decl, expected):
    assert rust_use_spec(decl) == expected


def test_ruby_class_module_methods_and_requires():
    pf = parse_file("lib/m.rb", RUBY_SRC, "Ruby")
    assert pf.parsed is True
    assert _syms(pf) == {("M.K", "class"), ("M", "class"), ("M.K.run", "method"),
                         ("M.K.make", "method"), ("top", "function")}
    assert [i.module_spec for i in pf.imports] == ["json", "./lib/util"]
    assert ("M.K.run", "go") in {(c.caller_qualname, c.callee_name) for c in pf.calls if c.is_self}
    assert ("M.K.run", "helper") in _calls(pf)


def test_unsupported_extension_returns_empty():
    pf = parse_file("notes.md", "# hi\n", None)
    assert pf.parsed is False and pf.symbols == () and pf.imports == ()


def test_grammar_unavailable_falls_back_to_regex_imports(monkeypatch):
    monkeypatch.setattr(lang_parse, "get_parser", None)
    pf = parse_file("web/app.js", "import { x } from './util';\nconst React = require('react');\n", "JavaScript")
    assert pf.parsed is False
    assert pf.symbols == () and pf.calls == ()
    assert [(i.module_spec, i.line) for i in pf.imports] == [("./util", 1), ("react", 2)]


def test_grammar_load_error_falls_back(monkeypatch):
    def boom(name):
        raise RuntimeError(f"no grammar {name}")

    monkeypatch.setattr(lang_parse, "get_parser", boom)
    pf = parse_file("web/app.js", "import x from 'pkg';\n", "JavaScript")
    assert pf.parsed is False
    assert [i.module_spec for i in pf.imports] == ["pkg"]


def test_non_js_fallback_has_no_imports(monkeypatch):
    monkeypatch.setattr(lang_parse, "get_parser", None)
    pf = parse_file("a.go", GO_SRC, "Go")
    assert pf.parsed is False and pf.imports == () and pf.symbols == ()


def test_parse_error_falls_back_to_regex():
    pf = parse_file("web/broken.js", "import a from './a';\nfunction (((( {\n", "JavaScript")
    assert pf.parsed is False
    assert [i.module_spec for i in pf.imports] == ["./a"]


def test_oversized_file_is_skipped():
    big = "x = 1\n" * (MAX_BYTES // 6 + 10)
    pf = parse_file("huge.js", "import a from 'a';\n" + big, "JavaScript")
    assert pf.parsed is False and pf.symbols == () and pf.imports == ()
