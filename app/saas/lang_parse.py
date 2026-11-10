"""Multi-language symbol, import and call extraction (tree-sitter).

Covers JavaScript, TypeScript/TSX, Go, Java, Rust and Ruby. Python stays on the
stdlib ``ast`` path in the indexer. When a grammar is unavailable or the file
has parse errors, the result falls back to regex JS import extraction
(``parsed=False``), so graph building never fails on a bad file.
"""

from __future__ import annotations

import logging
import re
from typing import List, NamedTuple, Optional, Tuple

from app.saas.lang_queries import (
    FUNCTION, METHOD, CLASS, SCOPE, JS_FAMILY, RULES, Decl, LangRules,
    call_parts, grammar_for,
)

try:
    from tree_sitter_language_pack import get_parser
except ImportError:  # pragma: no cover - optional dependency
    get_parser = None

logger = logging.getLogger(__name__)

MAX_BYTES = 500 * 1024
_NAME_RE = re.compile(r"^[A-Za-z_$][\w$]*[?!]?$")
_JS_SPEC_RE = re.compile(
    r"""(?:import|export)\s[^'"`;]*?\bfrom\s*['"]([^'"]+)['"]"""
    r"""|\bimport\s*['"]([^'"]+)['"]"""
    r"""|\brequire\(\s*['"]([^'"]+)['"]\s*\)"""
    r"""|\bimport\(\s*['"]([^'"]+)['"]\s*\)"""
)


class Symbol(NamedTuple):
    qualname: str
    kind: str  # class | function | method
    line: int
    end_line: int


class Import(NamedTuple):
    module_spec: str  # raw spec; resolved later by lang_queries.resolve_import
    line: int


class Call(NamedTuple):
    caller_qualname: Optional[str]
    callee_name: str
    line: int
    is_self: bool  # receiver is this/self or the enclosing method's receiver variable


class ParsedFile(NamedTuple):
    symbols: Tuple[Symbol, ...]
    imports: Tuple[Import, ...]
    calls: Tuple[Call, ...]
    parsed: bool


EMPTY = ParsedFile((), (), (), False)
# Scope entry: (qualname, kind, receiver variable name)
_Scope = Tuple[Tuple[str, str, Optional[str]], ...]


def _line(node) -> int:
    return node.start_point[0] + 1


def _enter(node, decl: Decl, scope: _Scope, symbols: List[Symbol], seen: set) -> _Scope:
    parent = scope[-1] if scope else None
    qual = decl.name if (decl.absolute or parent is None) else f"{parent[0]}.{decl.name}"
    kind = METHOD if decl.kind == FUNCTION and parent is not None and parent[1] == CLASS else decl.kind
    if decl.kind != SCOPE and qual not in seen:  # first definition wins on duplicate names
        seen.add(qual)
        symbols.append(Symbol(qual, kind, _line(node), node.end_point[0] + 1))
    entry_kind = CLASS if decl.kind in (CLASS, SCOPE) else kind
    return scope + ((qual, entry_kind, decl.self_name),)


def _make_call(node, scope: _Scope, rules: LangRules) -> Optional[Call]:
    recv, name = call_parts(node)
    if not name or not _NAME_RE.match(name):
        return None
    inner = scope[-1] if scope else None
    caller = inner[0] if inner is not None and inner[1] in (FUNCTION, METHOD) else None
    is_self = bool(recv) and (recv in ("self", "this") or (inner is not None and inner[2] == recv))
    return Call(caller, name, _line(node), is_self)


def _walk(root, rules: LangRules) -> Tuple[List[Symbol], List[Import], List[Call]]:
    symbols: List[Symbol] = []
    imports: List[Import] = []
    calls: List[Call] = []
    seen: set = set()
    stack: list = [(root, ())]  # explicit stack: no recursion limit on deep trees
    while stack:
        node, scope = stack.pop()
        inner = scope
        decl = rules.decl(node)
        if decl is not None:
            inner = _enter(node, decl, scope, symbols, seen)
        imports.extend(Import(spec, _line(node)) for spec in rules.imports(node))
        if node.type in rules.call_types:
            call = _make_call(node, scope, rules)
            if call is not None:
                calls.append(call)
        stack.extend((child, inner) for child in reversed(node.children))
    return symbols, imports, calls


def _fallback(grammar: str, source: str) -> ParsedFile:
    """Regex import extraction for JS-family files; nothing else without a parse tree."""
    if grammar not in JS_FAMILY:
        return EMPTY
    imports = []
    for m in _JS_SPEC_RE.finditer(source):
        spec = next(g for g in m.groups() if g)
        imports.append(Import(spec, source.count("\n", 0, m.start()) + 1))
    return ParsedFile((), tuple(imports), (), False)


def _load_parser(grammar: str):
    if get_parser is None:
        logger.debug("tree-sitter language pack not installed; regex fallback for %s", grammar)
        return None
    try:
        return get_parser(grammar)
    except Exception:  # grammar missing or failed to load: degrade, never raise
        logger.debug("tree-sitter grammar %s unavailable; regex fallback", grammar, exc_info=True)
        return None


def parse_file(path_rel: str, source: str, language: Optional[str] = None) -> ParsedFile:
    """Extract symbols, imports and calls from one non-Python source file."""
    grammar = grammar_for(path_rel, language)
    if grammar is None:
        return EMPTY
    data = source.encode("utf-8")
    if len(data) > MAX_BYTES:
        logger.debug("skipping %s: %d bytes exceeds parse cap", path_rel, len(data))
        return EMPTY
    parser = _load_parser(grammar)
    if parser is None:
        return _fallback(grammar, source)
    tree = parser.parse(data)
    if tree.root_node.has_error:
        logger.debug("parse errors in %s; regex fallback", path_rel)
        return _fallback(grammar, source)
    symbols, imports, calls = _walk(tree.root_node, RULES[grammar])
    return ParsedFile(tuple(symbols), tuple(imports), tuple(calls), True)
