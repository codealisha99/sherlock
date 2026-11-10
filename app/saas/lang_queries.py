"""Per-language tree-sitter rules and import resolution for ``lang_parse`` and ``graph``.

Rules classify tree-sitter nodes into declarations (class / function / method /
scope), import specs and call sites. Resolution maps import specs to repository
files (or an external package name) using a precomputed ``PathIndex``.
"""

from __future__ import annotations

import posixpath
import re
from typing import Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Set, Tuple

from tree_sitter import Node

CLASS, FUNCTION, METHOD = "class", "function", "method"
SCOPE = "scope"  # container without a symbol of its own (Rust impl blocks)
JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
GRAMMAR_BY_EXT = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".java": "java", ".rs": "rust", ".rb": "ruby",
}
GRAMMAR_BY_LANGUAGE = {
    "JavaScript": "javascript", "TypeScript": "typescript", "Go": "go",
    "Java": "java", "Rust": "rust", "Ruby": "ruby",
}
JS_FAMILY = frozenset({"javascript", "typescript", "tsx"})
_SIMPLE_NAMES = frozenset({"identifier", "field_identifier", "property_identifier", "constant", "type_identifier"})
_RECEIVER_FIELDS = ("object", "operand", "value", "receiver")
_NAME_FIELDS = ("property", "field", "name")
_JS_CLASSES = frozenset({"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration"})
_JS_FUNC_VALUES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})
_JAVA_CLASSES = frozenset({"class_declaration", "interface_declaration", "enum_declaration",
                           "record_declaration", "annotation_type_declaration"})
_JAVA_FUNCS = frozenset({"method_declaration", "constructor_declaration"})
_RUST_CLASSES = frozenset({"struct_item", "enum_item", "trait_item", "union_item"})


class Decl(NamedTuple):
    name: str
    kind: str  # class | function | method | scope
    absolute: bool = False  # name already fully qualified (Go methods)
    self_name: Optional[str] = None  # receiver variable name (Go)


class LangRules(NamedTuple):
    decl: Callable[[Node], Optional[Decl]]
    imports: Callable[[Node], List[str]]
    call_types: FrozenSet[str]


def text(node: Node) -> str:
    return node.text.decode("utf-8", "replace")


def _field(node: Node, name: str) -> Optional[Node]:
    return node.child_by_field_name(name)


def _first_field(node: Node, names: Iterable[str]) -> Optional[Node]:
    return next((n for n in (_field(node, f) for f in names) if n is not None), None)


def _first_of(node: Node, node_type: str) -> Optional[Node]:
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == node_type:
            return cur
        stack.extend(reversed(cur.children))
    return None


def string_value(node: Node) -> str:
    return text(node).strip("'\"`")


def call_parts(node: Node) -> Tuple[Optional[str], Optional[str]]:
    """(receiver text or None, callee name or None) for a call-like node."""
    fn = _field(node, "function") or _field(node, "method") or _field(node, "name")
    if fn is None:
        return None, None
    recv = _first_field(node, _RECEIVER_FIELDS)
    if fn.type in _SIMPLE_NAMES:
        name_node = fn
    else:
        name_node = _first_field(fn, _NAME_FIELDS)
        recv = _first_field(fn, _RECEIVER_FIELDS) or recv
    if name_node is None:
        return None, None
    return (text(recv) if recv is not None else None), text(name_node)


def _named(node: Node, kind: str) -> Optional[Decl]:
    name = _field(node, "name")
    return Decl(text(name).split("::")[-1], kind) if name is not None else None


def _no_imports(_: Node) -> List[str]:
    return []


# ---- JavaScript / TypeScript -------------------------------------------------

def _js_decl(node: Node) -> Optional[Decl]:
    t = node.type
    if t in _JS_CLASSES:
        return _named(node, CLASS)
    if t in ("function_declaration", "generator_function_declaration"):
        return _named(node, FUNCTION)
    if t == "method_definition" and node.parent is not None and node.parent.type == "class_body":
        return _named(node, METHOD)
    if t == "variable_declarator":
        name, value = _field(node, "name"), _field(node, "value")
        if name is not None and name.type == "identifier" and value is not None and value.type in _JS_FUNC_VALUES:
            return Decl(text(name), FUNCTION)
    return None


def _js_imports(node: Node) -> List[str]:
    if node.type in ("import_statement", "export_statement"):
        src = _field(node, "source")
        return [string_value(src)] if src is not None else []
    if node.type == "call_expression":
        fn, args = _field(node, "function"), _field(node, "arguments")
        if fn is not None and args is not None and (fn.type == "import" or text(fn) == "require"):
            first = args.named_children[0] if args.named_children else None
            if first is not None and first.type == "string":
                return [string_value(first)]
    return []


# ---- Go ----------------------------------------------------------------------

def _go_decl(node: Node) -> Optional[Decl]:
    if node.type == "function_declaration":
        return _named(node, FUNCTION)
    if node.type == "method_declaration":
        recv, name = _field(node, "receiver"), _field(node, "name")
        owner = _first_of(recv, "type_identifier") if recv is not None else None
        if owner is None or name is None:
            return None
        params = [c for c in recv.named_children if c.type == "parameter_declaration"]
        recv_name = _field(params[0], "name") if params else None
        self_name = text(recv_name) if recv_name is not None else None
        return Decl(f"{text(owner)}.{text(name)}", METHOD, absolute=True, self_name=self_name)
    if node.type == "type_spec":
        typ = _field(node, "type")
        if typ is not None and typ.type in ("struct_type", "interface_type"):
            return _named(node, CLASS)
    return None


def _go_imports(node: Node) -> List[str]:
    if node.type == "import_spec":
        path = _field(node, "path")
        return [string_value(path)] if path is not None else []
    return []


# ---- Java --------------------------------------------------------------------

def _java_decl(node: Node) -> Optional[Decl]:
    if node.type in _JAVA_CLASSES:
        return _named(node, CLASS)
    if node.type in _JAVA_FUNCS:
        return _named(node, FUNCTION)
    return None


def _java_imports(node: Node) -> List[str]:
    if node.type != "import_declaration":
        return []
    part = next((c for c in node.named_children if c.type in ("scoped_identifier", "identifier")), None)
    return [text(part)] if part is not None else []


# ---- Rust --------------------------------------------------------------------

def _rust_decl(node: Node) -> Optional[Decl]:
    if node.type == "function_item":
        return _named(node, FUNCTION)
    if node.type in _RUST_CLASSES:
        return _named(node, CLASS)
    if node.type == "impl_item":
        typ = _field(node, "type")
        owner = _first_of(typ, "type_identifier") if typ is not None else None
        return Decl(text(owner), SCOPE) if owner is not None else None
    return None


_USE_PREFIX_RE = re.compile(r"^(pub(\([^)]*\))?\s+)?use\s+")


def rust_use_spec(decl_text: str) -> str:
    """'use crate::a::{b, c};' -> 'crate::a'; 'use std::io as x;' -> 'std::io'."""
    body = _USE_PREFIX_RE.sub("", decl_text.strip()).rstrip(";").split(" as ")[0]
    body = re.sub(r"\s+", "", body)
    body = body.split("{")[0].removesuffix("::*").rstrip(":")
    return body


def _rust_imports(node: Node) -> List[str]:
    if node.type == "use_declaration":
        spec = rust_use_spec(text(node))
        return [spec] if spec else []
    if node.type == "mod_item" and _field(node, "body") is None:
        name = _field(node, "name")
        return [f"self::{text(name)}"] if name is not None else []
    return []


# ---- Ruby --------------------------------------------------------------------

def _ruby_decl(node: Node) -> Optional[Decl]:
    if node.type in ("class", "module"):
        return _named(node, CLASS)
    if node.type in ("method", "singleton_method"):
        return _named(node, FUNCTION)
    return None


def _ruby_imports(node: Node) -> List[str]:
    if node.type != "call":
        return []
    method, args = _field(node, "method"), _field(node, "arguments")
    if method is None or args is None or text(method) not in ("require", "require_relative"):
        return []
    first = args.named_children[0] if args.named_children else None
    if first is None or first.type != "string":
        return []
    value = string_value(first)
    if text(method) == "require_relative" and not value.startswith(("./", "../")):
        value = f"./{value}"
    return [value]


RULES: Dict[str, LangRules] = {
    "javascript": LangRules(_js_decl, _js_imports, frozenset({"call_expression"})),
    "typescript": LangRules(_js_decl, _js_imports, frozenset({"call_expression"})),
    "tsx": LangRules(_js_decl, _js_imports, frozenset({"call_expression"})),
    "go": LangRules(_go_decl, _go_imports, frozenset({"call_expression"})),
    "java": LangRules(_java_decl, _java_imports, frozenset({"method_invocation"})),
    "rust": LangRules(_rust_decl, _rust_imports, frozenset({"call_expression"})),
    "ruby": LangRules(_ruby_decl, _ruby_imports, frozenset({"call"})),
}


def grammar_for(path: str, language: Optional[str] = None) -> Optional[str]:
    """Grammar name for a repo path; extension wins, indexer language name is the fallback."""
    ext = posixpath.splitext(path)[1].lower()
    return GRAMMAR_BY_EXT.get(ext) or (GRAMMAR_BY_LANGUAGE.get(language) if language else None)


# ---- Import resolution -------------------------------------------------------

class PathIndex(NamedTuple):
    paths: frozenset
    by_base: Dict[str, List[str]]
    go_dirs: Dict[str, List[str]]  # dir ('' = repo root) -> non-test .go files
    go_module: Optional[str]
    rust_roots: frozenset  # dirs containing lib.rs / main.rs


def build_index(paths: Set[str], go_mod_text: str = "") -> PathIndex:
    by_base: Dict[str, List[str]] = {}
    go_dirs: Dict[str, List[str]] = {}
    roots: Set[str] = set()
    for p in sorted(paths):
        base = posixpath.basename(p)
        by_base.setdefault(base, []).append(p)
        if p.endswith(".go") and not p.endswith("_test.go"):
            go_dirs.setdefault(posixpath.dirname(p), []).append(p)
        if base in ("lib.rs", "main.rs"):
            roots.add(posixpath.dirname(p))
    mod = re.search(r"^module\s+(\S+)", go_mod_text, re.MULTILINE)
    return PathIndex(frozenset(paths), by_base, go_dirs, mod.group(1) if mod else None, frozenset(roots))


def _relative(rel: str, spec: str, paths: frozenset, suffixes: Tuple[str, ...], index_files: bool) -> List[str]:
    base = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
    if base == ".." or base.startswith("../"):
        return []
    cands = [base, *(base + e for e in suffixes)]
    if index_files:
        cands += [f"{base}/index{e}" for e in suffixes]
    return [next(c for c in cands if c in paths)] if any(c in paths for c in cands) else []


def _package_name(spec: str) -> str:
    parts = spec.split("/")
    return "/".join(parts[:2]) if spec.startswith("@") else parts[0]


def _resolve_js(rel: str, spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    if spec.startswith(("./", "../")):
        return _relative(rel, spec, idx.paths, JS_SUFFIXES, index_files=True), None
    if not spec or spec.startswith("/"):
        return [], None
    return [], _package_name(spec)


def _resolve_ruby(rel: str, spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    if spec.startswith(("./", "../")):
        return _relative(rel, spec, idx.paths, (".rb",), index_files=False), None
    return [], spec.split("/")[0]


def _resolve_java(spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    parts = spec.split(".")
    for k in range(len(parts), 1, -1):  # k >= 2: never match a bare 'Foo.java' by suffix
        want = "/".join(parts[:k]) + ".java"
        hits = sorted(p for p in idx.by_base.get(parts[k - 1] + ".java", []) if p == want or p.endswith("/" + want))
        if hits:
            return hits[:1], None
    return [], ".".join(parts[:2])


def _go_ext(spec: str) -> str:
    parts = spec.split("/")
    return "/".join(parts[:3] if "." in parts[0] else parts[:1])


def _resolve_go(spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    mod = idx.go_module
    if mod and (spec == mod or spec.startswith(mod + "/")):
        rel = "" if spec == mod else spec[len(mod) + 1:]
        return list(idx.go_dirs.get(rel, [])), None
    if not mod:
        parts = spec.split("/")
        for k in range(len(parts)):
            d = "/".join(parts[k:])
            if d in idx.go_dirs:
                return list(idx.go_dirs[d]), None
    return [], _go_ext(spec)


def _rust_mod_dir(rel: str) -> str:
    stem = posixpath.splitext(posixpath.basename(rel))[0]
    d = posixpath.dirname(rel)
    return d if stem in ("mod", "lib", "main") else posixpath.join(d, stem)


def _rust_crate_root(rel: str, roots: frozenset) -> Optional[str]:
    d = posixpath.dirname(rel)
    while True:
        if d in roots:
            return d
        if not d:
            return None
        d = posixpath.dirname(d)


def _rust_find(base: str, segs: List[str], paths: frozenset) -> List[str]:
    for k in range(len(segs), 0, -1):  # longest module prefix wins
        stem = posixpath.join(base, *segs[:k])
        for cand in (f"{stem}.rs", f"{stem}/mod.rs"):
            if cand in paths:
                return [cand]
    return []


def _resolve_rust(rel: str, spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    if spec.startswith("self::"):
        return _rust_find(_rust_mod_dir(rel), spec[6:].split("::"), idx.paths), None
    if spec.startswith("crate::"):
        root = _rust_crate_root(rel, idx.rust_roots)
        return (_rust_find(root, spec[7:].split("::"), idx.paths) if root is not None else []), None
    if spec.startswith("super::"):
        return [], None
    return [], spec.split("::")[0]


def resolve_import(grammar: str, rel: str, spec: str, idx: PathIndex) -> Tuple[List[str], Optional[str]]:
    """Map an import spec to (internal repo files, external package name or None)."""
    if grammar in JS_FAMILY:
        return _resolve_js(rel, spec, idx)
    if grammar == "ruby":
        return _resolve_ruby(rel, spec, idx)
    if grammar == "java":
        return _resolve_java(spec, idx)
    if grammar == "go":
        return _resolve_go(spec, idx)
    if grammar == "rust":
        return _resolve_rust(rel, spec, idx)
    return [], None
