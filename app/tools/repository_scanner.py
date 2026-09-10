"""Repository Scanner for Sherlock.

Performs static analysis, AST inspection, directory hierarchy extraction,
and metadata indexing on target codebases.
"""

import ast
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field

from config import config, SherlockConfig
from app.tools.filesys import SafeFilesystem


class FunctionDefInfo(BaseModel):
    name: str
    is_async: bool = False
    args: List[str] = Field(default_factory=list)
    decorators: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    start_line: int
    end_line: int


class ClassDefInfo(BaseModel):
    name: str
    bases: List[str] = Field(default_factory=list)
    methods: List[FunctionDefInfo] = Field(default_factory=list)
    docstring: Optional[str] = None
    start_line: int
    end_line: int


class FileAstInfo(BaseModel):
    imports: List[str] = Field(default_factory=list)
    from_imports: List[str] = Field(default_factory=list)
    classes: List[ClassDefInfo] = Field(default_factory=list)
    functions: List[FunctionDefInfo] = Field(default_factory=list)
    route_decorators: List[str] = Field(default_factory=list)
    todos: List[Dict[str, Any]] = Field(default_factory=list)


class ScannedFile(BaseModel):
    path: str  # relative path
    name: str
    extension: str
    size_bytes: int
    line_count: int = 0
    language: str
    is_binary: bool = False
    is_secret: bool = False
    is_entry_point: bool = False
    is_config: bool = False
    is_dependency_file: bool = False
    ast_info: Optional[FileAstInfo] = None


class RepositoryIndex(BaseModel):
    root_dir: str
    project_name: str
    total_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)
    files: List[ScannedFile] = Field(default_factory=list)
    directories: List[str] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    config_files: List[str] = Field(default_factory=list)
    dependency_files: List[str] = Field(default_factory=list)
    todos_and_fixmes: List[Dict[str, Any]] = Field(default_factory=list)
    directory_tree_repr: str = ""


# Common extension to language mapping
EXTENSION_LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".md": "Markdown",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
}

ENTRY_POINT_CANDIDATES = {
    "main.py", "app.py", "wsgi.py", "asgi.py", "server.py", "manage.py", "cli.py",
    "run.py", "index.js", "index.ts", "server.js", "main.go", "main.rs", "App.java",
}

CONFIG_CANDIDATES = {
    "config.py", "settings.py", "config.json", "config.yaml", "config.yml",
    "application.yml", "application.properties", "tsconfig.json", "vite.config.js",
    "vite.config.ts", "webpack.config.js", "Dockerfile", "docker-compose.yml",
    "docker-compose.yaml", ".eslintrc.json",
}

DEPENDENCY_FILE_CANDIDATES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
    "package.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "Gemfile",
}


class RepositoryScanner:
    """Recursively scans and parses repositories into structured indices."""

    def __init__(self, root_dir: Path | str, sherlock_cfg: Optional[SherlockConfig] = None):
        self.root_dir = Path(root_dir).resolve()
        self.cfg = sherlock_cfg or config
        self.fs = SafeFilesystem(self.root_dir, self.cfg)
        self.gitignore_patterns = self._load_gitignore()

    def _load_gitignore(self) -> List[str]:
        patterns = []
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line.rstrip("/"))
            except Exception:
                pass
        return patterns

    def _is_ignored(self, path: Path) -> bool:
        """Check if directory or file should be ignored."""
        # Check parent directory names against ignored_dirs
        for part in path.parts:
            if part in self.cfg.ignored_dirs:
                return True

        # Check gitignore patterns
        rel_path = self.fs.get_relative_path(path)
        for pat in self.gitignore_patterns:
            if fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel_path, pat):
                return True

        return False

    def scan(self) -> RepositoryIndex:
        """Run full scan and produce structured RepositoryIndex."""
        scanned_files: List[ScannedFile] = []
        directories: Set[str] = set()
        entry_points: List[str] = []
        config_files: List[str] = []
        dependency_files: List[str] = []
        todos: List[Dict[str, Any]] = []
        language_bytes: Dict[str, int] = {}
        total_size = 0
        total_lines = 0

        # Traverse directory
        for p in self.root_dir.rglob("*"):
            if self._is_ignored(p):
                continue

            if p.is_dir():
                rel_dir = self.fs.get_relative_path(p)
                if rel_dir and rel_dir != ".":
                    directories.add(rel_dir)
                continue

            if not p.is_file():
                continue

            # File checks
            rel_path = self.fs.get_relative_path(p)
            file_name = p.name
            ext = p.suffix.lower()
            is_bin = self.fs.is_binary(p)
            is_sec = self.fs.is_secret_file(p)
            size = p.stat().st_size
            total_size += size

            lang = EXTENSION_LANGUAGE_MAP.get(ext, "Other" if not is_bin else "Binary")
            language_bytes[lang] = language_bytes.get(lang, 0) + size

            # Classification flags
            is_entry = file_name.lower() in ENTRY_POINT_CANDIDATES
            is_cfg = file_name.lower() in CONFIG_CANDIDATES
            is_dep = file_name.lower() in DEPENDENCY_FILE_CANDIDATES

            if is_entry:
                entry_points.append(rel_path)
            if is_cfg:
                config_files.append(rel_path)
            if is_dep:
                dependency_files.append(rel_path)

            lines_count = 0
            ast_info = None

            # For text source files, count lines and parse AST / look for TODOs
            if not is_bin and not is_sec and size <= self.cfg.max_file_size_bytes:
                content, _ = self.fs.read_file(p)
                lines = content.splitlines()
                lines_count = len(lines)
                total_lines += lines_count

                # Extract TODOs & FIXMEs
                for line_idx, line_text in enumerate(lines, start=1):
                    upper = line_text.upper()
                    if "TODO" in upper or "FIXME" in upper or "NOTIMPLEMENTED" in upper or "HACK" in upper:
                        todos.append({
                            "file": rel_path,
                            "line": line_idx,
                            "text": line_text.strip(),
                            "type": "FIXME" if "FIXME" in upper else ("TODO" if "TODO" in upper else "NOTE")
                        })

                # Python AST parsing
                if ext == ".py":
                    ast_info = self._parse_python_ast(content, rel_path)

            scanned_files.append(
                ScannedFile(
                    path=rel_path,
                    name=file_name,
                    extension=ext,
                    size_bytes=size,
                    line_count=lines_count,
                    language=lang,
                    is_binary=is_bin,
                    is_secret=is_sec,
                    is_entry_point=is_entry,
                    is_config=is_cfg,
                    is_dependency_file=is_dep,
                    ast_info=ast_info,
                )
            )

            if len(scanned_files) >= self.cfg.max_files_to_index:
                break

        # Generate ASCII directory tree representation
        tree_repr = self._generate_tree_repr(scanned_files)

        return RepositoryIndex(
            root_dir=str(self.root_dir),
            project_name=self.root_dir.name,
            total_files=len(scanned_files),
            total_lines=total_lines,
            total_size_bytes=total_size,
            languages=language_bytes,
            files=scanned_files,
            directories=sorted(list(directories)),
            entry_points=sorted(entry_points),
            config_files=sorted(config_files),
            dependency_files=sorted(dependency_files),
            todos_and_fixmes=todos,
            directory_tree_repr=tree_repr,
        )

    def _parse_python_ast(self, code: str, rel_path: str) -> Optional[FileAstInfo]:
        """Extract classes, functions, decorators, and imports using Python AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        imports: List[str] = []
        from_imports: List[str] = []
        classes: List[ClassDefInfo] = []
        functions: List[FunctionDefInfo] = []
        route_decorators: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_imports.append(f"{module}.{alias.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_info = self._extract_function_info(node)
                functions.append(fn_info)
                for dec in fn_info.decorators:
                    if any(kw in dec.lower() for kw in ("get", "post", "put", "delete", "route", "api")):
                        route_decorators.append(f"{fn_info.name} -> @{dec}")
            elif isinstance(node, ast.ClassDef):
                cls_info = self._extract_class_info(node)
                classes.append(cls_info)
                for m in cls_info.methods:
                    for dec in m.decorators:
                        if any(kw in dec.lower() for kw in ("get", "post", "put", "delete", "route", "api")):
                            route_decorators.append(f"{cls_info.name}.{m.name} -> @{dec}")

        return FileAstInfo(
            imports=imports,
            from_imports=from_imports,
            classes=classes,
            functions=functions,
            route_decorators=route_decorators,
        )

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionDefInfo:
        decorators = []
        for dec in node.decorator_list:
            decorators.append(ast.unparse(dec))

        args = [arg.arg for arg in node.args.args]
        docstring = ast.get_docstring(node)

        return FunctionDefInfo(
            name=node.name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            args=args,
            decorators=decorators,
            docstring=docstring,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )

    def _extract_class_info(self, node: ast.ClassDef) -> ClassDefInfo:
        bases = [ast.unparse(base) for base in node.bases]
        docstring = ast.get_docstring(node)
        methods = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function_info(item))

        return ClassDefInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            docstring=docstring,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )

    def _generate_tree_repr(self, files: List[ScannedFile], max_lines: int = 150) -> str:
        """Create a compact tree representation of repository structure."""
        paths = sorted([f.path for f in files])
        lines = [f"{self.root_dir.name}/"]
        for p in paths[:max_lines]:
            parts = p.split("/")
            indent = "  " * len(parts)
            lines.append(f"{indent}├── {parts[-1]}")
        if len(paths) > max_lines:
            lines.append(f"  ... [and {len(paths) - max_lines} more files]")
        return "\n".join(lines)
