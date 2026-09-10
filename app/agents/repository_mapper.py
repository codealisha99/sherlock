"""Agent 1: Repository Mapper.

Builds the structural map of the repository: folder hierarchy, languages,
frameworks, entry points, configuration, and architectural layers.
"""

from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import RepositoryMap, DirectoryInfo


class RepositoryMapperAgent(BaseAgent):
    """Agent responsible for mapping repository structure and technology stack."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Repository Mapper",
            description="Analyzes folder hierarchy, file types, entry points, and architectural layers.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Repository Mapper Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your responsibility is to build the structural map of an unfamiliar repository.\n"
            "You must identify:\n"
            "- What this project is and its primary purpose.\n"
            "- The technology stack (languages, frameworks, libraries).\n"
            "- Where execution begins (entry points).\n"
            "- Major components and architectural layers.\n"
            "- The purpose of each major directory.\n"
            "- Configuration files that control behavior.\n"
            "Output strictly valid JSON matching the RepositoryMap schema.\n"
            "Be precise, objective, and reference actual directory names and file paths."
        )

    def run(self, ctx: AnalysisContext) -> RepositoryMap:
        """Execute repository mapping."""
        if not ctx.index:
            ctx.initialize_index()

        index = ctx.index
        # Gather context
        prompt = self._build_prompt(ctx)

        result = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=RepositoryMap,
            fallback_factory=self._synthesize_from_index,
            ctx=ctx,
        )

        ctx.repository_map = result
        return result

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        index = ctx.index
        files_sample = [f"{f.path} ({f.language}, {f.size_bytes}B)" for f in index.files[:100]]

        # Read dependency files if available
        deps_content = []
        for dep_file in index.dependency_files[:3]:
            content, _ = ctx.fs.read_file(dep_file, max_bytes=3000)
            deps_content.append(f"--- {dep_file} ---\n{content}\n")

        # Read config files sample
        cfg_content = []
        for cfg_file in index.config_files[:3]:
            content, _ = ctx.fs.read_file(cfg_file, max_bytes=2000)
            cfg_content.append(f"--- {cfg_file} ---\n{content}\n")

        return (
            f"Project Name: {index.project_name}\n"
            f"Total Files: {index.total_files}, Total Lines: {index.total_lines}\n"
            f"Languages Detected: {list(index.languages.keys())}\n"
            f"Candidate Entry Points: {index.entry_points}\n"
            f"Candidate Config Files: {index.config_files}\n"
            f"Candidate Dependency Files: {index.dependency_files}\n\n"
            f"Directory Tree:\n{index.directory_tree_repr}\n\n"
            f"Files Sample:\n" + "\n".join(files_sample) + "\n\n"
            f"Dependencies Snippets:\n" + "\n".join(deps_content) + "\n\n"
            f"Config Snippets:\n" + "\n".join(cfg_content) + "\n\n"
            f"Analyze this codebase and return the RepositoryMap JSON object."
        )

    def _synthesize_from_index(self, ctx: AnalysisContext) -> RepositoryMap:
        """Fallback static synthesis when LLM is offline."""
        index = ctx.index
        top_languages = sorted(index.languages.items(), key=lambda x: x[1], reverse=True)
        langs = [lang for lang, _ in top_languages if lang not in ("Other", "Binary")] or ["Python"]

        # Detect frameworks from dependencies & imports
        frameworks = set()
        deps = []
        for dep_file in index.dependency_files:
            content, _ = ctx.fs.read_file(dep_file, max_bytes=5000)
            for line in content.splitlines():
                line = line.strip().split("==")[0].split(">=")[0].strip()
                if line and not line.startswith("#"):
                    deps.append(line)
                    lower = line.lower()
                    if "fastapi" in lower:
                        frameworks.add("FastAPI")
                    if "flask" in lower:
                        frameworks.add("Flask")
                    if "django" in lower:
                        frameworks.add("Django")
                    if "ollama" in lower:
                        frameworks.add("Ollama")
                    if "pydantic" in lower:
                        frameworks.add("Pydantic")
                    if "react" in lower:
                        frameworks.add("React")
                    if "torch" in lower or "pytorch" in lower:
                        frameworks.add("PyTorch")

        # Fallback frameworks
        if not frameworks:
            frameworks.add("Standard Python / Modular Architecture")

        # Build directory info
        dirs_info: List[DirectoryInfo] = []
        for d in index.directories[:15]:
            matching_files = [f.path for f in index.files if f.path.startswith(f"{d}/")][:5]
            purpose = f"Module containing {d.split('/')[-1]} logic and related components."
            if "tool" in d:
                purpose = "Utility tools and helper routines."
            elif "agent" in d:
                purpose = "Autonomous and specialized agent implementations."
            elif "llm" in d:
                purpose = "Language model integrations and client wrappers."
            elif "api" in d:
                purpose = "API routing and service endpoints."
            elif "test" in d:
                purpose = "Automated test suites and verification cases."
            elif "frontend" in d:
                purpose = "User interface and presentation layer."
            dirs_info.append(DirectoryInfo(path=d, purpose=purpose, key_files=matching_files))

        # Major components
        major_components = []
        for f in index.files:
            if f.ast_info:
                for cls in f.ast_info.classes:
                    major_components.append(f"{cls.name} ({f.path})")
                for fn in f.ast_info.functions:
                    if fn.decorators or fn.name in ("main", "run", "serve"):
                        major_components.append(f"{fn.name}() ({f.path})")

        if not major_components:
            major_components = [f.path for f in index.files if not f.is_binary][:10]

        layers = [
            "Entry / CLI Layer",
            "API / Controller Layer",
            "Orchestration & Agent Pipeline Layer",
            "Domain / Analysis Engine Layer",
            "Data & Integration Layer",
            "Presentation / Frontend Layer",
        ]

        summary = (
            f"{index.project_name} is a software system written in {', '.join(langs[:3])} "
            f"comprising {index.total_files} files across {len(index.directories)} directories. "
            f"It utilizes {', '.join(sorted(list(frameworks))[:4])} with structured modular layers."
        )

        return RepositoryMap(
            project_name=index.project_name,
            project_summary=summary,
            languages=langs,
            frameworks=sorted(list(frameworks)),
            entry_points=index.entry_points or ["app/main.py"],
            directories=dirs_info,
            major_components=major_components[:20],
            configuration=index.config_files,
            dependencies=deps[:25],
            architectural_layers=layers,
            directory_tree=index.directory_tree_repr,
        )
