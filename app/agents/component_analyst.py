"""Agent 3: Component Analyst.

Examines key files, classes, and functions to determine their concrete
responsibilities, inputs/outputs, caller relationships, side effects,
assumptions, and failure modes.
"""

from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import ComponentAnalysis, ComponentDetail


class ComponentAnalystAgent(BaseAgent):
    """Agent that analyzes internal behavior and contracts of major components."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Component Analyst",
            description="Examines actual behaviors, dependencies, side effects, and failure modes of core components.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Component Analyst Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your responsibility is to understand what important components ACTUALLY DO, not merely what they are named.\n"
            "For every major class or function, determine:\n"
            "- Concrete responsibility (transforms, coordinates, validates, renders)\n"
            "- Inputs and outputs\n"
            "- Callees (what it calls) and callers (who calls it)\n"
            "- Side effects (filesystem writes, network requests, state mutation)\n"
            "- Architectural assumptions (e.g. Ollama running, UTF-8 encoding, synchronous execution)\n"
            "- Failure modes (exceptions, timeouts, malformed payloads)\n"
            "Output strictly valid JSON matching the ComponentAnalysis schema."
        )

    def run(self, ctx: AnalysisContext) -> ComponentAnalysis:
        """Execute component analysis."""
        prompt = self._build_prompt(ctx)

        result = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=ComponentAnalysis,
            fallback_factory=self._synthesize_components,
            ctx=ctx,
        )

        ctx.component_analysis = result
        return result

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        index = ctx.index
        # Gather top classes and functions from AST
        component_summaries = []
        for f in index.files:
            if f.ast_info:
                for cls in f.ast_info.classes:
                    methods = [m.name for m in cls.methods]
                    component_summaries.append(
                        f"Class: {cls.name} in {f.path} (bases: {cls.bases}, methods: {methods}, doc: {cls.docstring})"
                    )
                for fn in f.ast_info.functions:
                    if fn.decorators or fn.name in ("main", "run", "scan", "generate"):
                        component_summaries.append(
                            f"Function: {fn.name}({', '.join(fn.args)}) in {f.path} (decorators: {fn.decorators})"
                        )

        # Grab code snippets of top 3 files
        sample_code = []
        for f in index.files[:5]:
            if not f.is_binary and f.extension in (".py", ".js", ".ts"):
                content, _ = ctx.fs.read_file(f.path, max_bytes=2500)
                sample_code.append(f"=== {f.path} ===\n{content}\n")

        return (
            f"Project: {index.project_name}\n"
            f"AST Extracted Components:\n" + "\n".join(component_summaries[:25]) + "\n\n"
            f"Sample Code:\n" + "\n".join(sample_code) + "\n\n"
            f"Analyze the major components in detail and return the ComponentAnalysis JSON object."
        )

    def _synthesize_components(self, ctx: AnalysisContext) -> ComponentAnalysis:
        """Fallback static synthesis from AST data."""
        index = ctx.index
        components: List[ComponentDetail] = []

        for f in index.files:
            if not f.ast_info:
                continue

            for cls in f.ast_info.classes:
                method_names = [m.name for m in cls.methods]
                doc = cls.docstring or f"Component responsible for {cls.name.lower()} operations."

                # Heuristic calls & callers
                calls = [f"{m}()" for m in method_names if not m.startswith("_")]
                side_effects = []
                assumptions = ["Input parameters adhere to expected types", "Runtime environment matches requirements"]
                failure_modes = ["Unhandled TypeError or ValueError", "Resource unavailability"]

                name_lower = cls.name.lower()
                if "scanner" in name_lower or "filesys" in name_lower:
                    side_effects.append("Reads filesystem contents and metadata")
                    assumptions.append("Target directory has read permissions and accessible paths")
                    failure_modes.append("PermissionError on restricted folders")
                elif "client" in name_lower or "llm" in name_lower:
                    side_effects.append("Performs outbound HTTP requests to Ollama API")
                    assumptions.append("Local Ollama server is running at configured host/port")
                    failure_modes.append("httpx.ConnectError / Connection refused")
                    failure_modes.append("Ollama timeout on complex prompts")
                elif "orchestrator" in name_lower:
                    side_effects.append("Coordinates pipeline and triggers disk persistence of artifacts")
                    assumptions.append("Pipeline agents execute sequentially and yield valid models")
                    failure_modes.append("Agent exception halts pipeline progression")
                elif "agent" in name_lower:
                    side_effects.append("Consumes LLM tokens and populates AnalysisContext")
                    assumptions.append("AnalysisContext has valid RepositoryIndex")
                    failure_modes.append("LLM output validation failure")

                components.append(
                    ComponentDetail(
                        component=cls.name,
                        file=f.path,
                        responsibility=doc,
                        inputs=[f"{m}(args)" for m in method_names[:3]] or ["Context / Config"],
                        outputs=["Structured results / Pydantic models / State updates"],
                        calls=calls[:5] or ["Internal helper methods"],
                        called_by=["Orchestrator / Main Entrypoint / Base classes"],
                        side_effects=side_effects or ["In-memory state manipulation"],
                        assumptions=assumptions,
                        failure_modes=failure_modes,
                    )
                )

        # If no AST classes found (e.g. non-python repo), synthesize from files
        if not components:
            for f in index.files[:8]:
                if not f.is_binary:
                    components.append(
                        ComponentDetail(
                            component=f.name,
                            file=f.path,
                            responsibility=f"Source module providing logic for {f.name}.",
                            inputs=["Module import arguments / parameters"],
                            outputs=["Exports / functions"],
                            calls=[],
                            called_by=["Parent modules"],
                            side_effects=["Execution of top-level code"],
                            assumptions=["Dependencies are satisfied"],
                            failure_modes=["ImportError", "Runtime execution error"],
                        )
                    )

        # Generate Mermaid graph of components
        mermaid_nodes = []
        for c in components[:12]:
            safe_id = "".join(ch for ch in c.component if ch.isalnum())
            mermaid_nodes.append(f"  {safe_id}[\"{c.component}<br/><small>{c.file}</small>\"]")

        mermaid_edges = []
        for i in range(min(len(components) - 1, 8)):
            id_a = "".join(ch for ch in components[i].component if ch.isalnum())
            id_b = "".join(ch for ch in components[i+1].component if ch.isalnum())
            mermaid_edges.append(f"  {id_a} --> {id_b}")

        mermaid_chart = "graph TD;\n" + "\n".join(mermaid_nodes) + "\n" + "\n".join(mermaid_edges)

        return ComponentAnalysis(
            components=components,
            component_graph_mermaid=mermaid_chart,
            summary=f"Analyzed {len(components)} core components with their responsibilities, inputs, outputs, side effects, and failure conditions.",
        )
