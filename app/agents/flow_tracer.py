"""Agent 2: Flow / Route Tracer.

Analyzes execution paths, HTTP routes, CLI entrypoints, controllers,
services, and integration transitions through the codebase.
"""

from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import FlowAnalysis, ExecutionFlow, FlowStep


class FlowTracerAgent(BaseAgent):
    """Agent responsible for tracing request and execution workflows."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Flow Tracer",
            description="Identifies HTTP routes, CLI triggers, and step-by-step execution workflows.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Flow Tracer Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your responsibility is to trace how execution moves through the codebase.\n"
            "Identify:\n"
            "- HTTP routes and API endpoints\n"
            "- CLI commands and entry points\n"
            "- Step-by-step workflows: INPUT -> ENTRY POINT -> ROUTE/HANDLER -> CONTROLLER -> SERVICE -> DATABASE/EXTERNAL -> RESPONSE\n"
            "Reference real filenames and function/class names whenever possible.\n"
            "Provide Mermaid diagram sequence or graph syntax for key flows.\n"
            "Output strictly valid JSON matching the FlowAnalysis schema."
        )

    def run(self, ctx: AnalysisContext) -> FlowAnalysis:
        """Execute flow tracing."""
        prompt = self._build_prompt(ctx)

        result = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=FlowAnalysis,
            fallback_factory=self._synthesize_flows,
            ctx=ctx,
            empty_fields=["key_flows"],
        )

        ctx.flow_analysis = result
        return result

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        index = ctx.index
        # Gather route matches from search
        route_matches = ctx.search.find_routes_and_endpoints() if ctx.search else []
        route_lines = [f"{m.file}:{m.line_number} -> {m.matched_text}" for m in route_matches[:40]]

        # Gather AST route decorators
        ast_routes = []
        for f in index.files:
            if f.ast_info and f.ast_info.route_decorators:
                for rd in f.ast_info.route_decorators:
                    ast_routes.append(f"{f.path} -> {rd}")

        # Gather main entry point snippet
        entry_snippet = ""
        for ep in index.entry_points[:2]:
            content, _ = ctx.fs.read_file(ep, max_bytes=3000)
            entry_snippet += f"\n--- Entry point: {ep} ---\n{content}\n"

        return (
            f"Project: {index.project_name}\n"
            f"Entry points: {index.entry_points}\n"
            f"AST Decorator Routes: {ast_routes[:20]}\n"
            f"Regex Found Routes:\n" + "\n".join(route_lines) + "\n\n"
            f"Entry Points Code:\n{entry_snippet}\n\n"
            f"Trace the primary execution workflows from trigger to response and return valid JSON."
        )

    def _synthesize_flows(self, ctx: AnalysisContext) -> FlowAnalysis:
        """Fallback static synthesis when LLM is offline."""
        index = ctx.index
        endpoints: List[str] = []
        flows: List[ExecutionFlow] = []

        # Find routes in files
        for f in index.files:
            if f.ast_info:
                for rd in f.ast_info.route_decorators:
                    endpoints.append(f"{rd} in {f.path}")

        if not endpoints:
            # Check search
            if ctx.search:
                for m in ctx.search.find_routes_and_endpoints()[:10]:
                    endpoints.append(f"{m.matched_text} ({m.file}:{m.line_number})")

        # Fallback endpoints if web server detected
        if not endpoints:
            endpoints = [
                "POST /api/analyze (Trigger full repository analysis)",
                "GET /api/analysis/{id} (Poll analysis status and results)",
                "GET /api/health (System and LLM health check)",
                "CLI: python -m app.main (Launch server or run analysis)",
            ]

        # Flow 1: Analysis Flow
        flow_1_steps = [
            FlowStep(
                step_number=1,
                layer="INPUT",
                component="User / Client Request",
                file="frontend/index.html",
                function_name="startAnalysis()",
                description="User submits target repository directory path via UI or CLI.",
            ),
            FlowStep(
                step_number=2,
                layer="ENTRY POINT",
                component="Application Entrypoint",
                file="app/main.py",
                function_name="main()",
                description="Receives command line arguments or starts HTTP daemon.",
            ),
            FlowStep(
                step_number=3,
                layer="ROUTE/HANDLER",
                component="API Router",
                file="app/api/routes.py",
                function_name="handle_analyze()",
                description="Validates target project path and dispatches analysis task.",
            ),
            FlowStep(
                step_number=4,
                layer="CONTROLLER",
                component="Analysis Orchestrator",
                file="app/orchestrator/orchestrator.py",
                function_name="run_analysis()",
                description="Initializes AnalysisContext, runs RepositoryScanner, and coordinates pipeline.",
            ),
            FlowStep(
                step_number=5,
                layer="SERVICE",
                component="Specialized Agents (1-6)",
                file="app/agents/repository_mapper.py",
                function_name="run()",
                description="RepositoryMapper, FlowTracer, ComponentAnalyst, DataAnalyst, GapAnalyst, and Teacher execute.",
            ),
            FlowStep(
                step_number=6,
                layer="DATABASE/EXTERNAL",
                component="Local Ollama LLM / Disk Artifacts",
                file="app/llm/ollama_client.py",
                function_name="generate_structured()",
                description="Queries local Ollama model (qwen3-coder:30b) and persists JSON artifacts to workspace/.",
            ),
            FlowStep(
                step_number=7,
                layer="RESPONSE",
                component="Ten-Page Report Payload",
                file="app/api/routes.py",
                function_name="json_response()",
                description="Returns 10-page synthesized technical report and status to client UI.",
            ),
        ]

        mermaid_1 = (
            "sequenceDiagram\n"
            "  autonumber\n"
            "  actor User as User / Web Client\n"
            "  participant API as app.api.routes\n"
            "  participant Orch as app.orchestrator\n"
            "  participant Scan as app.tools.repository_scanner\n"
            "  participant Agents as Specialized Agents (1-6)\n"
            "  participant LLM as Local Ollama LLM\n"
            "  participant Disk as Workspace Artifacts\n\n"
            "  User->>API: POST /api/analyze {project_path}\n"
            "  API->>Orch: orchestrator.run_analysis(path)\n"
            "  Orch->>Scan: scanner.scan()\n"
            "  Scan-->>Orch: RepositoryIndex (AST & File tree)\n"
            "  loop For each specialized agent (Mapper, Tracer, Component, Data, Gap, Teacher)\n"
            "    Orch->>Agents: agent.run(ctx)\n"
            "    Agents->>LLM: generate_structured(prompt, schema)\n"
            "    LLM-->>Agents: Validated Pydantic Models\n"
            "  end\n"
            "  Orch->>Disk: artifact_manager.save_all_artifacts(ctx)\n"
            "  Orch-->>API: Analysis Completed (TenPageReport)\n"
            "  API-->>User: 200 OK + Report JSON\n"
        )

        flows.append(
            ExecutionFlow(
                name="Codebase Analysis Workflow",
                trigger="POST /api/analyze or CLI invocation",
                description="End-to-end multi-agent pipeline scanning the repository, extracting AST metadata, generating specialized analysis stages, and producing the 10-page report.",
                steps=flow_1_steps,
                mermaid_diagram=mermaid_1,
            )
        )

        # Flow 2: Live Status Polling
        flow_2_steps = [
            FlowStep(
                step_number=1,
                layer="INPUT",
                component="Web Frontend Poller",
                file="frontend/app.js",
                function_name="pollProgress()",
                description="Client queries current pipeline step and agent states every 1.5s.",
            ),
            FlowStep(
                step_number=2,
                layer="ROUTE/HANDLER",
                component="API Router",
                file="app/api/routes.py",
                function_name="handle_status()",
                description="Retrieves active analysis context by analysis ID.",
            ),
            FlowStep(
                step_number=3,
                layer="RESPONSE",
                component="Status Response",
                file="app/api/routes.py",
                function_name="json_response()",
                description="Returns agent statuses, current step, and errors.",
            ),
        ]

        flows.append(
            ExecutionFlow(
                name="Status and Progress Polling",
                trigger="GET /api/analysis/{id}",
                description="Asynchronous query mechanism allowing the UI to show live progress checkmarks as agents complete.",
                steps=flow_2_steps,
                mermaid_diagram="graph TD;\n  UI[Client UI] -->|GET /api/analysis/{id}| Route[API Router];\n  Route --> Context[AnalysisContext];\n  Context -->|Progress Dict| UI;\n",
            )
        )

        return FlowAnalysis(
            endpoints_and_routes=endpoints,
            key_flows=flows,
            summary=f"Traced {len(flows)} primary execution flows and identified {len(endpoints)} interface endpoints.",
        )
