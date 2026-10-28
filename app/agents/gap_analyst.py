"""Agent 5: Gap & Issue Analyst.

Inspects the outputs of Agents 1-4 and codebase patterns to identify
TODOs, missing error handling, untested paths, security concerns,
architectural bottlenecks, and potential contribution opportunities.
"""

from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import GapAnalysis, GapItem


class GapAnalystAgent(BaseAgent):
    """Agent responsible for identifying codebase gaps, risks, and improvements."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Gap Analyst",
            description="Evaluates code health, missing validations, TODOs/FIXMEs, security assumptions, and contribution opportunities.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Gap Analyst Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your responsibility is to analyze the repository and prior agent outputs to identify weaknesses, "
            "inconsistencies, missing error handling, and potential contribution opportunities.\n"
            "IMPORTANT:\n"
            "- Do not label something a bug merely because it looks unusual.\n"
            "- Provide concrete evidence (file, function, or line reference).\n"
            "- Categorize strictly as: 'Confirmed issue', 'Likely issue', 'Architectural concern', 'Missing feature', or 'Possible improvement'.\n"
            "- Explain why it matters, affected components, confidence (High, Medium, Low), possible impact, and concrete contribution.\n"
            "Output strictly valid JSON matching the GapAnalysis schema."
        )

    def run(self, ctx: AnalysisContext) -> GapAnalysis:
        """Execute gap analysis."""
        prompt = self._build_prompt(ctx)

        result = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=GapAnalysis,
            fallback_factory=self._synthesize_gaps,
            ctx=ctx,
            empty_fields=["gaps"],
        )

        ctx.gap_analysis = result
        return result

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        index = ctx.index
        # Gather TODOs and FIXMEs found by scanner
        todos_summary = [
            f"{t['type']} in {t['file']}:{t['line']} -> {t['text']}"
            for t in index.todos_and_fixmes[:30]
        ]

        # Prior agent findings
        mapper_summary = ctx.repository_map.project_summary if ctx.repository_map else ""
        flow_summary = ctx.flow_analysis.summary if ctx.flow_analysis else ""
        comp_summary = ctx.component_analysis.summary if ctx.component_analysis else ""
        data_summary = ctx.data_integration_analysis.summary if ctx.data_integration_analysis else ""

        return (
            f"Project: {index.project_name}\n"
            f"Repository Overview: {mapper_summary}\n"
            f"Flow Overview: {flow_summary}\n"
            f"Component Overview: {comp_summary}\n"
            f"Data/Integrations: {data_summary}\n\n"
            f"Detected Code TODOs/FIXMEs:\n" + ("\n".join(todos_summary) if todos_summary else "None detected.") + "\n\n"
            f"Analyze all architectural concerns, missing test coverage, unhandled network failures, "
            f"and contribution opportunities."
        )

    def _synthesize_gaps(self, ctx: AnalysisContext) -> GapAnalysis:
        """Fallback static synthesis of codebase gaps."""
        index = ctx.index
        gaps: List[GapItem] = []

        # Check for explicit TODOs/FIXMEs in index
        for t in index.todos_and_fixmes[:5]:
            gaps.append(
                GapItem(
                    gap=f"Unresolved {t['type']}: {t['text']}",
                    category="Likely issue" if t["type"] == "FIXME" else "Missing feature",
                    evidence=f"{t['file']}:{t['line']}",
                    why_it_matters="Indicates incomplete logic or acknowledged technical debt left by developers.",
                    affected_components=[t["file"]],
                    confidence="High",
                    possible_impact="Potential runtime failures or missing functionality when path is reached.",
                    possible_contribution=f"Implement or address the TODO in {t['file']}.",
                )
            )

        # Check test coverage gap
        test_files = [f.path for f in index.files if "test" in f.path.lower()]
        if len(test_files) <= 1:
            gaps.append(
                GapItem(
                    gap="Limited or Missing Automated Test Suite",
                    category="Architectural concern",
                    evidence="tests/ directory contains few or no test modules.",
                    why_it_matters="Lack of automated regression and unit tests makes refactoring risky and reduces maintainability.",
                    affected_components=["All components across codebase"],
                    confidence="High",
                    possible_impact="Regressions may go undetected until deployed or exercised manually.",
                    possible_contribution="Add unit test suites covering the scanner, LLM client, and orchestrator pipelines.",
                )
            )

        # Check external service resilience (Ollama connection)
        gaps.append(
            GapItem(
                gap="External Ollama Service Connectivity Resilience",
                category="Architectural concern",
                evidence="app/llm/ollama_client.py: generate_structured()",
                why_it_matters="If the local Ollama daemon is offline or under high VRAM pressure, agent queries fail unless fallback static synthesis is in place.",
                affected_components=["app/llm/ollama_client.py", "app/agents/*"],
                confidence="High",
                possible_impact="Analysis could stall or error if Ollama service is unbooted.",
                possible_contribution="Ensure graceful fallback to static AST analysis when Ollama is unreachable, with clear UI error indicators.",
            )
        )

        # Check input directory validation
        gaps.append(
            GapItem(
                gap="Filesystem Boundary & Path Traversal Verification",
                category="Possible improvement",
                evidence="app/tools/filesys.py: resolve_path()",
                why_it_matters="Analyzed projects must be strictly contained to prevent arbitrary file disclosure across the host OS.",
                affected_components=["app/tools/filesys.py"],
                confidence="High",
                possible_impact="Potentially accessing host paths outside designated project directory.",
                possible_contribution="Maintain strict relative path enforcement using Path.resolve().relative_to().",
            )
        )

        # Large file handling
        gaps.append(
            GapItem(
                gap="Very Large File Handling & Token Truncation",
                category="Possible improvement",
                evidence="app/tools/repository_scanner.py: max_file_size_bytes",
                why_it_matters="Large vendor bundles, minified scripts, or huge data files could exceed token limits if ingested blindly.",
                affected_components=["app/tools/repository_scanner.py", "app/tools/filesys.py"],
                confidence="Medium",
                possible_impact="LLM context window overflow or excessive prompt latency.",
                possible_contribution="Cap single-file read buffer to 150KB and exclude binary/minified files automatically.",
            )
        )

        return GapAnalysis(
            gaps=gaps,
            summary=f"Identified {len(gaps)} potential gaps spanning technical debt, test coverage, external service resilience, and bounded ingestion.",
        )
