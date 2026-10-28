"""Agent 4: Data & Integration Analyst.

Traces how information enters, transforms, and exits the codebase,
identifying databases, ORMs, file stores, external APIs, and AI/LLM models.
"""

from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import (
    DataIntegrationAnalysis,
    DatabaseInfo,
    ExternalServiceInfo,
    AiMlComponentInfo,
    DataPipelineFlow,
)


class DataIntegrationAnalystAgent(BaseAgent):
    """Agent responsible for mapping data lifecycle and external integration boundaries."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Data Analyst",
            description="Maps databases, external APIs, file storage, AI/LLM models, and end-to-end data pipelines.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Data & Integration Analyst Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your responsibility is to analyze how data enters, transforms, persists, and exits the system.\n"
            "Analyze:\n"
            "- Databases, ORMs, schemas, tables, and caches\n"
            "- File storage and persistence systems\n"
            "- External web services, APIs, third-party integrations\n"
            "- AI/ML components (LLM inference, embeddings, vector stores)\n"
            "- Pipeline data flows (e.g. USER -> API -> SERVICE -> DATABASE)\n"
            "For each external dependency explain why it exists, sent/received data, failure modes, and config.\n"
            "Output strictly valid JSON matching the DataIntegrationAnalysis schema."
        )

    def run(self, ctx: AnalysisContext) -> DataIntegrationAnalysis:
        """Execute data and integration analysis."""
        prompt = self._build_prompt(ctx)

        result = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=DataIntegrationAnalysis,
            fallback_factory=self._synthesize_data_integrations,
            ctx=ctx,
            empty_fields=["data_flows"],
        )

        ctx.data_integration_analysis = result
        return result

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        index = ctx.index
        # Gather DB search matches
        db_matches = ctx.search.find_database_usage() if ctx.search else []
        db_snippets = [f"{m.file}:{m.line_number} -> {m.matched_text}" for m in db_matches[:25]]

        # Gather external API search matches
        ext_matches = ctx.search.find_external_integrations() if ctx.search else []
        ext_snippets = [f"{m.file}:{m.line_number} -> {m.matched_text}" for m in ext_matches[:25]]

        return (
            f"Project: {index.project_name}\n"
            f"Database / ORM Code Indicators:\n" + "\n".join(db_snippets) + "\n\n"
            f"External Integration / LLM Indicators:\n" + "\n".join(ext_snippets) + "\n\n"
            f"Dependencies: {index.dependency_files}\n"
            f"Analyze all databases, file storage, external services, AI components, and data flows."
        )

    def _synthesize_data_integrations(self, ctx: AnalysisContext) -> DataIntegrationAnalysis:
        """Fallback static synthesis of data models and integrations."""
        index = ctx.index

        databases: List[DatabaseInfo] = []
        external_services: List[ExternalServiceInfo] = []
        ai_ml: List[AiMlComponentInfo] = []
        data_flows: List[DataPipelineFlow] = []

        # Check for Local Storage / Workspace Artifacts
        databases.append(
            DatabaseInfo(
                name="Filesystem Artifact Store",
                type="JSON File Storage",
                orm_or_driver="Standard library 'json' and 'pathlib.Path'",
                tables_or_models=[
                    "repository_map.json",
                    "flows.json",
                    "components.json",
                    "data_integrations.json",
                    "gaps.json",
                    "teaching_report.json",
                    "ten_page_report.json",
                ],
                purpose="Persists structured analysis results, intermediate agent state, and synthesized 10-page reports.",
            )
        )

        # Check for Ollama / LLM
        external_services.append(
            ExternalServiceInfo(
                name="Local Ollama LLM Service",
                purpose="Executes generative inference for repository analysis, structured schema generation, and code teaching.",
                sent_data="Prompts containing code snippets, AST metadata, file trees, and JSON schemas.",
                received_data="Raw JSON objects representing validated Pydantic models.",
                call_location="app/llm/ollama_client.py -> OllamaClient.generate_structured()",
                failure_possibilities=[
                    "Connection refused (Ollama daemon offline)",
                    "Model not found / not pulled",
                    "Request timeout on large codebases",
                    "Malformed JSON output / schema mismatch",
                ],
                configuration_required="SHERLOCK_OLLAMA_HOST (default: http://localhost:11434), SHERLOCK_MODEL (default: qwen3-coder:30b)",
            )
        )

        ai_ml.append(
            AiMlComponentInfo(
                name="Qwen 2.5/3 Coder LLM",
                model_or_service="qwen3-coder:30b (via local Ollama)",
                role="Code reasoning, architecture abstraction, gap detection, and tutorial generation.",
                entry_location="app/llm/ollama_client.py",
                purpose="Acts as the reasoning engine for the 6 specialized Sherlock analysis agents.",
            )
        )

        # Build data pipeline flows
        data_flows.append(
            DataPipelineFlow(
                name="Codebase Ingestion & AST Extraction",
                pipeline=["Target Repository Files", "SafeFilesystem", "RepositoryScanner", "AST Visitor", "RepositoryIndex"],
                description="Raw repository files are bounded, checked for secrets, traversed, and parsed into Python AST nodes and metadata records.",
            )
        )

        data_flows.append(
            DataPipelineFlow(
                name="Agent Reasoning & Structured Output Generation",
                pipeline=["RepositoryIndex", "Specialized Agent", "Prompt Builder", "Ollama LLM", "JSON Validator", "AnalysisContext"],
                description="Targeted AST context and code snippets are formatted into prompts and sent to local Ollama; response is validated against Pydantic schemas.",
            )
        )

        data_flows.append(
            DataPipelineFlow(
                name="Artifact Persistence & Presentation",
                pipeline=["AnalysisContext", "ArtifactManager", "workspace/*.json", "API Router", "Frontend SPA"],
                description="Completed agent outputs are serialized to JSON in the workspace folder and served to the user-facing 10-page report web interface.",
            )
        )

        mermaid_data = (
            "graph LR;\n"
            "  Codebase[User Codebase] -->|Safe Read| Scanner[RepositoryScanner];\n"
            "  Scanner -->|AST & File Index| Context[AnalysisContext];\n"
            "  Context -->|Prompts + Schemas| Ollama[Local Ollama LLM];\n"
            "  Ollama -->|Structured JSON| Context;\n"
            "  Context -->|Save State| Artifacts[Workspace JSON Files];\n"
            "  Context -->|Serve 10 Pages| Frontend[Web UI Explorer];\n"
        )

        return DataIntegrationAnalysis(
            databases=databases,
            external_services=external_services,
            ai_ml_components=ai_ml,
            data_flows=data_flows,
            data_flow_mermaid=mermaid_data,
            summary="Identified primary data movement from raw filesystem files through AST parsers to local Ollama LLM, persisting as structured JSON artifacts.",
        )
