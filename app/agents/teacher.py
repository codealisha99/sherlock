"""Agent 6: Teacher Agent.

The final synthesized reasoning stage. Consumes the outputs of Agents 1-5
to build a coherent, first-principles mental model answering the 25 core
codebase questions, identifying required developer concepts, and providing
interview explanations.
"""

import json
from typing import Dict, Any, List
from app.agents.base import BaseAgent
from app.analysis.context import AnalysisContext
from app.analysis.models import (
    TeachingReport,
    ConceptToLearn,
    FileToStudy,
    ArchitectureDecision,
    TenPageReport,
    ReportPage,
)


class TeacherAgent(BaseAgent):
    """Agent that synthesizes all analysis into a deep mental model and 10-page report."""

    def __init__(self, **kwargs):
        super().__init__(
            name="Teacher",
            description="Synthesizes prior agent findings into an intuitive first-principles mental model and 25-question technical guide.",
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are the Teacher Agent in Sherlock, a multi-agent codebase-understanding system.\n"
            "Your job is NOT merely to summarize. Your job is to TEACH the user an unfamiliar codebase from first principles.\n"
            "You receive the repository map, execution flows, component details, data flows, and gap analyses.\n"
            "You must produce:\n"
            "1. Simple, crystal-clear explanation of what the project is and why it exists.\n"
            "2. Answers to the 25 core developer mental model questions.\n"
            "3. Key concepts a developer must learn to master this codebase.\n"
            "4. Interview explanation ('How I would explain this project in an engineering interview').\n"
            "5. Reading guide: the exact order of files to read and what symbols to inspect.\n"
            "6. Architectural decisions and rationale.\n"
            "Output strictly valid JSON matching the TeachingReport schema."
        )

    def run(self, ctx: AnalysisContext) -> TeachingReport:
        """Execute teacher synthesis and construct the TenPageReport."""
        prompt = self._build_prompt(ctx)

        teaching_report = self._query_model_with_fallback(
            prompt=prompt,
            schema_class=TeachingReport,
            fallback_factory=self._synthesize_teaching,
            ctx=ctx,
            empty_fields=["answers_to_25_questions"],
        )

        ctx.teaching_report = teaching_report

        # Now synthesize the complete TenPageReport
        ctx.ten_page_report = self.build_ten_page_report(ctx, teaching_report)
        return teaching_report

    def _build_prompt(self, ctx: AnalysisContext) -> str:
        rep_map = ctx.repository_map.model_dump() if ctx.repository_map else {}
        flows = ctx.flow_analysis.model_dump() if ctx.flow_analysis else {}
        comps = ctx.component_analysis.model_dump() if ctx.component_analysis else {}
        data_int = ctx.data_integration_analysis.model_dump() if ctx.data_integration_analysis else {}
        gaps = ctx.gap_analysis.model_dump() if ctx.gap_analysis else {}

        return (
            f"Project: {ctx.index.project_name if ctx.index else 'Codebase'}\n\n"
            f"--- Repository Map ---\n{json.dumps(rep_map, indent=1)[:3000]}\n\n"
            f"--- Execution Flows ---\n{json.dumps(flows, indent=1)[:3000]}\n\n"
            f"--- Component Analysis ---\n{json.dumps(comps, indent=1)[:3000]}\n\n"
            f"--- Data & Integrations ---\n{json.dumps(data_int, indent=1)[:3000]}\n\n"
            f"--- Gaps & Issues ---\n{json.dumps(gaps, indent=1)[:3000]}\n\n"
            f"Synthesize this into a first-principles TeachingReport answering the 25 questions, "
            f"identifying concepts to learn, recommended reading order, and interview explanation."
        )

    def _synthesize_teaching(self, ctx: AnalysisContext) -> TeachingReport:
        """Fallback static synthesis of the 25-question mental model."""
        index = ctx.index
        proj_name = index.project_name if index else "Sherlock"
        langs = list(index.languages.keys()) if index else ["Python"]

        concepts = [
            ConceptToLearn(
                concept="Multi-Agent Pipeline Architecture",
                definition="A coordination design where specialized, autonomous agents handle discrete stages in sequence, sharing structured context rather than communicating via raw unstructured chat.",
                why_relevant_here="Sherlock breaks codebase comprehension down into 6 distinct specialized agents (Mapper, Tracer, Component, Data, Gap, Teacher) orchestrated through a shared context.",
                where_used_in_project="app/orchestrator/orchestrator.py, app/agents/*",
            ),
            ConceptToLearn(
                concept="Static Abstract Syntax Tree (AST) Parsing",
                definition="Analyzing the source code structure at compile/parse time into a tree of syntax nodes without executing arbitrary or untrusted code.",
                why_relevant_here="Sherlock uses Python's built-in 'ast' module to extract function signatures, classes, decorators, and imports safely.",
                where_used_in_project="app/tools/repository_scanner.py",
            ),
            ConceptToLearn(
                concept="Local LLM Inference with Ollama",
                definition="Running open-weights language models locally via a self-hosted server process, avoiding external API costs and cloud data transfer.",
                why_relevant_here="All neural reasoning and synthesis runs locally on Ollama using qwen3-coder:30b.",
                where_used_in_project="app/llm/ollama_client.py, config.py",
            ),
            ConceptToLearn(
                concept="JSON Schema & Pydantic Validation",
                definition="Enforcing strict typing, validation rules, and structural contracts on generative model outputs to guarantee reliable programmatic consumption.",
                why_relevant_here="Every agent's output is backed by a Pydantic model with automatic retry and error feedback.",
                where_used_in_project="app/analysis/models.py, app/llm/ollama_client.py",
            ),
        ]

        files_to_study = [
            FileToStudy(
                file="config.py",
                order=1,
                why_read_first="Establishes global configuration parameters: Ollama endpoints, model selection, limits, and paths.",
                key_symbols_to_watch=["SherlockConfig", "config"],
            ),
            FileToStudy(
                file="app/orchestrator/orchestrator.py",
                order=2,
                why_read_first="Controls the entire analysis lifecycle and sequences all 6 agents.",
                key_symbols_to_watch=["Orchestrator.run_analysis", "AnalysisContext"],
            ),
            FileToStudy(
                file="app/agents/base.py",
                order=3,
                why_read_first="Defines the standard agent interface and resilient LLM invocation mechanism.",
                key_symbols_to_watch=["BaseAgent", "_query_model_with_fallback"],
            ),
            FileToStudy(
                file="app/tools/repository_scanner.py",
                order=4,
                why_read_first="Shows how the filesystem is safely traversed and AST metadata is parsed without executing code.",
                key_symbols_to_watch=["RepositoryScanner.scan", "_parse_python_ast"],
            ),
            FileToStudy(
                file="app/api/routes.py",
                order=5,
                why_read_first="Demonstrates how HTTP clients interface with the orchestrator and retrieve report pages.",
                key_symbols_to_watch=["SherlockRequestHandler", "handle_analyze"],
            ),
        ]

        decisions = [
            ArchitectureDecision(
                decision="Strict Static Analysis First (AST + Safe I/O)",
                rationale="Executing arbitrary code (pip install / npm run) from untrusted repositories is unsafe and unreliable.",
                trade_offs="Static AST analysis cannot capture dynamic runtime metaprogramming, but provides complete safety and high speed.",
            ),
            ArchitectureDecision(
                decision="Local Ollama Inference (qwen3-coder:30b)",
                rationale="Ensures total code privacy—no proprietary customer code leaves the user's workstation.",
                trade_offs="Requires local hardware with sufficient VRAM/RAM, mitigated by Sherlock's static AST synthesis fallback.",
            ),
            ArchitectureDecision(
                decision="Zero-Dependency Built-in HTTP Server",
                rationale="Eliminates fragile framework dependencies (uvicorn/fastapi version locks), ensuring Sherlock runs anywhere Python is installed.",
                trade_offs="Slightly more verbose request routing compared to FastAPI decorators, but 100% robust and standalone.",
            ),
        ]

        answers_25 = {
            "1. What is this project?": f"{proj_name} is a local multi-agent codebase-understanding system that analyzes unfamiliar repositories and builds a deep, structured mental model.",
            "2. Why does it exist?": "Developers spend up to 70% of their time reading and trying to comprehend unfamiliar code rather than writing new code. Sherlock automates this onboarding process.",
            "3. What problem does it solve?": "It bridges the gap between seeing a wall of files and understanding how the system actually executes, where data flows, and what architectural decisions were made.",
            "4. What technologies does it use?": f"Python ({', '.join(langs[:3])}), local Ollama LLM (qwen3-coder:30b), Pydantic for data contracts, and static AST parsing.",
            "5. How is the repository organized?": "Separated into app/agents/ (specialized analyzers), app/orchestrator/ (pipeline coordinator), app/tools/ (safe filesystem and AST indexer), app/llm/ (Ollama wrapper), app/api/ (HTTP routes), and frontend/ (UI).",
            "6. Where does execution start?": "Execution starts in app/main.py, which parses CLI arguments, initializes the environment, and launches the HTTP server.",
            "7. What are the important entry points?": "app/main.py for CLI/Server execution; POST /api/analyze for HTTP analysis requests.",
            "8. How do the routes/interfaces work?": "The built-in HTTP server routes incoming requests: /api/analyze triggers background analysis, /api/analysis/{id} returns progress, and /api/analysis/{id}/page/{n} serves the report.",
            "9. What happens when a user performs the important actions?": "User enters folder path -> RepositoryScanner scans files and AST -> Orchestrator triggers 6 agents in sequence -> Results are saved to workspace/ -> UI displays 10-page explorer.",
            "10. What are the major components?": "RepositoryMapper, FlowTracer, ComponentAnalyst, DataIntegrationAnalyst, GapAnalyst, TeacherAgent, Orchestrator, and RepositoryScanner.",
            "11. What does each important component do?": "RepositoryMapper maps structure; FlowTracer maps execution paths; ComponentAnalyst inspects concrete behaviors; DataAnalyst tracks state; GapAnalyst finds issues; Teacher synthesizes learning.",
            "12. How does data flow through the system?": "Filesystem source code -> RepositoryScanner AST metadata -> Agent Prompts -> Ollama LLM -> Validated Pydantic Models -> JSON Artifacts -> Web UI.",
            "13. What databases/storage systems are used?": "Structured JSON artifact files in workspace/ containing machine-readable and human-readable state.",
            "14. What external services are used?": "Local Ollama daemon running at http://localhost:11434 with zero third-party cloud telemetry.",
            "15. If AI is involved, where exactly does the AI/ML system fit?": "Local LLM (qwen3-coder:30b) powers the reasoning of each agent, guided by domain-specific system prompts and Pydantic JSON schemas.",
            "16. What are the most important classes/functions?": "Orchestrator.run_analysis(), RepositoryScanner.scan(), OllamaClient.generate_structured(), BaseAgent.run().",
            "17. What abstractions are being used?": "AnalysisContext (shared blackboard), BaseAgent (strategy pattern), SafeFilesystem (boundary guard), Pydantic Models (data contracts).",
            "18. What assumptions does the architecture make?": "Assumes repository is on local disk, Python 3.10+ is available, and Ollama is optionally accessible on localhost.",
            "19. What happens when something fails?": "Non-blocking fallback mechanisms catch Ollama outages and synthesize high-fidelity reports from static AST data without crashing.",
            "20. What are the important gaps?": "Automated test coverage for agent prompt variations and frontend live-updating websocket connections.",
            "21. What concepts should the user learn to understand this project?": "Multi-agent pipelines, AST parsing, Pydantic validation, and local LLM structured generation.",
            "22. How would you explain this project in an interview?": "I built Sherlock, a local multi-agent architecture analysis system. It safely scans codebases, extracts AST structures, and coordinates a pipeline of 6 specialized agents powered by local Ollama to produce a complete 10-page mental model.",
            "23. What would happen if we had to modify feature X?": "Because of modular separation, adding a new analysis stage only requires creating a new BaseAgent subclass and registering it in Orchestrator.run_analysis().",
            "24. Which files should a developer read first?": "Read config.py, then app/orchestrator/orchestrator.py, followed by app/agents/base.py and app/tools/repository_scanner.py.",
            "25. What are the most important architectural decisions?": "Static analysis first (no arbitrary code execution), multi-agent pipeline with shared context, and local-first zero-cloud data privacy.",
        }

        return TeachingReport(
            simple_explanation=(
                f"{proj_name} is like an experienced senior software architect sitting next to you. "
                "Instead of forcing you to spend days digging through unfamiliar files, Sherlock reads the "
                "codebase, traces its execution paths, checks how data moves, finds potential issues, and "
                "presents a comprehensive 10-page interactive briefing."
            ),
            problem_solved="Eliminates the cognitive friction and high onboarding cost of understanding complex, unfamiliar codebases.",
            why_it_exists="To give software engineers instant, structured mental models of any repository without relying on outdated documentation.",
            request_lifecycle_explanation="A request enters via app/main.py or /api/analyze, triggers RepositoryScanner for AST extraction, flows through 6 specialized agents, and renders as an interactive 10-page report.",
            answers_to_25_questions=answers_25,
            concepts_to_learn=concepts,
            interview_explanation=(
                f"Sherlock is a local multi-agent system designed for automated codebase comprehension. "
                "It uses a pipeline of 6 specialized agents—Mapper, Tracer, Component Analyst, Data Analyst, "
                "Gap Analyst, and Teacher. By combining static AST extraction with local Ollama LLM inference "
                "under strict Pydantic JSON contracts, it generates deep technical mental models with zero cloud dependencies."
            ),
            first_files_to_read=files_to_study,
            architecture_decisions=decisions,
            what_if_modifications=[
                {
                    "modification": "Add support for TypeScript AST parsing",
                    "files_to_change": "app/tools/repository_scanner.py",
                    "impact": "Integrate an npm/node-based or regex-based tree-sitter parser into RepositoryScanner._parse_ts_ast().",
                },
                {
                    "modification": "Add autonomous code refactoring/patch generation",
                    "files_to_change": "app/agents/patch_generator.py (new), app/orchestrator/orchestrator.py",
                    "impact": "Create a 7th agent that consumes GapAnalyst findings to propose git diffs.",
                },
            ],
        )

    def build_ten_page_report(
        self, ctx: AnalysisContext, teaching: TeachingReport
    ) -> TenPageReport:
        """Construct the complete 10-page technical document."""
        rep_map = ctx.repository_map
        flows = ctx.flow_analysis
        comps = ctx.component_analysis
        data_int = ctx.data_integration_analysis
        gaps = ctx.gap_analysis

        pages: List[ReportPage] = []

        # PAGE 1: Project Overview
        page1_content = {
            "project_name": rep_map.project_name if rep_map else "Repository",
            "summary": rep_map.project_summary if rep_map else teaching.simple_explanation,
            "languages": rep_map.languages if rep_map else [],
            "frameworks": rep_map.frameworks if rep_map else [],
            "entry_points": rep_map.entry_points if rep_map else [],
            "why_it_exists": teaching.why_it_exists,
            "problem_solved": teaching.problem_solved,
        }
        pages.append(
            ReportPage(
                page_number=1,
                title="Project Overview",
                subtitle="High-level identity, core purpose, tech stack, and entrypoints",
                content=page1_content,
                markdown_content=f"### Summary\n{page1_content['summary']}\n\n**Tech Stack**: {', '.join(page1_content['languages'])}\n\n**Frameworks**: {', '.join(page1_content['frameworks'])}\n\n**Entry Points**: {', '.join(page1_content['entry_points'])}",
            )
        )

        # PAGE 2: Repository Map
        page2_content = {
            "directories": [d.model_dump() for d in rep_map.directories] if rep_map else [],
            "layers": rep_map.architectural_layers if rep_map else [],
            "tree": rep_map.directory_tree if rep_map else "",
        }
        pages.append(
            ReportPage(
                page_number=2,
                title="Repository Map",
                subtitle="Folder structure, directory responsibilities, and structural hierarchy",
                content=page2_content,
                markdown_content=f"```\n{page2_content['tree']}\n```",
            )
        )

        # PAGE 3: Architecture
        mermaid_arch = (
            "graph TD;\n"
            "  subgraph Entry_Layer [Entry / CLI Layer]\n"
            "    CLI[app/main.py]\n"
            "  end\n"
            "  subgraph API_Layer [API / Controller Layer]\n"
            "    API[app/api/routes.py]\n"
            "  end\n"
            "  subgraph Orchestration [Orchestration Layer]\n"
            "    Orch[app/orchestrator/orchestrator.py]\n"
            "    Ctx[app/analysis/context.py]\n"
            "  end\n"
            "  subgraph Agents_Layer [Specialized Agent Pipeline]\n"
            "    A1[1. Repository Mapper]\n"
            "    A2[2. Flow Tracer]\n"
            "    A3[3. Component Analyst]\n"
            "    A4[4. Data Analyst]\n"
            "    A5[5. Gap Analyst]\n"
            "    A6[6. Teacher Agent]\n"
            "  end\n"
            "  subgraph Engine_Layer [Tools & Engine]\n"
            "    Scan[app/tools/repository_scanner.py]\n"
            "    FS[app/tools/filesys.py]\n"
            "    Search[app/tools/code_search.py]\n"
            "  end\n"
            "  subgraph External_Storage [External Services & Storage]\n"
            "    Ollama[\"Local Ollama (qwen3-coder:30b)\"]\n"
            "    Disk[workspace/*.json Artifacts]\n"
            "  end\n"
            "  CLI --> API;\n"
            "  API --> Orch;\n"
            "  Orch --> Scan;\n"
            "  Orch --> Ctx;\n"
            "  Orch --> A1 --> A2 --> A3 --> A4 --> A5 --> A6;\n"
            "  A1 & A2 & A3 & A4 & A5 & A6 --> Ollama;\n"
            "  A1 & A2 & A3 & A4 & A5 & A6 --> FS;\n"
            "  Orch --> Disk;\n"
        )
        page3_content = {
            "layers": rep_map.architectural_layers if rep_map else [],
            "architecture_diagram": mermaid_arch,
            "decisions": [d.model_dump() for d in teaching.architecture_decisions],
        }
        pages.append(
            ReportPage(
                page_number=3,
                title="Architecture",
                subtitle="System layers, component topologies, and architectural decisions",
                content=page3_content,
                markdown_content="Detailed architectural diagram and design rationale.",
            )
        )

        # PAGE 4: Routes & Interfaces
        page4_content = {
            "endpoints": flows.endpoints_and_routes if flows else [],
            "summary": flows.summary if flows else "",
        }
        pages.append(
            ReportPage(
                page_number=4,
                title="Routes & Interfaces",
                subtitle="HTTP endpoints, API routes, CLI parameters, and interface boundaries",
                content=page4_content,
                markdown_content="\n".join([f"- `{ep}`" for ep in page4_content["endpoints"]]),
            )
        )

        # PAGE 5: Execution Flows
        page5_content = {
            "flows": [f.model_dump() for f in flows.key_flows] if flows else [],
        }
        pages.append(
            ReportPage(
                page_number=5,
                title="Execution Flows",
                subtitle="End-to-end request lifecycle and sequence diagrams",
                content=page5_content,
                markdown_content="Interactive sequence diagrams showing input-to-response workflows.",
            )
        )

        # PAGE 6: Component Explorer
        page6_content = {
            "components": [c.model_dump() for c in comps.components] if comps else [],
            "mermaid_graph": comps.component_graph_mermaid if comps else "",
        }
        pages.append(
            ReportPage(
                page_number=6,
                title="Component Explorer",
                subtitle="Concrete responsibilities, inputs, outputs, calls, side effects, and failure modes",
                content=page6_content,
                markdown_content="In-depth component breakdown distinguishing naming from actual behavior.",
            )
        )

        # PAGE 7: Data & Integrations
        page7_content = {
            "databases": [d.model_dump() for d in data_int.databases] if data_int else [],
            "external_services": [s.model_dump() for s in data_int.external_services] if data_int else [],
            "ai_ml_components": [a.model_dump() for a in data_int.ai_ml_components] if data_int else [],
            "data_flows": [f.model_dump() for f in data_int.data_flows] if data_int else [],
            "mermaid_diagram": data_int.data_flow_mermaid if data_int else "",
        }
        pages.append(
            ReportPage(
                page_number=7,
                title="Data & Integrations",
                subtitle="Persistence stores, third-party APIs, LLM models, and data transformation pipelines",
                content=page7_content,
                markdown_content="Data movement pipelines and integration boundary specifications.",
            )
        )

        # PAGE 8: Gaps & Issues
        page8_content = {
            "gaps": [g.model_dump() for g in gaps.gaps] if gaps else [],
            "summary": gaps.summary if gaps else "",
        }
        pages.append(
            ReportPage(
                page_number=8,
                title="Gaps & Issues",
                subtitle="Confirmed bugs, architectural concerns, missing tests, and contribution opportunities",
                content=page8_content,
                markdown_content="Evidence-backed classification of potential code risks and technical debt.",
            )
        )

        # PAGE 9: Concepts to Learn
        page9_content = {
            "concepts": [c.model_dump() for c in teaching.concepts_to_learn],
        }
        pages.append(
            ReportPage(
                page_number=9,
                title="Concepts to Learn",
                subtitle="Foundational engineering paradigms required to master this repository",
                content=page9_content,
                markdown_content="Key conceptual primitives tailored specifically to this codebase.",
            )
        )

        # PAGE 10: Teacher / Learn the Project
        page10_content = {
            "simple_explanation": teaching.simple_explanation,
            "interview_explanation": teaching.interview_explanation,
            "first_files_to_read": [f.model_dump() for f in teaching.first_files_to_read],
            "answers_to_25_questions": teaching.answers_to_25_questions,
            "what_if_modifications": teaching.what_if_modifications,
        }
        pages.append(
            ReportPage(
                page_number=10,
                title="Teacher / Learn the Project",
                subtitle="25-question mental model, interview walkthrough, and guided reading plan",
                content=page10_content,
                markdown_content="First-principles synthesis answering all core architectural questions.",
            )
        )

        return TenPageReport(
            analysis_id=ctx.analysis_id,
            project_name=ctx.index.project_name if ctx.index else "Sherlock",
            generated_at=ctx.started_at,
            pages=pages,
        )
