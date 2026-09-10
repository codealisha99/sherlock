"""Sherlock Orchestrator.

Coordinates the multi-agent pipeline:
1. Validates project path
2. Scans and indexes codebase (AST & hierarchy)
3. Sequentially executes the 6 specialized agents
4. Aggregates findings into the shared AnalysisContext
5. Persists structured JSON and Markdown artifacts
6. Emits live progress events for web UI
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from config import config, SherlockConfig
from app.analysis.context import AnalysisContext
from app.analysis.artifacts import ArtifactManager
from app.analysis.models import TenPageReport
from app.agents.repository_mapper import RepositoryMapperAgent
from app.agents.flow_tracer import FlowTracerAgent
from app.agents.component_analyst import ComponentAnalystAgent
from app.agents.data_integration_analyst import DataIntegrationAnalystAgent
from app.agents.gap_analyst import GapAnalystAgent
from app.agents.teacher import TeacherAgent
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central pipeline coordinator for Sherlock."""

    def __init__(
        self,
        sherlock_cfg: Optional[SherlockConfig] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.cfg = sherlock_cfg or config
        self.llm = llm_client or OllamaClient(self.cfg)
        self.artifact_mgr = ArtifactManager(self.cfg.workspace_dir)

        # In-memory registry of analyses
        self.active_analyses: Dict[str, AnalysisContext] = {}

        # Instantiate specialized agents
        self.agent_mapper = RepositoryMapperAgent(llm_client=self.llm, sherlock_cfg=self.cfg)
        self.agent_flow = FlowTracerAgent(llm_client=self.llm, sherlock_cfg=self.cfg)
        self.agent_component = ComponentAnalystAgent(llm_client=self.llm, sherlock_cfg=self.cfg)
        self.agent_data = DataIntegrationAnalystAgent(llm_client=self.llm, sherlock_cfg=self.cfg)
        self.agent_gap = GapAnalystAgent(llm_client=self.llm, sherlock_cfg=self.cfg)
        self.agent_teacher = TeacherAgent(llm_client=self.llm, sherlock_cfg=self.cfg)

    def run_analysis(
        self,
        project_path: Path | str,
        analysis_id: Optional[str] = None,
        progress_callback: Optional[Callable[[AnalysisContext], None]] = None,
    ) -> TenPageReport:
        """
        Execute the full 6-agent analysis pipeline synchronously.
        Can be run in a background thread by the web server.
        """
        path = Path(project_path).resolve()
        if not path.exists():
            raise ValueError(f"Project directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Project path is not a directory: {path}")

        ctx = AnalysisContext(project_path=path, analysis_id=analysis_id)
        self.active_analyses[ctx.analysis_id] = ctx

        def notify(stage: str):
            ctx.status = stage
            if progress_callback:
                try:
                    progress_callback(ctx)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        try:
            # Stage 0: Indexing
            logger.info(f"[{ctx.analysis_id}] Indexing codebase at {path}...")
            notify("indexing")
            ctx.initialize_index()

            # Stage 1: Repository Mapper
            logger.info(f"[{ctx.analysis_id}] Running Agent 1: Repository Mapper...")
            ctx.update_agent_status("Repository Mapper", "running")
            notify("running_mapper")
            self.agent_mapper.run(ctx)
            ctx.update_agent_status("Repository Mapper", "completed")

            # Stage 2: Flow Tracer
            logger.info(f"[{ctx.analysis_id}] Running Agent 2: Flow Tracer...")
            ctx.update_agent_status("Flow Tracer", "running")
            notify("running_flows")
            self.agent_flow.run(ctx)
            ctx.update_agent_status("Flow Tracer", "completed")

            # Stage 3: Component Analyst
            logger.info(f"[{ctx.analysis_id}] Running Agent 3: Component Analyst...")
            ctx.update_agent_status("Component Analyst", "running")
            notify("running_components")
            self.agent_component.run(ctx)
            ctx.update_agent_status("Component Analyst", "completed")

            # Stage 4: Data & Integration Analyst
            logger.info(f"[{ctx.analysis_id}] Running Agent 4: Data & Integration Analyst...")
            ctx.update_agent_status("Data Analyst", "running")
            notify("running_data")
            self.agent_data.run(ctx)
            ctx.update_agent_status("Data Analyst", "completed")

            # Stage 5: Gap Analyst
            logger.info(f"[{ctx.analysis_id}] Running Agent 5: Gap Analyst...")
            ctx.update_agent_status("Gap Analyst", "running")
            notify("running_gaps")
            self.agent_gap.run(ctx)
            ctx.update_agent_status("Gap Analyst", "completed")

            # Stage 6: Teacher Agent
            logger.info(f"[{ctx.analysis_id}] Running Agent 6: Teacher Agent...")
            ctx.update_agent_status("Teacher", "running")
            notify("running_teacher")
            self.agent_teacher.run(ctx)
            ctx.update_agent_status("Teacher", "completed")

            # Finalize & Persist
            ctx.status = "completed"
            ctx.completed_at = datetime.datetime.now().isoformat()
            self.artifact_mgr.save_all_artifacts(ctx)
            notify("completed")
            logger.info(f"[{ctx.analysis_id}] Analysis complete! Artifacts saved.")

            return ctx.ten_page_report

        except Exception as e:
            logger.error(f"[{ctx.analysis_id}] Analysis failed: {e}", exc_info=True)
            ctx.status = "failed"
            ctx.errors.append(str(e))
            notify("failed")
            raise

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisContext]:
        """Retrieve analysis context by ID."""
        return self.active_analyses.get(analysis_id)
