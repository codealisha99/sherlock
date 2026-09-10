"""Analysis Context for Sherlock.

Maintains the shared state across all pipeline stages, including
the scanned repository index, tool handles, and agent outputs.
"""

from pathlib import Path
import uuid
import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from app.tools.filesys import SafeFilesystem
from app.tools.repository_scanner import RepositoryIndex, RepositoryScanner
from app.tools.code_search import CodeSearch
from app.analysis.models import (
    RepositoryMap,
    FlowAnalysis,
    ComponentAnalysis,
    DataIntegrationAnalysis,
    GapAnalysis,
    TeachingReport,
    TenPageReport,
)


class AnalysisContext:
    """Shared state container for a single repository analysis run."""

    def __init__(self, project_path: Path | str, analysis_id: Optional[str] = None):
        self.project_path = Path(project_path).resolve()
        self.analysis_id = analysis_id or f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.fs = SafeFilesystem(self.project_path)
        self.scanner = RepositoryScanner(self.project_path)
        self.index: Optional[RepositoryIndex] = None
        self.search: Optional[CodeSearch] = None

        # Agent outputs
        self.repository_map: Optional[RepositoryMap] = None
        self.flow_analysis: Optional[FlowAnalysis] = None
        self.component_analysis: Optional[ComponentAnalysis] = None
        self.data_integration_analysis: Optional[DataIntegrationAnalysis] = None
        self.gap_analysis: Optional[GapAnalysis] = None
        self.teaching_report: Optional[TeachingReport] = None
        self.ten_page_report: Optional[TenPageReport] = None

        # Tracking execution progress
        self.status: str = "initialized"
        self.current_agent: str = "Idle"
        self.agent_statuses: Dict[str, str] = {
            "Repository Mapper": "pending",
            "Flow Tracer": "pending",
            "Component Analyst": "pending",
            "Data Analyst": "pending",
            "Gap Analyst": "pending",
            "Teacher": "pending",
        }
        self.errors: List[str] = []
        self.started_at: str = datetime.datetime.now().isoformat()
        self.completed_at: Optional[str] = None

    def initialize_index(self) -> RepositoryIndex:
        """Scan repository and create index and search tools."""
        self.status = "indexing"
        self.index = self.scanner.scan()
        self.search = CodeSearch(self.index, self.fs)
        return self.index

    def update_agent_status(self, agent_name: str, status: str):
        """Update live status for an agent ('running', 'completed', 'failed')."""
        self.agent_statuses[agent_name] = status
        self.current_agent = agent_name if status == "running" else self.current_agent

    def to_progress_dict(self) -> Dict[str, Any]:
        """Return live progress snapshot for frontend polling/streaming."""
        return {
            "analysis_id": self.analysis_id,
            "project_name": self.project_path.name,
            "project_path": str(self.project_path),
            "status": self.status,
            "current_agent": self.current_agent,
            "agent_statuses": self.agent_statuses,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "has_report": self.ten_page_report is not None,
        }
