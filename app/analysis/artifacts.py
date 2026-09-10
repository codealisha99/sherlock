"""Artifact persistence manager for Sherlock.

Saves and retrieves structured JSON artifacts and markdown reports
for each stage of the analysis pipeline.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from config import config
from app.analysis.context import AnalysisContext
from app.analysis.models import (
    RepositoryMap,
    FlowAnalysis,
    ComponentAnalysis,
    DataIntegrationAnalysis,
    GapAnalysis,
    TeachingReport,
    TenPageReport,
)


class ArtifactManager:
    """Handles writing and loading analysis artifacts to/from disk."""

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or config.workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def get_analysis_dir(self, analysis_id: str) -> Path:
        target_dir = self.workspace_dir / analysis_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def save_all_artifacts(self, ctx: AnalysisContext) -> Path:
        """Write all completed agent outputs and synthesized report to disk."""
        target_dir = self.get_analysis_dir(ctx.analysis_id)

        # 1. Repository Map
        if ctx.repository_map:
            self._save_json(target_dir / "repository_map.json", ctx.repository_map.model_dump())

        # 2. Flows
        if ctx.flow_analysis:
            self._save_json(target_dir / "flows.json", ctx.flow_analysis.model_dump())

        # 3. Components
        if ctx.component_analysis:
            self._save_json(target_dir / "components.json", ctx.component_analysis.model_dump())

        # 4. Data & Integrations
        if ctx.data_integration_analysis:
            self._save_json(target_dir / "data_integrations.json", ctx.data_integration_analysis.model_dump())

        # 5. Gaps
        if ctx.gap_analysis:
            self._save_json(target_dir / "gaps.json", ctx.gap_analysis.model_dump())

        # 6. Teaching Report
        if ctx.teaching_report:
            self._save_json(target_dir / "teaching_report.json", ctx.teaching_report.model_dump())

        # 7. Ten-Page Synthesized Report
        if ctx.ten_page_report:
            self._save_json(target_dir / "ten_page_report.json", ctx.ten_page_report.model_dump())
            self._save_markdown_summary(target_dir / "README_ANALYSIS.md", ctx.ten_page_report)

        return target_dir

    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_markdown_summary(self, file_path: Path, report: TenPageReport):
        lines = [
            f"# Sherlock Codebase Understanding: {report.project_name}",
            f"*Generated on {report.generated_at} (Analysis ID: {report.analysis_id})*\n",
            "---",
        ]
        for page in report.pages:
            lines.append(f"\n## Page {page.page_number}: {page.title}")
            lines.append(f"*{page.subtitle}*\n")
            if page.markdown_content:
                lines.append(page.markdown_content)
            lines.append("\n---")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def load_ten_page_report(self, analysis_id: str) -> Optional[TenPageReport]:
        target_file = self.get_analysis_dir(analysis_id) / "ten_page_report.json"
        if not target_file.is_file():
            return None
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return TenPageReport.model_validate(data)
