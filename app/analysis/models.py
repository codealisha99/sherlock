"""Data models for Sherlock analysis artifacts and 10-page report.

All agent inputs/outputs are strongly typed with Pydantic for validation,
serialization, and schema generation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Agent 1: Repository Map Models
# ==============================================================================

class DirectoryInfo(BaseModel):
    path: str
    purpose: str
    key_files: List[str] = Field(default_factory=list)


class RepositoryMap(BaseModel):
    project_name: str
    project_summary: str
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    directories: List[DirectoryInfo] = Field(default_factory=list)
    major_components: List[str] = Field(default_factory=list)
    configuration: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    architectural_layers: List[str] = Field(default_factory=list)
    directory_tree: str = ""


# ==============================================================================
# Agent 2: Flow Tracer Models
# ==============================================================================

class FlowStep(BaseModel):
    step_number: int
    layer: str  # e.g., "INPUT", "ENTRY POINT", "ROUTE/HANDLER", "CONTROLLER", "SERVICE", "DATABASE/EXT", "RESPONSE"
    component: str
    file: str
    function_name: str
    description: str


class ExecutionFlow(BaseModel):
    name: str
    trigger: str  # e.g., "POST /api/analyze", "CLI: python -m app.main"
    description: str
    steps: List[FlowStep] = Field(default_factory=list)
    mermaid_diagram: str = ""


class FlowAnalysis(BaseModel):
    endpoints_and_routes: List[str] = Field(default_factory=list)
    key_flows: List[ExecutionFlow] = Field(default_factory=list)
    summary: str = ""


# ==============================================================================
# Agent 3: Component Analyst Models
# ==============================================================================

class ComponentDetail(BaseModel):
    component: str
    file: str
    responsibility: str  # What the code actually does vs what it is called
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    calls: List[str] = Field(default_factory=list)
    called_by: List[str] = Field(default_factory=list)
    side_effects: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)


class ComponentAnalysis(BaseModel):
    components: List[ComponentDetail] = Field(default_factory=list)
    component_graph_mermaid: str = ""
    summary: str = ""


# ==============================================================================
# Agent 4: Data & Integration Models
# ==============================================================================

class DatabaseInfo(BaseModel):
    name: str
    type: str  # SQL, NoSQL, Vector DB, In-Memory, etc.
    orm_or_driver: str
    tables_or_models: List[str] = Field(default_factory=list)
    purpose: str


class ExternalServiceInfo(BaseModel):
    name: str
    purpose: str
    sent_data: str
    received_data: str
    call_location: str
    failure_possibilities: List[str] = Field(default_factory=list)
    configuration_required: str


class AiMlComponentInfo(BaseModel):
    name: str
    model_or_service: str
    role: str
    entry_location: str
    purpose: str


class DataPipelineFlow(BaseModel):
    name: str
    pipeline: List[str] = Field(default_factory=list)  # e.g. ["USER", "API", "SERVICE", "DATABASE"]
    description: str


class DataIntegrationAnalysis(BaseModel):
    databases: List[DatabaseInfo] = Field(default_factory=list)
    external_services: List[ExternalServiceInfo] = Field(default_factory=list)
    ai_ml_components: List[AiMlComponentInfo] = Field(default_factory=list)
    data_flows: List[DataPipelineFlow] = Field(default_factory=list)
    data_flow_mermaid: str = ""
    summary: str = ""


# ==============================================================================
# Agent 5: Gap & Issue Analyst Models
# ==============================================================================

class GapItem(BaseModel):
    gap: str
    category: str  # "Confirmed issue", "Likely issue", "Architectural concern", "Missing feature", "Possible improvement"
    evidence: str
    why_it_matters: str
    affected_components: List[str] = Field(default_factory=list)
    confidence: str  # "High", "Medium", "Low"
    possible_impact: str
    possible_contribution: str


class GapAnalysis(BaseModel):
    gaps: List[GapItem] = Field(default_factory=list)
    summary: str = ""


# ==============================================================================
# Agent 6: Teacher Agent Models
# ==============================================================================

class ConceptToLearn(BaseModel):
    concept: str
    definition: str
    why_relevant_here: str
    where_used_in_project: str


class FileToStudy(BaseModel):
    file: str
    order: int
    why_read_first: str
    key_symbols_to_watch: List[str] = Field(default_factory=list)


class ArchitectureDecision(BaseModel):
    decision: str
    rationale: str
    trade_offs: str


class TeachingReport(BaseModel):
    simple_explanation: str
    problem_solved: str
    why_it_exists: str
    request_lifecycle_explanation: str
    answers_to_25_questions: Dict[str, str] = Field(default_factory=dict)
    concepts_to_learn: List[ConceptToLearn] = Field(default_factory=list)
    interview_explanation: str
    first_files_to_read: List[FileToStudy] = Field(default_factory=list)
    architecture_decisions: List[ArchitectureDecision] = Field(default_factory=list)
    what_if_modifications: List[Dict[str, str]] = Field(default_factory=list)


# ==============================================================================
# Synthesized 10-Page Report Model
# ==============================================================================

class ReportPage(BaseModel):
    page_number: int
    title: str
    subtitle: str
    content: Dict[str, Any]
    markdown_content: str = ""


class TenPageReport(BaseModel):
    analysis_id: str
    project_name: str
    generated_at: str
    pages: List[ReportPage] = Field(default_factory=list)
