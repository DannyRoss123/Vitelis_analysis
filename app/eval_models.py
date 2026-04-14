from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class KPIGroundTruthComparison(BaseModel):
    kpi_id: str
    kpi_name: str
    pipeline_score: float
    pipeline_confidence: float = 0.0  # LLM/enhanced confidence score 0.0-1.0
    pipeline_rationale: str
    pipeline_sources: List[str] = Field(default_factory=list)  # URLs cited by pipeline
    ground_truth_name: str            # matched data point name from raw_data_points.json
    ground_truth_answer: str          # e.g. "77", "15.5%", "Formal charter exists..."
    ground_truth_explanation: str
    ground_truth_sources: List[str] = Field(default_factory=list)  # URLs the analyst used
    match_confidence: float           # difflib ratio 0.0-1.0


class GroundTruthEvalReport(BaseModel):
    company_name: str
    run_id: str
    timestamp: str
    comparisons: List[KPIGroundTruthComparison] = Field(default_factory=list)
    unmatched_kpis: List[str] = Field(default_factory=list)      # KPI names with no GT match
    unmatched_data_points: List[str] = Field(default_factory=list)  # GT points with no KPI match
