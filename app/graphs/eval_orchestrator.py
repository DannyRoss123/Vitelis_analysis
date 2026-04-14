"""
eval_orchestrator.py
LangGraph pipeline for ground-truth evaluation mode.

Flow:
    load_from_export → index_sources → eval_score_kpis → evaluate_rag → eval_aggregate_report

Imports existing nodes unchanged; only the entry/exit nodes differ from the main pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# Existing nodes (imported unchanged — no modifications)
from app.nodes.evaluate_rag_node import evaluate_rag_node

# New eval-specific nodes
from app.nodes.load_from_export import load_from_export_node
from app.nodes.eval_index_sources import eval_index_sources_node
from app.nodes.eval_score_kpis import eval_score_kpis_node
from app.nodes.eval_aggregate_report import eval_aggregate_report_node


class EvalOrchestratorState(TypedDict, total=False):
    # Identification
    run_id: str
    company_name: str
    company_domain: str      # derived from sources by load_from_export_node
    company_folder: str      # e.g. "Orange S.A" — set by run_eval.py
    max_urls: int            # optional cap for load_from_export_node
    kpi_limit: int           # optional cap for eval_score_kpis_node
    kpi_ids: List[str]       # optional explicit KPI IDs to score

    # Source pipeline
    target_urls: List[str]
    url_count: int
    sources: List[Dict]
    collection_id: str

    # Scoring
    kpi_results: List[Dict]
    missing_evidence: List[str]
    kpi_definitions: List[Dict]  # passed to evaluate_rag_node

    # RAG evaluation
    rag_evaluation: Dict

    # Output paths
    report_path: str
    eval_report_path: Optional[str]
    overall_score: float


def build_eval_graph():
    builder = StateGraph(EvalOrchestratorState)

    builder.add_node("load_from_export", load_from_export_node)
    builder.add_node("index_sources", eval_index_sources_node)
    builder.add_node("eval_score_kpis", eval_score_kpis_node)
    builder.add_node("evaluate_rag", evaluate_rag_node)
    builder.add_node("eval_aggregate_report", eval_aggregate_report_node)

    builder.set_entry_point("load_from_export")
    builder.add_edge("load_from_export", "index_sources")
    builder.add_edge("index_sources", "eval_score_kpis")
    builder.add_edge("eval_score_kpis", "evaluate_rag")
    builder.add_edge("evaluate_rag", "eval_aggregate_report")
    builder.add_edge("eval_aggregate_report", END)

    return builder.compile()
