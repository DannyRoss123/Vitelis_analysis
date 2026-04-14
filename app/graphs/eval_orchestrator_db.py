from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.nodes.evaluate_rag_node import evaluate_rag_node
from app.nodes.eval_aggregate_report import eval_aggregate_report_node
from app.nodes.load_from_db import load_from_db_node


class EvalDbState(TypedDict, total=False):
    run_id: str
    db_run_id: str
    company_name: str
    company_domain: str
    company_folder: str
    target_urls: List[str]
    url_count: int
    sources: List[Dict]
    kpi_results: List[Dict]
    missing_evidence: List[str]
    kpi_definitions: List[Dict]
    rag_evaluation: Dict
    report_path: str
    eval_report_path: Optional[str]
    overall_score: float


def build_eval_db_graph():
    builder = StateGraph(EvalDbState)
    builder.add_node("load_from_db", load_from_db_node)
    builder.add_node("evaluate_rag", evaluate_rag_node)
    builder.add_node("eval_aggregate_report", eval_aggregate_report_node)

    builder.set_entry_point("load_from_db")
    builder.add_edge("load_from_db", "evaluate_rag")
    builder.add_edge("evaluate_rag", "eval_aggregate_report")
    builder.add_edge("eval_aggregate_report", END)
    return builder.compile()

