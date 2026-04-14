from __future__ import annotations

from app.snapshots import build_snapshot, write_snapshot

import os
from datetime import datetime, timezone
from typing import Dict, List

import yaml

from app.debug_log import get_debug
from app.models import AggregatedKPIResult, KPIDriverResult, ReportArtifact
# code change for RAG Eval by SN
from app.models import RagEvaluationReport
# code change end for RAG Eval by SN


def _aggregate_by_pillar(results: List[KPIDriverResult]) -> List[AggregatedKPIResult]:
    pillar_map: Dict[str, List[KPIDriverResult]] = {}
    for result in results:
        pillar_map.setdefault(result.pillar, []).append(result)

    aggregated: List[AggregatedKPIResult] = []
    for pillar, items in pillar_map.items():
        total_weight = sum(item.confidence for item in items)
        if total_weight == 0:
            score = 0.0
        else:
            # Use the statistical mean score (live_score) when available so the report
            # reflects mean ± σ rather than a single-point estimate.
            def _effective_score(x: KPIDriverResult) -> float:
                return float(x.live_score) if x.live_score is not None else float(x.score)

            score = sum(_effective_score(item) * item.confidence for item in items) / total_weight
        confidence = sum(item.confidence for item in items) / len(items)
        aggregated.append(
            AggregatedKPIResult(
                pillar=pillar,
                score=round(score, 2),
                confidence=round(confidence, 2),
                kpis=[item.kpi_id for item in items],
            )
        )

    return aggregated


def _overall_score(pillars: List[AggregatedKPIResult]) -> float:
    total_weight = sum(pillar.confidence for pillar in pillars)
    if total_weight == 0:
        return 0.0
    return round(
        sum(pillar.score * pillar.confidence for pillar in pillars) / total_weight,
        2,
    )


def aggregate_report_node(state: Dict) -> Dict:
    run_id = state["run_id"]
    company_name = state["company_name"]
    company_domain = state["company_domain"]
    url_count = state.get("url_count", 0)

    kpi_results = [KPIDriverResult(**item) for item in state.get("kpi_results", [])]
    pillar_scores = _aggregate_by_pillar(kpi_results)
    overall_score = _overall_score(pillar_scores)

    # code change for RAG Eval by SN
    # Read the RAG evaluation result from state (populated by evaluate_rag_node).
    # If the node did not run or returned nothing, this will be None and the
    # rag_evaluation section will simply be absent from the YAML report.
    rag_eval_dict = state.get("rag_evaluation")
    rag_evaluation = RagEvaluationReport(**rag_eval_dict) if rag_eval_dict else None
    # code change end for RAG Eval by SN

    kpi_definitions = state.get("kpi_definitions", [])

    # Feature 9: surface the collection fingerprint at report level so every
    # generated report is traceable to the exact ChromaDB snapshot it used.
    chromadb_snapshot_id: str = state.get("chromadb_snapshot_id", "")
    report = ReportArtifact(
        run_id=run_id,
        company_name=company_name,
        company_domain=company_domain,
        timestamp=datetime.now(timezone.utc).isoformat(),
        url_count=url_count,
        kpi_results=kpi_results,
        pillar_scores=pillar_scores,
        overall_score=overall_score,
        missing_evidence=state.get("missing_evidence", []),
        debug_log=get_debug() if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"} else None,
        kpi_definitions=kpi_definitions if kpi_definitions else None,
        # code change for RAG Eval by SN
        # Include RAG evaluation in the final report (None if node was skipped)
        rag_evaluation=rag_evaluation,
        # code change end for RAG Eval by SN
        # Feature 9: collection fingerprint — makes report traceable to source data
        chromadb_snapshot_id=chromadb_snapshot_id if chromadb_snapshot_id else None,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"report_{run_id}.yaml")

    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(report.model_dump(), handle, sort_keys=False)

    snapshot = build_snapshot(report.model_dump(), report_path=output_path)
    snapshot_path = write_snapshot(snapshot)

    return {"report_path": output_path, "overall_score": overall_score, "snapshot_path": snapshot_path}
