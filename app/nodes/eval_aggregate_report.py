"""
eval_aggregate_report.py
Aggregates KPI results into a YAML report (same format as aggregate_report_node)
AND writes an eval_{run_id}.json with ground-truth comparison data.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import yaml

from app.models import AggregatedKPIResult, KPIDriverResult, RagEvaluationReport, ReportArtifact
from app.nodes.compare_ground_truth import compare_with_ground_truth
from app.nodes.load_ground_truth import load_ground_truth
from app.mlflow_logger import log_pipeline_run


def _aggregate_by_pillar(results: List[KPIDriverResult]) -> List[AggregatedKPIResult]:
    pillar_map: Dict[str, List[KPIDriverResult]] = {}
    for result in results:
        pillar_map.setdefault(result.pillar, []).append(result)

    aggregated: List[AggregatedKPIResult] = []
    for pillar, items in pillar_map.items():
        total_weight = sum(item.confidence for item in items)
        score = (
            sum(
                (float(item.live_score) if item.live_score is not None else float(item.score))
                * item.confidence
                for item in items
            )
            / total_weight
            if total_weight > 0
            else 0.0
        )
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
    total_weight = sum(p.confidence for p in pillars)
    if total_weight == 0:
        return 0.0
    return round(sum(p.score * p.confidence for p in pillars) / total_weight, 2)


def eval_aggregate_report_node(state: Dict) -> Dict:
    """
    1. Writes app/output/report_{run_id}.yaml  (standard format)
    2. Writes app/output/eval_{run_id}.json    (ground-truth comparison)

    State keys read:  run_id, company_name, company_domain, company_folder,
                      url_count, kpi_results, missing_evidence, rag_evaluation
    State keys set:   report_path, eval_report_path, overall_score
    """
    run_id = state["run_id"]
    company_name = state["company_name"]
    company_domain = state.get("company_domain", "")
    company_folder = state.get("company_folder", "")
    url_count = state.get("url_count", 0)

    # ── 1. Build standard YAML report ────────────────────────────────── #
    raw_results = state.get("kpi_results", [])
    # Strip internal _name field before building KPIDriverResult objects
    kpi_results = [KPIDriverResult(**{k: v for k, v in r.items() if not k.startswith("_")})
                   for r in raw_results]

    pillar_scores = _aggregate_by_pillar(kpi_results)
    overall_score = _overall_score(pillar_scores)

    rag_eval_dict = state.get("rag_evaluation")
    rag_evaluation = RagEvaluationReport(**rag_eval_dict) if rag_eval_dict else None

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
        rag_evaluation=rag_evaluation,
    )

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"report_{run_id}.yaml")

    with open(report_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report.model_dump(), f, sort_keys=False)

    print(f"[eval_aggregate] Report written: {report_path}")

    # ── 1b. MLflow pipeline summary run (optional) ─────────────────────── #
    try:
        # Prefer the run-level Chroma snapshot if present; else leave empty.
        snapshot_id = str(state.get("chromadb_snapshot_id") or "")
        log_pipeline_run(
            run_id=run_id,
            company_id=company_domain or company_name,
            overall_score=float(overall_score),
            kpi_count=len(kpi_results),
            chromadb_snapshot_id=snapshot_id,
            # We don't currently keep a single "run trace" id in state; KPI traces are stored per KPI.
            langfuse_trace_id=None,
            extra_metrics={
                "url_count": float(url_count),
            },
        )
    except Exception:
        pass

    # ── 2. Eval JSON (always) + optional ground-truth comparison ───────── #
    eval_report_path = None
    eval_path = os.path.join(output_dir, f"eval_{run_id}.json")
    eval_payload: Dict = {
        "run_id": run_id,
        "company_name": company_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_score": float(overall_score),
        "url_count": url_count,
        "rag_evaluation": rag_eval_dict,
        "kpi_count": len(kpi_results),
    }

    if company_folder:
        try:
            ground_truth_points = load_ground_truth(company_folder)
            eval_report = compare_with_ground_truth(
                kpi_results=raw_results,   # includes _name field
                ground_truth_points=ground_truth_points,
                company_name=company_name,
                run_id=run_id,
            )
            eval_payload["ground_truth_comparison"] = eval_report.model_dump()
            n_matched = len(eval_report.comparisons)
            n_unmatched_kpis = len(eval_report.unmatched_kpis)
            n_unmatched_gt = len(eval_report.unmatched_data_points)
            print(
                f"[eval_aggregate] Ground-truth comparison merged into eval JSON.\n"
                f"  Matched: {n_matched} KPIs | "
                f"Unmatched KPIs: {n_unmatched_kpis} | "
                f"Unmatched GT points: {n_unmatched_gt}"
            )
        except FileNotFoundError as exc:
            print(f"[eval_aggregate] WARNING: Could not load ground truth — {exc}")

    try:
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2, ensure_ascii=False)
        eval_report_path = eval_path
        print(f"[eval_aggregate] Eval JSON written: {eval_path}")
    except OSError as exc:
        print(f"[eval_aggregate] WARNING: Could not write eval JSON — {exc}")

    return {
        "report_path": report_path,
        "eval_report_path": eval_report_path,
        "overall_score": overall_score,
    }
