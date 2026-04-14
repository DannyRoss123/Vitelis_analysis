"""
compare_ground_truth.py
Matches pipeline KPI results against analyst ground-truth data points
using fuzzy string matching (difflib).
"""
from __future__ import annotations

import difflib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from app.eval_models import GroundTruthEvalReport, KPIGroundTruthComparison


def _best_match(
    kpi_name: str,
    candidates: List[str],
    cutoff: float = 0.35,
) -> Tuple[str, float]:
    """
    Return (best_candidate, ratio) or ("", 0.0) if nothing exceeds cutoff.
    Uses SequenceMatcher for a ratio score and get_close_matches for filtering.
    """
    matches = difflib.get_close_matches(kpi_name, candidates, n=1, cutoff=cutoff)
    if not matches:
        return "", 0.0
    best = matches[0]
    ratio = difflib.SequenceMatcher(None, kpi_name.lower(), best.lower()).ratio()
    return best, ratio


def compare_with_ground_truth(
    kpi_results: List[Dict],
    ground_truth_points: List[Dict],
    company_name: str,
    run_id: str,
) -> GroundTruthEvalReport:
    """
    Match each KPI result to the closest ground-truth data point by name.

    Returns a GroundTruthEvalReport with per-KPI comparisons and unmatched lists.
    """
    gt_by_name: Dict[str, Dict] = {pt["name"]: pt for pt in ground_truth_points}
    gt_names = list(gt_by_name.keys())

    comparisons: List[KPIGroundTruthComparison] = []
    matched_gt_names: set = set()
    unmatched_kpis: List[str] = []

    for kpi in kpi_results:
        kpi_name: str = kpi.get("kpi_id", "")
        # Prefer a human-readable name if available from kpi_definitions lookup
        # (score_kpis stores kpi_id; we use that to look up the name from kpi_definitions)
        display_name = kpi.get("_name", kpi_name)  # set by eval_score_kpis if available

        best_gt_name, ratio = _best_match(display_name, gt_names)

        # Extract pipeline sources from citations
        citations = kpi.get("citations") or []
        pipeline_sources = [c.get("url", "") for c in citations if c.get("url")]

        if not best_gt_name:
            unmatched_kpis.append(display_name)
            continue

        matched_gt_names.add(best_gt_name)
        gt = gt_by_name[best_gt_name]

        comparisons.append(
            KPIGroundTruthComparison(
                kpi_id=kpi.get("kpi_id", ""),
                kpi_name=display_name,
                pipeline_score=float(kpi.get("score", 0)),
                pipeline_confidence=float(kpi.get("confidence", 0.0)),
                pipeline_rationale=str(kpi.get("rationale", "")),
                pipeline_sources=pipeline_sources,
                ground_truth_name=best_gt_name,
                ground_truth_answer=gt.get("answer", ""),
                ground_truth_explanation=gt.get("explanation", ""),
                ground_truth_sources=gt.get("sources", []),
                match_confidence=round(ratio, 3),
            )
        )

    # Data points with no matching KPI
    unmatched_data_points = [n for n in gt_names if n not in matched_gt_names]

    return GroundTruthEvalReport(
        company_name=company_name,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        comparisons=comparisons,
        unmatched_kpis=unmatched_kpis,
        unmatched_data_points=unmatched_data_points,
    )
