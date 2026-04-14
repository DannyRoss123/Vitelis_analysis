"""
eval_score_kpis.py
Evaluation-mode scorer that reuses the full extension-enabled score_kpis_node
and adds human-readable KPI names for ground-truth matching.
"""
from __future__ import annotations

from typing import Dict, List

from app.nodes.score_kpis import score_kpis_node


def eval_score_kpis_node(state: Dict) -> Dict:
    """
    Reuse the extension-enabled score_kpis_node in eval mode, then add _name
    for each KPI result so compare_ground_truth can fuzzy-match display names.

    State keys read:  run_id, company_domain, sources
    State keys set:   kpi_results, missing_evidence, kpi_definitions
    """
    scored = score_kpis_node(state)
    results: List[Dict] = scored.get("kpi_results", [])
    defs: List[Dict] = scored.get("kpi_definitions", [])
    name_by_id = {str(d.get("kpi_id", "")): d.get("name", "") for d in defs}
    for r in results:
        rid = str(r.get("kpi_id", ""))
        r["_name"] = name_by_id.get(rid, rid)
    return scored
