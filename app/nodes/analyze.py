from __future__ import annotations

import hashlib
import random
from typing import Dict, List

from app.kpi_catalog import load_kpi_catalog
from app.models import AggregatedKPIResult, Citation, KPIDriver, KPIDriverResult
from app.observability import get_tracer
from app.vectorstore import build_collection, index_sources, retrieve_evidence


def _stable_random(run_seed: str, kpi_id: str) -> random.Random:
    seed_input = f"{run_seed}:{kpi_id}".encode("utf-8")
    seed = int(hashlib.md5(seed_input).hexdigest()[:8], 16)
    return random.Random(seed)


def _score_from_evidence(kpi: KPIDriver, evidences: List[tuple], run_seed: str) -> KPIDriverResult:
    rng = _stable_random(run_seed, kpi.kpi_id)
    base_score = 60.0
    # retrieve_evidence now returns 3-tuples: (metadata, document, score)
    text_blob = " ".join(doc for _, doc, *_ in evidences).lower()

    positives = ["strong", "improve", "growth", "quality", "efficient", "optimized"]
    negatives = ["risk", "decline", "issue", "problem", "weak"]

    for word in positives:
        if word in text_blob:
            base_score += 5.0
    for word in negatives:
        if word in text_blob:
            base_score -= 5.0

    noise = rng.uniform(-2.0, 2.0)
    score = max(0.0, min(100.0, base_score + noise))

    if evidences:
        confidence = 0.65
        rationale = "Evidence references relevant page content."
    else:
        confidence = 0.2
        rationale = "No evidence retrieved; score is low confidence."

    citations = [
        Citation(source_id=metadata["source_id"], quote=doc[:200])
        for metadata, doc, *_ in evidences[:2]
    ]

    return KPIDriverResult(
        kpi_id=kpi.kpi_id,
        score=round(score, 2),
        confidence=confidence,
        rationale=rationale,
        citations=citations,
    )


def _aggregate(kpis: List[KPIDriver], results: List[KPIDriverResult]) -> List[AggregatedKPIResult]:
    result_map = {result.kpi_id: result for result in results}
    children_map: Dict[str, List[str]] = {}
    for kpi in kpis:
        if kpi.parent_id:
            children_map.setdefault(kpi.parent_id, []).append(kpi.kpi_id)

    aggregated: List[AggregatedKPIResult] = []
    for parent_id, children in children_map.items():
        total_weight = 0.0
        weighted_sum = 0.0
        confidences = []
        for child_id in children:
            child = result_map.get(child_id)
            if not child:
                continue
            weighted_sum += child.score * child.confidence
            total_weight += child.confidence
            confidences.append(child.confidence)

        if total_weight == 0:
            score = 0.0
        else:
            score = weighted_sum / total_weight
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        aggregated.append(
            AggregatedKPIResult(
                kpi_id=parent_id,
                score=round(score, 2),
                confidence=round(confidence, 2),
                children=children,
            )
        )

    return aggregated


def analyze_node(state: Dict) -> Dict:
    tracer = get_tracer()
    with tracer.span("analyze"):
        sources = state.get("sources", [])
        run_id = state["run_id"]
        run_seed = state.get("run_seed", run_id)

        collection = build_collection(run_id)
        index_sources(collection, sources)

        kpis = load_kpi_catalog()
        results: List[KPIDriverResult] = []
        empty_retrievals = 0

        for kpi in kpis:
            evidences = retrieve_evidence(collection, kpi.question, k=4)
            if not evidences:
                empty_retrievals += 1
            result = _score_from_evidence(kpi, evidences, run_seed)
            results.append(result)

        aggregated = _aggregate(kpis, results)

        return {
            "kpi_results": [result.model_dump() for result in results],
            "aggregated_results": [result.model_dump() for result in aggregated],
            "retrieval_stats": {
                "empty_retrievals": empty_retrievals,
                "total_kpis": len(kpis),
            },
        }
