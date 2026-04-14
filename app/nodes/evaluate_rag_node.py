# code change for RAG Eval by SN
"""
evaluate_rag_node.py — LangGraph Node: RAG Evaluation
======================================================
This node runs after score_kpis and before aggregate_report.
It calls eval_rag.run_all_evaluations() using the KPI results and sources
already present in pipeline state, then stores the structured results
back into state under the key "rag_evaluation".

The evaluate_rag_node does NOT modify any KPI scores or rationales.
It only measures quality and appends evaluation metadata to the report.

Pipeline position:
    score_kpis → [evaluate_rag] → aggregate_report
"""

from __future__ import annotations

import os
from typing import Dict, List

from app.models import (
    KPIDefinition,
    KPIDriverResult,
    Citation,
    RagEvaluationReport,
    RagKpiEval,
)

# Import the standalone evaluation module — no changes needed there
from eval_rag import run_all_evaluations


def _merge_eval_into_kpi_dicts(
    kpi_results_dicts: List[Dict],
    raw_results: Dict[str, Dict],
) -> List[Dict]:
    """Attach retrieval_metrics, BERTScore, CoT from eval_rag into kpi_results for the YAML/report."""
    out: List[Dict] = []
    for kd in kpi_results_dicts:
        kid = kd.get("kpi_id")
        res = raw_results.get(kid) if kid else None
        if not res:
            out.append(kd)
            continue
        det = dict(kd.get("details") or {})
        rm = res.get("retrieval_metrics")
        if rm:
            det["retrieval_metrics"] = rm
        new_kd = {**kd, "details": det}
        if res.get("bertscore_f1") is not None:
            new_kd["bertscore_f1"] = res["bertscore_f1"]
        if res.get("low_semantic_grounding") is not None:
            new_kd["low_semantic_grounding"] = res["low_semantic_grounding"]
        if res.get("cot_eval"):
            new_kd["cot_eval"] = res["cot_eval"]
        out.append(new_kd)
    return out


def evaluate_rag_node(state: Dict) -> Dict:
    """
    LangGraph node that runs all 9 RAG evaluation checks across every
    rubric KPI in the pipeline state, then returns the results as a
    structured dict to be included in the final YAML report.

    Inputs consumed from pipeline state:
        - kpi_results:    List of KPIDriverResult dicts (from score_kpis node)
        - sources:        List of SourceDoc dicts (from fetch_sources node)
        - kpi_definitions: List of KPIDefinition dicts (from score_kpis node)

    Output added to pipeline state:
        - rag_evaluation: Dict (RagEvaluationReport.model_dump()) containing
                          per-KPI evaluation scores and an overall verdict.
    """

    # --- Step 1: Pull all required inputs from pipeline state ---
    # kpi_results and kpi_definitions are stored as plain dicts in state;
    # we reconstruct the Pydantic objects so eval_rag.py can use them directly.
    kpi_results_dicts: List[Dict] = state.get("kpi_results", [])
    sources: List[Dict] = state.get("sources", [])
    kpi_def_dicts: List[Dict] = state.get("kpi_definitions", [])

    # --- Step 2: Reconstruct Pydantic objects from state dicts ---
    # KPIDriverResult objects contain citations as nested dicts — rebuild those too.
    kpi_results: List[KPIDriverResult] = []
    for r in kpi_results_dicts:
        # Reconstruct nested Citation objects if present
        raw_citations = r.get("citations", [])
        citations = [Citation(**c) for c in raw_citations if isinstance(c, dict)]
        kpi_results.append(KPIDriverResult(
            kpi_id=r["kpi_id"],
            pillar=r["pillar"],
            type=r["type"],
            score=r["score"],
            confidence=r["confidence"],
            rationale=r.get("rationale", ""),
            citations=citations,
            details=r.get("details"),
        ))

    # KPIDefinition objects — needed so eval_rag can extract Score-5 ground truths
    kpi_definitions: List[KPIDefinition] = [KPIDefinition(**d) for d in kpi_def_dicts]

    # --- Step 3: Run all 9 RAG evaluation checks ---
    # run_all_evaluations() returns a dict: {kpi_id -> {check_name -> result_dict}}
    # It prints a human-readable summary to stdout as it runs.
    run_llm_judge = os.getenv("RAG_EVAL_LLM_JUDGE", "true").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    raw_results: Dict[str, Dict] = run_all_evaluations(
        kpi_results=kpi_results,
        sources=sources,
        kpi_definitions=kpi_definitions,
        run_llm_judge=run_llm_judge,
        hallucination_threshold=0.4,
        verbose=False,  # Suppress per-KPI print output; results go into YAML only
    )

    merged_kpi_dicts = _merge_eval_into_kpi_dicts(kpi_results_dicts, raw_results)

    # --- Step 4: Flatten nested results into RagKpiEval objects ---
    # Each key in raw_results is a kpi_id; the value is the full evaluation dict.
    per_kpi: List[RagKpiEval] = []
    flagged_kpi_ids: List[str] = []

    for kpi_id, res in raw_results.items():
        # Unpack each check's sub-dict
        ragas = res.get("ragas", {})
        judge = res.get("llm_judge", {})
        rk = res.get("recall_at_k", {})
        f1 = res.get("f1", {})
        hall = res.get("hallucination", {})
        mmr = res.get("mmr", {})
        gt = res.get("ground_truth_checks", {})
        retm = res.get("retrieval_metrics") or {}
        bs = res.get("bertscore_f1")
        lsg = res.get("low_semantic_grounding")
        cot = res.get("cot_eval")

        # Track which KPIs were flagged for hallucination
        is_flagged = hall.get("is_flagged", False)
        if is_flagged:
            flagged_kpi_ids.append(kpi_id)

        # Build one RagKpiEval per KPI with all 9 check results
        per_kpi.append(RagKpiEval(
            kpi_id=kpi_id,
            # Ground truth used (Score-5 rubric text, if available)
            ground_truth_used=gt.get("ground_truth_used"),

            # Check 1 — RAGAS core
            ragas_faithfulness=ragas.get("faithfulness"),
            ragas_answer_relevancy=ragas.get("answer_relevancy"),
            ragas_context_precision=ragas.get("context_precision"),
            ragas_context_recall=ragas.get("context_recall"),

            # Check 2 — LLM judge
            llm_judge_overall=judge.get("overall"),
            llm_judge_feedback=judge.get("feedback"),

            # Check 3 — Recall@k (using k=3 as the headline number)
            recall_at_3=rk.get("recall@3"),

            # Check 4 — F1
            f1=f1.get("f1"),

            # Check 5 — Hallucination
            hallucination_score=hall.get("hallucination_score"),
            hallucination_flagged=is_flagged,

            # Check 6 — MMR diversity
            mmr_diversity_score=mmr.get("diversity_score"),

            # Checks 7, 8, 9 — Ground-truth-based (None if ground truth unavailable)
            factual_correctness=gt.get("factual_correctness"),
            noise_sensitivity=gt.get("noise_sensitivity"),
            semantic_similarity=gt.get("semantic_similarity"),
            retrieval_hit_rate=retm.get("hit_rate"),
            retrieval_mrr=retm.get("mrr"),
            retrieval_ndcg=retm.get("ndcg"),
            bertscore_f1=bs,
            low_semantic_grounding=lsg,
            cot_eval=cot if isinstance(cot, dict) else None,
        ))

    # --- Step 5: Build the batch-level summary and overall verdict ---
    # Counts derived from actual lists (no hardcoding)
    total = len(per_kpi)
    num_flagged = len(flagged_kpi_ids)

    # Compute aggregate stats across all evaluated KPIs for the summary
    f1_scores = [k.f1 for k in per_kpi if k.f1 is not None]
    sem_scores = [k.semantic_similarity for k in per_kpi if k.semantic_similarity is not None]
    avg_f1 = round(sum(f1_scores) / len(f1_scores), 2) if f1_scores else None
    avg_sem = round(sum(sem_scores) / len(sem_scores), 2) if sem_scores else None

    # Plain-English verdict for the executive section of the report
    if total == 0:
        overall_verdict = (
            "No rubric KPIs were available for RAG evaluation. "
            "Only quantitative KPIs were present, which do not generate LLM answers."
        )
    elif num_flagged == 0:
        overall_verdict = (
            f"All {total} KPIs were evaluated. "
            "Answers appear well-grounded in evidence with no reliability concerns flagged."
        )
    elif num_flagged <= total * 0.2:
        overall_verdict = (
            f"{total} KPIs evaluated. {num_flagged} KPI(s) were flagged because their "
            "answers may contain claims not fully supported by the retrieved evidence. "
            "These should be manually reviewed before sharing with stakeholders."
        )
    else:
        overall_verdict = (
            f"{total} KPIs evaluated. {num_flagged} of {total} KPIs were flagged. "
            "A significant portion of answers may go beyond what the evidence supports. "
            "A thorough manual review is recommended before presenting results externally."
        )

    # 2-3 line human-readable summary for the report
    f1_line = f"Average answer accuracy (F1) across evaluated KPIs: {avg_f1:.0%}." if avg_f1 is not None else ""
    sem_line = f"Average meaning similarity to ideal answers: {avg_sem:.0%}." if avg_sem is not None else ""
    flag_line = (
        f"{num_flagged} of {total} KPI answers were flagged as potentially unsupported by evidence — manual review recommended."
        if num_flagged > 0
        else f"All {total} evaluated KPI answers appear grounded in the retrieved evidence."
    )
    summary = " ".join(filter(None, [f1_line, sem_line, flag_line]))

    # Assemble the full RagEvaluationReport object (evaluated_kpi_count = actual list length)
    rag_report = RagEvaluationReport(
        evaluated_kpi_count=total,
        flagged_kpi_count=num_flagged,
        flagged_kpi_ids=flagged_kpi_ids,
        overall_verdict=overall_verdict,
        summary=summary,
        per_kpi=per_kpi,
    )

    # --- Step 6: Return the evaluation result into pipeline state ---
    # aggregate_report_node will pick this up and write it into the YAML report.
    return {
        "rag_evaluation": rag_report.model_dump(),
        "kpi_results": merged_kpi_dicts,
    }

# code change end for RAG Eval by SN
