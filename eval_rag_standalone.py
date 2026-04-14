"""
eval_rag_standalone.py
======================
Run RAG evaluation against an EXISTING report.yaml — no pipeline rerun needed.

Usage:
    python eval_rag_standalone.py app/output/report_<run_id>.yaml
    python eval_rag_standalone.py app/output/report_<run_id>.yaml --kpi strat_ai_vision
    python eval_rag_standalone.py app/output/report_<run_id>.yaml --all

This script loads the report YAML, reconstructs the KPI results and sources,
then runs eval_rag on them and prints the results + writes a new YAML with
rag_evaluation appended.

Why this exists:
    Running eval_rag inside the full pipeline means the OpenAI API is still
    recovering from the scoring phase's LLM calls, causing 429 rate-limit errors.
    Running eval standalone — after the pipeline has finished and the rate limit
    has recovered — gives the LLM judge a clean window to operate in.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import yaml
from pathlib import Path
from typing import List, Optional

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app.models import (
    KPIDriverResult,
    KPIDefinition,
    Citation,
    RagEvaluationReport,
    RagKpiEval,
)
from app.kpi_catalog import load_kpi_catalog
from eval_rag import run_all_evaluations


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reconstruct_kpi_results(report: dict) -> List[KPIDriverResult]:
    results = []
    for r in report.get("kpi_results", []):
        raw_citations = r.get("citations", []) or []
        citations = []
        for c in raw_citations:
            if isinstance(c, dict):
                citations.append(Citation(
                    source_id=c.get("source_id", ""),
                    url=c.get("url", ""),
                    quote=c.get("quote", ""),
                ))
        results.append(KPIDriverResult(
            kpi_id=r["kpi_id"],
            pillar=r["pillar"],
            type=r["type"],
            score=float(r["score"]),
            confidence=float(r["confidence"]),
            rationale=r.get("rationale", ""),
            citations=citations,
            details=r.get("details"),
        ))
    return results


def build_rag_report(raw_results: dict) -> RagEvaluationReport:
    per_kpi: List[RagKpiEval] = []
    flagged_kpi_ids: List[str] = []

    for kpi_id, res in raw_results.items():
        ragas = res.get("ragas", {})
        judge = res.get("llm_judge", {})
        rk    = res.get("recall_at_k", {})
        f1    = res.get("f1", {})
        hall  = res.get("hallucination", {})
        mmr   = res.get("mmr", {})
        gt    = res.get("ground_truth_checks", {})

        is_flagged = hall.get("is_flagged", False)
        if is_flagged:
            flagged_kpi_ids.append(kpi_id)

        per_kpi.append(RagKpiEval(
            kpi_id=kpi_id,
            ground_truth_used=gt.get("ground_truth_used"),
            ragas_faithfulness=ragas.get("faithfulness"),
            ragas_answer_relevancy=ragas.get("answer_relevancy"),
            ragas_context_precision=ragas.get("context_precision"),
            ragas_context_recall=ragas.get("context_recall"),
            llm_judge_overall=judge.get("overall"),
            llm_judge_feedback=judge.get("feedback"),
            recall_at_3=rk.get("recall@3"),
            f1=f1.get("f1"),
            hallucination_score=hall.get("hallucination_score"),
            hallucination_flagged=is_flagged,
            mmr_diversity_score=mmr.get("diversity_score"),
            factual_correctness=gt.get("factual_correctness"),
            noise_sensitivity=gt.get("noise_sensitivity"),
            semantic_similarity=gt.get("semantic_similarity"),
        ))

    total = len(per_kpi)
    num_flagged = len(flagged_kpi_ids)

    # Aggregate stats for summary
    f1_scores = [k.f1 for k in per_kpi if k.f1 is not None]
    sem_scores = [k.semantic_similarity for k in per_kpi if k.semantic_similarity is not None]
    avg_f1 = round(sum(f1_scores) / len(f1_scores), 2) if f1_scores else None
    avg_sem = round(sum(sem_scores) / len(sem_scores), 2) if sem_scores else None

    if total == 0:
        verdict = "No rubric KPIs evaluated."
    elif num_flagged == 0:
        verdict = f"All {total} KPIs evaluated. Answers appear well-grounded in evidence."
    elif num_flagged <= total * 0.2:
        verdict = (f"{total} KPIs evaluated. {num_flagged} KPI(s) flagged for manual review.")
    else:
        verdict = (
            f"{total} KPIs evaluated. {num_flagged} of {total} flagged. "
            "Manual review recommended before sharing externally."
        )

    f1_line = f"Average answer accuracy (F1) across evaluated KPIs: {avg_f1:.0%}." if avg_f1 is not None else ""
    sem_line = f"Average meaning similarity to ideal answers: {avg_sem:.0%}." if avg_sem is not None else ""
    flag_line = (
        f"{num_flagged} of {total} KPI answers were flagged as potentially unsupported by evidence — manual review recommended."
        if num_flagged > 0
        else f"All {total} evaluated KPI answers appear grounded in the retrieved evidence."
    )
    summary = " ".join(filter(None, [f1_line, sem_line, flag_line]))

    return RagEvaluationReport(
        evaluated_kpi_count=total,
        flagged_kpi_count=num_flagged,
        flagged_kpi_ids=flagged_kpi_ids,
        overall_verdict=verdict,
        summary=summary,
        per_kpi=per_kpi,
    )


def main():
    parser = argparse.ArgumentParser(description="Run RAG eval on an existing report.yaml")
    parser.add_argument("report", help="Path to existing report YAML file")
    parser.add_argument(
        "--kpi", default="strat_ai_vision",
        help="Single KPI ID to evaluate (default: strat_ai_vision)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all rubric KPIs (ignores --kpi)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Evaluate only the first N rubric KPIs (e.g. --limit 10)"
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM-as-judge calls (faster, no API cost)"
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}")
        sys.exit(1)

    print(f"Loading report: {report_path}")
    report = load_report(str(report_path))
    print(f"Company: {report.get('company_name')}  |  Run ID: {report.get('run_id')}")

    # Reconstruct KPI results from YAML
    kpi_results = reconstruct_kpi_results(report)
    rubric_kpis = [r for r in kpi_results if r.type == "rubric"]
    print(f"Rubric KPIs in report: {len(rubric_kpis)}")

    # Load KPI definitions (for questions + rubric ground truth)
    kpi_definitions: List[KPIDefinition] = load_kpi_catalog()

    # Determine which KPIs to evaluate
    kpi_ids_to_evaluate: Optional[List[str]] = None
    if args.all:
        print(f"Evaluating all {len(rubric_kpis)} rubric KPIs")
    elif args.limit is not None:
        n = min(args.limit, len(rubric_kpis))
        kpi_ids_to_evaluate = [r.kpi_id for r in rubric_kpis[:n]]
        print(f"Evaluating first {n} rubric KPIs (--limit {args.limit})")
    else:
        kpi_ids_to_evaluate = [args.kpi]
        print(f"Evaluating 1 KPI: {args.kpi}")

    # No sources in the YAML — rebuild context from citation quotes only
    # (the full source text is not stored in report.yaml)
    sources: list = []

    run_llm_judge = not args.no_llm_judge
    if run_llm_judge:
        print("\n[Note] LLM judge is ON. If you hit 429 errors, wait a few minutes")
        print("       after your last pipeline run before running this script.\n")

    raw_results = run_all_evaluations(
        kpi_results=kpi_results,
        sources=sources,
        kpi_definitions=kpi_definitions,
        run_llm_judge=run_llm_judge,
        hallucination_threshold=0.4,
        kpi_ids_to_evaluate=kpi_ids_to_evaluate,
        verbose=True,  # Standalone mode — print full per-KPI detail to terminal
    )

    # Build structured RagEvaluationReport
    rag_report = build_rag_report(raw_results)

    # Write updated report YAML with rag_evaluation section appended
    report["rag_evaluation"] = rag_report.model_dump()
    out_path = report_path.with_stem(report_path.stem + "_with_rag_eval")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\nUpdated report written to: {out_path}")
    print(f"RAG eval: {rag_report.evaluated_kpi_count} KPIs evaluated, "
          f"{rag_report.flagged_kpi_count} flagged")


if __name__ == "__main__":
    main()
