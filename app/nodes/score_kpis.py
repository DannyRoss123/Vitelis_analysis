"""
score_kpis.py — KPI scoring node with full extension suite.

Orchestrates the 10-feature extension stack around the existing
score_rubric_kpi / score_quant_kpi calls.  The existing functions are
called unchanged; all new behaviour is layered on top.

Extension features wired here (all optional / gracefully degrading):
  Feature 1  — Score splitting       (baseline vs live scores)
  Feature 2  — Statistical scoring   (N=5 runs → mean ± σ, stored in scoring_distribution)
  Feature 3  — Quality gates         (faithfulness / stability / coverage / bleed)
  Feature 4  — Score change attribution
  Feature 5  — MLflow versioning
  Feature 6  — Retrieval metrics     (hit rate, MRR, nDCG → LangFuse scores)
  Feature 7  — BERTScore
  Feature 8  — Chain-of-thought eval
  Feature 9  — ChromaDB snapshot ID
  Feature 10 — Prompt hash

All LangFuse interactions use the shared client from app/langfuse_client.py.
MLflow logging is wrapped in try/except and never blocks the pipeline.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from app.kpi_catalog import load_kpi_catalog
from app.kpi_scoring import build_rubric_prompt, score_quant_kpi, score_rubric_kpi
from app.vectorstore import build_collection, retrieve_evidence
from app.reranker import rerank

# ── Extension modules ──────────────────────────────────────────────────────
from app.langfuse_client import (
    create_trace,
    create_span_on_trace,
    end_span,
    flush_langfuse,
    get_trace_id,
    log_score_to_trace,
    update_trace_metadata,
)
from app.score_extensions import (
    DEFAULT_FEATURE_FLAGS,
    apply_quality_gates,
    compute_bertscore,
    compute_chromadb_snapshot_id,
    compute_prompt_hash,
    compute_score_attribution,
    compute_score_split,
    retrieve_baseline_evidence,
    run_cot_eval,
    run_scoring_with_stats,
)
from app.retrieval_metrics import compute_all_retrieval_metrics
from app.mlflow_logger import log_kpi_run


# Default RAGAS config logged to MLflow (Feature 5)
_RAGAS_CONFIG: Dict[str, object] = {
    "faithfulness_threshold": 0.8,
    "top_k": 10,
    "rerank_top_n": 3,
    "embedding_model": "text-embedding-3-small",
    "judge_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}


def _get_feature_flags() -> Dict:
    """Read feature flags from env; fall back to defaults."""
    flags = dict(DEFAULT_FEATURE_FLAGS)
    n_env = os.getenv("VITELIS_N_SCORING_RUNS")
    if n_env:
        try:
            flags["n_scoring_runs"] = int(n_env)
        except ValueError:
            pass
    temp_env = os.getenv("VITELIS_SCORING_TEMPERATURE")
    if temp_env:
        try:
            flags["scoring_temperature"] = float(temp_env)
        except ValueError:
            pass
    if os.getenv("VITELIS_BERTSCORE_DISABLED", "").lower() in {"1", "true"}:
        flags["bertscore_enabled"] = False
    if os.getenv("VITELIS_COT_EVAL_DISABLED", "").lower() in {"1", "true"}:
        flags["cot_eval_enabled"] = False
    return flags


def _embedding_model_version() -> str:
    """Return a canonical embedding model version string for MLflow."""
    return "openai-text-embedding-3-small-v1"


def score_kpis_node(state: Dict) -> Dict:
    """
    Score all KPIs and attach the full extension-suite metadata to each result.

    Reads from state:
        run_id, company_domain, sources

    Writes to state:
        kpi_results       — list of KPIDriverResult dicts with new extension fields
        missing_evidence  — list of kpi_ids with insufficient evidence
        kpi_definitions   — list of KPIDefinition dicts (for evaluate_rag_node)
        chromadb_snapshot_id — collection fingerprint for this run (Feature 9)

    All extension calls are individually try/excepted so a failure in any
    single feature never prevents the base score from being written.
    """
    run_id: str = state["run_id"]
    company_domain: str = state.get("company_domain", "")
    company_id: str = company_domain or run_id
    sources: List[Dict] = state.get("sources", [])

    flags = _get_feature_flags()
    collection = build_collection(run_id)
    # Load full KPI catalog (column N in CSV = complete list of drivers; expect 80 KPIs).
    kpis = load_kpi_catalog()
    requested_kpi_ids = {
        str(k).strip() for k in (state.get("kpi_ids") or []) if str(k).strip()
    }
    if requested_kpi_ids:
        kpis = [k for k in kpis if str(k.kpi_id) in requested_kpi_ids]
    # Some CSV catalogs can contain duplicate rows with the same KPI ID.
    # Keep first occurrence to ensure deterministic one-pass scoring per KPI.
    deduped_kpis = []
    seen_kpi_ids = set()
    for k in kpis:
        kid = str(k.kpi_id)
        if kid in seen_kpi_ids:
            continue
        seen_kpi_ids.add(kid)
        deduped_kpis.append(k)
    kpis = deduped_kpis
    kpi_limit_raw = state.get("kpi_limit", os.getenv("VITELIS_KPI_LIMIT", "0"))
    try:
        kpi_limit = int(kpi_limit_raw or 0)
    except (TypeError, ValueError):
        kpi_limit = 0
    if kpi_limit > 0:
        kpis = kpis[:kpi_limit]

    # ── Feature 9: Snapshot ID (computed once for the whole run) ──────────
    snapshot_id = ""
    try:
        snapshot_id = compute_chromadb_snapshot_id(collection)
    except Exception:
        pass

    # ── Feature 3 outer context: RAGAS faithfulness from state (if present)
    # evaluate_rag_node runs *after* this node, so faithfulness is unavailable
    # here on the first pass.  We use None and the gate will log "unavailable".
    # On re-runs where rag_evaluation is pre-populated, it will be picked up.
    rag_eval_dict = state.get("rag_evaluation") or {}

    # ── LangFuse trace for the entire scoring run ─────────────────────────
    run_trace = create_trace(
        name="kpi_scoring_run",
        metadata={
            "run_id": run_id,
            "company_id": company_id,
            "chromadb_snapshot_id": snapshot_id,
        },
        tags=[os.getenv("VITELIS_ENV", "dev"), "kpi_scoring"],
    )
    run_trace_id: Optional[str] = get_trace_id(run_trace)

    results: List[Dict] = []
    missing: List[str] = []

    for kpi in kpis:
        kpi_trace = create_trace(
            name=f"kpi_score_{kpi.kpi_id}",
            metadata={
                "run_id": run_id,
                "company_id": company_id,
                "kpi_id": kpi.kpi_id,
                "kpi_pillar": kpi.pillar,
                "chromadb_snapshot_id": snapshot_id,
            },
            tags=[os.getenv("VITELIS_ENV", "dev"), kpi.pillar],
        )
        kpi_trace_id: Optional[str] = get_trace_id(kpi_trace)

        # ── Base scoring (unchanged existing logic) ───────────────────────
        if kpi.type == "rubric":
            result, missing_flag = score_rubric_kpi(
                kpi, collection, company_domain=company_domain, full_sources=sources
            )
        else:
            result, missing_flag = score_quant_kpi(kpi, sources)

        result_dict = result.model_dump()
        if missing_flag:
            missing.append(kpi.kpi_id)

        # Store traceability IDs on all KPI types
        result_dict["chromadb_snapshot_id"] = snapshot_id
        result_dict["langfuse_trace_id"] = kpi_trace_id

        # Extensions are only run for rubric KPIs (quant KPIs have no LLM prompt)
        if kpi.type != "rubric":
            results.append(result_dict)
            continue

        # ── Retrieve evidence for extensions (reuse pipeline's top-k) ─────
        retrieval_span = create_span_on_trace(kpi_trace, "retrieval")
        live_evidences = []
        baseline_evidences = []
        try:
            live_evidences = retrieve_evidence(collection, kpi.question, k=10)
            live_evidences = rerank(kpi.question, live_evidences, top_n=3)
            baseline_evidences = retrieve_baseline_evidence(collection, kpi.question, k=10)
        except Exception:
            pass
        end_span(retrieval_span)

        # ── Feature 10: Prompt hash ───────────────────────────────────────
        prompt_hash = ""
        try:
            sys_p, usr_p = build_rubric_prompt(kpi, live_evidences)
            prompt_hash = compute_prompt_hash(sys_p, usr_p)
            result_dict["prompt_hash"] = prompt_hash
        except Exception:
            pass

        # ── Feature 6: Retrieval metrics → LangFuse scores ───────────────
        retrieval_metrics: Dict[str, Optional[float]] = {}
        if flags.get("retrieval_metrics_enabled", True):
            try:
                retrieval_metrics = compute_all_retrieval_metrics(kpi.kpi_id, live_evidences)
                result_dict.setdefault("details", {})["retrieval_metrics"] = retrieval_metrics
                if kpi_trace_id:
                    if retrieval_metrics.get("hit_rate") is not None:
                        log_score_to_trace(
                            kpi_trace_id, "retrieval_hit_rate", retrieval_metrics["hit_rate"]
                        )
                    if retrieval_metrics.get("mrr") is not None:
                        log_score_to_trace(kpi_trace_id, "retrieval_mrr", retrieval_metrics["mrr"])
                    if retrieval_metrics.get("ndcg") is not None:
                        log_score_to_trace(kpi_trace_id, "retrieval_ndcg", retrieval_metrics["ndcg"])
            except Exception:
                pass

        # ── Feature 1 + 2: Score split with mean/σ over N runs ───────────
        score_split: Dict = {}
        scoring_distribution: Dict = {}
        if flags.get("score_splitting_enabled", True):
            split_span = create_span_on_trace(kpi_trace, "score_split")
            try:
                rubric_text = "\n".join(kpi.rubric or [])
                score_split = compute_score_split(
                    kpi_name=kpi.name,
                    rubric=rubric_text,
                    baseline_evidences=baseline_evidences,
                    live_evidences=live_evidences,
                    config=flags,
                )
                result_dict["baseline_score"] = score_split.get("baseline_score")
                result_dict["live_score"] = score_split.get("live_score")
                result_dict["score_split_delta"] = score_split.get("delta")
                scoring_distribution = {
                    "baseline_mean": score_split.get("baseline_score"),
                    "baseline_std": score_split.get("baseline_std"),
                    "baseline_raw_scores": score_split.get("baseline_raw_scores", []),
                    "live_mean": score_split.get("live_score"),
                    "live_std": score_split.get("live_std"),
                    "live_raw_scores": score_split.get("live_raw_scores", []),
                }
                result_dict["scoring_distribution"] = scoring_distribution
            except Exception:
                pass
            end_span(split_span, metadata={"score_split": score_split})

            # Update LangFuse trace with distribution metadata
            if kpi_trace and scoring_distribution:
                update_trace_metadata(kpi_trace, {"scoring_distribution": scoring_distribution})

        # ── Feature 3: Quality gates ──────────────────────────────────────
        gate_result: Dict = {}
        if flags.get("quality_gates_enabled", True):
            try:
                # Pull faithfulness from rag_evaluation if available
                ragas_faithfulness: Optional[float] = None
                per_kpi_rag = {
                    e.get("kpi_id"): e
                    for e in (rag_eval_dict.get("per_kpi") or [])
                }
                if kpi.kpi_id in per_kpi_rag:
                    ragas_faithfulness = per_kpi_rag[kpi.kpi_id].get("ragas_faithfulness")

                live_std = score_split.get("live_std") if score_split else None
                live_mean = score_split.get("live_score") if score_split else None

                gate_result = apply_quality_gates(
                    kpi_id=kpi.kpi_id,
                    company_id=company_id,
                    ragas_faithfulness=ragas_faithfulness,
                    score_std=live_std,
                    score_mean=live_mean,
                    retrieved_chunks=live_evidences,
                    target_entity_id=company_id,
                    trace=kpi_trace,
                    trace_id=kpi_trace_id,
                )
                result_dict["quality_gates"] = gate_result
            except Exception:
                pass

        # ── Feature 4: Score change attribution ───────────────────────────
        if flags.get("score_attribution_enabled", True):
            try:
                live_mean_score = (
                    score_split.get("live_score")
                    if score_split
                    else float(result_dict.get("score", 0))
                )
                model_version = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                attribution = compute_score_attribution(
                    company_id=company_id,
                    kpi_id=kpi.kpi_id,
                    new_mean_score=live_mean_score or 0.0,
                    current_model_version=model_version,
                    current_prompt_hash=prompt_hash,
                    current_source_fingerprint=snapshot_id,
                    trace=kpi_trace,
                )
                if attribution:
                    result_dict["score_attribution"] = attribution
            except Exception:
                pass

        # ── Feature 7: BERTScore ──────────────────────────────────────────
        try:
            chunk_texts = [doc for _meta, doc, _score in live_evidences]
            rationale = result_dict.get("rationale", "")
            f1 = compute_bertscore(
                rationale=rationale,
                reference_chunks=chunk_texts,
                trace_id=kpi_trace_id,
                trace=kpi_trace,
                config=flags,
            )
            if f1 is not None:
                result_dict["bertscore_f1"] = f1
        except Exception:
            pass

        # ── Feature 8: Chain-of-thought eval ─────────────────────────────
        try:
            rubric_text = "\n".join(kpi.rubric or [])
            chunk_texts = [doc for _meta, doc, _score in live_evidences]
            cot = run_cot_eval(
                kpi_name=kpi.name,
                rubric=rubric_text,
                retrieved_chunks=chunk_texts,
                rationale=result_dict.get("rationale", ""),
                trace_id=kpi_trace_id,
                trace=kpi_trace,
                config=flags,
            )
            if cot:
                result_dict["cot_eval"] = cot
        except Exception:
            pass

        # ── Feature 5: MLflow versioning ──────────────────────────────────
        if flags.get("mlflow_enabled", True):
            try:
                mlflow_run_id = log_kpi_run(
                    run_id=run_id,
                    company_id=company_id,
                    kpi_id=kpi.kpi_id,
                    prompt_hash=prompt_hash,
                    embedding_model_version=_embedding_model_version(),
                    chromadb_snapshot_id=snapshot_id,
                    ragas_config=_RAGAS_CONFIG,
                    langfuse_trace_id=kpi_trace_id,
                    extra_params={
                        "baseline_score": score_split.get("baseline_score"),
                        "live_score": score_split.get("live_score"),
                        "live_std": score_split.get("live_std"),
                        "bertscore_f1": result_dict.get("bertscore_f1"),
                    },
                    extra_metrics={
                        "score": result_dict.get("score"),
                        "confidence": result_dict.get("confidence"),
                        "baseline_score": score_split.get("baseline_score"),
                        "live_score": score_split.get("live_score"),
                        "score_split_delta": score_split.get("delta"),
                        "live_std": score_split.get("live_std"),
                        "bertscore_f1": result_dict.get("bertscore_f1"),
                        "retrieval_hit_rate": retrieval_metrics.get("hit_rate"),
                        "retrieval_mrr": retrieval_metrics.get("mrr"),
                        "retrieval_ndcg": retrieval_metrics.get("ndcg"),
                    },
                )
                if mlflow_run_id:
                    result_dict["mlflow_run_id"] = mlflow_run_id
                    # Cross-reference: store mlflow_run_id on the LangFuse trace
                    if kpi_trace:
                        update_trace_metadata(kpi_trace, {"mlflow_run_id": mlflow_run_id})
            except Exception:
                pass

        results.append(result_dict)

    # Flush all pending LangFuse events at end of scoring run
    try:
        flush_langfuse()
    except Exception:
        pass

    return {
        "kpi_results": results,
        "missing_evidence": missing,
        "kpi_definitions": [kpi.model_dump() for kpi in kpis],
        "chromadb_snapshot_id": snapshot_id,
    }
