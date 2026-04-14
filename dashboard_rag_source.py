"""
dashboard_rag_source.py
Run with: streamlit run dashboard_rag_source.py

Single consolidated dashboard:
- **Report (YAML)**: all RAG / retrieval / BERTScore / CoT / mean±σ fields from `app/output/report_*.yaml`
- **Postgres (live DB)**: source coverage, score distribution, DB RAG columns (requires DATABASE_URL)
"""
from __future__ import annotations

from glob import glob
import math
import os

import pandas as pd
import streamlit as st
import yaml

from app.db_explorer_panel import render_postgres_explorer


st.set_page_config(
    page_title="Consolidated RAG Evaluation",
    page_icon="🧪",
    layout="wide",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "app", "output")


@st.cache_data(ttl=60)
def load_reports() -> list[dict]:
    reports: list[dict] = []
    for path in sorted(glob(os.path.join(OUTPUT_DIR, "report_*.yaml"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data["_path"] = path
            reports.append(data)
        except Exception:
            continue
    reports.sort(key=lambda r: (r.get("timestamp", ""), r.get("run_id", "")), reverse=True)
    return reports


def _fmt(v, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def _mean_std(series: pd.Series) -> tuple[str, str]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "—", "—"
    return f"{float(s.mean()):.4f}", f"{float(s.std(ddof=0)):.4f}"


st.title("🧪 Consolidated RAG evaluation")
st.caption("One app: YAML evaluation reports and optional live Postgres metrics.")

tab_report, tab_db = st.tabs(["Report (YAML output)", "Postgres (live DB)"])

with tab_report:
    reports = load_reports()
    if not reports:
        st.warning("No `report_*.yaml` files found in `app/output`. Run the eval pipeline or copy reports here.")
    else:
        labels = [
            f"{r.get('company_name', 'Unknown')} — {r.get('run_id', '?')} ({(r.get('timestamp') or '')[:10]})"
            for r in reports
        ]
        selected_label = st.selectbox("Select report", labels, index=0)
        report = reports[labels.index(selected_label)]

        kpi_results = report.get("kpi_results", []) or []
        rubric_kpis = [k for k in kpi_results if k.get("type") == "rubric"]
        rag = report.get("rag_evaluation") or {}
        per_kpi_rag = {str(x.get("kpi_id")): x for x in (rag.get("per_kpi") or [])}

        st.subheader(f"Run `{report.get('run_id', '?')}` — {report.get('company_name', 'Unknown')}")

        with st.expander("Where are mean ± σ (baseline / live)? Why are many cells empty?"):
            st.markdown(
                """
                - **Mean and standard deviation** for rubric **live** vs **baseline** scores come from
                  `kpi_results[].scoring_distribution` (`live_std`, `baseline_std`) together with
                  `live_score` and `baseline_score`. They are filled when the full **Score-5 / `score_kpis`**
                  stochastic loop ran (typically N repeated scores).
                - **DB-imported** KPI rows often have `scoring_distribution: null` and `baseline_score` /
                  `live_score` **null** because only stored DB scores were loaded—no multi-draw distribution.
                - **RAGAS / context recall / answer correctness** can be **null** when the metric was not
                  computed for that KPI (e.g. missing rubric context, RAGAS skip, or judge unavailable).
                - **Hit rate, MRR, nDCG** are **null** when there is no golden row for that `kpi_id` in the
                  **`golden_chunks`** Postgres table (or chunk IDs do not match retrieval). Optional YAML
                  only applies if you set `GOLDEN_CHUNKS_SOURCE=yaml` and `GOLDEN_CHUNKS_PATH`.
                - **LLM-as-judge** may be null with a note in **LLM judge note** (e.g. no API key).
                - **Chain-of-thought eval** is null if the CoT scorer is disabled in feature flags, the judge
                  model is unavailable, or the CoT scorer failed.
                """
            )

        with st.expander("Source coverage in this report (YAML) vs Postgres tab"):
            st.markdown(
                """
                In the **table below**, **Source coverage (YAML)** is
                `kpi_results[].quality_gates.gates.source_coverage_gate.primary_fraction`: the fraction of
                retrieved chunks tagged as **primary** tier after quality-gate evaluation in the scoring
                pipeline (pass threshold is typically ≥ 0.4). It is **empty (—)** when `quality_gates` is
                null (common for DB-only imports). The **Postgres** tab shows a different metric: share of
                `sources` rows with non-empty fetched **content** (operational crawl coverage).
                """
            )

        if rag:
            st.subheader("RAG evaluation batch summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Evaluated KPIs", rag.get("evaluated_kpi_count", "—"))
            c2.metric("Flagged KPIs", rag.get("flagged_kpi_count", "—"))
            c3.metric("Rubric KPI rows", len(rubric_kpis))
            c4.metric("Snapshot ID", (report.get("chromadb_snapshot_id", "") or "—")[:12] + "…")
            ov = rag.get("overall_verdict") or ""
            sm = rag.get("summary") or ""
            if ov:
                st.info(ov)
            if sm:
                st.caption(sm)

        rows = []
        for k in rubric_kpis:
            kid = str(k.get("kpi_id", ""))
            dist = k.get("scoring_distribution") or {}
            gates = (k.get("quality_gates") or {}).get("gates") or {}
            coverage = (gates.get("source_coverage_gate") or {}).get("primary_fraction")
            r = per_kpi_rag.get(kid, {})
            ret = (k.get("details") or {}).get("retrieval_metrics") or {}
            if not ret and r:
                ret = {
                    "hit_rate": r.get("retrieval_hit_rate"),
                    "mrr": r.get("retrieval_mrr"),
                    "ndcg": r.get("retrieval_ndcg"),
                }
            cot = k.get("cot_eval") or r.get("cot_eval") or {}
            rows.append(
                {
                    "KPI ID": kid,
                    "Pillar": k.get("pillar", ""),
                    "RAGAS faithfulness": r.get("ragas_faithfulness"),
                    "Answer relevance": r.get("ragas_answer_relevancy"),
                    "Context recall": r.get("ragas_context_recall"),
                    "Context precision": r.get("ragas_context_precision"),
                    "MMR diversity": r.get("mmr_diversity_score"),
                    "LLM-as-judge": r.get("llm_judge_overall"),
                    "LLM judge note": (r.get("llm_judge_feedback") or "")[:200],
                    "Answer correctness": r.get("factual_correctness"),
                    "Hallucination rate": r.get("hallucination_score"),
                    "Hit rate": ret.get("hit_rate"),
                    "MRR": ret.get("mrr"),
                    "nDCG": ret.get("ndcg"),
                    "Recall@k": r.get("recall_at_3"),
                    "Answer accuracy (F1)": r.get("f1"),
                    "Baseline mean": k.get("baseline_score"),
                    "Baseline σ": dist.get("baseline_std"),
                    "Live mean": k.get("live_score"),
                    "Live σ": dist.get("live_std"),
                    "Score delta": k.get("score_split_delta"),
                    "BERTScore F1": k.get("bertscore_f1") or r.get("bertscore_f1"),
                    "Low semantic grounding": k.get("low_semantic_grounding"),
                    "CoT specificity": cot.get("cot_specificity"),
                    "CoT evidence": cot.get("cot_evidence"),
                    "CoT alignment": cot.get("cot_alignment"),
                    "Source coverage (YAML)": coverage,
                    "Prompt hash": (k.get("prompt_hash") or "")[:12] + "…",
                    "MLflow run": (k.get("mlflow_run_id") or "")[:12] + "…",
                    "Langfuse trace": (k.get("langfuse_trace_id") or "")[:12] + "…",
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            st.info("No rubric KPI rows in this report.")
        else:
            main_cols = [
                "KPI ID",
                "Pillar",
                "RAGAS faithfulness",
                "Answer relevance",
                "Context recall",
                "Context precision",
                "MMR diversity",
                "LLM-as-judge",
                "LLM judge note",
                "Answer correctness",
                "Hallucination rate",
                "Hit rate",
                "MRR",
                "nDCG",
                "Recall@k",
                "Answer accuracy (F1)",
                "Baseline mean",
                "Baseline σ",
                "Live mean",
                "Live σ",
                "Score delta",
                "BERTScore F1",
                "Low semantic grounding",
                "CoT specificity",
                "CoT evidence",
                "CoT alignment",
                "Source coverage (YAML)",
            ]
            trace_cols = ["Prompt hash", "MLflow run", "Langfuse trace"]
            present_main = [c for c in main_cols if c in df.columns]
            st.subheader("Per-KPI metrics (all requested columns)")
            st.dataframe(df[present_main], width="stretch", height=520)

            st.subheader("Run-level mean ± σ (numeric columns, non-null KPIs only)")
            numeric_for_agg = [
                "RAGAS faithfulness",
                "Answer relevance",
                "Context recall",
                "Context precision",
                "MMR diversity",
                "LLM-as-judge",
                "Answer correctness",
                "Hallucination rate",
                "Hit rate",
                "MRR",
                "nDCG",
                "Recall@k",
                "Answer accuracy (F1)",
                "Baseline mean",
                "Baseline σ",
                "Live mean",
                "Live σ",
                "BERTScore F1",
            ]
            agg_rows = []
            for col in numeric_for_agg:
                if col not in df.columns:
                    continue
                m, s = _mean_std(df[col])
                agg_rows.append({"Metric": col, "Mean (this run)": m, "Std dev (across KPIs)": s})
            if agg_rows:
                st.dataframe(pd.DataFrame(agg_rows), width="stretch", hide_index=True)

            st.subheader("Mean ± σ (baseline / live) — compact view")
            dist_rows = []
            for _, row in df.iterrows():
                dist_rows.append(
                    {
                        "KPI ID": row["KPI ID"],
                        "Baseline mean ± σ": f"{_fmt(row['Baseline mean'], 3)} ± {_fmt(row['Baseline σ'], 3)}",
                        "Live mean ± σ": f"{_fmt(row['Live mean'], 3)} ± {_fmt(row['Live σ'], 3)}",
                        "Score delta": _fmt(row["Score delta"], 3),
                    }
                )
            st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)

            st.subheader("Retrieval + source coverage (YAML)")
            cov_cols = [c for c in ["KPI ID", "Source coverage (YAML)", "Hit rate", "MRR", "nDCG"] if c in df.columns]
            st.dataframe(df[cov_cols], width="stretch", hide_index=True)

            with st.expander("Traceability columns"):
                st.dataframe(df[[c for c in trace_cols if c in df.columns]], width="stretch", hide_index=True)

with tab_db:
    st.subheader("Live database")
    render_postgres_explorer(embedded=True)
