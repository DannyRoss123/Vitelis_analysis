"""
Postgres-backed RAG + source explorer for Streamlit.
Used by dashboard_rag_source.py (tab) and dashboard_db_eval.py (standalone).
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


def _get_db_url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


@st.cache_resource
def get_engine():
    db_url = _get_db_url()
    if not db_url:
        return None
    return create_engine(db_url)


def qdf(sql: str, quiet: bool = False) -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(sql, conn)
    except Exception as exc:
        if not quiet:
            st.warning(f"Query failed; using fallback scope. Details: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def table_columns(table_name: str) -> set[str]:
    df = qdf(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = '{table_name}'
        """,
        quiet=True,
    )
    if df.empty:
        return set()
    return set(df["column_name"].astype(str).tolist())


@st.cache_data(ttl=300)
def table_column_types(table_name: str) -> dict[str, str]:
    df = qdf(
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = '{table_name}'
        """,
        quiet=True,
    )
    if df.empty:
        return {}
    return {str(r["column_name"]): str(r["data_type"]) for _, r in df.iterrows()}


def _eq_filter(alias: str, col: str, value: int | str, types: dict[str, str]) -> str:
    dtype = (types.get(col) or "").lower()
    if "uuid" in dtype:
        return f"{alias}.{col}::text = '{value}'"
    if "character" in dtype or "text" in dtype:
        return f"{alias}.{col} = '{value}'"
    return f"{alias}.{col} = {value}"


def _rag_metric_cols(types: dict[str, str]) -> list[str]:
    rag_keywords = (
        "rag", "faith", "relev", "precision", "recall", "mrr", "ndcg",
        "mmr", "halluc", "factual", "semantic", "noise", "judge",
        "bertscore", "cot", "correctness",
    )
    numeric_types = {
        "smallint", "integer", "bigint", "real", "double precision", "numeric", "decimal",
    }
    cols: list[str] = []
    for c, t in types.items():
        lc = c.lower()
        if any(k in lc for k in rag_keywords) and t.lower() in numeric_types:
            cols.append(c)
    return sorted(cols)


def render_postgres_explorer(*, embedded: bool = False) -> None:
    db_url = _get_db_url()
    if not db_url:
        st.error("Set `DATABASE_URL` first, then reload.") if not embedded else st.warning(
            "Set `DATABASE_URL` to use this tab."
        )
        if not embedded:
            st.code("set DATABASE_URL=postgresql://postgres:***@localhost:5432/your_db")
            st.stop()
        return

    runs_df = qdf(
        """
        SELECT id, created_at
        FROM runs
        ORDER BY created_at DESC
        LIMIT 50
        """
    )
    if runs_df.empty:
        st.warning("No runs found in `runs`.")
        if not embedded:
            st.stop()
        return

    runs_df["label"] = runs_df.apply(lambda r: f"run={r['id']}  ({r['created_at']})", axis=1)
    selected = st.selectbox("Select run", runs_df["label"].tolist(), index=0)
    run_id = int(runs_df.loc[runs_df["label"] == selected, "id"].iloc[0])

    rr_cols = table_columns("report_runs")
    if {"company_id", "run_id"}.issubset(rr_cols):
        companies_df = qdf(
            f"""
            SELECT DISTINCT c.id, c.name
            FROM companies c
            JOIN report_runs rr ON rr.company_id = c.id
            WHERE rr.run_id = {run_id}
            ORDER BY c.name
            """
        )
    else:
        companies_df = pd.DataFrame()
    if companies_df.empty:
        companies_df = qdf("SELECT id, name FROM companies ORDER BY name")
        st.caption("`report_runs` has no `run_id`; company selector is using all companies.")
    company_name = st.selectbox("Select company", companies_df["name"].tolist(), index=0)
    company_id = int(companies_df.loc[companies_df["name"] == company_name, "id"].iloc[0])

    scope_notes: list[str] = []

    kpi_cols = table_columns("kpi_node_scores")
    kar_cols = table_columns("kpi_analysis_runs")
    rs_cols = table_columns("resolved_signals")
    src_cols = table_columns("sources")
    se_cols = table_columns("source_evaluations")
    kns_types = table_column_types("kpi_node_scores")
    kar_types = table_column_types("kpi_analysis_runs")
    rs_types = table_column_types("resolved_signals")
    src_types = table_column_types("sources")
    se_types = table_column_types("source_evaluations")

    kpi_scope_join = ""
    kpi_scope_filters = ["kns.status = 'success'"]
    if "run_id" in kpi_cols:
        kpi_scope_filters.append(_eq_filter("kns", "run_id", run_id, kns_types))
        scope_notes.append("`kpi_node_scores` filtered by `run_id`.")
    if "company_id" in kpi_cols:
        kpi_scope_filters.append(_eq_filter("kns", "company_id", company_id, kns_types))
        scope_notes.append("`kpi_node_scores` filtered by `company_id`.")
    elif "kpi_analysis_run_id" in kpi_cols and {"id", "run_id", "company_id"}.issubset(kar_cols):
        kpi_scope_join = "JOIN kpi_analysis_runs kar ON kar.id = kns.kpi_analysis_run_id"
        kpi_scope_filters.append(_eq_filter("kar", "run_id", run_id, kar_types))
        kpi_scope_filters.append(_eq_filter("kar", "company_id", company_id, kar_types))
        scope_notes.append("`kpi_node_scores` scoped via `kpi_analysis_runs`.")
    else:
        scope_notes.append("`kpi_node_scores` has no run/company linkage; showing broader scope.")

    score_dist_df = qdf(
        f"""
        SELECT
          kns.node_title,
          COUNT(*) AS n_scored,
          ROUND(AVG(kns.score)::numeric, 4) AS mean_score,
          ROUND(COALESCE(STDDEV_POP(kns.score), 0)::numeric, 4) AS sigma_score
        FROM kpi_node_scores kns
        {kpi_scope_join}
        WHERE {' AND '.join(kpi_scope_filters)}
        GROUP BY kns.node_title
        ORDER BY kns.node_title
        """
    )

    src_scope_filters = ["1=1"]
    if "run_id" in src_cols:
        src_scope_filters.append(_eq_filter("s", "run_id", run_id, src_types).replace("s.", ""))
        scope_notes.append("`sources` filtered by `run_id`.")
    if "company_id" in src_cols:
        src_scope_filters.append(_eq_filter("s", "company_id", company_id, src_types).replace("s.", ""))
        scope_notes.append("`sources` filtered by `company_id`.")
    if src_scope_filters == ["1=1"]:
        scope_notes.append("`sources` has no run/company linkage; showing broader scope.")

    source_cov_df = qdf(
        f"""
        SELECT
          COUNT(*) AS total_sources,
          SUM(CASE WHEN COALESCE(NULLIF(TRIM(content), ''), '') <> '' THEN 1 ELSE 0 END) AS non_empty_content_sources,
          SUM(CASE WHEN content ILIKE 'Oops, something went wrong%' THEN 1 ELSE 0 END) AS fetch_error_sources
        FROM sources
        WHERE {' AND '.join(src_scope_filters)}
        """
    )

    rs_scope_filters = ["1=1"]
    if "company_id" in rs_cols:
        rs_scope_filters.append(_eq_filter("rs", "company_id", company_id, rs_types).replace("rs.", ""))
    if "run_id" in rs_cols:
        rs_scope_filters.append(_eq_filter("rs", "run_id", run_id, rs_types).replace("rs.", ""))
    if rs_scope_filters == ["1=1"]:
        scope_notes.append("`resolved_signals` has no run/company linkage; showing broader scope.")
    else:
        scope_notes.append("`resolved_signals` filtered by available run/company keys.")

    signal_cov_df = qdf(
        f"""
        SELECT
          status,
          COUNT(*) AS cnt
        FROM resolved_signals
        WHERE {' AND '.join(rs_scope_filters)}
        GROUP BY status
        ORDER BY status
        """
    )

    se_scope_join = ""
    se_scope_filters = ["1=1"]
    if "run_id" in se_cols:
        se_scope_filters.append(_eq_filter("se", "run_id", run_id, se_types))
    if "company_id" in se_cols:
        se_scope_filters.append(_eq_filter("se", "company_id", company_id, se_types))
    if "source_id" in se_cols and ("run_id" in src_cols or "company_id" in src_cols):
        se_scope_join = "LEFT JOIN sources s ON s.id = se.source_id"
        if "run_id" in src_cols:
            se_scope_filters.append(_eq_filter("s", "run_id", run_id, src_types))
        if "company_id" in src_cols:
            se_scope_filters.append(_eq_filter("s", "company_id", company_id, src_types))

    src_eval_df = qdf(
        f"""
        SELECT
          COUNT(*) AS eval_rows,
          ROUND(AVG(COALESCE(se.score, 0))::numeric, 4) AS avg_score
        FROM source_evaluations se
        {se_scope_join}
        WHERE {' AND '.join(se_scope_filters)}
        """
    )

    kar_rag_cols = _rag_metric_cols(kar_types)
    kns_rag_cols = _rag_metric_cols(kns_types)
    rag_metrics_df = pd.DataFrame()
    rag_scope_note = ""

    if kar_rag_cols:
        run_filter = _eq_filter("kar", "run_id", run_id, kar_types) if "run_id" in kar_cols else "1=1"
        company_filter = _eq_filter("kar", "company_id", company_id, kar_types) if "company_id" in kar_cols else "1=1"
        select_sql = ", ".join([f"ROUND(AVG(kar.{c})::numeric, 4) AS {c}" for c in kar_rag_cols])
        rag_metrics_df = qdf(
            f"""
            SELECT {select_sql}
            FROM kpi_analysis_runs kar
            WHERE {run_filter} AND {company_filter}
            """
        )
        rag_scope_note = "`kpi_analysis_runs`"
    elif kns_rag_cols:
        filters = ["1=1"]
        if "run_id" in kpi_cols:
            filters.append(_eq_filter("kns", "run_id", run_id, kns_types))
        if "company_id" in kpi_cols:
            filters.append(_eq_filter("kns", "company_id", company_id, kns_types))
        select_sql = ", ".join([f"ROUND(AVG(kns.{c})::numeric, 4) AS {c}" for c in kns_rag_cols])
        rag_metrics_df = qdf(
            f"""
            SELECT {select_sql}
            FROM kpi_node_scores kns
            WHERE {' AND '.join(filters)}
            """
        )
        rag_scope_note = "`kpi_node_scores`"

    left, right = st.columns(2)
    with left:
        st.subheader("Source coverage (DB)")
        if not source_cov_df.empty:
            row = source_cov_df.iloc[0]
            total = int(row["total_sources"] or 0)
            non_empty = int(row["non_empty_content_sources"] or 0)
            errors = int(row["fetch_error_sources"] or 0)
            coverage = (non_empty / total) if total else 0.0
            c1, c2, c3 = st.columns(3)
            c1.metric("Total sources", total)
            c2.metric("Non-empty content", non_empty)
            c3.metric("Coverage %", f"{coverage:.1%}")
            st.metric("Fetch-error pages", errors)
        else:
            st.info("No rows in `sources`.")

        st.subheader("Resolved signals status")
        if not signal_cov_df.empty:
            st.dataframe(signal_cov_df, width="stretch", hide_index=True)
        else:
            st.info("No rows in `resolved_signals` for this company.")

    with right:
        st.subheader("Score distribution (`kpi_node_scores`)")
        if not score_dist_df.empty:
            st.dataframe(score_dist_df, width="stretch", height=430, hide_index=True)
        else:
            st.info("No successful rows in `kpi_node_scores`.")

        st.subheader("Source evaluation aggregate")
        if not src_eval_df.empty:
            st.dataframe(src_eval_df, width="stretch", hide_index=True)

    st.subheader("RAG metrics (DB columns)")
    if not rag_metrics_df.empty and len(rag_metrics_df.columns) > 0:
        st.caption(f"Auto-discovered numeric RAG-like columns from {rag_scope_note}.")
        melted = rag_metrics_df.melt(var_name="metric", value_name="avg_value").sort_values("metric")
        st.dataframe(melted, width="stretch", hide_index=True)
    else:
        st.info(
            "No explicit numeric RAG metric columns found in `kpi_analysis_runs` or `kpi_node_scores`.\n"
            "If metrics are stored in JSON fields, parse them in the pipeline or add typed columns."
        )

    st.divider()
    st.caption("Scope resolution")
    for note in sorted(set(scope_notes)):
        st.markdown(f"- {note}")

    st.subheader("Data quality checks")
    dq_df = qdf(
        f"""
        SELECT
          url,
          CASE
            WHEN content ILIKE 'Oops, something went wrong%' THEN 'fetch_error'
            WHEN COALESCE(NULLIF(TRIM(content), ''), '') = '' THEN 'empty_content'
            ELSE 'ok'
          END AS quality_flag
        FROM sources
        WHERE {' AND '.join(src_scope_filters)}
        ORDER BY quality_flag DESC, url
        LIMIT 200
        """
    )
    if not dq_df.empty:
        st.dataframe(dq_df, width="stretch", hide_index=True)
    else:
        st.info("No source rows for quality checks.")
