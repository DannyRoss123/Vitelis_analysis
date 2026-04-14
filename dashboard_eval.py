"""
dashboard_eval.py — Ground-Truth Evaluation Dashboard
Run with: streamlit run dashboard_eval.py

Loads all app/output/eval_*.json files and displays four tabs:
  1. KPI Comparison  — pipeline score vs analyst answer, side by side
  2. Source Analysis — which URLs were cited most (pipeline vs analyst)
  3. Unmatched       — KPIs or data points with no counterpart
  4. Run Comparison  — detailed diff between two runs for the same company
"""
from __future__ import annotations

import json
import os
from collections import Counter
from glob import glob

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────── #
# Page config
# ─────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Vitelis — Ground Truth Comparison",
    page_icon="🔍",
    layout="wide",
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "app", "output")


# ─────────────────────────────────────────────────────────────────────────── #
# Data loading — all runs per company, newest first
# ─────────────────────────────────────────────────────────────────────────── #
@st.cache_data(ttl=60)
def load_all_eval_reports() -> dict[str, list[dict]]:
    """Load all eval_*.json files keyed by company name; all runs, newest first."""
    pattern = os.path.join(EVAL_DIR, "eval_*.json")
    reports: dict[str, list[dict]] = {}
    for path in sorted(glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            company = data.get("company_name", os.path.basename(path))
            reports.setdefault(company, []).append(data)
        except Exception:
            pass
    # Sort each company's runs newest first (timestamp, then run_id as tiebreak)
    for company in reports:
        reports[company].sort(
            key=lambda r: (r.get("timestamp", ""), r.get("run_id", "")),
            reverse=True,
        )
    return reports


def _run_label(run: dict) -> str:
    ts = run.get("timestamp", "")[:10]
    rid = run.get("run_id", "?")
    return f"{rid}  ({ts})"


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: build comparison dataframe
# ─────────────────────────────────────────────────────────────────────────── #
def build_comparison_df(report: dict) -> pd.DataFrame:
    rows = []
    for c in report.get("comparisons", []):
        rows.append({
            "KPI ID": c.get("kpi_id", ""),
            "KPI Name (pipeline)": c.get("kpi_name", ""),
            "Pipeline Score": c.get("pipeline_score", 0),
            "Pipeline Confidence": c.get("pipeline_confidence", 0.0),
            "GT Data Point": c.get("ground_truth_name", ""),
            "GT Answer": c.get("ground_truth_answer", ""),
            "GT Explanation": c.get("ground_truth_explanation", ""),
            "Match Confidence": c.get("match_confidence", 0),
            "Pipeline Sources": c.get("pipeline_sources", []),
            "GT Sources": c.get("ground_truth_sources", []),
            "Pipeline Rationale": c.get("pipeline_rationale", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: build run-comparison dataframe
# ─────────────────────────────────────────────────────────────────────────── #
def build_run_diff_df(run_a: dict, run_b: dict) -> pd.DataFrame:
    """
    Align KPIs from two runs by kpi_id and compute deltas.
    Returns one row per KPI with scores, confidences, and source change counts.
    """
    def index_comparisons(run: dict) -> dict:
        return {c["kpi_id"]: c for c in run.get("comparisons", [])}

    a_by_id = index_comparisons(run_a)
    b_by_id = index_comparisons(run_b)
    all_ids = sorted(set(a_by_id) | set(b_by_id))

    rows = []
    for kid in all_ids:
        a = a_by_id.get(kid)
        b = b_by_id.get(kid)

        name = (a or b).get("kpi_name", kid)
        score_a = float(a["pipeline_score"]) if a else None
        score_b = float(b["pipeline_score"]) if b else None
        conf_a = float(a.get("pipeline_confidence", 0.0)) if a else None
        conf_b = float(b.get("pipeline_confidence", 0.0)) if b else None

        srcs_a = set(a.get("pipeline_sources", [])) if a else set()
        srcs_b = set(b.get("pipeline_sources", [])) if b else set()
        added = srcs_b - srcs_a      # sources in B not in A
        removed = srcs_a - srcs_b    # sources in A not in B

        score_delta = (score_b - score_a) if (score_a is not None and score_b is not None) else None
        conf_delta = (conf_b - conf_a) if (conf_a is not None and conf_b is not None) else None

        rows.append({
            "KPI ID": kid,
            "KPI Name": name,
            "Score A": score_a,
            "Score B": score_b,
            "Score Δ": score_delta,
            "Conf A": round(conf_a, 3) if conf_a is not None else None,
            "Conf B": round(conf_b, 3) if conf_b is not None else None,
            "Conf Δ": round(conf_delta, 3) if conf_delta is not None else None,
            "Sources Added": len(added),
            "Sources Removed": len(removed),
            "_srcs_added": sorted(added),
            "_srcs_removed": sorted(removed),
            "_srcs_shared": sorted(srcs_a & srcs_b),
            "_rationale_a": a.get("pipeline_rationale", "") if a else "",
            "_rationale_b": b.get("pipeline_rationale", "") if b else "",
            "_in_a": a is not None,
            "_in_b": b is not None,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────── #
# Main app
# ─────────────────────────────────────────────────────────────────────────── #
st.title("🔍 Ground Truth Evaluation Dashboard")
st.caption("Compares the pipeline's KPI scores against analyst research findings.")

all_company_runs = load_all_eval_reports()

if not all_company_runs:
    st.warning(
        "No evaluation reports found in `app/output/eval_*.json`.\n\n"
        "Run `python run_eval.py --company-folder \"Orange S.A\"` first."
    )
    st.stop()

# ── Company selector ─────────────────────────────────────────────────────── #
companies = sorted(all_company_runs.keys())
selected_company = st.selectbox("Select company", companies)

company_runs = all_company_runs[selected_company]  # newest first
report = company_runs[0]                           # show latest run in tabs 1-3

col1, col2, col3, col4 = st.columns(4)
comparisons = report.get("comparisons", [])
col1.metric("Run ID", report.get("run_id", "—"))
col2.metric("Matched KPIs", len(comparisons))
col3.metric("Unmatched KPIs", len(report.get("unmatched_kpis", [])))
col4.metric("Unmatched GT points", len(report.get("unmatched_data_points", [])))

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────── #
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 KPI Comparison",
    "🔗 Source Analysis",
    "⚠️ Unmatched",
    "🔄 Run Comparison",
])

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — KPI Comparison
# ════════════════════════════════════════════════════════════════════════════ #
with tab1:
    st.subheader("Pipeline Score vs Analyst Ground Truth")

    df = build_comparison_df(report)
    if df.empty:
        st.info("No matched comparisons to display.")
    else:
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        min_confidence = filter_col1.slider(
            "Minimum match confidence", 0.0, 1.0, 0.35, step=0.05
        )
        score_filter = filter_col2.multiselect(
            "Filter by pipeline score",
            options=[1, 2, 3, 4, 5],
            default=[],
        )

        filtered = df[df["Match Confidence"] >= min_confidence]
        if score_filter:
            filtered = filtered[filtered["Pipeline Score"].isin(score_filter)]

        st.caption(f"Showing {len(filtered)} of {len(df)} matched KPI pairs")

        # Display table — core columns
        display_cols = [
            "KPI Name (pipeline)", "Pipeline Score", "Pipeline Confidence",
            "GT Data Point", "GT Answer", "Match Confidence",
        ]
        st.dataframe(
            filtered[display_cols].style.background_gradient(
                subset=["Pipeline Score"], cmap="RdYlGn", vmin=1, vmax=5
            ).background_gradient(
                subset=["Pipeline Confidence"], cmap="Blues", vmin=0, vmax=1
            ).background_gradient(
                subset=["Match Confidence"], cmap="Purples", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=400,
        )

        # Detail expander — click to see full rationale & sources
        st.subheader("KPI Detail")
        kpi_names = filtered["KPI Name (pipeline)"].tolist()
        if kpi_names:
            selected_kpi = st.selectbox("Select a KPI for full detail", kpi_names)
            row = filtered[filtered["KPI Name (pipeline)"] == selected_kpi].iloc[0]

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Pipeline output**")
                st.metric("Score", row["Pipeline Score"])
                st.metric("Confidence", f"{row['Pipeline Confidence']:.2f}")
                st.markdown(f"*Rationale:* {row['Pipeline Rationale']}")
                if row["Pipeline Sources"]:
                    with st.expander("Pipeline sources cited"):
                        for url in row["Pipeline Sources"]:
                            st.markdown(f"- [{url}]({url})")

            with detail_col2:
                st.markdown("**Analyst ground truth**")
                st.markdown(f"**Data point:** {row['GT Data Point']}")
                st.markdown(f"**Answer:** `{row['GT Answer']}`")
                st.markdown(f"*Explanation:* {row['GT Explanation']}")
                if row["GT Sources"]:
                    with st.expander(f"Analyst sources ({len(row['GT Sources'])})"):
                        for url in row["GT Sources"][:20]:
                            st.markdown(f"- [{url}]({url})")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Source Analysis
# ════════════════════════════════════════════════════════════════════════════ #
with tab2:
    st.subheader("Source Attribution")

    if not comparisons:
        st.info("No comparisons available.")
    else:
        # Pipeline sources
        all_pipeline_urls: list[str] = []
        for c in comparisons:
            all_pipeline_urls.extend(c.get("pipeline_sources", []))

        # Analyst sources
        all_gt_urls: list[str] = []
        for c in comparisons:
            all_gt_urls.extend(c.get("ground_truth_sources", []))

        src_col1, src_col2 = st.columns(2)

        with src_col1:
            st.markdown("**Top Pipeline Sources**  *(URLs cited most across KPIs)*")
            pipeline_counts = Counter(all_pipeline_urls).most_common(20)
            if pipeline_counts:
                pipe_df = pd.DataFrame(pipeline_counts, columns=["URL", "Citations"])
                st.dataframe(pipe_df, use_container_width=True, height=350)
            else:
                st.info("No pipeline source citations found.")

        with src_col2:
            st.markdown("**Top Analyst Sources**  *(URLs used most across data points)*")
            gt_counts = Counter(all_gt_urls).most_common(20)
            if gt_counts:
                gt_df = pd.DataFrame(gt_counts, columns=["URL", "Citations"])
                st.dataframe(gt_df, use_container_width=True, height=350)
            else:
                st.info("No analyst source URLs found.")

        st.divider()
        st.subheader("Per-KPI Source Overlap")
        st.caption("How much do the pipeline's cited URLs overlap with the analyst's URLs for each KPI?")

        overlap_rows = []
        for c in comparisons:
            p_set = set(c.get("pipeline_sources", []))
            g_set = set(c.get("ground_truth_sources", []))
            if p_set or g_set:
                overlap = len(p_set & g_set)
                union = len(p_set | g_set)
                jaccard = round(overlap / union, 3) if union > 0 else 0.0
                overlap_rows.append({
                    "KPI": c.get("kpi_name", c.get("kpi_id", "")),
                    "Pipeline URLs": len(p_set),
                    "Analyst URLs": len(g_set),
                    "Shared URLs": overlap,
                    "Jaccard similarity": jaccard,
                })

        if overlap_rows:
            overlap_df = pd.DataFrame(overlap_rows).sort_values("Jaccard similarity", ascending=False)
            st.dataframe(
                overlap_df.style.background_gradient(
                    subset=["Jaccard similarity"], cmap="Greens", vmin=0, vmax=1
                ),
                use_container_width=True,
                height=350,
            )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Unmatched
# ════════════════════════════════════════════════════════════════════════════ #
with tab3:
    st.subheader("Unmatched Items")

    unmatched_kpis = report.get("unmatched_kpis", [])
    unmatched_gt = report.get("unmatched_data_points", [])

    um_col1, um_col2 = st.columns(2)

    with um_col1:
        st.markdown(f"**KPIs with no ground-truth match** ({len(unmatched_kpis)})")
        st.caption(
            "These KPIs were scored by the pipeline but could not be matched "
            "to any analyst data point by name."
        )
        if unmatched_kpis:
            st.dataframe(pd.DataFrame({"KPI Name": unmatched_kpis}), use_container_width=True)
        else:
            st.success("All pipeline KPIs were matched.")

    with um_col2:
        st.markdown(f"**Analyst data points with no KPI match** ({len(unmatched_gt)})")
        st.caption(
            "These analyst findings exist in the ground truth but no pipeline KPI "
            "was matched to them — they may represent gaps in the KPI set."
        )
        if unmatched_gt:
            st.dataframe(
                pd.DataFrame({"Ground Truth Data Point": unmatched_gt}),
                use_container_width=True,
            )
        else:
            st.success("All analyst data points were matched.")

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 4 — Run Comparison
# ════════════════════════════════════════════════════════════════════════════ #
with tab4:
    st.subheader("Run-to-Run Comparison")

    if len(company_runs) < 2:
        st.info(
            f"Only one run found for **{selected_company}**. "
            "Run the pipeline again to generate a second run for comparison."
        )
    else:
        run_labels = [_run_label(r) for r in company_runs]

        rc1, rc2 = st.columns(2)
        label_a = rc1.selectbox("Run A (baseline)", run_labels, index=0)
        label_b = rc2.selectbox("Run B (compare)", run_labels, index=1)

        if label_a == label_b:
            st.warning("Select two different runs to compare.")
        else:
            run_a = company_runs[run_labels.index(label_a)]
            run_b = company_runs[run_labels.index(label_b)]

            diff_df = build_run_diff_df(run_a, run_b)

            # ── Summary metrics ───────────────────────────────────────────── #
            n_score_changed = diff_df["Score Δ"].notna().sum() - (diff_df["Score Δ"] == 0).sum()
            n_conf_changed = diff_df["Conf Δ"].notna().sum() - (diff_df["Conf Δ"].fillna(0).abs() < 0.01).sum()
            n_src_changed = ((diff_df["Sources Added"] > 0) | (diff_df["Sources Removed"] > 0)).sum()
            only_in_a = (~diff_df["_in_b"]).sum()
            only_in_b = (~diff_df["_in_a"]).sum()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("KPIs compared", len(diff_df))
            m2.metric("Score changed", int(n_score_changed))
            m3.metric("Confidence changed", int(n_conf_changed))
            m4.metric("Source set changed", int(n_src_changed))
            m5.metric("Only in one run", int(only_in_a + only_in_b))

            st.divider()

            # ── Per-KPI diff table ────────────────────────────────────────── #
            st.markdown("**Per-KPI diff** — sorted by absolute score change")
            st.caption(
                "Score Δ and Conf Δ = B minus A. Positive = B is higher. "
                "Sources Added/Removed = sources present in B but not A, and vice versa."
            )

            display_df = diff_df[[
                "KPI Name", "Score A", "Score B", "Score Δ",
                "Conf A", "Conf B", "Conf Δ",
                "Sources Added", "Sources Removed",
            ]].copy()

            # Sort by |score delta| desc, then |conf delta| desc
            display_df["_abs_score"] = diff_df["Score Δ"].abs().fillna(0)
            display_df["_abs_conf"] = diff_df["Conf Δ"].abs().fillna(0)
            display_df = display_df.sort_values(["_abs_score", "_abs_conf"], ascending=False)
            display_df = display_df.drop(columns=["_abs_score", "_abs_conf"])

            def _colour_delta(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return ""
                if val > 0:
                    return "color: green; font-weight: bold"
                if val < 0:
                    return "color: red; font-weight: bold"
                return ""

            styled = display_df.style.applymap(
                _colour_delta, subset=["Score Δ", "Conf Δ"]
            ).background_gradient(
                subset=["Score A", "Score B"], cmap="RdYlGn", vmin=1, vmax=5
            ).background_gradient(
                subset=["Conf A", "Conf B"], cmap="Blues", vmin=0, vmax=1
            )

            st.dataframe(styled, use_container_width=True, height=420)

            st.divider()

            # ── Per-KPI detail expander ───────────────────────────────────── #
            st.markdown("**KPI detail — drill into what changed**")
            kpi_names_diff = diff_df["KPI Name"].tolist()
            selected_diff_kpi = st.selectbox(
                "Select a KPI to inspect", kpi_names_diff, key="diff_kpi_select"
            )

            row = diff_df[diff_df["KPI Name"] == selected_diff_kpi].iloc[0]

            d1, d2 = st.columns(2)

            with d1:
                st.markdown(f"**Run A** — `{label_a}`")
                if row["_in_a"]:
                    st.metric("Score", row["Score A"])
                    st.metric("Confidence", f"{row['Conf A']:.3f}" if row["Conf A"] is not None else "—")
                    st.markdown(f"*Rationale:*  \n{row['_rationale_a'] or '—'}")
                else:
                    st.warning("This KPI was not present in Run A.")

            with d2:
                st.markdown(f"**Run B** — `{label_b}`")
                if row["_in_b"]:
                    score_delta = row["Score Δ"]
                    delta_str = f" ({'+' if score_delta and score_delta > 0 else ''}{score_delta:.0f})" if score_delta is not None else ""
                    st.metric("Score", f"{row['Score B']}{delta_str}")
                    conf_delta = row["Conf Δ"]
                    cdelta_str = f" ({'+' if conf_delta and conf_delta > 0 else ''}{conf_delta:.3f})" if conf_delta is not None else ""
                    st.metric("Confidence", f"{row['Conf B']:.3f}{cdelta_str}" if row["Conf B"] is not None else "—")
                    st.markdown(f"*Rationale:*  \n{row['_rationale_b'] or '—'}")
                else:
                    st.warning("This KPI was not present in Run B.")

            # Source breakdown
            st.markdown("**Source changes**")
            srcs_added = row["_srcs_added"]
            srcs_removed = row["_srcs_removed"]
            srcs_shared = row["_srcs_shared"]

            sc1, sc2, sc3 = st.columns(3)

            with sc1:
                st.markdown(f"**Added in B** ({len(srcs_added)})")
                if srcs_added:
                    for url in srcs_added:
                        st.markdown(f"<span style='color:green'>+ [{url}]({url})</span>", unsafe_allow_html=True)
                else:
                    st.caption("None")

            with sc2:
                st.markdown(f"**Removed from A** ({len(srcs_removed)})")
                if srcs_removed:
                    for url in srcs_removed:
                        st.markdown(f"<span style='color:red'>− [{url}]({url})</span>", unsafe_allow_html=True)
                else:
                    st.caption("None")

            with sc3:
                st.markdown(f"**Shared** ({len(srcs_shared)})")
                if srcs_shared:
                    with st.expander("Show shared sources"):
                        for url in srcs_shared:
                            st.markdown(f"- [{url}]({url})")
                else:
                    st.caption("None")

            # Explanation hint
            if row["Score Δ"] is not None and row["Score Δ"] != 0:
                direction = "increased" if row["Score Δ"] > 0 else "decreased"
                src_note = ""
                if srcs_added and not srcs_removed:
                    src_note = f" {len(srcs_added)} new source(s) added in Run B."
                elif srcs_removed and not srcs_added:
                    src_note = f" {len(srcs_removed)} source(s) from Run A dropped in Run B."
                elif srcs_added and srcs_removed:
                    src_note = f" {len(srcs_added)} source(s) added, {len(srcs_removed)} removed."
                st.info(
                    f"Score {direction} by {abs(row['Score Δ']):.0f} "
                    f"(confidence {'↑' if (row['Conf Δ'] or 0) > 0 else '↓'} "
                    f"{abs(row['Conf Δ'] or 0):.3f}).{src_note}"
                )
