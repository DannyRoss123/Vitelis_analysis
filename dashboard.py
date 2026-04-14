"""Vitelis AI Maturity Dashboard v2 — interactive Streamlit visualization."""

from __future__ import annotations

from app.snapshots import build_snapshot, find_previous_snapshot_for_domain, load_snapshot, diff_snapshots
from app.kpi_catalog import load_kpi_catalog

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vitelis — AI Maturity Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preload KPI catalog so we can show friendly KPI names/questions in the dashboard.
_KPI_DEFS = {k.kpi_id: k for k in load_kpi_catalog() or []}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- global ---------- */
.block-container { padding-top: 1.5rem; }

/* ---------- score gauge ---------- */
.score-card {
    text-align: center;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #333;
    background: #0e1117;
}
.score-card .score-value {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.1;
}
.score-card .score-label {
    font-size: 0.9rem;
    color: #999;
    margin-top: 4px;
}

/* ---------- pillar card ---------- */
.pillar-card {
    padding: 1rem 1.2rem;
    border-radius: 10px;
    border: 1px solid #333;
    background: #0e1117;
    margin-bottom: 0.6rem;
}
.pillar-card .pillar-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 4px;
}
.pillar-card .pillar-score {
    font-size: 1.8rem;
    font-weight: 800;
}

/* ---------- source chip ---------- */
.tier-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.tier-1 { background: #1b4332; color: #95d5b2; }
.tier-2 { background: #3a2d12; color: #f0c040; }
.tier-3 { background: #3a1212; color: #f08080; }

.auth-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.auth-first_party { background: #1a1a3e; color: #8888ff; }
.auth-third_party_news { background: #1b4332; color: #95d5b2; }
.auth-third_party_analyst { background: #2d1b43; color: #c595d5; }
.auth-regulatory { background: #3a2d12; color: #f0c040; }
.auth-unknown { background: #222; color: #888; }

.fresh-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.fresh-high { background: #1b4332; color: #95d5b2; }
.fresh-mid { background: #3a2d12; color: #f0c040; }
.fresh-low { background: #3a1212; color: #f08080; }

/* ---------- contradiction ---------- */
.contradiction-card {
    padding: 0.8rem 1rem;
    border-radius: 8px;
    border: 1px solid #f0808044;
    background: #3a121222;
    margin-bottom: 8px;
}

/* ---------- progress ---------- */
.stage-item {
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.9rem;
}
.stage-done { background: #1b4332; color: #95d5b2; }
.stage-active { background: #3a2d12; color: #f0c040; }
.stage-pending { background: #1a1a2e; color: #666; }

/* ---------- eval breakdown ---------- */
.eval-metric {
    text-align: center;
    padding: 0.6rem;
    border-radius: 8px;
    border: 1px solid #333;
    background: #0e1117;
}
.eval-metric .eval-value {
    font-size: 1.4rem;
    font-weight: 700;
}
.eval-metric .eval-label {
    font-size: 0.7rem;
    color: #888;
}

/* ---------- custom eval tab ---------- */
.gate-pass {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    background: #1b4332; color: #95d5b2; font-size: 0.78rem; font-weight: 700;
}
.gate-fail {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    background: #3a1212; color: #f08080; font-size: 0.78rem; font-weight: 700;
}
.gate-warn {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    background: #3a2d12; color: #f0c040; font-size: 0.78rem; font-weight: 700;
}
.gate-na {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    background: #1a1a2e; color: #666; font-size: 0.78rem; font-weight: 700;
}
.attr-badge {
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 700;
}
.attr-model_change  { background: #2d1b43; color: #c595d5; }
.attr-prompt_change { background: #3a2d12; color: #f0c040; }
.attr-data_change   { background: #1a2d3a; color: #64b5f6; }
.attr-external_noise{ background: #222;    color: #888;    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PILLAR_COLORS = {
    "Strategy & Governance": "#6C63FF",
    "Product & Delivery": "#00C49A",
    "People & Operations": "#FF6B6B",
}

AUTHORITY_LABELS = {
    "first_party": "1st Party",
    "third_party_news": "3rd Party News",
    "third_party_analyst": "3rd Party Analyst",
    "regulatory": "Regulatory",
    "unknown": "Unknown",
}


def score_color(score: float, max_score: float = 5.0) -> str:
    ratio = score / max_score
    if ratio >= 0.75:
        return "#00C49A"
    if ratio >= 0.5:
        return "#f0c040"
    return "#FF6B6B"


def boost_color(val: float) -> str:
    if val > 0.01:
        return "#00C49A"
    if val < -0.01:
        return "#FF6B6B"
    return "#888"


def tier_badge(tier: int) -> str:
    labels = {1: "Tier 1 — High", 2: "Tier 2 — Mid", 3: "Tier 3 — Low"}
    return f'<span class="tier-badge tier-{tier}">{labels.get(tier, f"Tier {tier}")}</span>'


def auth_badge(auth_type: str) -> str:
    label = AUTHORITY_LABELS.get(auth_type, auth_type)
    return f'<span class="auth-badge auth-{auth_type}">{label}</span>'


def fresh_badge(score: float) -> str:
    if score >= 0.7:
        return '<span class="fresh-badge fresh-high">Fresh</span>'
    if score >= 0.4:
        return '<span class="fresh-badge fresh-mid">Moderate</span>'
    return '<span class="fresh-badge fresh-low">Stale</span>'


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_reports() -> list[Path]:
    output_dir = Path("app/output")
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("report_*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# Pipeline runner (threaded so Streamlit stays responsive)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    ("discover_urls", "Discovering URLs"),
    ("fetch_sources", "Fetching Sources"),
    ("index_sources", "Indexing Sources"),
    ("score_kpis", "Scoring KPIs"),
    ("aggregate_report", "Generating Report"),
]


def run_pipeline(company_name: str, company_domain: str, run_id: str, progress: dict):
    """Run the full orchestrator pipeline, updating *progress* dict in-place."""
    from dotenv import load_dotenv

    load_dotenv()
    from app.nodes.discover_urls import discover_urls_node
    from app.nodes.fetch_sources import fetch_sources_node
    from app.nodes.index_sources import index_sources_node
    from app.nodes.score_kpis import score_kpis_node
    from app.nodes.aggregate_report import aggregate_report_node

    state: dict = {
        "run_id": run_id,
        "company_name": company_name,
        "company_domain": company_domain,
    }

    nodes = [
        ("discover_urls", discover_urls_node),
        ("fetch_sources", fetch_sources_node),
        ("index_sources", index_sources_node),
        ("score_kpis", score_kpis_node),
        ("aggregate_report", aggregate_report_node),
    ]

    try:
        for stage_key, node_fn in nodes:
            progress["current_stage"] = stage_key
            progress["stages_done"].append(stage_key)
            result = node_fn(state)
            state.update(result)
            progress["state"] = dict(state)
    except Exception as exc:
        progress["error"] = str(exc)
    finally:
        progress["done"] = True
        progress["state"] = dict(state)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## VITELIS")
st.sidebar.markdown("##### AI Maturity Assessment v2")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["New Analysis", "View Past Report"], label_visibility="collapsed")

report_data: dict | None = None

if mode == "New Analysis":
    st.sidebar.markdown("### Run New Analysis")
    company_name = st.sidebar.text_input("Company Name", value="Vodafone")
    company_domain = st.sidebar.text_input("Company Domain", value="vodafone.com")
    start = st.sidebar.button("Start Web Crawl", type="primary", use_container_width=True)

    if "pipeline_progress" not in st.session_state:
        st.session_state["pipeline_progress"] = None
    if "pipeline_thread" not in st.session_state:
        st.session_state["pipeline_thread"] = None

    if start:
        run_id = str(uuid.uuid4())[:8]
        progress = {
            "current_stage": None,
            "stages_done": [],
            "state": {},
            "done": False,
            "error": None,
        }
        st.session_state["pipeline_progress"] = progress
        t = threading.Thread(
            target=run_pipeline,
            args=(company_name, company_domain, run_id, progress),
            daemon=True,
        )
        t.start()
        st.session_state["pipeline_thread"] = t

    progress = st.session_state.get("pipeline_progress")

    if progress is not None and not progress["done"]:
        st.markdown("## Pipeline Running...")
        cols = st.columns(5)
        for i, (key, label) in enumerate(PIPELINE_STAGES):
            if key in progress["stages_done"]:
                if key == progress["current_stage"] and not progress["done"]:
                    cols[i].markdown(f'<div class="stage-item stage-active">⏳ {label}</div>', unsafe_allow_html=True)
                else:
                    cols[i].markdown(f'<div class="stage-item stage-done">✅ {label}</div>', unsafe_allow_html=True)
            else:
                cols[i].markdown(f'<div class="stage-item stage-pending">⬜ {label}</div>', unsafe_allow_html=True)

        snap = progress.get("state", {})
        if snap.get("target_urls"):
            st.markdown(f"**URLs discovered:** {snap.get('url_count', len(snap['target_urls']))}")
        if snap.get("sources"):
            st.markdown(f"**Sources fetched:** {len(snap['sources'])}")

        time.sleep(2)
        st.rerun()

    elif progress is not None and progress["done"]:
        if progress.get("error"):
            st.error(f"Pipeline failed: {progress['error']}")
        else:
            snap = progress["state"]
            report_path = snap.get("report_path")
            if report_path and Path(report_path).exists():
                report_data = load_report(report_path)
                st.sidebar.success(f"Report ready: {report_path}")
            else:
                st.warning("Pipeline finished but no report file found.")

else:
    reports = list_reports()
    if not reports:
        st.sidebar.warning("No reports found in app/output/")
    else:
        options = {f"{p.stem} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))})": p for p in reports}
        choice = st.sidebar.selectbox("Select Report", list(options.keys()))
        if choice:
            report_data = load_report(str(options[choice]))

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
if report_data is None:
    if mode == "New Analysis" and (st.session_state.get("pipeline_progress") is None):
        st.markdown("# Vitelis — AI Maturity Dashboard")
        st.markdown(
            """
            Enter a **company name** and **domain** in the sidebar, then press
            **Start Web Crawl** to begin the analysis pipeline.

            **v2 Source Evaluation** includes:
            - Content-based tier classification (not just URL matching)
            - Semantic corroboration (claim-level cross-source agreement)
            - Source freshness weighting (date-aware confidence)
            - Authority signal detection (1st-party vs 3rd-party validation)
            - Contradiction detection (conflicting evidence flagging)
            """
        )
        st.markdown("---")
        st.markdown("##### Or select **View Past Report** in the sidebar to explore a previous run.")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
company = report_data.get("company_name", "Unknown")
domain = report_data.get("company_domain", "")
overall = report_data.get("overall_score", 0)
url_count = report_data.get("url_count", 0)
ts = report_data.get("timestamp", "")
run_id = report_data.get("run_id", "")

st.markdown(f"# {company} — AI Maturity Report")
st.caption(f"Domain: {domain}  |  Run: {run_id}  |  {ts}")

# ---------------------------------------------------------------------------
# Row 1 — Overall score + Pillar scores
# ---------------------------------------------------------------------------
st.markdown("---")

pillar_scores = report_data.get("pillar_scores", [])
col_overall, *pillar_cols = st.columns([1.3] + [1] * len(pillar_scores))

with col_overall:
    color = score_color(overall)
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-value" style="color:{color}">{overall:.2f}</div>
            <div class="score-label">OVERALL SCORE (out of 5)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for i, ps in enumerate(pillar_scores):
    with pillar_cols[i]:
        pname = ps["pillar"]
        pscore = ps["score"]
        pconf = ps.get("confidence", 0)
        pcolor = PILLAR_COLORS.get(pname, "#888")
        st.markdown(
            f"""
            <div class="pillar-card">
                <div class="pillar-title" style="color:{pcolor}">{pname}</div>
                <div class="pillar-score" style="color:{score_color(pscore)}">{pscore:.2f}</div>
                <div style="color:#888;font-size:0.8rem">Confidence: {pconf:.0%} &nbsp;|&nbsp; {len(ps.get('kpis',[]))} KPIs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Row 2 — Pillar bar chart
# ---------------------------------------------------------------------------
st.markdown("---")

# If only a few KPIs were scored (e.g. fast run with 3 KPIs),
# show a KPI-level comparison instead of only pillar-level.
kpi_results = report_data.get("kpi_results", [])
if len(kpi_results) <= 3 and kpi_results:
    st.markdown("### KPI Comparison")
    kpi_rows = []
    for k in kpi_results:
        kpi_id = str(k.get("kpi_id"))
        kpi_def = _KPI_DEFS.get(kpi_id)
        label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
        kpi_rows.append(
            {
                "KPI": label,
                "Pillar": k.get("pillar", ""),
                "Score": k.get("score", 0),
                "Confidence": k.get("confidence", 0),
            }
        )
    df_kpis = pd.DataFrame(kpi_rows)
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_kpis, x="KPI", y="Score", color="Pillar", horizontal=False)
    with col_table:
        st.dataframe(
            df_kpis.style.format({"Score": "{:.2f}", "Confidence": "{:.0%}"}),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.markdown("### Pillar Comparison")
    bar_data = []
    for ps in pillar_scores:
        bar_data.append({"Pillar": ps["pillar"], "Score": ps["score"], "Confidence": ps.get("confidence", 0)})

    if bar_data:
        df_pillars = pd.DataFrame(bar_data)
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.bar_chart(df_pillars, x="Pillar", y="Score", color="Pillar", horizontal=False)
        with col_table:
            st.dataframe(
                df_pillars.style.format({"Score": "{:.2f}", "Confidence": "{:.0%}"}),
                hide_index=True,
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Row 3 — Tabs
# ---------------------------------------------------------------------------
st.markdown("---")

kpi_results = report_data.get("kpi_results", [])
missing_evidence = report_data.get("missing_evidence", [])
debug_log = report_data.get("debug_log", []) or []

tab_kpi, tab_diff, tab_eval, tab_rag_eval, tab_custom_eval, tab_sources, tab_citations, tab_raw, tab_debug = st.tabs(
    ["KPI Scores", "Run Diffs", "Source Evaluation", "RAG Evaluation", "🧪 Custom Eval", "Sources", "Citations", "Raw Report", "Debug Log"]
)

# ============================================================================
# TAB: KPI Scores
# ============================================================================
with tab_kpi:
    st.markdown("### KPI Scores by Pillar")

    kpis_by_pillar: dict[str, list] = defaultdict(list)
    for kpi in kpi_results:
        kpis_by_pillar[kpi.get("pillar", "Other")].append(kpi)

    for pillar_name, kpis in kpis_by_pillar.items():
        pcolor = PILLAR_COLORS.get(pillar_name, "#888")
        st.markdown(f"#### <span style='color:{pcolor}'>{pillar_name}</span>", unsafe_allow_html=True)

        for kpi in kpis:
            kpi_id = str(kpi["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            kpi_label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            score = kpi["score"]
            conf = kpi.get("confidence", 0)
            ktype = kpi.get("type", "")
            rationale = kpi.get("rationale", "")
            details = kpi.get("details", {}) or {}
            citations = kpi.get("citations", [])
            source_eval = details.get("source_evaluation", {})

            is_missing = kpi_id in missing_evidence

            with st.expander(
                f"{'🔴' if is_missing else '📊'} {kpi_label}  Score: {score:.1f}/5  Confidence: {conf:.0%}  {ktype}"
            ):
                st.progress(min(score / 5.0, 1.0))
                if kpi_def and getattr(kpi_def, "question", None):
                    st.markdown(f"**Question:** {kpi_def.question}")
                st.markdown(f"Rationale: {rationale}")

                # v2: Source evaluation breakdown
                if source_eval:
                    st.markdown("---")
                    st.markdown("Source Evaluation Breakdown")

                    m1, m2, m3, m4 = st.columns(4)

                    corr = source_eval.get("semantic_corroboration", {})
                    corr_score = corr.get("corroboration_score", 0)
                    corr_claims = len(corr.get("corroborated_claims", []))
                    with m1:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{score_color(corr_score, 1.0)}">{corr_score:.0%}</div>'
                            f'<div class="eval-label">Corroboration<br>{corr_claims} matched claims</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    fresh = source_eval.get("freshness", {})
                    fresh_boost = fresh.get("boost", 0)
                    with m2:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{boost_color(fresh_boost)}">{fresh_boost:+.3f}</div>'
                            f'<div class="eval-label">Freshness Boost</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    auth = source_eval.get("authority", {})
                    auth_boost = auth.get("boost", 0)
                    with m3:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{boost_color(auth_boost)}">{auth_boost:+.3f}</div>'
                            f'<div class="eval-label">Authority Boost</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    contrad = source_eval.get("contradictions", {})
                    contrad_count = contrad.get("contradiction_count", 0)
                    contrad_penalty = contrad.get("confidence_penalty", 0)
                    with m4:
                        contrad_color = "#FF6B6B" if contrad_count > 0 else "#00C49A"
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{contrad_color}">{contrad_count}</div>'
                            f'<div class="eval-label">Contradictions<br>Penalty: {contrad_penalty:+.3f}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    if corr.get("corroborated_claims"):
                        with st.popover("View Corroborated Claims"):
                            for claim in corr["corroborated_claims"][:5]:
                                st.markdown(
                                    f"{claim.get('source_a', '')}: {claim.get('claim_a', '')[:120]}\n\n"
                                    f"{claim.get('source_b', '')}: {claim.get('claim_b', '')[:120]}\n\n"
                                    f"Similarity: {claim.get('similarity', 0):.0%}"
                                )
                                st.markdown("---")

                    if contrad.get("contradictions"):
                        with st.popover("View Contradictions"):
                            for c in contrad["contradictions"]:
                                st.markdown(
                                    f'<div class="contradiction-card">'
                                    f'{c.get("source_a", "")}: "{c.get("claim_a", "")}"<br>'
                                    f'{c.get("source_b", "")}: "{c.get("claim_b", "")}"'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                if citations:
                    st.markdown("Citations")
                    for c in citations:
                        url = c.get("url", "")
                        quote = c.get("quote", "")[:200]
                        st.markdown(f"- [{c.get('source_id', 'source')}]({url}): {quote}")

                if details:
                    with st.popover("Full Details JSON"):
                        st.json(details)

    if missing_evidence:
        st.markdown("---")
        st.markdown("### Missing Evidence")
        st.warning(f"The following KPIs had no evidence found: {', '.join(missing_evidence)}")

# ============================================================================
# TAB: Run Diffs
# ============================================================================
with tab_diff:
    st.markdown("### Run to run diffs")

    current_snap = build_snapshot(report_data, report_path="")
    prev_path = find_previous_snapshot_for_domain(domain, exclude_run_id=run_id)

    if not prev_path:
        st.info("No previous snapshot found for this domain yet. Run it again with a new run id to see diffs.")
    else:
        prev_snap = load_snapshot(str(prev_path))
        d = diff_snapshots(prev_snap, current_snap)

        st.write(
            f"Overall score: {d['overall_old']:.2f} to {d['overall_new']:.2f}  delta {d['overall_delta']:+.2f}"
        )

        st.markdown("#### Biggest score changes")
        for r in (d.get("top_score_changes") or [])[:10]:
            st.write(
                f"{r.get('kpi_id','')}: score {r.get('score_delta',0):+.2f}  conf {r.get('confidence_delta',0):+.2f}"
            )

        st.markdown("#### Biggest confidence changes")
        for r in (d.get("top_confidence_changes") or [])[:10]:
            st.write(
                f"{r.get('kpi_id','')}: conf {r.get('confidence_delta',0):+.2f}  score {r.get('score_delta',0):+.2f}"
            )

        with st.expander("Show full diff json"):
            st.json(d)
            
# ============================================================================
# TAB: Source Evaluation (NEW — v2 deep dive)
# ============================================================================
with tab_eval:
    st.markdown("### Source Evaluation Engine v2")
    st.markdown(
        "Deep analysis of evidence quality across all KPIs. "
        "Each scored KPI includes evaluation of **corroboration**, **freshness**, "
        "**authority**, and **contradictions**."
    )

    # --- Aggregate stats across all KPIs ---
    all_corr_scores = []
    all_fresh_boosts = []
    all_auth_boosts = []
    all_contrad_counts = []
    all_contrad_penalties = []
    all_corroborated_claims = []
    all_contradictions = []
    all_authority_sources: dict[str, dict] = {}

    for kpi in kpi_results:
        details = kpi.get("details", {}) or {}
        se = details.get("source_evaluation", {})
        if not se:
            continue

        corr = se.get("semantic_corroboration", {})
        all_corr_scores.append(corr.get("corroboration_score", 0))
        for claim in corr.get("corroborated_claims", []):
            claim["kpi_id"] = kpi["kpi_id"]
            all_corroborated_claims.append(claim)

        fresh = se.get("freshness", {})
        all_fresh_boosts.append(fresh.get("boost", 0))

        auth = se.get("authority", {})
        all_auth_boosts.append(auth.get("boost", 0))
        for sid, info in auth.get("per_source", {}).items():
            if sid not in all_authority_sources:
                all_authority_sources[sid] = info

        contrad = se.get("contradictions", {})
        all_contrad_counts.append(contrad.get("contradiction_count", 0))
        all_contrad_penalties.append(contrad.get("confidence_penalty", 0))
        for c in contrad.get("contradictions", []):
            c["kpi_id"] = kpi["kpi_id"]
            all_contradictions.append(c)

    has_eval_data = bool(all_corr_scores)

    if not has_eval_data:
        st.info(
            "No v2 source evaluation data found in this report. "
            "Run a new analysis to generate evaluation data with the upgraded engine."
        )
    else:
        # --- Summary metrics ---
        st.markdown("#### Aggregate Quality Metrics")
        s1, s2, s3, s4, s5 = st.columns(5)

        avg_corr = sum(all_corr_scores) / len(all_corr_scores) if all_corr_scores else 0
        avg_fresh = sum(all_fresh_boosts) / len(all_fresh_boosts) if all_fresh_boosts else 0
        avg_auth = sum(all_auth_boosts) / len(all_auth_boosts) if all_auth_boosts else 0
        total_contrad = sum(all_contrad_counts)
        total_corr_claims = len(all_corroborated_claims)

        s1.metric("Avg Corroboration", f"{avg_corr:.0%}")
        s2.metric("Corroborated Claims", total_corr_claims)
        s3.metric("Avg Freshness Boost", f"{avg_fresh:+.3f}")
        s4.metric("Avg Authority Boost", f"{avg_auth:+.3f}")
        s5.metric("Total Contradictions", total_contrad)

        st.markdown("---")

        # ==============================================
        # Section 1: SEMANTIC CORROBORATION
        # ==============================================
        st.markdown("#### 1. Semantic Corroboration")
        st.markdown(
            "Claims extracted from each source and compared across different sources. "
            "When two independent sources make the same factual claim, it's corroborated."
        )

        if all_corroborated_claims:
            rows = []
            for claim in all_corroborated_claims:
                rows.append({
                    "KPI": claim.get("kpi_id", ""),
                    "Source A": claim.get("source_a", ""),
                    "Claim A": claim.get("claim_a", "")[:120],
                    "Source B": claim.get("source_b", ""),
                    "Claim B": claim.get("claim_b", "")[:120],
                    "Similarity": claim.get("similarity", 0),
                })
            df_corr = pd.DataFrame(rows)
            st.dataframe(
                df_corr.style.format({"Similarity": "{:.0%}"}).background_gradient(
                    subset=["Similarity"], cmap="Greens", vmin=0, vmax=1
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No cross-source corroborated claims detected.")

        # Per-KPI corroboration bar chart
        corr_per_kpi = []
        for kpi in kpi_results:
            se = (kpi.get("details") or {}).get("source_evaluation", {})
            corr = se.get("semantic_corroboration", {})
            if corr:
                corr_per_kpi.append({
                    "KPI": kpi["kpi_id"],
                    "Corroboration Score": corr.get("corroboration_score", 0),
                    "Matched Claims": len(corr.get("corroborated_claims", [])),
                })
        if corr_per_kpi:
            st.bar_chart(pd.DataFrame(corr_per_kpi), x="KPI", y="Corroboration Score")

        st.markdown("---")

        # ==============================================
        # Section 2: SOURCE FRESHNESS
        # ==============================================
        st.markdown("#### 2. Source Freshness")
        st.markdown(
            "Each source's content is scanned for dates. Recent content boosts confidence; "
            "stale content (>1–2 years old) penalizes it."
        )

        # Collect freshness per source across all KPIs
        freshness_sources: dict[str, dict] = {}
        for kpi in kpi_results:
            se = (kpi.get("details") or {}).get("source_evaluation", {})
            for sid, info in se.get("freshness", {}).get("per_source", {}).items():
                if sid not in freshness_sources:
                    freshness_sources[sid] = info

        if freshness_sources:
            rows = []
            for sid, info in freshness_sources.items():
                rows.append({
                    "Source": sid,
                    "Freshness Score": info.get("freshness_score", 0),
                    "Newest Date": info.get("newest_date", "N/A"),
                    "Age (days)": info.get("age_days") if info.get("age_days") is not None else "Unknown",
                    "Dates Found": info.get("dates_found", 0),
                })
            df_fresh = pd.DataFrame(rows).sort_values("Freshness Score", ascending=False)
            st.dataframe(
                df_fresh.style.background_gradient(subset=["Freshness Score"], cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No freshness data available.")

        st.markdown("---")

        # ==============================================
        # Section 3: AUTHORITY SIGNALS
        # ==============================================
        st.markdown("#### 3. Authority Signals")
        st.markdown(
            "Each source is classified by authority type. Third-party validation "
            "(news, analysts, regulators) is worth more than first-party self-claims."
        )

        if all_authority_sources:
            rows = []
            for sid, info in all_authority_sources.items():
                rows.append({
                    "Source": sid,
                    "Authority Type": AUTHORITY_LABELS.get(info.get("authority_type", ""), info.get("authority_type", "")),
                    "Authority Score": info.get("authority_score", 0),
                    "Reason": info.get("authority_reason", ""),
                    "3rd Party": "Yes" if info.get("is_third_party") else "No",
                })
            df_auth = pd.DataFrame(rows).sort_values("Authority Score", ascending=False)

            col_table, col_pie = st.columns([2, 1])
            with col_table:
                st.dataframe(
                    df_auth.style.background_gradient(subset=["Authority Score"], cmap="Purples", vmin=0, vmax=1),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_pie:
                auth_type_counts = pd.DataFrame(rows)["Authority Type"].value_counts()
                st.markdown("**Authority Distribution:**")
                for atype, count in auth_type_counts.items():
                    pct = count / len(rows) * 100
                    st.markdown(f"- **{atype}**: {count} sources ({pct:.0f}%)")
        else:
            st.info("No authority data available.")

        st.markdown("---")

        # ==============================================
        # Section 4: CONTRADICTION DETECTION
        # ==============================================
        st.markdown("#### 4. Contradiction Detection")
        st.markdown(
            "Evidence is scanned for opposing claims between different sources. "
            "Contradictions reduce confidence scores."
        )

        if all_contradictions:
            st.error(f"**{len(all_contradictions)} contradiction(s) detected across evidence**")
            for c in all_contradictions:
                st.markdown(
                    f'<div class="contradiction-card">'
                    f'<strong>KPI:</strong> {c.get("kpi_id", "")}<br>'
                    f'<strong>{c.get("source_a", "")}</strong> says: <em>"{c.get("claim_a", "")}"</em><br>'
                    f'<strong>{c.get("source_b", "")}</strong> says: <em>"{c.get("claim_b", "")}"</em><br>'
                    f'<strong>Type:</strong> {c.get("type", "opposing_claims")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No contradictions detected in the evidence.")

        st.markdown("---")

        # ==============================================
        # Section 5: CONFIDENCE IMPACT SUMMARY
        # ==============================================
        st.markdown("#### 5. Confidence Impact Summary")
        st.markdown(
            "How each evaluation factor adjusted KPI confidence scores."
        )

        impact_rows = []
        for kpi in kpi_results:
            se = (kpi.get("details") or {}).get("source_evaluation", {})
            if not se:
                continue
            corr = se.get("semantic_corroboration", {})
            corr_max = 0.15
            impact_rows.append({
                "KPI": kpi["kpi_id"],
                "Corroboration": f"+{corr.get('corroboration_score', 0) * corr_max:.3f}",
                "Freshness": f"{se.get('freshness', {}).get('boost', 0):+.3f}",
                "Authority": f"{se.get('authority', {}).get('boost', 0):+.3f}",
                "Contradictions": f"{se.get('contradictions', {}).get('confidence_penalty', 0):+.3f}",
                "Final Confidence": f"{kpi.get('confidence', 0):.0%}",
            })

        if impact_rows:
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)


# ============================================================================
# TAB: RAG Evaluation
# ============================================================================
def _rag_metric_summary(metric: str, value) -> str:
    """One-line human-readable summary for a RAG evaluation metric."""
    if value is None:
        return "Not available"
    if metric == "ragas_faithfulness":
        return "Answer well grounded in evidence" if value >= 0.6 else "Answer poorly grounded in evidence" if value >= 0.2 else "Answer barely grounded in evidence"
    if metric == "ragas_answer_relevancy":
        return "Answer addresses the question" if value >= 0.5 else "Answer partly addresses the question" if value >= 0.2 else "Answer does not address the question"
    if metric == "ragas_context_precision":
        return "Retrieved sources are on-topic" if value >= 0.4 else "Retrieved sources partly on-topic" if value >= 0.1 else "Retrieved sources mostly off-topic"
    if metric == "ragas_context_recall":
        return "Evidence covers the expected content" if value >= 0.5 else "Evidence partly covers expected content" if value >= 0.2 else "Evidence misses expected content"
    if metric == "llm_judge_overall":
        return f"AI reviewer rated {int(value)}/5" if value is not None else "LLM judge unavailable"
    if metric == "recall_at_3":
        return "Key info in top results" if value >= 0.5 else "Key info not in top results" if value >= 0.1 else "Key info missing from top results"
    if metric == "f1":
        return "Answer matches reference well" if value >= 0.4 else "Answer partly matches reference" if value >= 0.15 else "Answer does not match reference"
    if metric == "hallucination_score":
        return "Low risk of unsupported content" if value <= 0.3 else "Some unsupported content" if value <= 0.6 else "High risk of unsupported content (flagged)"
    if metric == "mmr_diversity_score":
        # Values can be slightly above 1 due to approximations; treat 1.0+ as "extremely diverse"
        if value >= 0.9:
            return "Sources are extremely diverse"
        if value >= 0.6:
            return "Sources are diverse"
        if value >= 0.3:
            return "Sources have moderate diversity"
        return "Sources are repetitive"
    if metric == "factual_correctness":
        return "Claims align with ideal answer" if value >= 0.5 else "Claims partly align" if value >= 0.2 else "Claims do not align with ideal"
    if metric == "noise_sensitivity":
        return "Not misled by low-quality sources" if value <= 0.2 else "Slightly affected by noise" if value <= 0.5 else "Noticeably misled by poor sources"
    if metric == "semantic_similarity":
        return "Meaning matches ideal answer" if value >= 0.6 else "Meaning partly matches" if value >= 0.4 else "Meaning diverges from ideal"
    return str(value)


with tab_rag_eval:
    st.markdown("### RAG Evaluation")
    rag = report_data.get("rag_evaluation") or {}
    if not rag:
        st.info("No RAG evaluation data in this report. Run a pipeline with RAG evaluation enabled.")
    else:
        # --- Overall summary and score at top ---
        # Derive counts from actual data (no hardcoding)
        per_kpi_list = rag.get("per_kpi", []) or []
        eval_count = len(per_kpi_list)
        flagged_count = len(rag.get("flagged_kpi_ids", []) or [])
        verdict = rag.get("overall_verdict", "")
        summary = rag.get("summary", "")

        st.markdown("#### Overall Summary")
        total_kpis = len(report_data.get("kpi_results", []) or [])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total KPIs in report", total_kpis)
        with c2:
            st.metric("RAG evaluated", eval_count)
        with c3:
            st.metric("Flagged for review", flagged_count)
        with c4:
            pct = (flagged_count / eval_count * 100) if eval_count else 0
            st.metric("Flagged %", f"{pct:.0f}%")
        st.markdown(f"**Verdict:** {verdict}")
        if summary:
            st.caption(summary)
        st.markdown("---")

        # --- Pillar-level aggregate RAG quality (0–1) ---
        per_kpi_raw = rag.get("per_kpi", []) or []
        if per_kpi_raw:
            # Map kpi_id -> pillar from main KPI results
            pillar_by_kpi: dict[str, str] = {str(k["kpi_id"]): k.get("pillar", "") for k in kpi_results}

            # Metrics where higher is better
            good_metrics = [
                "ragas_faithfulness",
                "ragas_answer_relevancy",
                "ragas_context_precision",
                "ragas_context_recall",
                "semantic_similarity",
            ]
            # Metrics where lower is better (we invert as 1 - value)
            inverted_metrics = [
                "hallucination_score",
                "noise_sensitivity",
            ]

            pillar_scores_rag: dict[str, list[float]] = defaultdict(list)

            for item in per_kpi_raw:
                kpi_id = str(item.get("kpi_id", ""))
                pillar = pillar_by_kpi.get(kpi_id)
                if not pillar:
                    continue

                values: list[float] = []
                for key in good_metrics:
                    v = item.get(key)
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                for key in inverted_metrics:
                    v = item.get(key)
                    if isinstance(v, (int, float)):
                        values.append(max(0.0, min(1.0, 1.0 - float(v))))

                if not values:
                    continue

                kpi_rag_score = sum(values) / len(values)
                pillar_scores_rag[pillar].append(kpi_rag_score)

            if pillar_scores_rag:
                bar_rows = []
                for pillar, vals in pillar_scores_rag.items():
                    score = sum(vals) / len(vals)
                    if score >= 0.7:
                        meaning = "High-quality, well-grounded answers; strong RAG performance"
                    elif score >= 0.4:
                        meaning = "Mixed quality; some answers are reliable, others need review"
                    else:
                        meaning = "Low RAG quality; most answers require manual review"
                    bar_rows.append({"Pillar": pillar, "RAG score": score, "Meaning": meaning})

                df_rag_pillars = pd.DataFrame(bar_rows)
                st.markdown("#### Pillar-level RAG Quality")
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(df_rag_pillars, x="Pillar", y="RAG score", color="Pillar", horizontal=False)
                with col_table:
                    st.dataframe(
                        df_rag_pillars.sort_values("RAG score", ascending=False).style.format({"RAG score": "{:.2f}"}),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.markdown("---")

        # --- First 7 unique KPIs: reorganised by metric into compact tables ---
        per_kpi_all = rag.get("per_kpi", []) or []

        # Deduplicate by KPI ID and keep only the first 7 unique KPIs
        seen_kpis = set()
        per_kpi = []
        for item in per_kpi_all:
            kpi_id = item.get("kpi_id")
            if not kpi_id or kpi_id in seen_kpis:
                continue
            seen_kpis.add(kpi_id)
            per_kpi.append(item)
            if len(per_kpi) >= 7:
                break

        if not per_kpi:
            st.info("No per-KPI RAG evaluation data.")
        else:
            st.markdown("#### KPI-level RAG Scores (first 7 KPIs)")
            kpi_id_to_pillar = {str(k["kpi_id"]): k.get("pillar", "") for k in kpi_results}
            # KPI Driver = question from column N: prefer report's kpi_definitions, then catalog
            kpi_id_to_question: dict[str, str] = {}
            for defn in report_data.get("kpi_definitions", []) or []:
                kid = defn.get("kpi_id")
                q = defn.get("question") or defn.get("name")
                if kid and q:
                    kpi_id_to_question[str(kid)] = str(q)
            for kid, kpi_def in _KPI_DEFS.items():
                if kid not in kpi_id_to_question:
                    q = getattr(kpi_def, "question", None) or getattr(kpi_def, "name", None)
                    if q:
                        kpi_id_to_question[kid] = str(q)

            # Metrics to display as sections: name and the key used in rag_evaluation
            metrics = [
                ("ragas_faithfulness", "Faithfulness"),
                ("ragas_answer_relevancy", "Answer relevance"),
                ("ragas_context_precision", "Context precision"),
                ("ragas_context_recall", "Context recall"),
                ("semantic_similarity", "Semantic similarity"),
                ("f1", "F1"),
                ("recall_at_3", "Recall@3"),
                ("hallucination_score", "Hallucination score"),
                ("mmr_diversity_score", "MMR diversity"),
                ("factual_correctness", "Factual correctness"),
                ("noise_sensitivity", "Noise sensitivity"),
            ]

            for key, label in metrics:
                rows = []
                for item in per_kpi:
                    kpi_id = str(item.get("kpi_id", ""))
                    val = item.get(key)
                    if val is None:
                        continue

                    pillar = kpi_id_to_pillar.get(kpi_id, "")
                    driver = kpi_id_to_question.get(kpi_id) or kpi_id

                    meaning = _rag_metric_summary(key, val)
                    score_str = f"{float(val):.3f}" if isinstance(val, (int, float)) else str(val)

                    rows.append(
                        {
                            "KPI Category": pillar or "—",
                            "KPI Driver": driver,
                            "Score": score_str,
                            "What the score means": meaning,
                        }
                    )

                if not rows:
                    continue

                st.markdown(f"##### {label}")
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown("---")

        # --- Cross-company RAG comparison (up to 3 reports) ---
        st.markdown("#### Cross-Company RAG Comparison (up to 3 reports)")

        # Helper: compute average RAG metrics from a report (same 7 metrics for radar).
        # Do not invert noise or hallucination: use raw values (higher = worse).
        def _rag_summary(r: dict) -> dict | None:
            rag_block = r.get("rag_evaluation") or {}
            rows = rag_block.get("per_kpi") or []
            if not rows:
                return None

            def _avg(field: str) -> float:
                vals = [row.get(field) for row in rows if row.get(field) is not None]
                return float(sum(vals) / len(vals)) if vals else 0.0

            return {
                "faithfulness": _avg("ragas_faithfulness"),
                "answer_relevancy": _avg("ragas_answer_relevancy"),
                "context_precision": _avg("ragas_context_precision"),
                "context_recall": _avg("ragas_context_recall"),
                "semantic_similarity": _avg("semantic_similarity"),
                "noise": _avg("noise_sensitivity"),  # raw: higher = worse
                "hallucination": _avg("hallucination_score"),  # raw: higher = worse
            }

        # Build list of available reports, grouped by company (one latest per company)
        all_reports = list_reports()
        if all_reports:
            latest_by_company: dict[str, tuple[float, Path]] = {}
            for p in all_reports:
                try:
                    data = load_report(str(p))
                except Exception:
                    continue
                company = data.get("company_name", "Unknown")
                mtime = p.stat().st_mtime
                prev = latest_by_company.get(company)
                if not prev or mtime > prev[0]:
                    latest_by_company[company] = (mtime, p)

            options = {company: entry[1] for company, entry in latest_by_company.items()}

            # Preselect the current report if present
            default_labels = [lbl for lbl, p in options.items() if str(p) == report_path] if "report_path" in locals() else []
            selected = st.multiselect(
                "Select up to 3 companies",
                list(options.keys()),
                default=default_labels[:1],
                max_selections=3,
            )

            if selected:
                # Same 7 metrics: faithfulness, answer relevance, context precision, context recall,
                # semantic similarity, noise (raw), hallucination (raw). For noise/hallucination higher = worse.
                categories = [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "semantic_similarity",
                    "noise",
                    "hallucination",
                ]
                axis_labels = [
                    "Faithfulness",
                    "Answer relevance",
                    "Context precision",
                    "Context recall",
                    "Semantic similarity",
                    "Noise",
                    "Hallucination",
                ]

                # Compact radar chart so labels and legend stay outside the data area
                fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(2.3, 2.3))
                angles = [n / float(len(categories)) * 2 * 3.14159265 for n in range(len(categories))]
                angles += angles[:1]  # close the loop

                for label in selected:
                    p = options[label]
                    try:
                        data = load_report(str(p))
                    except Exception:
                        continue
                    summary = _rag_summary(data)
                    if not summary:
                        continue
                    values = [summary[c] for c in categories]
                    values += values[:1]
                    values = [max(0.0, min(1.0, v)) for v in values]
                    # So all-zero companies (e.g. no evidence) still show: use a tiny radius so the line is visible
                    if max(values) == 0.0:
                        values = [0.02] * len(values)
                    ax.plot(angles, values, linewidth=1.5, label=label)
                    ax.fill(angles, values, alpha=0.1)

                ax.set_xticks(angles[:-1])
                labels = axis_labels
                ax.set_xticklabels(
                    [
                        "\n".join(lbl.split(" ", 1)) if " " in lbl else lbl
                        for lbl in labels
                    ],
                    fontsize=3,
                )
                # Push labels slightly outside the circle to avoid overlap
                for lbl in ax.get_xticklabels():
                    x, y = lbl.get_position()
                    lbl.set_position((x, y + 0.15))
                ax.set_yticklabels([])
                ax.set_ylim(0, 1)
                ax.set_title("RAG Quality Comparison", fontsize=9, pad=10)
                # Place legend under the chart so it never obscures lines
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=max(1, len(selected)),
                    fontsize=6,
                    frameon=True,
                )
                fig.subplots_adjust(top=0.78, bottom=0.28, left=0.05, right=0.95)

                # Render chart in a narrow column so the visual footprint matches roughly the green box
                chart_col, _ = st.columns([1, 2])
                with chart_col:
                    st.pyplot(fig, clear_figure=True)


# TAB: Custom Eval
# Displays all 10-feature extension metrics: score split, scoring distribution,
# quality gates, score attribution, retrieval metrics (Hit Rate / MRR / nDCG),
# BERTScore F1, and chain-of-thought evaluation sub-scores.
# ============================================================================
with tab_custom_eval:
    st.markdown("### Custom Evaluation Metrics")
    st.markdown(
        "Per-KPI results from the extension suite: **score splitting**, "
        "**statistical scoring** (mean ± σ), **quality gates**, "
        "**score attribution**, **retrieval metrics**, "
        "**BERTScore**, and **chain-of-thought evaluation**."
    )

    # ── helper: gate badge HTML ───────────────────────────────────────────
    def _gate_badge(passed, reason: str = "") -> str:
        if passed is True:
            return '<span class="gate-pass">✓ PASS</span>'
        if passed is False:
            label = reason or "FAIL"
            return f'<span class="gate-fail">✗ {label}</span>'
        return '<span class="gate-na">— N/A</span>'

    def _fmt(val, decimals: int = 3) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.{decimals}f}"
        except (TypeError, ValueError):
            return str(val)

    # ── collect data from kpi_results ────────────────────────────────────
    ce_kpis = [k for k in kpi_results if k.get("type") == "rubric"]

    if not ce_kpis:
        st.info(
            "No custom evaluation data found. This tab is populated after a "
            "pipeline run with the extension suite active (rubric KPIs only)."
        )
    else:
        # ─────────────────────────────────────────────────────────────────
        # SECTION 0 — Summary bar
        # ─────────────────────────────────────────────────────────────────
        n_split   = sum(1 for k in ce_kpis if k.get("baseline_score") is not None)
        n_gates   = sum(1 for k in ce_kpis if k.get("quality_gates"))
        n_bscore  = sum(1 for k in ce_kpis if k.get("bertscore_f1") is not None)
        n_cot     = sum(1 for k in ce_kpis if k.get("cot_eval"))
        n_attr    = sum(1 for k in ce_kpis if k.get("score_attribution"))
        n_dist    = sum(1 for k in ce_kpis if k.get("scoring_distribution"))

        total = len(ce_kpis)
        m0, m1, m2, m3, m4, m5 = st.columns(6)
        m0.metric("Rubric KPIs", total)
        m1.metric("Score Split", f"{n_split}/{total}")
        m2.metric("Quality Gates", f"{n_gates}/{total}")
        m3.metric("BERTScore", f"{n_bscore}/{total}")
        m4.metric("CoT Eval", f"{n_cot}/{total}")
        m5.metric("Attributions", n_attr)

        # Report-level snapshot ID
        snap_id = report_data.get("chromadb_snapshot_id", "")
        if snap_id:
            st.caption(f"ChromaDB snapshot: `{snap_id}`")

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 1 — Score Split & Scoring Distribution
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 1. Score Split & Scoring Distribution")
        st.markdown(
            "**baseline_score** is computed from primary-tier (tier=1) chunks only — "
            "the frozen reference corpus. **live_score** uses all retrieved chunks. "
            "Each is the mean of N=5 LLM runs at temperature > 0."
        )

        split_rows = []
        for k in ce_kpis:
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id

            b = k.get("baseline_score")
            l = k.get("live_score")
            delta = k.get("score_split_delta")
            dist  = k.get("scoring_distribution") or {}

            b_std = dist.get("baseline_std")
            l_std = dist.get("live_std")
            l_raw = dist.get("live_raw_scores") or []

            split_rows.append({
                "KPI": label,
                "Pillar": k.get("pillar", ""),
                "Baseline (mean)": _fmt(b, 3),
                "Baseline σ": _fmt(b_std, 3),
                "Live (mean)": _fmt(l, 3),
                "Live σ": _fmt(l_std, 3),
                "Δ (live−base)": _fmt(delta, 3),
                "N runs": len(l_raw) if l_raw else "—",
                "Point score": _fmt(k.get("score"), 2),
            })

        if split_rows:
            df_split = pd.DataFrame(split_rows)

            def _colour_delta_val(val):
                try:
                    v = float(val)
                    if v > 0.1:  return "color: #00C49A; font-weight:bold"
                    if v < -0.1: return "color: #FF6B6B; font-weight:bold"
                except (TypeError, ValueError):
                    pass
                return ""

            styled_split = df_split.style.applymap(_colour_delta_val, subset=["Δ (live−base)"])
            st.dataframe(styled_split, use_container_width=True, hide_index=True)
        else:
            st.info("No score split data available yet.")

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 2 — Quality Gates
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 2. Quality Gates")
        st.markdown(
            "Four gates run in sequence after scoring. "
            "**Gate 1** (faithfulness) and **Gate 4** (competitor bleed) are hard blocks. "
            "**Gate 2** (stability) shows a score range when σ > 0.4. "
            "**Gate 3** (source coverage) suppresses the structural score when primary-tier fraction < 40 %."
        )

        gate_rows = []
        for k in ce_kpis:
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            gates = (k.get("quality_gates") or {}).get("gates") or {}

            def _g(name):
                g = gates.get(name, {})
                return g.get("passed"), g.get("reason", "")

            f_pass, f_reason = _g("faithfulness_gate")
            s_pass, s_reason = _g("stability_gate")
            c_pass, c_reason = _g("source_coverage_gate")
            b_pass, b_reason = _g("competitor_bleed_gate")

            gate_meta = k.get("quality_gates") or {}
            range_display = gate_meta.get("score_range_display", "")
            blocked = gate_meta.get("blocked", False)
            gate_rows.append({
                "KPI": label,
                "Faithfulness": _gate_badge(f_pass, f_reason),
                "Stability": _gate_badge(s_pass, s_reason),
                "Coverage": _gate_badge(c_pass, c_reason),
                "Bleed": _gate_badge(b_pass, b_reason),
                "Score Display": range_display or _fmt(k.get("live_score") or k.get("score"), 2),
                "Hard Block": "🚫" if blocked else "✓",
            })

        if gate_rows:
            df_gates = pd.DataFrame(gate_rows)
            # Render badge columns as HTML
            html_cols = ["Faithfulness", "Stability", "Coverage", "Bleed"]
            st.write(
                df_gates.to_html(
                    escape=False,
                    index=False,
                    columns=["KPI"] + html_cols + ["Score Display", "Hard Block"],
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No quality gate data available yet.")

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 3 — Retrieval Metrics
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 3. Retrieval Metrics")
        st.markdown(
            "Computed against the **golden chunk set** in Postgres (`golden_chunks` table by default, "
            "via `DATABASE_URL`). "
            "**Hit Rate** = fraction of golden chunks in top-k. "
            "**MRR** = reciprocal rank of the first golden chunk. "
            "**nDCG** = normalised discounted cumulative gain using relevance labels."
        )

        # The retrieval metrics are logged to LangFuse scores — show them from
        # the kpi_results details if present, or indicate where to find them.
        retrieval_rows = []
        for k in ce_kpis:
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            details = k.get("details") or {}
            rm = details.get("retrieval_metrics") or {}

            hit  = rm.get("hit_rate")
            mrr  = rm.get("mrr")
            ndcg = rm.get("ndcg")

            if hit is None and mrr is None and ndcg is None:
                continue

            retrieval_rows.append({
                "KPI": label,
                "Pillar": k.get("pillar", ""),
                "Hit Rate": _fmt(hit, 4),
                "MRR": _fmt(mrr, 4),
                "nDCG": _fmt(ndcg, 4),
            })

        if retrieval_rows:
            df_ret = pd.DataFrame(retrieval_rows)
            st.dataframe(
                df_ret.style.background_gradient(
                    subset=[c for c in ["Hit Rate", "MRR", "nDCG"] if c in df_ret.columns],
                    cmap="YlGn",
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No retrieval metric data in this report. "
                "Populate the `golden_chunks` table (and `DATABASE_URL`) and re-run eval to compute "
                "Hit Rate, MRR, and nDCG. "
                "Scores are also logged as LangFuse trace scores (`retrieval_hit_rate`, "
                "`retrieval_mrr`, `retrieval_ndcg`) for each KPI."
            )

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 4 — BERTScore
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 4. BERTScore F1")
        st.markdown(
            "BERTScore F1 measures the **semantic similarity** between the generated "
            "score rationale (hypothesis) and the top-3 retrieved chunks (reference). "
            "Uses `distilbert-base-uncased`. Scores below 0.75 trigger a "
            "`low_semantic_grounding` warning in LangFuse."
        )

        bert_rows = []
        for k in ce_kpis:
            f1 = k.get("bertscore_f1")
            if f1 is None:
                continue
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            bert_rows.append({
                "KPI": label,
                "Pillar": k.get("pillar", ""),
                "BERTScore F1": float(f1),
                "Flag": "⚠️ low_semantic_grounding" if float(f1) < 0.75 else "✓",
            })

        if bert_rows:
            df_bert = pd.DataFrame(bert_rows).sort_values("BERTScore F1")
            st.dataframe(
                df_bert.style.background_gradient(
                    subset=["BERTScore F1"], cmap="RdYlGn", vmin=0.5, vmax=1.0
                ).format({"BERTScore F1": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

            # Summary bar
            avg_f1 = sum(r["BERTScore F1"] for r in bert_rows) / len(bert_rows)
            n_low  = sum(1 for r in bert_rows if r["BERTScore F1"] < 0.75)
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Avg BERTScore F1", f"{avg_f1:.4f}")
            bc2.metric("Below threshold (< 0.75)", n_low)
            bc3.metric("Pass rate", f"{(len(bert_rows) - n_low) / len(bert_rows):.0%}")
        else:
            st.info(
                "No BERTScore data in this report. "
                "Install `bert-score` and ensure `bertscore_enabled=True` in feature flags."
            )

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 5 — Chain-of-Thought Evaluation
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 5. Chain-of-Thought Evaluation")
        st.markdown(
            "A second LLM call evaluates the score rationale on three dimensions (1–5): "
            "**Specificity** (precise claims), **Evidence** (grounded in retrieved chunks), "
            "**Alignment** (maps to rubric criteria). "
            "Any sub-score < 3 fires a `cot_weak_reasoning` flag."
        )

        cot_rows = []
        for k in ce_kpis:
            cot = k.get("cot_eval")
            if not cot:
                continue
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id

            spec  = cot.get("cot_specificity")
            ev    = cot.get("cot_evidence")
            align = cot.get("cot_alignment")
            avg_cot = (
                sum(v for v in [spec, ev, align] if v is not None)
                / sum(1 for v in [spec, ev, align] if v is not None)
                if any(v is not None for v in [spec, ev, align]) else None
            )
            weak = any(
                v is not None and v < 3
                for v in [spec, ev, align]
            )

            cot_rows.append({
                "KPI": label,
                "Pillar": k.get("pillar", ""),
                "Specificity": spec,
                "Evidence": ev,
                "Alignment": align,
                "Avg CoT": round(avg_cot, 2) if avg_cot else None,
                "Flag": "⚠️ cot_weak_reasoning" if weak else "✓",
            })

        if cot_rows:
            df_cot = pd.DataFrame(cot_rows)
            num_cols = [c for c in ["Specificity", "Evidence", "Alignment", "Avg CoT"]
                        if c in df_cot.columns and df_cot[c].notna().any()]

            styled_cot = df_cot.style
            if num_cols:
                styled_cot = styled_cot.background_gradient(
                    subset=num_cols, cmap="RdYlGn", vmin=1, vmax=5
                )
            st.dataframe(styled_cot, use_container_width=True, hide_index=True)

            # Radar / bar chart — avg scores per dimension
            dim_avgs = {
                "Specificity": df_cot["Specificity"].dropna().mean() if df_cot["Specificity"].notna().any() else 0,
                "Evidence":    df_cot["Evidence"].dropna().mean()    if df_cot["Evidence"].notna().any()    else 0,
                "Alignment":   df_cot["Alignment"].dropna().mean()   if df_cot["Alignment"].notna().any()   else 0,
            }
            st.markdown("**Average CoT sub-scores across all KPIs:**")
            cc1, cc2, cc3 = st.columns(3)
            for col, (dim, avg) in zip([cc1, cc2, cc3], dim_avgs.items()):
                col.metric(dim, f"{avg:.2f} / 5")
        else:
            st.info(
                "No CoT evaluation data in this report. "
                "Ensure `cot_eval_enabled=True` in feature flags and re-run the pipeline."
            )

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 6 — Score Change Attribution
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 6. Score Change Attribution")
        st.markdown(
            "When |Δ mean score| > 0.2 vs the previous run, the change is attributed to one of: "
            "**model_change**, **prompt_change**, **data_change**, or **external_noise**."
        )

        attr_rows = []
        for k in ce_kpis:
            attr = k.get("score_attribution")
            if not attr:
                continue
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            atype = attr.get("attribution_type", "")
            attr_rows.append({
                "KPI": label,
                "Prev Score": _fmt(attr.get("previous_score"), 3),
                "New Score": _fmt(attr.get("new_score"), 3),
                "Δ": _fmt(attr.get("delta"), 3),
                "Attribution": f'<span class="attr-badge attr-{atype}">{atype}</span>',
            })

        if attr_rows:
            df_attr = pd.DataFrame(attr_rows)
            st.write(
                df_attr.to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No attribution events in this report. Attribution fires when "
                "|Δ mean score| > 0.2 vs the previous run for the same company."
            )

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # SECTION 7 — Traceability
        # ─────────────────────────────────────────────────────────────────
        st.markdown("#### 7. Traceability")
        st.markdown(
            "Prompt hashes, ChromaDB snapshot IDs, and MLflow run IDs make every "
            "evaluation fully reproducible. The snapshot ID on the report header "
            "is the fingerprint of the collection at report-generation time."
        )

        trace_rows = []
        for k in ce_kpis:
            ph = k.get("prompt_hash")
            sid = k.get("chromadb_snapshot_id")
            mlid = k.get("mlflow_run_id")
            lfid = k.get("langfuse_trace_id")
            if not any([ph, sid, mlid, lfid]):
                continue
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id
            trace_rows.append({
                "KPI": label,
                "Prompt Hash": (ph or "")[:16] + "…" if ph else "—",
                "Snapshot ID": (sid or "")[:16] + "…" if sid else "—",
                "MLflow Run ID": (mlid or "")[:16] + "…" if mlid else "—",
                "Langfuse Trace ID": (lfid or "")[:16] + "…" if lfid else "—",
            })

        if trace_rows:
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
            with st.expander("Show full hashes"):
                for r in trace_rows:
                    kpi_id_r = r["KPI"]
                    kpi_r = next((k for k in ce_kpis if (_KPI_DEFS.get(str(k["kpi_id"])) and
                                 _KPI_DEFS[str(k["kpi_id"])].name == kpi_id_r) or str(k["kpi_id"]) == kpi_id_r), None)
                    if kpi_r:
                        st.markdown(f"**{kpi_id_r}**")
                        st.code(
                            f"prompt_hash:          {kpi_r.get('prompt_hash') or '—'}\n"
                            f"chromadb_snapshot_id: {kpi_r.get('chromadb_snapshot_id') or '—'}\n"
                            f"mlflow_run_id:        {kpi_r.get('mlflow_run_id') or '—'}\n"
                            f"langfuse_trace_id:    {kpi_r.get('langfuse_trace_id') or '—'}"
                        )
        else:
            st.info("No traceability data found. Traceability IDs are populated on rubric KPIs.")

        # ─────────────────────────────────────────────────────────────────
        # Per-KPI drill-down expanders
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### KPI-level Detail")
        st.caption("Expand any KPI to see the full custom evaluation breakdown.")

        for k in ce_kpis:
            kpi_id = str(k["kpi_id"])
            kpi_def = _KPI_DEFS.get(kpi_id)
            label = kpi_def.name if kpi_def and getattr(kpi_def, "name", None) else kpi_id

            has_data = any([
                k.get("baseline_score") is not None,
                k.get("quality_gates"),
                k.get("bertscore_f1") is not None,
                k.get("cot_eval"),
                k.get("score_attribution"),
            ])
            if not has_data:
                continue

            gates_meta = k.get("quality_gates") or {}
            blocked    = gates_meta.get("blocked", False)
            icon = "🚫" if blocked else "🧪"
            f1_str = f"  BERTScore={_fmt(k.get('bertscore_f1'), 3)}" if k.get("bertscore_f1") is not None else ""

            with st.expander(f"{icon} {label}  (score {_fmt(k.get('score'), 1)}/5  |  live {_fmt(k.get('live_score'), 2)}{f1_str})"):

                d1, d2, d3 = st.columns(3)

                # Score split
                with d1:
                    st.markdown("**Score Split**")
                    dist = k.get("scoring_distribution") or {}
                    b_mean = k.get("baseline_score")
                    l_mean = k.get("live_score")
                    b_std  = dist.get("baseline_std")
                    l_std  = dist.get("live_std")
                    delta  = k.get("score_split_delta")

                    if b_mean is not None:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{score_color(b_mean, 5.0)}">{_fmt(b_mean, 2)}</div>'
                            f'<div class="eval-label">Baseline (± {_fmt(b_std, 3)})</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if l_mean is not None:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{score_color(l_mean, 5.0)}">{_fmt(l_mean, 2)}</div>'
                            f'<div class="eval-label">Live (± {_fmt(l_std, 3)})</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if delta is not None:
                        dcolor = "#00C49A" if delta > 0 else "#FF6B6B" if delta < 0 else "#888"
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{dcolor}">{delta:+.3f}</div>'
                            f'<div class="eval-label">Δ live − baseline</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    gate_score_display = gates_meta.get("score_range_display")
                    if gate_score_display:
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:#f0c040">{gate_score_display}</div>'
                            f'<div class="eval-label">Score range (unstable σ)</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # BERTScore + CoT
                with d2:
                    st.markdown("**Semantic & Reasoning Quality**")
                    f1_val = k.get("bertscore_f1")
                    if f1_val is not None:
                        bcolor = score_color(float(f1_val), 1.0)
                        flag = " ⚠️" if float(f1_val) < 0.75 else ""
                        st.markdown(
                            f'<div class="eval-metric">'
                            f'<div class="eval-value" style="color:{bcolor}">{_fmt(f1_val, 4)}{flag}</div>'
                            f'<div class="eval-label">BERTScore F1</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    cot = k.get("cot_eval") or {}
                    for dim, label_dim in [
                        ("cot_specificity", "Specificity"),
                        ("cot_evidence", "Evidence"),
                        ("cot_alignment", "Alignment"),
                    ]:
                        val = cot.get(dim)
                        if val is not None:
                            ccolor = score_color(float(val), 5.0)
                            flag_cot = " ⚠️" if int(val) < 3 else ""
                            st.markdown(
                                f'<div class="eval-metric">'
                                f'<div class="eval-value" style="color:{ccolor}">{val}{flag_cot}</div>'
                                f'<div class="eval-label">CoT {label_dim} / 5</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Quality gates
                with d3:
                    st.markdown("**Quality Gates**")
                    gates = gates_meta.get("gates") or {}
                    for gate_name, gate_label in [
                        ("faithfulness_gate", "Faithfulness"),
                        ("stability_gate",    "Stability"),
                        ("source_coverage_gate", "Source Coverage"),
                        ("competitor_bleed_gate", "Competitor Bleed"),
                    ]:
                        g = gates.get(gate_name, {})
                        badge = _gate_badge(g.get("passed"), g.get("reason", ""))
                        st.markdown(f"{badge} &nbsp; {gate_label}", unsafe_allow_html=True)

                    if gates_meta.get("suppress_structural_score"):
                        st.markdown(
                            '<span class="gate-warn">⚠ Structural score suppressed</span>',
                            unsafe_allow_html=True,
                        )
                    if gates_meta.get("competitor_bleed_detected"):
                        st.error("Competitor bleed detected — result blocked.")

                # Attribution
                attr = k.get("score_attribution")
                if attr:
                    atype = attr.get("attribution_type", "")
                    st.markdown(
                        f"**Score Attribution:** "
                        f'<span class="attr-badge attr-{atype}">{atype}</span>  '
                        f"Δ = {_fmt(attr.get('delta'), 3)}  "
                        f"({_fmt(attr.get('previous_score'), 2)} → {_fmt(attr.get('new_score'), 2)})",
                        unsafe_allow_html=True,
                    )

                # Raw JSON
                raw_keys = [
                    "baseline_score", "live_score", "score_split_delta",
                    "scoring_distribution", "quality_gates", "score_attribution",
                    "bertscore_f1", "cot_eval", "prompt_hash",
                    "chromadb_snapshot_id", "mlflow_run_id", "langfuse_trace_id",
                ]
                raw_payload = {k2: k.get(k2) for k2 in raw_keys if k.get(k2) is not None}
                if raw_payload:
                    with st.popover("Raw JSON"):
                        st.json(raw_payload)

# TAB: Sources
# ============================================================================
with tab_sources:
    st.markdown("### Crawled Sources")
    st.markdown(f"**Total URLs crawled:** {url_count}")

    source_urls: dict[str, dict] = {}
    for kpi in kpi_results:
        for c in kpi.get("citations", []):
            sid = c.get("source_id", "")
            if sid and sid not in source_urls:
                source_urls[sid] = {
                    "source_id": sid,
                    "url": c.get("url", ""),
                    "cited_in": [],
                }
            if sid in source_urls:
                source_urls[sid]["cited_in"].append(kpi["kpi_id"])

    if source_urls:
        rows = []
        for sid, info in source_urls.items():
            row = {
                "Source ID": sid,
                "URL": info["url"],
                "Cited In (# KPIs)": len(set(info["cited_in"])),
                "KPIs": ", ".join(sorted(set(info["cited_in"]))),
            }
            # Add authority info if available
            auth_info = all_authority_sources.get(sid, {}) if has_eval_data else {}
            if auth_info:
                row["Authority"] = AUTHORITY_LABELS.get(auth_info.get("authority_type", ""), "")
                row["Auth Score"] = auth_info.get("authority_score", 0)

            # Add freshness info if available
            fresh_info = freshness_sources.get(sid, {}) if has_eval_data else {}
            if fresh_info:
                row["Freshness"] = fresh_info.get("freshness_score", 0)
                age = fresh_info.get("age_days")
                row["Age (days)"] = age if age is not None else "?"

            rows.append(row)

        df_sources = pd.DataFrame(rows).sort_values("Cited In (# KPIs)", ascending=False)
        st.dataframe(df_sources, use_container_width=True, hide_index=True)
    else:
        st.info("No source citation data available.")

    fetch_entries = [d for d in debug_log if d.startswith("[fetch]")]
    if fetch_entries:
        st.markdown("### Fetch Log")
        for entry in fetch_entries:
            parts = entry.split(": ", 1)
            status = "✅" if "ok" in parts[0] else "❌"
            url_part = parts[1] if len(parts) > 1 else entry
            st.markdown(f"{status} `{url_part}`")


# ============================================================================
# TAB: Citations
# ============================================================================
with tab_citations:
    st.markdown("### All Citations")

    all_citations = []
    for kpi in kpi_results:
        for c in kpi.get("citations", []):
            all_citations.append({
                "KPI": kpi["kpi_id"],
                "Pillar": kpi.get("pillar", ""),
                "Source": c.get("source_id", ""),
                "URL": c.get("url", ""),
                "Quote": c.get("quote", "")[:300],
            })

    if all_citations:
        df_cit = pd.DataFrame(all_citations)

        pillar_filter = st.multiselect(
            "Filter by Pillar",
            options=sorted(df_cit["Pillar"].unique()),
            default=sorted(df_cit["Pillar"].unique()),
        )
        filtered = df_cit[df_cit["Pillar"].isin(pillar_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.metric("Total Citations", len(filtered))
    else:
        st.info("No citations found.")


# ============================================================================
# TAB: Raw Report
# ============================================================================
with tab_raw:
    st.markdown("### Raw YAML Report")
    st.code(yaml.dump(report_data, default_flow_style=False, allow_unicode=True), language="yaml")


# ============================================================================
# TAB: Debug Log
# ============================================================================
with tab_debug:
    st.markdown("### Pipeline Debug Log")
    if debug_log:
        for entry in debug_log:
            if "[fetch]" in entry:
                st.markdown(f"🌐 `{entry}`")
            elif "[llm]" in entry:
                st.markdown(f"🤖 `{entry}`")
            else:
                st.markdown(f"📝 `{entry}`")
    else:
        st.info("No debug log available. Set `VITELIS_DEBUG=1` to enable.")
