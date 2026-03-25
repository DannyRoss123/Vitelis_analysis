"""Vitelis AI Maturity Dashboard v2 — interactive Streamlit visualization."""

from __future__ import annotations

from app.snapshots import build_snapshot, find_previous_snapshot_for_domain, load_snapshot, diff_snapshots

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

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

# FIX 1: number of tab variables must match number of tab labels
tab_kpi, tab_diff, tab_eval, tab_sources, tab_citations, tab_raw, tab_debug = st.tabs(
    ["KPI Scores", "Run Diffs", "Source Evaluation", "Sources", "Citations", "Raw Report", "Debug Log"]
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
            kpi_id = kpi["kpi_id"]
            score = kpi["score"]
            conf = kpi.get("confidence", 0)
            ktype = kpi.get("type", "")
            rationale = kpi.get("rationale", "")
            details = kpi.get("details", {}) or {}
            citations = kpi.get("citations", [])
            source_eval = details.get("source_evaluation", {})

            is_missing = kpi_id in missing_evidence

            with st.expander(
                f"{'🔴' if is_missing else '📊'} {kpi_id}  Score: {score:.1f}/5  Confidence: {conf:.0%}  {ktype}"
            ):
                st.progress(min(score / 5.0, 1.0))
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

# This tab compares the current run to the most recent previous run for the same company domain.
# It summarizes what changed in overall score, KPI scores and confidence, and which citation sources 
# were added or removed, with a simple one line reason for the biggest movers. It also warns if the 
# two runs used different settings like model or temperature, and lets you drill down via a
# filtered table and citation details.

with tab_diff:
    st.markdown("### Run to run diffs")

    current_snap = build_snapshot(report_data, report_path="")  # build a compact snapshot from the currently loaded report
    prev_path = find_previous_snapshot_for_domain(domain, exclude_run_id=run_id)  # find the most recent prior snapshot for same domain

    if not prev_path:
        st.info("No previous snapshot found for this domain yet. Run again with a new run id to see diffs.")
        st.stop()  # nothing to compare against

    prev_snap = load_snapshot(str(prev_path))  # load the prior snapshot JSON from disk
    d = diff_snapshots(prev_snap, current_snap)  # compute a diff dict (summary + table + reasons)

    cfg = d.get("config", {}) or {}  # old vs new run settings (model, temperature, etc.)
    if cfg.get("changed"):
        st.warning("Run settings differ between runs. Diffs may reflect config changes, not evidence changes.")
        st.json(cfg)

    st.write(  # headline score change
        f"Overall score: {d['overall_old']:.2f} to {d['overall_new']:.2f}  delta {d['overall_delta']:+.2f}"
    )

    s = d.get("summary", {}) or {}  # short summary counts
    st.write(
        f"KPIs changed: {s.get('total_kpis_changed', 0)}  "
        f"Citations added: {s.get('total_citations_added', 0)}  "
        f"Citations removed: {s.get('total_citations_removed', 0)}"
    )

    rag = d.get("rag_eval", {}) or {}  # optional: RAG eval summary deltas if report has rag_evaluation
    if rag.get("old") or rag.get("new"):
        st.write("RAG eval delta summary")
        st.json(rag.get("delta", {}))

    st.markdown("#### Biggest movers")  # top 5 KPI changes with a 1 line explanation
    for r in (d.get("biggest_movers") or [])[:5]:
        st.write(
            f"{r.get('kpi_id','')}: score {r.get('score_delta',0):+.2f}, conf {r.get('confidence_delta',0):+.2f}, {r.get('reason','')}"
        )

    with st.expander("Show table view"):
        rows = d.get("table_rows", []) or []  # full KPI diff rows
        df = pd.DataFrame(rows)  # make it easy to filter and display

        only_changes = st.toggle("Show only changed KPIs", value=True)  # hide noise
        pillar_options = ["All"] + sorted(df["pillar"].fillna("Unknown").unique().tolist()) if len(df) else ["All"]
        pillar_choice = st.selectbox("Pillar", pillar_options, index=0)  # filter by pillar
        min_abs = st.slider("Min absolute score delta", 0.0, 5.0, 0.1, 0.1)  # filter by magnitude

        if len(df):
            if only_changes:
                df = df[
                    (df["score_delta"].abs() > 1e-9)
                    | (df["confidence_delta"].abs() > 1e-9)
                    | (df["new_citations_count"] > 0)
                    | (df["removed_citations_count"] > 0)
                ]
            if pillar_choice != "All":
                df = df[df["pillar"] == pillar_choice]
            df = df[df["score_delta"].abs() >= min_abs]

            show_cols = [  # keep the table compact and readable
                "kpi_id",
                "pillar",
                "score_old",
                "score_new",
                "score_delta",
                "confidence_old",
                "confidence_new",
                "confidence_delta",
                "new_citations_count",
                "removed_citations_count",
                "reason",
            ]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    with st.expander("Show citation changes for one KPI"):
        rows = d.get("table_rows", []) or []
        if rows:
            pick = st.selectbox("Pick KPI", sorted({r.get("kpi_id", "") for r in rows if r.get("kpi_id")}))
            r = next((x for x in rows if x.get("kpi_id") == pick), None)
            if r:
                st.write("New citations")
                for u in (r.get("new_citations") or [])[:50]:
                    st.write(u)
                st.write("Removed citations")
                for u in (r.get("removed_citations") or [])[:50]:
                    st.write(u)

    with st.expander("Show full diff json"):
        st.json(d)  # full raw diff for debugging
            
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
