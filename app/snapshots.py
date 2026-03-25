from __future__ import annotations

"""
snapshots.py

Purpose
- Save a compact per-run "snapshot" JSON so comparing runs is fast.
- Compute a diff between two snapshots:
  - summary numbers (how many KPIs changed, citations added/removed)
  - top movers
  - a table with reasons for changes (heuristics, no extra LLM calls)
  - config comparison (model, temperature) so you can tell if runs are comparable
  - RAG eval summary diff if present in the report

Snapshot location
- app/output/snapshots/snapshot_<run_id>.json
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _output_dir() -> Path:
    """
    Returns app/output, creating it if missing.
    We keep snapshots next to YAML reports.
    """
    here = Path(__file__).resolve().parent
    out_dir = (here / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _snapshots_dir() -> Path:
    """
    Returns app/output/snapshots, creating it if missing.
    """
    d = _output_dir() / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_quality(details: Any) -> Dict[str, Any]:
    """
    Extract a small set of evidence-quality signals from KPI details.
    This is optional. If the report does not contain these fields, return {}.
    """
    if not isinstance(details, dict):
        return {}

    se = details.get("source_evaluation", {})
    if not isinstance(se, dict) or not se:
        return {}

    corr = se.get("semantic_corroboration", {}) or {}
    fresh = se.get("freshness", {}) or {}
    auth = se.get("authority", {}) or {}
    contrad = se.get("contradictions", {}) or {}

    quality: Dict[str, Any] = {
        "corroboration_score": float(corr.get("corroboration_score", 0) or 0),
        "freshness_boost": float(fresh.get("boost", 0) or 0),
        "authority_boost": float(auth.get("boost", 0) or 0),
        "contradiction_count": int(contrad.get("contradiction_count", 0) or 0),
        "contradiction_penalty": float(contrad.get("confidence_penalty", 0) or 0),
    }

    # Optional details that some versions of the pipeline include
    td = details.get("tier_distribution")
    if isinstance(td, dict):
        quality["tier_distribution"] = td

    us = details.get("unique_sources")
    if us is not None:
        try:
            quality["unique_sources"] = int(us)
        except Exception:
            pass

    return quality


def _extract_rag_eval(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact summary of rag_evaluation if present in the report.
    We store averages over per_kpi metrics so diffs stay small.
    """
    reval = report.get("rag_evaluation")
    if not isinstance(reval, dict) or not reval:
        return {}

    per_kpi = reval.get("per_kpi", []) or []

    def avg(key: str) -> float:
        vals: List[float] = []
        for r in per_kpi:
            if not isinstance(r, dict):
                continue
            v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "evaluated_kpi_count": int(reval.get("evaluated_kpi_count", 0) or 0),
        "flagged_kpi_count": int(reval.get("flagged_kpi_count", 0) or 0),
        "avg_f1": avg("f1"),
        "avg_semantic_similarity": avg("semantic_similarity"),
        "avg_ragas_faithfulness": avg("ragas_faithfulness"),
        "avg_ragas_context_recall": avg("ragas_context_recall"),
        "avg_ragas_context_precision": avg("ragas_context_precision"),
    }


def build_snapshot(report: Dict[str, Any], report_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a full report dict into a compact snapshot dict.
    The report input is typically report.model_dump() from the pipeline.
    """
    kpis: Dict[str, Any] = {}

    for k in report.get("kpi_results", []) or []:
        kpi_id = k.get("kpi_id")
        if not kpi_id:
            continue

        # Pull citations and compress them to only URLs and distinct source_ids
        citations = k.get("citations", []) or []
        urls = [c.get("url", "") for c in citations if isinstance(c, dict)]
        urls = sorted({u for u in urls if u})

        sids = [c.get("source_id", "") for c in citations if isinstance(c, dict)]
        distinct_sources = sorted({s for s in sids if s})

        details = k.get("details", {}) or {}
        evidence_gate = details.get("evidence_gate") if isinstance(details, dict) else None
        quality = _extract_quality(details)

        # Store only what is needed for diffs and dashboard summaries
        kpis[kpi_id] = {
            "pillar": k.get("pillar", ""),
            "type": k.get("type", ""),
            "score": float(k.get("score", 0) or 0),
            "confidence": float(k.get("confidence", 0) or 0),
            "citation_urls": urls,
            "distinct_sources": distinct_sources,
            "evidence_gate": evidence_gate,
            "quality": quality,
        }

    # Pillar summary is useful for high-level deltas
    pillar_scores: Dict[str, Any] = {}
    for p in report.get("pillar_scores", []) or []:
        pname = p.get("pillar")
        if not pname:
            continue
        pillar_scores[pname] = {
            "score": float(p.get("score", 0) or 0),
            "confidence": float(p.get("confidence", 0) or 0),
            "kpis": list(p.get("kpis", []) or []),
        }

    # Record run settings so diffs can warn when runs are not comparable
    run_settings = {
        "openai_model": os.getenv("OPENAI_MODEL", ""),
        "openai_temperature": os.getenv("OPENAI_TEMPERATURE", ""),
        "openai_min_delay_seconds": os.getenv("OPENAI_MIN_DELAY_SECONDS", ""),
        "max_urls": os.getenv("MAX_URLS", ""),
    }

    return {
        "snapshot_version": 3,
        "run_id": report.get("run_id", ""),
        "timestamp": report.get("timestamp", ""),
        "company_name": report.get("company_name", ""),
        "company_domain": report.get("company_domain", ""),
        "url_count": int(report.get("url_count", 0) or 0),
        "overall_score": float(report.get("overall_score", 0) or 0),
        "pillar_scores": pillar_scores,
        "kpis": kpis,
        "rag_eval": _extract_rag_eval(report),
        "run_settings": run_settings,
        "report_path": report_path or "",
    }


def write_snapshot(snapshot: Dict[str, Any]) -> str:
    """
    Write snapshot JSON to disk and return the file path.
    """
    run_id = snapshot.get("run_id") or "unknown"
    path = _snapshots_dir() / f"snapshot_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=False)
    return str(path)


def list_snapshots() -> List[Path]:
    """
    List snapshots ordered by most recently modified first.
    """
    d = _snapshots_dir()
    return sorted(d.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_snapshot(path: str) -> Dict[str, Any]:
    """
    Read a snapshot JSON file into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_previous_snapshot_for_domain(domain: str, exclude_run_id: str) -> Optional[Path]:
    """
    Find the newest snapshot for this domain, excluding the current run_id.
    Used by the dashboard to compare current run to the previous run.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return None

    for p in list_snapshots():
        try:
            snap = load_snapshot(str(p))
        except Exception:
            continue
        if (snap.get("company_domain", "") or "").strip().lower() != domain:
            continue
        if (snap.get("run_id", "") or "") == exclude_run_id:
            continue
        return p

    return None


def _reason_for_change(row: Dict[str, Any]) -> str:
    """
    Heuristic explanation for why a KPI changed.
    This uses evidence changes and quality deltas, not an LLM call.
    """
    parts: List[str] = []

    new_c = int(row.get("new_citations_count", 0) or 0)
    rem_c = int(row.get("removed_citations_count", 0) or 0)
    ds_old = int(row.get("distinct_sources_old", 0) or 0)
    ds_new = int(row.get("distinct_sources_new", 0) or 0)

    s_delta = float(row.get("score_delta", 0) or 0)
    c_delta = float(row.get("confidence_delta", 0) or 0)

    contrad_old = int(row.get("contradiction_count_old", 0) or 0)
    contrad_new = int(row.get("contradiction_count_new", 0) or 0)

    fresh_old = float(row.get("freshness_boost_old", 0) or 0)
    fresh_new = float(row.get("freshness_boost_new", 0) or 0)

    corr_old = float(row.get("corroboration_score_old", 0) or 0)
    corr_new = float(row.get("corroboration_score_new", 0) or 0)

    if new_c > 0 and s_delta > 0:
        parts.append("score up with new evidence")
    if rem_c > 0 and s_delta < 0:
        parts.append("score down with evidence removed")
    if ds_new > ds_old and c_delta > 0:
        parts.append("confidence up with more source diversity")
    if contrad_new > contrad_old and c_delta < 0:
        parts.append("confidence down due to more contradictions")
    if fresh_new < fresh_old and c_delta < 0:
        parts.append("confidence down due to older evidence")
    if corr_new > corr_old and c_delta > 0:
        parts.append("confidence up due to stronger cross source agreement")
    if rem_c > 0 and not parts:
        parts.append("retrieval changed or sources unavailable")

    if not parts:
        parts.append("small shift from different retrieved chunks")

    return "; ".join(parts)


def diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two snapshots and return:
    - overall score deltas
    - summary counts
    - a compact top movers list
    - a full table of KPI diffs with a reason string
    - config and rag eval diffs
    """
    old_kpis = old.get("kpis", {}) or {}
    new_kpis = new.get("kpis", {}) or {}

    all_ids = sorted(set(old_kpis.keys()) | set(new_kpis.keys()))
    rows: List[Dict[str, Any]] = []

    for kpi_id in all_ids:
        o = old_kpis.get(kpi_id) or {}
        n = new_kpis.get(kpi_id) or {}

        old_urls = set(o.get("citation_urls", []) or [])
        new_urls = set(n.get("citation_urls", []) or [])
        new_citations = sorted(new_urls - old_urls)
        removed_citations = sorted(old_urls - new_urls)

        q_old = o.get("quality", {}) or {}
        q_new = n.get("quality", {}) or {}

        # Build one row per KPI so the dashboard can show a table
        row: Dict[str, Any] = {
            "kpi_id": kpi_id,
            "pillar": n.get("pillar", o.get("pillar", "")),
            "type": n.get("type", o.get("type", "")),
            "score_old": float(o.get("score", 0) or 0),
            "score_new": float(n.get("score", 0) or 0),
            "confidence_old": float(o.get("confidence", 0) or 0),
            "confidence_new": float(n.get("confidence", 0) or 0),
            "distinct_sources_old": len(o.get("distinct_sources", []) or []),
            "distinct_sources_new": len(n.get("distinct_sources", []) or []),
            "new_citations_count": len(new_citations),
            "removed_citations_count": len(removed_citations),
            "new_citations": new_citations,
            "removed_citations": removed_citations,
            "corroboration_score_old": float(q_old.get("corroboration_score", 0) or 0),
            "corroboration_score_new": float(q_new.get("corroboration_score", 0) or 0),
            "freshness_boost_old": float(q_old.get("freshness_boost", 0) or 0),
            "freshness_boost_new": float(q_new.get("freshness_boost", 0) or 0),
            "authority_boost_old": float(q_old.get("authority_boost", 0) or 0),
            "authority_boost_new": float(q_new.get("authority_boost", 0) or 0),
            "contradiction_count_old": int(q_old.get("contradiction_count", 0) or 0),
            "contradiction_count_new": int(q_new.get("contradiction_count", 0) or 0),
        }

        # Compute deltas
        row["score_delta"] = row["score_new"] - row["score_old"]
        row["confidence_delta"] = row["confidence_new"] - row["confidence_old"]

        # Compute a short explanation string
        row["reason"] = _reason_for_change(row)

        rows.append(row)

    # Overall deltas
    overall_old = float(old.get("overall_score", 0) or 0)
    overall_new = float(new.get("overall_score", 0) or 0)

    # Summary counts
    total_kpis_changed = sum(
        1
        for r in rows
        if abs(r["score_delta"]) > 1e-9
        or abs(r["confidence_delta"]) > 1e-9
        or r["new_citations_count"] > 0
        or r["removed_citations_count"] > 0
    )
    total_citations_added = sum(int(r["new_citations_count"]) for r in rows)
    total_citations_removed = sum(int(r["removed_citations_count"]) for r in rows)

    # Aggregate by pillar so you can see where movement came from
    changes_by_pillar: Dict[str, Any] = {}
    for r in rows:
        p = r.get("pillar", "") or "Unknown"
        changes_by_pillar.setdefault(
            p,
            {"kpis_changed": 0, "abs_score_delta_sum": 0.0, "abs_conf_delta_sum": 0.0},
        )
        changed = (
            abs(r["score_delta"]) > 1e-9
            or abs(r["confidence_delta"]) > 1e-9
            or r["new_citations_count"] > 0
            or r["removed_citations_count"] > 0
        )
        if changed:
            changes_by_pillar[p]["kpis_changed"] += 1
            changes_by_pillar[p]["abs_score_delta_sum"] += abs(float(r["score_delta"]))
            changes_by_pillar[p]["abs_conf_delta_sum"] += abs(float(r["confidence_delta"]))

    # Biggest movers helps make the dashboard short by default
    def mover_score(r: Dict[str, Any]) -> float:
        return abs(float(r.get("score_delta", 0) or 0)) + 0.5 * abs(float(r.get("confidence_delta", 0) or 0))

    biggest_movers = sorted(rows, key=mover_score, reverse=True)[:5]

    # Compare run settings so we can warn if temperature or model changed
    old_cfg = old.get("run_settings", {}) or {}
    new_cfg = new.get("run_settings", {}) or {}
    changed_keys = sorted([k for k in set(old_cfg) | set(new_cfg) if old_cfg.get(k, "") != new_cfg.get(k, "")])

    # Compare RAG eval summary if it exists
    old_rag = old.get("rag_eval", {}) or {}
    new_rag = new.get("rag_eval", {}) or {}
    rag_delta = {
        k: float(new_rag.get(k, 0) or 0) - float(old_rag.get(k, 0) or 0)
        for k in set(old_rag) | set(new_rag)
    }

    return {
        "old_run_id": old.get("run_id", ""),
        "new_run_id": new.get("run_id", ""),
        "old_timestamp": old.get("timestamp", ""),
        "new_timestamp": new.get("timestamp", ""),
        "company_domain": new.get("company_domain", old.get("company_domain", "")),
        "overall_old": overall_old,
        "overall_new": overall_new,
        "overall_delta": overall_new - overall_old,
        "summary": {
            "total_kpis": len(rows),
            "total_kpis_changed": total_kpis_changed,
            "total_citations_added": total_citations_added,
            "total_citations_removed": total_citations_removed,
            "changes_by_pillar": changes_by_pillar,
        },
        "biggest_movers": biggest_movers,
        "table_rows": rows,
        "config": {
            "old": old_cfg,
            "new": new_cfg,
            "changed_keys": changed_keys,
            "changed": bool(changed_keys),
        },
        "rag_eval": {
            "old": old_rag,
            "new": new_rag,
            "delta": rag_delta,
        },
    }