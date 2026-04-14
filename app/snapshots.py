from __future__ import annotations

"""
snapshots.py

Goal
This module creates and compares small "snapshot" JSON files for each run.
A snapshot is a compact summary of the run results that is easy to diff between runs.

Why it exists
The YAML report is great for full detail, but it is bulky and harder to compare.
Snapshots keep only the fields you usually care about when answering:
- Did the overall score change?
- Which KPIs changed score or confidence?
- Which citations were added or removed?
- Did evidence gating trigger?

Where snapshots are stored
app/output/snapshots/snapshot_<run_id>.json

How it is used in the codebase
- aggregate_report_node writes the YAML report
- then it calls build_snapshot(report_dict) and write_snapshot(snapshot)
- later, the dashboard or diff_snapshot.py loads two snapshots and calls diff_snapshots(old, new)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _output_dir() -> Path:
    """
    Returns the absolute path to app/output/.

    This folder already exists in the repo logic:
    - YAML reports are written to app/output/report_<run_id>.yaml
    We reuse the same folder for snapshots.
    """
    here = Path(__file__).resolve().parent
    out_dir = (here / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _snapshots_dir() -> Path:
    """
    Returns the path to the snapshots folder: app/output/snapshots/.

    If it does not exist, it is created.
    """
    d = _output_dir() / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_snapshot(report: Dict[str, Any], report_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a full YAML report dict into a compact "snapshot" dict.

    Input
    - report: the same dict that gets written into YAML (report.model_dump()).
    - report_path: optional path to the YAML file for traceability.

    Output snapshot fields
    - run metadata: run_id, timestamp, company_name, company_domain, url_count
    - overall_score
    - pillar_scores: score/confidence per pillar
    - kpis: a per-KPI summary including score, confidence, citations, and evidence gate info

    Note on citations
    We store only citation URLs and distinct source_ids.
    That is enough to answer "what evidence changed" between runs.
    """
    kpis: Dict[str, Any] = {}

    # report["kpi_results"] is typically a list of dicts (because we model_dump() it).
    for k in report.get("kpi_results", []) or []:
        kpi_id = k.get("kpi_id")
        if not kpi_id:
            continue

        citations = k.get("citations", []) or []

        # We keep citation URLs because they are human-readable and stable across runs.
        citation_urls = [c.get("url", "") for c in citations if isinstance(c, dict)]
        citation_urls = sorted({u for u in citation_urls if u})

        # We keep distinct source_ids so we can measure evidence diversity.
        source_ids = [c.get("source_id", "") for c in citations if isinstance(c, dict)]
        distinct_sources = sorted({s for s in source_ids if s})

        # Evidence gating is an optional detail, depends on whether you implemented that feature.
        details = k.get("details", {}) or {}
        evidence_gate = details.get("evidence_gate") if isinstance(details, dict) else None

        kpis[kpi_id] = {
            "pillar": k.get("pillar", ""),
            "type": k.get("type", ""),
            "score": float(k.get("score", 0) or 0),
            "confidence": float(k.get("confidence", 0) or 0),
            "citation_urls": citation_urls,
            "distinct_sources": distinct_sources,
            "evidence_gate": evidence_gate,
        }

    # Summarize pillars so we can see high-level movement without reading all KPIs.
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

    return {
        "snapshot_version": 1,
        "run_id": report.get("run_id", ""),
        "timestamp": report.get("timestamp", ""),
        "company_name": report.get("company_name", ""),
        "company_domain": report.get("company_domain", ""),
        "url_count": int(report.get("url_count", 0) or 0),
        "overall_score": float(report.get("overall_score", 0) or 0),
        # Feature 9: include snapshot ID so attribution state can detect data changes
        "chromadb_snapshot_id": report.get("chromadb_snapshot_id", ""),
        "pillar_scores": pillar_scores,
        "kpis": kpis,
        "report_path": report_path or "",
    }


def write_snapshot(snapshot: Dict[str, Any]) -> str:
    """
    Write a snapshot JSON to disk and return the file path as a string.

    File naming
    snapshot_<run_id>.json

    This makes it easy to diff two runs by run_id.
    """
    run_id = snapshot.get("run_id") or "unknown"
    path = _snapshots_dir() / f"snapshot_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=False)
    return str(path)


def list_snapshots() -> List[Path]:
    """
    Return snapshot files sorted by most recently modified first.

    Used by find_previous_snapshot_for_domain to find the "last run" quickly.
    """
    d = _snapshots_dir()
    return sorted(d.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_snapshot(path: str) -> Dict[str, Any]:
    """
    Load a snapshot JSON file from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_previous_snapshot_for_domain(domain: str, exclude_run_id: str) -> Optional[Path]:
    """
    Find the most recent snapshot for a specific company_domain, excluding the current run_id.

    Example
    - you run test2 for goldmansachs.com
    - you want to compare it to test1 for goldmansachs.com
    This function returns snapshot_test1.json.
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


def diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two snapshots and return a diff summary.

    Output includes:
    - overall score old vs new
    - top KPI score changes
    - top KPI confidence changes
    - new vs removed citation URLs per KPI

    This is intentionally simple:
    it does not try to "understand" text, it just tracks numeric changes and evidence changes.
    """
    old_kpis = old.get("kpis", {}) or {}
    new_kpis = new.get("kpis", {}) or {}

    all_ids = sorted(set(old_kpis.keys()) | set(new_kpis.keys()))
    kpi_diffs: List[Dict[str, Any]] = []

    for kpi_id in all_ids:
        o = old_kpis.get(kpi_id)
        n = new_kpis.get(kpi_id)

        # KPI exists only in new snapshot
        if not o and n:
            kpi_diffs.append(
                {
                    "kpi_id": kpi_id,
                    "status": "added",
                    "score_delta": float(n.get("score", 0) or 0),
                    "confidence_delta": float(n.get("confidence", 0) or 0),
                    "new_citations": n.get("citation_urls", []) or [],
                    "removed_citations": [],
                }
            )
            continue

        # KPI exists only in old snapshot
        if o and not n:
            kpi_diffs.append(
                {
                    "kpi_id": kpi_id,
                    "status": "removed",
                    "score_delta": -float(o.get("score", 0) or 0),
                    "confidence_delta": -float(o.get("confidence", 0) or 0),
                    "new_citations": [],
                    "removed_citations": o.get("citation_urls", []) or [],
                }
            )
            continue

        # KPI exists in both
        old_urls = set(o.get("citation_urls", []) or [])
        new_urls = set(n.get("citation_urls", []) or [])

        score_old = float(o.get("score", 0) or 0)
        score_new = float(n.get("score", 0) or 0)
        conf_old = float(o.get("confidence", 0) or 0)
        conf_new = float(n.get("confidence", 0) or 0)

        kpi_diffs.append(
            {
                "kpi_id": kpi_id,
                "status": "changed",
                "pillar": n.get("pillar", o.get("pillar", "")),
                "score_old": score_old,
                "score_new": score_new,
                "score_delta": score_new - score_old,
                "confidence_old": conf_old,
                "confidence_new": conf_new,
                "confidence_delta": conf_new - conf_old,
                "new_citations": sorted(new_urls - old_urls),
                "removed_citations": sorted(old_urls - new_urls),
            }
        )

    # Sort by largest absolute changes, keep top 15 for easy display.
    top_score_changes = sorted(kpi_diffs, key=lambda x: abs(float(x.get("score_delta", 0) or 0)), reverse=True)[:15]
    top_conf_changes = sorted(
        kpi_diffs, key=lambda x: abs(float(x.get("confidence_delta", 0) or 0)), reverse=True
    )[:15]

    overall_old = float(old.get("overall_score", 0) or 0)
    overall_new = float(new.get("overall_score", 0) or 0)

    return {
        "old_run_id": old.get("run_id", ""),
        "new_run_id": new.get("run_id", ""),
        "old_timestamp": old.get("timestamp", ""),
        "new_timestamp": new.get("timestamp", ""),
        "company_domain": new.get("company_domain", old.get("company_domain", "")),
        "overall_old": overall_old,
        "overall_new": overall_new,
        "overall_delta": overall_new - overall_old,
        "top_score_changes": top_score_changes,
        "top_confidence_changes": top_conf_changes,
        "kpi_diffs_all": kpi_diffs,
    }