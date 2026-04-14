"""
run_eval.py — Entry point for ground-truth evaluation mode.

Usage:
    python run_eval.py --company-folder "Orange S.A"
    python run_eval.py --all

The script detects company folders automatically (any directory in the project
root that contains a _sources_export.json file).

For each company it:
  1. Loads pre-fetched sources from _sources_export.json
  2. Runs the full eval pipeline (index → score 47 KPIs → RAG eval → compare GT)
  3. Writes app/output/report_{run_id}.yaml  and  app/output/eval_{run_id}.json

At the end a summary table is printed showing matches vs unmatched per company.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (where run_eval.py lives) so provider keys are picked up.
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


def _find_company_folders(root: str = ".") -> list[str]:
    """Return folder names that contain a *_sources_export.json file."""
    import glob as _glob
    folders = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        pattern = os.path.join(entry.path, "*_sources_export.json")
        if _glob.glob(pattern):
            folders.append(entry.name)
    return folders


def _run_company(
    company_folder: str,
    max_urls: int = 0,
    kpi_limit: int = 0,
    kpi_ids: list[str] | None = None,
) -> dict:
    """Run the eval pipeline for a single company folder. Returns a summary dict."""
    from app.graphs.eval_orchestrator import build_eval_graph

    run_id = uuid.uuid4().hex[:8]

    # Derive a display name from the folder (strip trailing whitespace / dots)
    company_name = company_folder.strip().rstrip(".")

    print(f"\n{'='*60}")
    print(f"  Company : {company_name}")
    print(f"  Run ID  : {run_id}")
    print(f"{'='*60}")

    graph = build_eval_graph()
    result = graph.invoke({
        "run_id": run_id,
        "company_name": company_name,
        "company_folder": company_folder,
        "max_urls": max_urls,
        "kpi_limit": kpi_limit,
        "kpi_ids": kpi_ids or [],
    })

    summary = {
        "company": company_name,
        "run_id": run_id,
        "overall_score": result.get("overall_score", 0.0),
        "report_path": result.get("report_path", ""),
        "eval_report_path": result.get("eval_report_path", ""),
        "matched": 0,
        "unmatched_kpis": 0,
        "unmatched_gt": 0,
    }

    eval_path = result.get("eval_report_path")
    if eval_path and os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        summary["matched"] = len(eval_data.get("comparisons", []))
        summary["unmatched_kpis"] = len(eval_data.get("unmatched_kpis", []))
        summary["unmatched_gt"] = len(eval_data.get("unmatched_data_points", []))

    return summary


def _print_summary(summaries: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    header = f"{'Company':<35} {'Score':>5} {'Matched':>8} {'!KPI':>5} {'!GT':>5}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['company']:<35} "
            f"{s['overall_score']:>5.2f} "
            f"{s['matched']:>8} "
            f"{s['unmatched_kpis']:>5} "
            f"{s['unmatched_gt']:>5}"
        )
    print(f"{'='*60}")
    print("  !KPI = KPIs with no ground-truth match")
    print("  !GT  = Ground-truth points with no KPI match")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Vitelis ground-truth evaluation pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--company-folder",
        metavar="FOLDER",
        help='Company folder name, e.g. "Orange S.A"',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation for all detected company folders",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=0,
        metavar="N",
        help="Max sources to load per company from export (0 = all)",
    )
    parser.add_argument(
        "--kpi-limit",
        type=int,
        default=0,
        metavar="N",
        help="Limit number of KPIs scored per run (0 = all)",
    )
    parser.add_argument(
        "--kpi-ids",
        default="",
        metavar="ID1,ID2,...",
        help="Comma-separated KPI IDs to run (applied before --kpi-limit)",
    )
    args = parser.parse_args()
    kpi_ids = [x.strip() for x in str(args.kpi_ids or "").split(",") if x.strip()]

    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    google_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    use_gemini = provider in ("gemini", "google") or (google_key and provider != "openai")
    print(
        f"[Eval Pipeline] OPENAI_API_KEY: {'Yes' if openai_key else 'No'}  |  "
        f"Google/Gemini: {'Yes' if google_key else 'No'}  |  "
        f"KPI LLM: {'Gemini' if use_gemini and google_key else 'OpenAI' if openai_key else 'None'} "
        f"(.env: {_env_path})"
    )

    if args.all:
        folders = _find_company_folders()
        if not folders:
            print("No company folders found (looking for dirs with _sources_export.json)")
            sys.exit(1)
        print(f"Found {len(folders)} company folder(s): {', '.join(folders)}")
    else:
        folders = [args.company_folder]
        if not os.path.isdir(folders[0]):
            print(f"ERROR: Folder '{folders[0]}' not found")
            sys.exit(1)

    summaries = []
    for folder in folders:
        try:
            summary = _run_company(
                folder,
                max_urls=args.max_urls,
                kpi_limit=args.kpi_limit,
                kpi_ids=kpi_ids,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"\nERROR processing '{folder}': {exc}")
            summaries.append({
                "company": folder, "run_id": "—", "overall_score": 0,
                "report_path": "", "eval_report_path": "",
                "matched": 0, "unmatched_kpis": 0, "unmatched_gt": 0,
            })

    _print_summary(summaries)


if __name__ == "__main__":
    main()
