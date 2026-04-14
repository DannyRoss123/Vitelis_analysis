from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from app.graphs.eval_orchestrator_db import build_eval_db_graph


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(
        description="Run DB-backed RAG evaluation pipeline",
        epilog="Golden retrieval labels: PostgreSQL table golden_chunks (kpi_id, chunk_id, relevance) "
        "by default; override with GOLDEN_CHUNKS_TABLE / *_COLUMN env vars.",
    )
    parser.add_argument(
        "--db-run-id",
        required=True,
        help="Run key: integer runs.id (resolved to runs.request_id for KPI/sources) or UUID/string",
    )
    parser.add_argument("--company-name", required=True, help="Exact company name in DB")
    args = parser.parse_args()

    if not (os.getenv("DATABASE_URL") or "").strip():
        raise RuntimeError("DATABASE_URL is required.")

    run_id = uuid.uuid4().hex[:8]
    graph = build_eval_db_graph()
    result = graph.invoke(
        {
            "run_id": run_id,
            "db_run_id": str(args.db_run_id),
            "company_name": args.company_name,
        }
    )

    print("\n=== DB Evaluation Completed ===")
    print(f"Pipeline run_id: {run_id}")
    print(f"Report: {result.get('report_path', '')}")
    print(f"Eval JSON: {result.get('eval_report_path', '')}")
    print(f"Overall score: {result.get('overall_score', 0)}")


if __name__ == "__main__":
    main()

