from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv
from app.graphs.orchestrator import build_orchestrator_graph

# Load .env from project root (where run.py lives) so it works regardless of cwd
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


def main() -> None:
    import os
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    google_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    use_gemini = provider in ("gemini", "google") or (google_key and provider != "openai")
    print(f"[Pipeline] OPENAI_API_KEY: {'Yes' if openai_key else 'No'}  |  Google/Gemini: {'Yes' if google_key else 'No'}  |  KPI LLM: {'Gemini' if use_gemini and google_key else 'OpenAI' if openai_key else 'None'} (.env: {_env_path})")
    parser = argparse.ArgumentParser(description="Run Vitelis v0.2 pipeline")
    parser.add_argument("--company-name", required=True, help="Company name")
    parser.add_argument("--company-domain", required=True, help="Company domain (e.g., vodafone.com)")
    parser.add_argument("--run-id", default=str(uuid.uuid4())[:8], help="Optional run id")
    parser.add_argument("--max-urls", type=int, default=80, metavar="N", help="Max URLs to use (0 = all in export file; default 80)")
    args = parser.parse_args()

    graph = build_orchestrator_graph()
    state = graph.invoke(
        {
            "run_id": args.run_id,
            "company_name": args.company_name,
            "company_domain": args.company_domain,
            "max_urls": args.max_urls,
        }
    )

    print("Run complete")
    print(f"Report: {state.get('report_path')}")
    print(f"Overall score: {state.get('overall_score')}")


if __name__ == "__main__":
    main()
