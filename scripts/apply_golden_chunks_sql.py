"""
Create public.golden_chunks without psql (uses DATABASE_URL + SQLAlchemy).

Run from repo root RAG-Evaluation:
    python scripts/apply_golden_chunks_sql.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _sql_statements(sql_text: str) -> list[str]:
    """Drop full-line -- comments, split on semicolons, return non-empty statements."""
    lines = [ln for ln in sql_text.splitlines() if not ln.strip().startswith("--")]
    blob = "\n".join(lines)
    parts = re.split(r";\s*", blob)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise SystemExit("DATABASE_URL is missing. Set it in RAG-Evaluation/.env")

    sql_path = root / "sql" / "golden_chunks.sql"
    if not sql_path.is_file():
        raise SystemExit(f"Not found: {sql_path}")

    statements = _sql_statements(sql_path.read_text(encoding="utf-8"))
    if not statements:
        raise SystemExit("No SQL statements found (check sql/golden_chunks.sql).")

    engine = create_engine(url)
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    print(f"Applied {len(statements)} statement(s) from {sql_path.name} — table public.golden_chunks is ready.")


if __name__ == "__main__":
    main()
