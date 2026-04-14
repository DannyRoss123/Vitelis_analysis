from __future__ import annotations

from typing import List
import csv
import os
import re
from pathlib import Path

import yaml

from app.models import KPIDefinition


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _data_folder() -> Path | None:
    """
    Resolve the Data folder (no hardcoded filenames).

    Lookup order:
      1. KPI_CSV_PATH env var: if it points to a directory, use it; if to a file, use its parent.
      2. Fixed path: <repo>/../Data (e.g. Team/Data when repo is Team/0304/RAG-Evaluation).
      3. Search for a folder named "Data" under repo root, then repo's parent, then grandparent.
    """
    env_path = os.getenv("KPI_CSV_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p.parent if p.is_file() else p
        return None

    repo_root = Path(__file__).resolve().parents[2]
    default = repo_root.parent / "Data"
    if default.exists() and default.is_dir():
        return default

    for base in (repo_root, repo_root.parent, repo_root.parent.parent):
        if not base.exists() or not base.is_dir():
            continue
        for path in base.iterdir():
            if path.is_dir() and path.name == "Data":
                return path
    return None


def _pick_csv_from_folder(folder: Path) -> Path | None:
    """Pick a single CSV from the folder. Prefer one with 'KPI' in the name; else use the first by name."""
    csvs = sorted(folder.glob("*.csv"), key=lambda p: p.name.lower())
    if not csvs:
        return None
    for p in csvs:
        if "kpi" in p.name.lower():
            return p
    return csvs[0]


def _default_kpi_csv_path() -> Path | None:
    """Resolve the KPI catalog CSV by picking a CSV from the Data folder (no hardcoded filename)."""
    data_dir = _data_folder()
    if not data_dir:
        return None
    return _pick_csv_from_folder(data_dir)


def _load_kpis_from_csv() -> List[KPIDefinition]:
    """
    Load KPI definitions directly from the AlixPartners KPI Drivers & Quality Criteria CSV.

    Each row becomes a rubric KPI:
      - kpi_id: slug from column N (KPI Drivers) with row-index suffix for uniqueness
      - name:  'Definition (KPI)' (fallback to question)
      - pillar: 'KPI Category'
      - type:  'rubric'
      - question: latest KPI driver text (KPI Drivers as of 2.10.2025, then fallbacks)
      - rubric: 5 quality-level descriptions (1–5) from the Quality Criteria columns
    """
    path = _default_kpi_csv_path()
    if not path:
        return []

    kpis: List[KPIDefinition] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            status = (row.get("Status") or "").strip().lower()
            if status and status != "confirmed":
                # Only keep confirmed KPIs
                continue

            # Choose the most up-to-date driver text as the KPI question
            question_fields = [
                "KPI Drivers as of 2.10.2025",
                "KPI Drivers",
                "KPI Drivers ",
                "KPI Drivers - Initial input (Alix Partners)",
                "Definition (KPI)",
            ]
            question = ""
            for field in question_fields:
                if field in row and row[field]:
                    candidate = row[field].strip()
                    if candidate:
                        question = candidate
                        break

            if not question:
                # Skip rows without a usable driver/question
                continue

            name = (row.get("Definition (KPI)") or question).strip()
            pillar = (row.get("KPI Category") or "Uncategorized").strip()

            # Column N (KPI Drivers / KPI Drivers ) has unique values per row — use for unique kpi_id
            driver_col_n = (row.get("KPI Drivers ") or row.get("KPI Drivers") or "").strip()
            base_id = driver_col_n or question
            slug = _slugify(base_id)
            # Append row index so every row gets a unique id even if slug would collide
            kpi_id = f"{slug}_{idx}" if slug else f"kpi_{idx}"

            # Build rubric from quality criteria columns (1–5)
            rubric_texts = []
            quality_cols = [
                ("1", "Quality Criteria (1=Low)"),
                ("2", "Quality Criteria (2=Low-Medium)"),
                ("3", "Quality Criteria (3=Medium)"),
                ("4", "Quality Criteria (4=Medium-High)"),
                ("5", "Quality Criteria (5=High)"),
            ]
            for score_label, col_name in quality_cols:
                text = (row.get(col_name) or "").strip()
                if text:
                    rubric_texts.append(f"{score_label}: {text}")

            # If no rubric text is present, still include the KPI as rubric with an empty list
            rubric = rubric_texts or None

            kpis.append(
                KPIDefinition(
                    kpi_id=kpi_id,
                    name=name,
                    pillar=pillar,
                    type="rubric",
                    question=question,
                    rubric=rubric,
                    evidence_requirements=None,
                )
            )

    return kpis


def load_kpi_catalog(path: str | None = None) -> List[KPIDefinition]:
    """
    Load KPI definitions for scoring.

    Priority:
      1. If KPI_CSV_PATH (or the default AlixPartners CSV) exists, load KPIs from it.
      2. Otherwise, fall back to the legacy YAML catalog (app/kpis.yaml or custom path).
    """
    # Preferred source: the AlixPartners KPI Drivers & Quality Criteria CSV
    csv_kpis = _load_kpis_from_csv()
    if csv_kpis:
        return csv_kpis

    # Fallback: original YAML-based catalog
    if path is None:
        path = "app/kpis.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    return [KPIDefinition(**item) for item in data]
