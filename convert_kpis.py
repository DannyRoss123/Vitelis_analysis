"""
convert_kpis.py — one-time utility
Reads the AlixPartners KPI CSV and writes app/kpis_47.yaml.

Usage:
    python convert_kpis.py

Output:
    app/kpis_47.yaml   (47 rubric KPIs ready for the eval pipeline)
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd
import yaml


CSV_PATH = "14.11.2025 AlixPartners_Model_Master_V5.1.xlsx - KPI Drivers & Quality Criteria.csv"
OUTPUT_PATH = "app/kpis_47.yaml"

# Pillar normalisation — map raw CSV category names to consistent pillar labels
PILLAR_MAP = {
    "Strategy": "Strategy",
    "Execution": "Execution",
    "Technical": "Technical",
    "Organizational": "Organizational",
    "Innovation": "Innovation",
}


def _clean(value) -> str:
    """Return a clean string or empty string for NaN / None."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_gt_sources(raw: str) -> list[str]:
    """Extract URLs from the comma-separated or newline-separated sources string."""
    if not raw:
        return []
    urls = re.findall(r"https?://[^\s,\n\"']+", raw)
    return [u.rstrip(".,;)") for u in urls]


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at '{CSV_PATH}'")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, header=0, dtype=str)

    print(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {repr(col)}")

    # ------------------------------------------------------------------ #
    # Column layout (0-indexed, confirmed by live CSV inspection):       #
    #  0  #                  row number (= KPI number)                   #
    #  2  Status             filter to "confirmed" rows only             #
    #  3  KPI Category       pillar                                       #
    #  4  Definition (KPI)   evidence_requirements                        #
    #  6  KPI Drivers as of 2.10.2025   name                             #
    # 13  KPI Drivers        the quality-criteria QUESTION               #
    # 14  Quality Criteria (1=Low)       rubric level 1                  #
    # 15  Quality Criteria (2=Low-Medium) rubric level 2                 #
    # 16  Quality Criteria (3=Medium)    rubric level 3                  #
    # 17  Quality Criteria (4=Medium-High) rubric level 4                #
    # 18  Quality Criteria (5=High)      rubric level 5                  #
    # 19  Unnamed: 19        unused trailing column                      #
    # ------------------------------------------------------------------ #
    cols = list(df.columns)
    n_cols = len(cols)

    kpis = []

    for _, row in df.iterrows():
        row_num = _clean(row.iloc[0])
        if not row_num or not row_num.isdigit():
            continue  # skip header-repeat or blank rows

        # Only include confirmed KPIs
        status = _clean(row.iloc[2]).lower() if n_cols > 2 else ""
        if status != "confirmed":
            continue

        kpi_number = int(row_num)

        # Name: prefer "KPI Drivers as of 2.10.2025" (col 6), fallback col 13
        name = _clean(row.iloc[6]) if n_cols > 6 else ""
        if not name and n_cols > 13:
            name = _clean(row.iloc[13])
        if not name:
            name = f"KPI {kpi_number}"

        # Pillar
        raw_pillar = _clean(row.iloc[3]) if n_cols > 3 else ""
        pillar = PILLAR_MAP.get(raw_pillar, raw_pillar or "Unknown")

        # Evidence requirements
        evidence_requirements = _clean(row.iloc[4]) if n_cols > 4 else ""

        # Question — col 13 ("KPI Drivers " column holds the assessment question)
        question = _clean(row.iloc[13]) if n_cols > 13 else ""
        if not question:
            # Fallback: use name as question
            question = f"What is the company's maturity level for: {name}?"

        # Rubric levels — cols 14-18 (Quality Criteria 1=Low through 5=High)
        rubric_raw = []
        for col_idx in range(14, min(19, n_cols)):
            val = _clean(row.iloc[col_idx])
            if val:
                rubric_raw.append(val)

        # Build rubric list in the format the scoring engine expects:
        # ["1: <level-1 text>", "2: <level-2 text>", ..., "5: <level-5 text>"]
        rubric: list[str] = []
        for level, text in enumerate(rubric_raw[:5], start=1):
            rubric.append(f"{level}: {text}")

        # If we only got fewer than 5 levels, pad with placeholders so the LLM
        # always has a full 1-5 rubric to reason against.
        while len(rubric) < 5:
            rubric.append(f"{len(rubric) + 1}: (no criteria defined)")

        kpi_dict = {
            "kpi_id": f"kpi_{kpi_number}",
            "name": name,
            "pillar": pillar,
            "type": "rubric",
            "question": question,
            "rubric": rubric,
            "evidence_requirements": evidence_requirements or None,
        }
        kpis.append(kpi_dict)

    print(f"\nConverted {len(kpis)} KPIs")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(kpis, f, allow_unicode=True, sort_keys=False)

    print(f"Written to {OUTPUT_PATH}")

    # Quick sanity check
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    print(f"Verified: {len(loaded)} KPIs in output file")
    print("\nFirst KPI preview:")
    first = loaded[0]
    print(f"  id:      {first['kpi_id']}")
    print(f"  name:    {first['name']}")
    print(f"  pillar:  {first['pillar']}")
    print(f"  question:{first['question'][:80]}...")
    print(f"  rubric:  {len(first['rubric'])} levels")


if __name__ == "__main__":
    main()
