"""
load_ground_truth.py
Loads analyst research findings from a company's _raw_data_points.json file.
"""
from __future__ import annotations

import json
import os
import re
from typing import List


def _parse_sources(raw: str) -> List[str]:
    """Extract URLs from the various formats used in sources fields."""
    if not raw:
        return []
    return [u.rstrip(".,;)") for u in re.findall(r"https?://[^\s,\n\"'<>]+", raw)]


def load_ground_truth(company_folder: str) -> List[dict]:
    """
    Reads {company_folder}/{company_folder}_raw_data_points.json.

    Returns a list of dicts, each with:
        name, answer, explanation, sources (list of URLs), definition, output_variable
    """
    import glob as _glob
    matches = _glob.glob(os.path.join(company_folder, "*_raw_data_points.json"))
    if not matches:
        raise FileNotFoundError(
            f"No *_raw_data_points.json found in folder '{company_folder}'"
        )
    path = matches[0]

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    points = []
    for entry in raw:
        name = (entry.get("name") or "").strip()
        if not name:
            continue

        answer = str(entry.get("answer") or "").strip()
        explanation = str(entry.get("explanation") or "").strip()
        definition = str(entry.get("definition") or "").strip()
        output_variable = str(entry.get("output_variable") or "").strip()

        raw_sources = entry.get("sources") or ""
        source_urls = _parse_sources(str(raw_sources))

        points.append({
            "name": name,
            "answer": answer,
            "explanation": explanation,
            "sources": source_urls,
            "definition": definition,
            "output_variable": output_variable,
        })

    return points
