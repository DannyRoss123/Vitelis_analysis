"""
load_from_export.py
Loads pre-fetched sources from a company's _sources_export.json file.
Replaces the discover_urls + fetch_sources steps for eval mode.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List
from urllib.parse import urlparse


def _source_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def _dominant_domain(sources: List[Dict]) -> str:
    """Return the most common domain across all sources as a proxy for company_domain."""
    from collections import Counter
    domains = []
    for s in sources:
        parsed = urlparse(s.get("url", ""))
        if parsed.netloc:
            # Strip 'www.' prefix so vodafone.com and www.vodafone.com merge
            domains.append(parsed.netloc.lstrip("www."))
    if not domains:
        return "unknown"
    return Counter(domains).most_common(1)[0][0]


def load_from_export_node(state: Dict) -> Dict:
    """
    Reads {company_folder}/{company_folder}_sources_export.json and converts
    each entry into a SourceDoc-compatible dict for index_sources_node.

    State keys read:  company_folder
    State keys set:   sources, target_urls, url_count, company_domain
    """
    company_folder: str = state["company_folder"]

    # Locate the export file — search folder for any *_sources_export.json
    # (handles naming variants like "Orange S.A._sources_export.json")
    import glob as _glob
    matches = _glob.glob(os.path.join(company_folder, "*_sources_export.json"))
    if not matches:
        raise FileNotFoundError(
            f"No *_sources_export.json found in folder '{company_folder}'"
        )
    export_path = matches[0]

    with open(export_path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    sources: List[Dict] = []
    seen_ids: set = set()

    max_urls_raw = state.get("max_urls", 0)
    try:
        max_urls = int(max_urls_raw or 0)
    except (TypeError, ValueError):
        max_urls = 0

    for entry in raw_entries:
        url = (entry.get("url") or "").strip()
        content = (entry.get("content") or "").strip()

        if not url or not content:
            continue

        sid = _source_id(url)
        if sid in seen_ids:
            continue
        seen_ids.add(sid)

        parsed = urlparse(url)
        domain = parsed.netloc.lstrip("www.")
        tier = int(entry.get("tier", 2))
        raw_ts = entry.get("created_at") or entry.get("date") or ""
        # Normalise to timezone-aware ISO format to match what kpi_scoring expects
        if raw_ts and "+" not in raw_ts and raw_ts.upper()[-1] != "Z":
            raw_ts = raw_ts.split(".")[0] + "+00:00"
        retrieved_at = raw_ts

        sources.append({
            "source_id": sid,
            "url": url,
            "title": domain,          # use domain as title (no page title in export)
            "text": content,          # 'content' in export → 'text' in SourceDoc
            "domain": domain,
            "retrieved_at": retrieved_at,
            "tier": tier,
        })
        if max_urls > 0 and len(sources) >= max_urls:
            break

    target_urls = [s["url"] for s in sources]
    company_domain = _dominant_domain(sources)

    print(f"[load_from_export] Loaded {len(sources)} sources for '{company_folder}'")

    return {
        "sources": sources,
        "target_urls": target_urls,
        "url_count": len(sources),
        "company_domain": company_domain,
    }
