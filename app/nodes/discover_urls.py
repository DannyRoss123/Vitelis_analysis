from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import json

from app.url_discovery import discover_urls


def _urls_from_export(company_name: str, max_urls: int | None = None) -> List[str]:
    """
    If a pre-collected sources export exists for this company, load URLs from it
    instead of discovering new ones.

    Expected file pattern (under repo root /data):
        "<company_name>_sources_export.json"
        "<company_name>._sources_export.json"
    e.g. "Orange S.A._sources_export.json"
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"
        if not data_dir.exists():
            return []

        urls: List[str] = []

        # Prefer files whose name starts with the company name and ends with _sources_export.json
        for path in data_dir.glob("*_sources_export.json"):
            if not path.name.startswith(company_name):
                continue
            with path.open("r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if not url:
                    continue
                urls.append(url)
                if max_urls is not None and len(urls) >= max_urls:
                    break
            break  # use first matching file only

        return urls
    except Exception:
        # On any error, fall back to normal discovery
        return []


def discover_urls_node(state: Dict) -> Dict:
    company_name = state["company_name"]
    company_domain = state["company_domain"]
    # 0 = use all URLs from export; otherwise cap at N (default 80).
    max_urls = state.get("max_urls", 80)
    cap_export = None if max_urls == 0 else max_urls
    cap_live = 100 if max_urls == 0 else max_urls

    # First, try to load URLs from a pre-exported sources file (if available).
    urls = _urls_from_export(company_name=company_name, max_urls=cap_export)

    # If no export exists, fall back to live discovery.
    if not urls:
        urls = discover_urls(company_name=company_name, domain=company_domain, max_urls=cap_live)

    return {
        "target_urls": urls,
        "url_count": len(urls),
    }
