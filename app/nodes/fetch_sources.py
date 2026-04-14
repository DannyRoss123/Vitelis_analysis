from __future__ import annotations

from typing import Dict, List
from urllib.parse import urlparse

from app.models import SourceDoc
from app.source_eval import classify_tier_content
from app.web_fetcher import build_source_record, fetch_page


def _page_type(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else "home"


def fetch_sources_node(state: Dict) -> Dict:
    urls: List[str] = state.get("target_urls", [])

    sources: List[SourceDoc] = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        page = fetch_page(url)
        if not page:
            continue
        text = page["text"].strip()
        if not text:
            continue

        # v2: Content-based tier classification
        title = page.get("title", "")
        tier_info = classify_tier_content(url, text, title)
        tier = tier_info["tier"]

        record = build_source_record(url, title, text, tier)
        source = SourceDoc(
            source_id=record["source_id"],
            url=record["url"],
            title=record["title"],
            text=record["text"],
            domain=record["domain"],
            retrieved_at=record["retrieved_at"],
            tier=record["tier"],
            page_type=_page_type(url),
            # v2: Store content analysis metadata
            tier_reason=tier_info["tier_reason"],
            content_score=tier_info["content_score"],
            content_signals=tier_info.get("content_signals"),
        )
        sources.append(source)
        # Allow a larger batch when URLs come from a curated export (e.g., Orange S.A data file).
        # Cap at 70 sources to keep runtime reasonable.
        if len(sources) >= 70:
            break

    return {
        "sources": [source.model_dump() for source in sources],
    }
