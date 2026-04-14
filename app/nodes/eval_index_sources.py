from __future__ import annotations

import os
from typing import Dict, List

from app.vectorstore import build_collection, chunk_text
from app.debug_log import add_debug

# Batch size for ChromaDB adds — keep small to manage OpenAI embedding API calls
_BATCH_SIZE = 100


def eval_index_sources_node(state: Dict) -> Dict:
    """Index sources into ChromaDB with semantic chunking + OpenAI embeddings.

    Uses batched adds to avoid ChromaDB batch-size limits and to manage
    OpenAI embedding API rate limits when sources contain very long content.
    """
    run_id = state["run_id"]
    sources: List[Dict] = state.get("sources", [])

    collection = build_collection(run_id)

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []

    for source in sources:
        chunks = chunk_text(source["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source['source_id']}::chunk_{idx}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "title": source.get("title", ""),
                    "domain": source.get("domain", ""),
                    "retrieved_at": source.get("retrieved_at", ""),
                    "tier": source["tier"],
                }
            )

    for start in range(0, len(ids), _BATCH_SIZE):
        end = start + _BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug(f"[eval_index] indexed chunks {start}–{min(end, len(ids))} of {len(ids)}")

    return {"collection_id": f"collection_{run_id}"}
