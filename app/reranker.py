"""
Cross-Encoder Reranker
======================
Takes top-k retrieval results and reranks them using a cross-encoder model
for much higher precision on the final top-n results.

Pipeline:  retrieve top-10 (bi-encoder) → rerank → top-3 (cross-encoder)

Cross-encoders score (query, document) pairs jointly, which is far more
accurate than bi-encoder cosine similarity but too slow for the full corpus.
That's why we use them as a second stage on the shortlist.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, fast, strong)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from app.debug_log import add_debug

# Lazy-load the model to avoid import cost when not needed
_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model on first use."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv(
            "VITELIS_CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug(f"[reranker] loading cross-encoder: {model_name}")

        _cross_encoder = CrossEncoder(model_name)

        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug("[reranker] cross-encoder loaded")

    return _cross_encoder


def rerank(
    query: str,
    evidences: List[Tuple[dict, str, float]],
    top_n: int = 3,
) -> List[Tuple[dict, str, float]]:
    """
    Rerank retrieved evidence using a cross-encoder.

    Args:
        query: The KPI question or search query.
        evidences: List of (metadata, document, bi_encoder_score) from retrieval.
        top_n: Number of results to return after reranking.

    Returns:
        Top-n results sorted by cross-encoder relevance score.
        The float in each tuple is now the cross-encoder score (higher = more relevant).
    """
    if not evidences:
        return []

    if len(evidences) <= top_n:
        # Nothing to rerank — already at or below top_n
        return evidences

    try:
        model = _get_cross_encoder()
    except Exception as exc:
        # Graceful fallback for environments without a compatible torch runtime.
        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug(f"[reranker] cross-encoder unavailable, using retrieval order fallback ({exc})")
        ranked = sorted(evidences, key=lambda x: x[2], reverse=True)
        return ranked[:top_n]

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, doc) for _, doc, _ in evidences]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach cross-encoder scores to evidences
    scored = [
        (meta, doc, float(ce_score))
        for (meta, doc, _bi_score), ce_score in zip(evidences, scores)
    ]

    # Sort by cross-encoder score descending
    scored.sort(key=lambda x: x[2], reverse=True)

    if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
        add_debug(
            f"[reranker] reranked {len(evidences)} → top {top_n} | "
            f"best={scored[0][2]:.3f} worst_kept={scored[min(top_n-1, len(scored)-1)][2]:.3f}"
        )

    return scored[:top_n]
