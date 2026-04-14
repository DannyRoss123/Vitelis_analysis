"""
Tier-based source quality assessment for confidence scoring.

In the v2 pipeline, tier is NO LONGER used at retrieval time.
Retrieval is purely semantic (OpenAI embeddings + cross-encoder reranking).

Tier is now used exclusively at CONFIDENCE time to assess evidence quality:
- Tier 1 sources (investor reports, whitepapers) boost confidence
- Tier 3 sources (thin content) provide no boost
- This is applied via calculate_tier_quality() in the confidence calculation
"""
import os
from typing import List, Tuple, Dict, Any


def calculate_tier_quality(evidences: List[Tuple[dict, str, float]]) -> float:
    """
    Calculate confidence boost based on source tier quality.

    Args:
        evidences: List of (metadata, document, score) tuples

    Returns:
        Confidence boost value (0.0 to 0.15)
    """
    if not evidences:
        return 0.0

    max_boost = float(os.getenv("VITELIS_TIER_BOOST_MAX", "0.15"))

    # Calculate average tier (lower tier number = higher quality)
    tiers = [meta.get("tier", 3) for meta, _, _ in evidences]
    avg_tier = sum(tiers) / len(tiers)

    # Convert to boost: tier 1 avg = max boost, tier 3 avg = 0 boost
    # Formula: max_boost - (avg_tier - 1) * (max_boost / 2)
    tier_boost = max(0.0, max_boost - (avg_tier - 1.0) * (max_boost / 2.0))

    return round(tier_boost, 3)


def get_tier_distribution(evidences: List[Tuple[dict, str, float]]) -> Dict[str, Any]:
    """
    Get distribution statistics of tiers in evidence set.

    Args:
        evidences: List of (metadata, document, score) tuples

    Returns:
        Dictionary with tier counts and average
    """
    if not evidences:
        return {"tier1": 0, "tier2": 0, "tier3": 0, "avg": 0.0}

    tiers = [meta.get("tier", 3) for meta, _, _ in evidences]

    tier_counts = {
        "tier1": tiers.count(1),
        "tier2": tiers.count(2),
        "tier3": tiers.count(3),
    }

    tier_counts["avg"] = round(sum(tiers) / len(tiers), 2)

    return tier_counts
