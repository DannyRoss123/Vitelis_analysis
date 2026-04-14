"""
Simple test script to verify enhanced source evaluation features.
"""

import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.tier_weighting import (
    calculate_tier_quality,
    get_tier_distribution,
)
from app.corroboration import (
    detect_corroboration,
    calculate_source_diversity,
    find_keyword_overlap,
)
from app.dynamic_retrieval import determine_optimal_k
from app.models import KPIDefinition


def test_tier_weighting():
    """Test tier weighting functionality."""
    print("\n=== Testing Tier Weighting ===")

    # Test tier quality calculation
    evidences_with_score = [
        ({"source_id": "src1", "tier": 1}, "doc1", 0.8),
        ({"source_id": "src2", "tier": 1}, "doc2", 0.7),
        ({"source_id": "src3", "tier": 3}, "doc3", 0.5),
    ]

    tier_dist = get_tier_distribution(evidences_with_score)
    print(f"Tier distribution: {tier_dist}")
    assert tier_dist["tier1"] == 2
    assert tier_dist["tier3"] == 1
    print("[OK] Tier distribution calculated correctly")

    tier_quality = calculate_tier_quality(evidences_with_score)
    print(f"Tier quality boost: {tier_quality}")
    assert tier_quality > 0.0  # Should have positive boost for tier 1 sources
    print("[OK] Tier quality boost calculated correctly")


def test_corroboration():
    """Test cross-source corroboration detection."""
    print("\n=== Testing Corroboration ===")

    # Test with evidence from multiple sources sharing keywords
    multi_source_evidences = [
        ({"source_id": "src1"}, "artificial intelligence machine learning automation", 0.8),
        ({"source_id": "src2"}, "machine learning models and artificial intelligence", 0.7),
        ({"source_id": "src3"}, "deep learning and automation systems", 0.6),
    ]

    unique_sources = calculate_source_diversity(multi_source_evidences)
    print(f"Unique sources: {unique_sources}")
    assert unique_sources == 3
    print("[OK] Source diversity calculated correctly")

    shared_keywords = find_keyword_overlap(multi_source_evidences, min_sources=2)
    print(f"Shared keywords: {shared_keywords}")
    assert len(shared_keywords) > 0  # Should find some shared keywords
    print("[OK] Shared keywords detected")

    corroboration_score = detect_corroboration(multi_source_evidences, min_sources=2)
    print(f"Corroboration score: {corroboration_score}")
    assert corroboration_score > 0.0  # Should detect corroboration
    print("[OK] Corroboration detected")

    # Test with evidence from single source (no corroboration)
    single_source_evidences = [
        ({"source_id": "src1"}, "artificial intelligence", 0.8),
        ({"source_id": "src1"}, "machine learning", 0.7),
    ]

    corroboration_score_single = detect_corroboration(single_source_evidences, min_sources=2)
    print(f"Single source corroboration score: {corroboration_score_single}")
    assert corroboration_score_single == 0.0  # No corroboration from single source
    print("[OK] No corroboration for single source")


def test_dynamic_k():
    """Test dynamic k-parameter tuning."""
    print("\n=== Testing Dynamic K ===")

    # Test rubric KPI
    rubric_kpi = KPIDefinition(
        kpi_id="test_rubric",
        pillar="Strategy",
        type="rubric",
        name="Test Rubric KPI",
        question="How well does the company perform?",
        rubric=["1: Poor", "2: Fair", "3: Good", "4: Very Good", "5: Excellent"] * 20,  # Complex rubric
    )

    k_rubric = determine_optimal_k(rubric_kpi, default_k=6)
    print(f"K for complex rubric KPI: {k_rubric}")

    # Test quant KPI
    quant_kpi = KPIDefinition(
        kpi_id="test_quant",
        pillar="Product",
        type="quant",
        name="Test Quant KPI",
        question="Count mentions of AI features",
        rubric=[],
    )

    k_quant = determine_optimal_k(quant_kpi, default_k=6)
    print(f"K for quant KPI: {k_quant}")

    # Dynamic k should only work when enabled
    if os.getenv("VITELIS_ENABLE_DYNAMIC_K", "false").lower() in {"1", "true"}:
        assert k_rubric != k_quant or k_rubric != 6  # At least one should be different
        print("[OK] Dynamic k tuning active")
    else:
        assert k_rubric == 6 and k_quant == 6
        print("[OK] Dynamic k disabled, using default")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Enhanced Source Evaluation Features")
    print("=" * 60)

    try:
        test_tier_weighting()
        test_corroboration()
        test_dynamic_k()

        print("\n" + "=" * 60)
        print("[OK] All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
