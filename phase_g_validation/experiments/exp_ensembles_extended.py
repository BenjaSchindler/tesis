#!/usr/bin/env python3
"""
Extended Ensemble Experiments - Categories 1-5

Tests 55 ensemble configurations across 5 categories:
- Category 1: Weighted Top-K (12 tests)
- Category 2: Diversity-Maximizing (10 tests)
- Category 3: Hybrid Strategy (15 tests)
- Category 4: Deduplication-Based (8 tests)
- Category 5: Class-Targeted (10 tests - excluding 2 MLP configs for now)

Total: 55 tests (~12-14 hours with 4 parallel configs)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from validation_runner import load_data, EmbeddingCache
from config_definitions import ENSEMBLES
from base_config import RESULTS_DIR

# Import ensemble validation function
from exp11_ensemble_validation import run_ensemble_validation

# Import diversity selector
try:
    from diversity_selector import (
        load_all_configs,
        select_diverse_configs,
        select_class_balanced_configs,
        select_strategy_diverse_configs
    )
    DIVERSITY_AVAILABLE = True
except ImportError:
    DIVERSITY_AVAILABLE = False
    print("WARNING: diversity_selector.py not found, diversity features disabled")


def get_category_ensembles(category: int):
    """Get ensemble names for a specific category."""

    if category == 1:
        # Weighted Top-K (12 tests)
        return [
            "WGT_Top3_equal", "WGT_Top3_perf", "WGT_Top3_exp",
            "WGT_Top5_equal", "WGT_Top5_perf", "WGT_Top5_rank",
            "WGT_Top7_equal", "WGT_Top7_perf",
            "WGT_Top10_equal", "WGT_Top10_perf",
            "WGT_Prompting_only", "WGT_Temperature_only"
        ]

    elif category == 2:
        # Diversity-Maximizing (10 tests)
        return [
            "DIV_k3_maxdist", "DIV_k5_maxdist", "DIV_k7_maxdist",
            "DIV_k5_class_balanced", "DIV_prompting_temp_vol",
            "DIV_wave_diverse", "DIV_ENTJ_ISFJ_focus",
            "DIV_rare_vs_common", "DIV_orthogonal",
            "DIV_complementary_pairs"
        ]

    elif category == 3:
        # Hybrid Strategy (15 tests)
        return [
            "HYB_prompt_temp", "HYB_prompt_vol", "HYB_prompt_filter",
            "HYB_temp_vol", "HYB_temp_filter", "HYB_vol_filter",
            "HYB_all_strategies", "HYB_manyshot_hightemp",
            "HYB_manyshot_yolo", "HYB_manyshot_ultra",
            "HYB_triple_prompting", "HYB_triple_temp",
            "HYB_conservative", "HYB_aggressive", "HYB_balanced"
        ]

    elif category == 4:
        # Deduplication-Based (8 tests)
        return [
            "DEDUP_Top5_sim095", "DEDUP_Top5_sim098",
            "DEDUP_Top5_classwise", "DEDUP_Top5_cluster",
            "DEDUP_WaveChampions_sim095", "DEDUP_diverse_k7",
            "DEDUP_hybrid_all", "DEDUP_aggressive"
        ]

    elif category == 5:
        # Class-Targeted (10 tests - excluding MLP configs for now)
        return [
            "TOP_ENTJ_focus", "TOP_ISTJ_focus", "TOP_ISFJ_focus",
            "TOP_all_common", "BAL_16class_coverage",
            "BAL_weighted_need", "BAL_equal_improvement",
            "BAL_rare_plus_common",
            "NOVEL_boosted_rare", "NOVEL_temperature_ladder"  # Add 2 from Cat 7
        ]

    else:
        return []


def populate_diversity_ensemble(ensemble_name: str, cache: EmbeddingCache):
    """
    Populate dynamic diversity-based ensemble components.

    Uses diversity_selector.py to choose components based on per-class performance.
    """
    if not DIVERSITY_AVAILABLE:
        print(f"  WARNING: Cannot populate {ensemble_name} - diversity_selector unavailable")
        return None

    ensemble_info = ENSEMBLES[ensemble_name]
    strategy = ensemble_info.get("strategy", None)

    if not strategy:
        # Already has static components
        return ensemble_info

    # Load all configs for selection
    all_configs = load_all_configs()

    # Select components based on strategy
    if strategy == "diversity":
        k = ensemble_info.get("k", 5)
        metric = ensemble_info.get("metric", "euclidean")
        selected = select_diverse_configs(all_configs, k=k, metric=metric)
        components = [cfg["config"] for cfg in selected]

    elif strategy == "class_balanced":
        k = ensemble_info.get("k", 5)
        selected = select_class_balanced_configs(all_configs, k=k)
        components = [cfg["config"] for cfg in selected]

    elif strategy == "strategy_diverse":
        strategy_types = ensemble_info.get("strategy_types", {})
        selected = select_strategy_diverse_configs(all_configs, strategy_types)
        components = [cfg["config"] for cfg in selected]

    else:
        print(f"  WARNING: Unknown strategy '{strategy}' for {ensemble_name}")
        return None

    # Update ensemble_info with selected components
    ensemble_info_copy = ensemble_info.copy()
    ensemble_info_copy["components"] = components

    print(f"  {ensemble_name}: Selected {len(components)} components via '{strategy}'")
    print(f"    Components: {components}")

    return ensemble_info_copy


def run_weighted_ensemble(ensemble_name: str, cache: EmbeddingCache, verbose: bool = True):
    """Run weighted ensemble (Category 1)."""
    ensemble_info = ENSEMBLES[ensemble_name]
    components = ensemble_info["components"]
    weights = ensemble_info.get("weights", None)

    return run_ensemble_validation(
        ensemble_name,
        cache,
        verbose=verbose,
        weights=weights,
        dedup_method=None,
        dedup_params=None
    )


def run_diversity_ensemble(ensemble_name: str, cache: EmbeddingCache, verbose: bool = True):
    """Run diversity-maximizing ensemble (Category 2)."""

    # Populate components dynamically
    populated_info = populate_diversity_ensemble(ensemble_name, cache)

    if populated_info is None:
        print(f"  ERROR: Failed to populate {ensemble_name}")
        return None

    # Temporarily update ENSEMBLES dict
    original_info = ENSEMBLES[ensemble_name].copy()
    ENSEMBLES[ensemble_name] = populated_info

    try:
        result = run_ensemble_validation(
            ensemble_name,
            cache,
            verbose=verbose,
            weights=None,  # No weighting for diversity ensembles
            dedup_method=None,
            dedup_params=None
        )
    finally:
        # Restore original
        ENSEMBLES[ensemble_name] = original_info

    return result


def run_hybrid_ensemble(ensemble_name: str, cache: EmbeddingCache, verbose: bool = True):
    """Run hybrid strategy ensemble (Category 3)."""
    return run_ensemble_validation(
        ensemble_name,
        cache,
        verbose=verbose,
        weights=None,  # No weighting unless specified
        dedup_method=None,
        dedup_params=None
    )


def run_dedup_ensemble(ensemble_name: str, cache: EmbeddingCache, verbose: bool = True):
    """Run deduplication-based ensemble (Category 4)."""
    ensemble_info = ENSEMBLES[ensemble_name]

    # Check if this is a diversity ensemble that needs population
    if ensemble_info.get("strategy"):
        populated_info = populate_diversity_ensemble(ensemble_name, cache)
        if populated_info:
            components = populated_info["components"]
            ENSEMBLES[ensemble_name]["components"] = components

    dedup_method = ensemble_info.get("dedup_method", None)
    dedup_params = ensemble_info.get("dedup_params", {})

    return run_ensemble_validation(
        ensemble_name,
        cache,
        verbose=verbose,
        weights=None,
        dedup_method=dedup_method,
        dedup_params=dedup_params
    )


def run_class_targeted_ensemble(ensemble_name: str, cache: EmbeddingCache, verbose: bool = True):
    """Run class-targeted ensemble (Category 5)."""
    ensemble_info = ENSEMBLES[ensemble_name]

    # Check if this is a diversity ensemble that needs population
    if ensemble_info.get("strategy"):
        populated_info = populate_diversity_ensemble(ensemble_name, cache)
        if populated_info:
            components = populated_info["components"]
            ENSEMBLES[ensemble_name]["components"] = components

    # Note: MLP classifier support would require modifying KFoldEvaluator
    # For now, these ensembles use default LogisticRegression

    return run_ensemble_validation(
        ensemble_name,
        cache,
        verbose=verbose,
        weights=None,
        dedup_method=None,
        dedup_params=None
    )


def run_category(category: int, cache: EmbeddingCache):
    """Run all ensembles for a specific category."""

    category_names = {
        1: "Weighted Top-K",
        2: "Diversity-Maximizing",
        3: "Hybrid Strategy",
        4: "Deduplication-Based",
        5: "Class-Targeted"
    }

    print(f"\n{'='*70}")
    print(f"CATEGORY {category}: {category_names[category]}")
    print(f"{'='*70}")

    ensemble_names = get_category_ensembles(category)
    print(f"\nEnsembles to test: {len(ensemble_names)}")
    for i, name in enumerate(ensemble_names, 1):
        print(f"  {i}. {name}")

    results = {}

    for ensemble_name in ensemble_names:
        try:
            if category == 1:
                result = run_weighted_ensemble(ensemble_name, cache, verbose=True)
            elif category == 2:
                result = run_diversity_ensemble(ensemble_name, cache, verbose=True)
            elif category == 3:
                result = run_hybrid_ensemble(ensemble_name, cache, verbose=True)
            elif category == 4:
                result = run_dedup_ensemble(ensemble_name, cache, verbose=True)
            elif category == 5:
                result = run_class_targeted_ensemble(ensemble_name, cache, verbose=True)
            else:
                continue

            if result:
                results[ensemble_name] = result

        except Exception as e:
            print(f"\n  ERROR running {ensemble_name}: {e}")
            import traceback
            traceback.print_exc()

    # Category summary
    print(f"\n{'='*70}")
    print(f"CATEGORY {category} SUMMARY - {category_names[category]}")
    print(f"{'='*70}")

    if results:
        # Sort by delta_pct
        sorted_results = sorted(results.items(), key=lambda x: x[1].delta_pct, reverse=True)

        for ensemble_name, result in sorted_results:
            sig = "✓" if result.significant else "✗"
            print(f"  {ensemble_name:30} delta={result.delta_pct:+.2f}% "
                  f"p={result.p_value:.6f} {sig} n={result.n_synthetic}")

    return results


def main():
    print("="*70)
    print("EXTENDED ENSEMBLE EXPERIMENTS - Categories 1-5")
    print("="*70)
    print(f"\nTotal tests: 55")
    print(f"Estimated time: ~12-14 hours (4 parallel configs)")
    print(f"Categories: 1=Weighted, 2=Diversity, 3=Hybrid, 4=Dedup, 5=ClassTarget")

    # Load data once
    print(f"\nLoading data...")
    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    print(f"  Data loaded: {len(texts)} samples, {len(np.unique(labels))} classes")
    print(f"  Embeddings: {cache.embeddings.shape}")

    # Run all categories
    all_results = {}

    for category in [1, 2, 3, 4, 5]:
        category_results = run_category(category, cache)
        all_results.update(category_results)

    # Final summary across all categories
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - ALL CATEGORIES")
    print(f"{'='*70}")

    print(f"\nTotal ensembles tested: {len(all_results)}")
    print(f"Significant results: {sum(1 for r in all_results.values() if r.significant)}")

    # Top 10 overall
    print(f"\n{'='*70}")
    print(f"TOP 10 ENSEMBLES (All Categories)")
    print(f"{'='*70}")

    sorted_all = sorted(all_results.items(), key=lambda x: x[1].delta_pct, reverse=True)[:10]

    for i, (name, result) in enumerate(sorted_all, 1):
        sig = "✓" if result.significant else "✗"
        print(f"  {i:2}. {name:30} delta={result.delta_pct:+.2f}% "
              f"p={result.p_value:.6f} {sig}")

    # Save combined summary
    summary_path = RESULTS_DIR / "ensembles" / "extended_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(all_results),
        "significant_count": sum(1 for r in all_results.values() if r.significant),
        "ensembles": {
            name: {
                "delta_pct": r.delta_pct,
                "p_value": r.p_value,
                "significant": r.significant,
                "n_synthetic": r.n_synthetic,
            }
            for name, r in all_results.items()
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Extended summary saved to: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
