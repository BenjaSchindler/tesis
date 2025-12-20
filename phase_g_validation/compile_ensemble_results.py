#!/usr/bin/env python3
"""
Compile Extended Ensemble Results

Aggregates results from all ensemble experiments and generates comprehensive reports.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results/ensembles")
OUTPUT_FILE = RESULTS_DIR / "ensemble_extended_full.json"


def load_all_results():
    """Load all ensemble result files."""
    results = []

    for json_file in RESULTS_DIR.glob("*.json"):
        if json_file.name in ["ensemble_summary.json", "extended_summary.json",
                               "advanced_summary.json", "ensemble_extended_full.json"]:
            # Skip summary files
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_file"] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"  WARNING: Failed to load {json_file.name}: {e}")

    return results


def categorize_ensemble(config_name: str) -> str:
    """Determine category from config name."""
    if config_name.startswith("WGT_"):
        return "Category 1: Weighted Top-K"
    elif config_name.startswith("DIV_"):
        return "Category 2: Diversity-Maximizing"
    elif config_name.startswith("HYB_"):
        return "Category 3: Hybrid Strategy"
    elif config_name.startswith("DEDUP_"):
        return "Category 4: Deduplication-Based"
    elif config_name.startswith(("TOP_", "BAL_", "RARE_MLP")):
        return "Category 5: Class-Targeted"
    elif config_name.startswith(("STACK_", "VOTE_", "SELECT_")):
        return "Category 6: Advanced Combination"
    elif config_name.startswith("NOVEL_"):
        return "Category 7: Experimental"
    elif config_name.startswith("ENS_"):
        return "Original Ensembles"
    else:
        return "Unknown"


def compile_results(results):
    """Compile results into structured format."""

    compiled = {
        "total_ensembles": len(results),
        "significant_count": 0,
        "by_category": defaultdict(list),
        "top_10": [],
        "statistics": {},
        "ensembles": []
    }

    for result in results:
        config_name = result.get("config", "unknown")
        category = categorize_ensemble(config_name)

        ensemble_summary = {
            "config": config_name,
            "category": category,
            "baseline_mean": result.get("baseline_mean", 0),
            "augmented_mean": result.get("augmented_mean", 0),
            "delta_mean": result.get("delta_mean", 0),
            "delta_pct": result.get("delta_pct", 0),
            "p_value": result.get("p_value", 1.0),
            "significant": result.get("significant", False),
            "n_synthetic": result.get("n_synthetic", 0),
            "per_class_delta": result.get("per_class_delta", {}),
        }

        compiled["ensembles"].append(ensemble_summary)
        compiled["by_category"][category].append(ensemble_summary)

        if ensemble_summary["significant"]:
            compiled["significant_count"] += 1

    # Sort ensembles by delta_pct
    compiled["ensembles"].sort(key=lambda x: x["delta_pct"], reverse=True)

    # Get top 10
    compiled["top_10"] = compiled["ensembles"][:10]

    # Compute statistics
    deltas = [e["delta_pct"] for e in compiled["ensembles"]]
    p_values = [e["p_value"] for e in compiled["ensembles"] if not np.isnan(e["p_value"])]

    compiled["statistics"] = {
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "median_delta": float(np.median(deltas)) if deltas else 0,
        "std_delta": float(np.std(deltas)) if deltas else 0,
        "min_delta": float(np.min(deltas)) if deltas else 0,
        "max_delta": float(np.max(deltas)) if deltas else 0,
        "mean_p_value": float(np.mean(p_values)) if p_values else 1.0,
        "significant_rate": compiled["significant_count"] / len(results) if results else 0,
    }

    return compiled


def print_summary(compiled):
    """Print comprehensive summary."""

    print("\n" + "="*70)
    print("EXTENDED ENSEMBLE RESULTS - COMPREHENSIVE SUMMARY")
    print("="*70)

    print(f"\nTotal Ensembles: {compiled['total_ensembles']}")
    print(f"Significant Results: {compiled['significant_count']} ({compiled['statistics']['significant_rate']:.1%})")

    print(f"\n--- Overall Statistics ---")
    stats = compiled["statistics"]
    print(f"  Mean Delta:   {stats['mean_delta']:+.2f}%")
    print(f"  Median Delta: {stats['median_delta']:+.2f}%")
    print(f"  Std Delta:    {stats['std_delta']:.2f}%")
    print(f"  Min Delta:    {stats['min_delta']:+.2f}%")
    print(f"  Max Delta:    {stats['max_delta']:+.2f}%")

    print(f"\n--- By Category ---")
    for category, ensembles in sorted(compiled["by_category"].items()):
        sig_count = sum(1 for e in ensembles if e["significant"])
        avg_delta = np.mean([e["delta_pct"] for e in ensembles])
        print(f"  {category:35} n={len(ensembles):2}  sig={sig_count:2}  avg={avg_delta:+.2f}%")

    print(f"\n{'='*70}")
    print(f"TOP 10 ENSEMBLES")
    print(f"{'='*70}")

    for i, ensemble in enumerate(compiled["top_10"], 1):
        sig = "✓" if ensemble["significant"] else "✗"
        print(f"  {i:2}. {ensemble['config']:30} "
              f"delta={ensemble['delta_pct']:+.2f}% "
              f"p={ensemble['p_value']:.6f} {sig}")

    # Compare with best individual config
    best_individual = 5.98  # W5_many_shot_10
    if compiled["top_10"]:
        best_ensemble = compiled["top_10"][0]["delta_pct"]
        print(f"\n{'='*70}")
        print(f"COMPARISON TO BEST INDIVIDUAL CONFIG")
        print(f"{'='*70}")
        print(f"  Best Individual (W5_many_shot_10): {best_individual:+.2f}%")
        print(f"  Best Ensemble ({compiled['top_10'][0]['config']}): {best_ensemble:+.2f}%")

        if best_ensemble > best_individual:
            diff = best_ensemble - best_individual
            print(f"  ✅ ENSEMBLE WINS by {diff:+.2f} percentage points!")
        elif best_ensemble >= best_individual - 0.5:
            diff = best_individual - best_ensemble
            print(f"  ⚖️  COMPETITIVE (within {diff:.2f} pp of best individual)")
        else:
            diff = best_individual - best_ensemble
            print(f"  ❌ Individual config wins by {diff:.2f} pp")

    print("\n" + "="*70)


def main():
    print("="*70)
    print("COMPILING EXTENDED ENSEMBLE RESULTS")
    print("="*70)

    print(f"\nScanning: {RESULTS_DIR}")

    # Load results
    results = load_all_results()
    print(f"  Loaded {len(results)} ensemble results")

    if not results:
        print("\n  No results found!")
        return

    # Compile
    print(f"\nCompiling results...")
    compiled = compile_results(results)

    # Print summary
    print_summary(compiled)

    # Save compiled results
    print(f"\nSaving compiled results...")

    # Convert defaultdict to dict for JSON serialization
    compiled_for_json = dict(compiled)
    compiled_for_json["by_category"] = dict(compiled["by_category"])

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(compiled_for_json, f, indent=2)

    print(f"  ✓ Saved to: {OUTPUT_FILE}")
    print("="*70)


if __name__ == "__main__":
    main()
