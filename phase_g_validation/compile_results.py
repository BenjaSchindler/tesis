#!/usr/bin/env python3
"""
Compile all Phase G Validation results into a comprehensive summary
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results")

def load_result(json_path):
    """Load a single result JSON"""
    with open(json_path) as f:
        data = json.load(f)
    return data

def compile_all_results():
    """Compile all results by category"""
    results = {
        "wave1": [],
        "wave2": [],
        "wave3": [],
        "wave4": [],
        "wave5": [],
        "wave6": [],
        "wave7": [],
        "wave8": [],
        "wave9": [],
        "rare_class": [],
        "component": [],
        "pf_derived": [],
        "ensembles": [],
        "multiclassifier": []
    }

    # Scan all subdirectories
    for category in results.keys():
        cat_dir = RESULTS_DIR / category
        if not cat_dir.exists():
            continue

        for json_file in cat_dir.glob("*.json"):
            try:
                data = load_result(json_file)
                config_name = json_file.stem.replace("_kfold", "")

                # Extract key metrics
                summary = {
                    "config": config_name,
                    "baseline_mean": data.get("baseline_mean", 0),
                    "augmented_mean": data.get("augmented_mean", 0),
                    "delta_mean": data.get("delta_mean", 0),
                    "delta_pct": data.get("delta_pct", 0),
                    "p_value": data.get("p_value", 1.0),
                    "significant": data.get("significant", False),
                    "per_class_delta": data.get("per_class_delta", {}),
                }

                results[category].append(summary)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    return results

def print_summary(results):
    """Print formatted summary"""
    print("=" * 100)
    print("PHASE G VALIDATION - COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    total_configs = sum(len(v) for v in results.values())
    significant_configs = sum(sum(1 for r in v if r["significant"]) for v in results.values())

    print(f"\nTotal Configurations Tested: {total_configs}")
    print(f"Statistically Significant: {significant_configs}/{total_configs} ({100*significant_configs/total_configs:.1f}%)")

    # Summary by category
    for category, configs in results.items():
        if not configs:
            continue

        print(f"\n{'=' * 100}")
        print(f"{category.upper()}: {len(configs)} configs")
        print("=" * 100)

        # Sort by delta_pct descending
        configs_sorted = sorted(configs, key=lambda x: x["delta_pct"], reverse=True)

        print(f"\n{'Config':<30} {'Baseline':<10} {'Delta':<12} {'p-value':<12} {'Sig?':<5}")
        print("-" * 100)

        for cfg in configs_sorted:
            sig = "✓" if cfg["significant"] else ""
            print(f"{cfg['config']:<30} {cfg['baseline_mean']:.4f}    "
                  f"{cfg['delta_pct']:+7.2f}%    {cfg['p_value']:.6f}    {sig}")

    # Top 10 overall
    print(f"\n{'=' * 100}")
    print("TOP 10 CONFIGURATIONS (by delta %)")
    print("=" * 100)

    all_configs = []
    for category, configs in results.items():
        for cfg in configs:
            cfg["category"] = category
            all_configs.append(cfg)

    top10 = sorted(all_configs, key=lambda x: x["delta_pct"], reverse=True)[:10]

    print(f"\n{'Rank':<6} {'Config':<35} {'Category':<15} {'Delta':<12} {'Sig?':<5}")
    print("-" * 100)

    for i, cfg in enumerate(top10, 1):
        sig = "✓" if cfg["significant"] else ""
        print(f"{i:<6} {cfg['config']:<35} {cfg['category']:<15} {cfg['delta_pct']:+7.2f}%    {sig}")

    # Problem classes analysis
    print(f"\n{'=' * 100}")
    print("PROBLEM CLASSES ANALYSIS (ESFJ, ESFP, ESTJ)")
    print("=" * 100)

    problem_classes = ["ESFJ", "ESFP", "ESTJ"]

    for cls in problem_classes:
        print(f"\n--- {cls} ---")

        # Find configs that improved this class
        improved = []
        for cfg in all_configs:
            delta = cfg["per_class_delta"].get(cls, 0)
            if delta > 0:
                improved.append((cfg["config"], cfg["category"], delta, cfg["significant"]))

        improved_sorted = sorted(improved, key=lambda x: x[2], reverse=True)[:5]

        if improved_sorted:
            print(f"Top 5 configs for {cls}:")
            for config, cat, delta, sig in improved_sorted:
                sig_mark = "✓" if sig else ""
                print(f"  {config:<35} ({cat:<12}) {delta:+.4f} {sig_mark}")
        else:
            print(f"  No configurations improved {cls}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    results = compile_all_results()
    print_summary(results)

    # Save to JSON
    output_file = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results/FULL_SUMMARY.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: {output_file}")
