#!/usr/bin/env python3
"""
Analyze variance across multiple replications.
Calculates statistics to measure reproducibility of results.

Usage:
    python3 analyze_replication_variance.py --runs 3

This will look for:
    replication_run1/results/ENS_*_holdout.json
    replication_run2/results/ENS_*_holdout.json
    replication_run3/results/ENS_*_holdout.json
"""
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats


def load_results(run_dirs: list, ensemble_name: str) -> list:
    """Load holdout results from multiple runs."""
    results = []
    for run_dir in run_dirs:
        json_path = run_dir / "results" / f"{ensemble_name}_holdout.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            results.append({
                'run': run_dir.name,
                'delta_percent': data.get('delta_percent', 0),
                'baseline_f1': data.get('baseline_f1', 0),
                'augmented_f1': data.get('augmented_f1', 0),
                'n_synthetic': data.get('n_synthetic', 0),
            })
        else:
            print(f"  WARNING: {json_path} not found")
    return results


def analyze_ensemble(ensemble_name: str, results: list, original_delta: float):
    """Analyze variance for a single ensemble."""
    if not results:
        print(f"\n{ensemble_name}: No results found")
        return None

    deltas = [r['delta_percent'] for r in results]
    synth_counts = [r['n_synthetic'] for r in results]

    print(f"\n{'=' * 70}")
    print(f"  {ensemble_name}")
    print(f"{'=' * 70}")

    print(f"\n  Runs: {len(results)}")
    for r in results:
        print(f"    {r['run']}: {r['delta_percent']:+.2f}% ({r['n_synthetic']} synthetics)")

    print(f"\n  Statistics:")
    print(f"    Original result:   {original_delta:+.2f}%")
    print(f"    Mean:              {np.mean(deltas):+.2f}%")
    print(f"    Std:               {np.std(deltas):.2f}%")
    print(f"    Range:             [{min(deltas):+.2f}%, {max(deltas):+.2f}%]")
    print(f"    Median:            {np.median(deltas):+.2f}%")

    # 95% CI
    if len(deltas) >= 2:
        se = np.std(deltas) / np.sqrt(len(deltas))
        ci_lower = np.mean(deltas) - 1.96 * se
        ci_upper = np.mean(deltas) + 1.96 * se
        print(f"    95% CI:            [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")

    # Coefficient of variation
    cv = (np.std(deltas) / abs(np.mean(deltas))) * 100 if np.mean(deltas) != 0 else 0
    print(f"    CV (variability):  {cv:.1f}%")

    # Synthetic count variance
    print(f"\n  Synthetic count variation:")
    print(f"    Mean:  {np.mean(synth_counts):.0f}")
    print(f"    Range: [{min(synth_counts)}, {max(synth_counts)}]")

    # Assessment
    print(f"\n  Assessment:")
    if np.std(deltas) < 1.0:
        print("    ROBUST - Low variance (<1%)")
    elif np.std(deltas) < 2.0:
        print("    ACCEPTABLE - Moderate variance (1-2%)")
    else:
        print("    HIGH VARIANCE - Results may not be reproducible (>2%)")

    # Is original within expected range?
    if len(deltas) >= 2:
        if ci_lower <= original_delta <= ci_upper:
            print(f"    Original {original_delta:+.2f}% is within 95% CI")
        else:
            print(f"    WARNING: Original {original_delta:+.2f}% is outside 95% CI")

    return {
        'ensemble': ensemble_name,
        'n_runs': len(results),
        'mean': float(np.mean(deltas)),
        'std': float(np.std(deltas)),
        'min': float(min(deltas)),
        'max': float(max(deltas)),
        'cv': float(cv),
        'original': original_delta,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze replication variance')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs to analyze')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Find run directories
    run_dirs = []
    for i in range(1, args.runs + 1):
        run_dir = base_dir / f"replication_run{i}"
        if run_dir.exists():
            run_dirs.append(run_dir)

    print("=" * 70)
    print("  REPLICATION VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"\n  Found {len(run_dirs)} replication runs")

    if len(run_dirs) < 2:
        print("  ERROR: Need at least 2 runs to analyze variance")
        print("  Run: ./run_replication.sh 1")
        print("       ./run_replication.sh 2")
        return

    # Original results to compare against
    original_results = {
        'ENS_TopG5_Extended': 13.73,
        'ENS_SUPER_G5_F7_v2': 13.32,
        'ENS_Top3_G5': 5.98,  # Estimated from K-fold
    }

    all_results = {}

    for ensemble_name, original_delta in original_results.items():
        results = load_results(run_dirs, ensemble_name)
        analysis = analyze_ensemble(ensemble_name, results, original_delta)
        if analysis:
            all_results[ensemble_name] = analysis

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Ensemble':<25} {'Original':>10} {'Mean':>10} {'Std':>8} {'CV':>8}")
    print("  " + "-" * 65)

    for name, data in all_results.items():
        print(f"  {name:<25} {data['original']:>+9.2f}% {data['mean']:>+9.2f}% {data['std']:>7.2f}% {data['cv']:>7.1f}%")

    # Save results
    output_path = base_dir / "replication_variance_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # Final assessment
    print("\n" + "=" * 70)
    print("  FINAL ASSESSMENT")
    print("=" * 70)

    avg_cv = np.mean([d['cv'] for d in all_results.values()])
    if avg_cv < 10:
        print("\n  EXCELLENT REPRODUCIBILITY")
        print("  Results are highly consistent across runs.")
    elif avg_cv < 20:
        print("\n  GOOD REPRODUCIBILITY")
        print("  Results show acceptable variance.")
    else:
        print("\n  MODERATE REPRODUCIBILITY")
        print("  Results show significant variance - interpret with caution.")


if __name__ == "__main__":
    main()
