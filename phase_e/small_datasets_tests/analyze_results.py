#!/usr/bin/env python3
"""
Phase E: Experiment Results Analyzer
====================================
Aggregates results across all experiments and seeds, calculates statistics,
and generates comparison tables.
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Any, Optional
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_metrics(filepath: str) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def extract_experiment_name(filename: str) -> str:
    """Extract experiment name from filename like 'A1_gpt5_none_s42_20251202_*.json'"""
    parts = filename.split('_s')
    if len(parts) >= 2:
        return parts[0]
    return filename


def aggregate_results() -> Dict[str, List[Dict]]:
    """Aggregate all experiment results by experiment name."""
    results = defaultdict(list)

    # Find all metrics files
    pattern = os.path.join(RESULTS_DIR, "*_metrics.json")
    files = glob.glob(pattern)

    for filepath in files:
        filename = os.path.basename(filepath)
        exp_name = extract_experiment_name(filename)

        metrics = load_metrics(filepath)
        if metrics:
            # Extract key metrics
            baseline_f1 = metrics.get('baseline', {}).get('macro_f1', 0)
            augmented_f1 = metrics.get('augmented', {}).get('macro_f1', 0)
            delta_pct = metrics.get('improvement', {}).get('f1_delta_pct', 0)
            synthetics = metrics.get('synthetic_data', {}).get('accepted_count', 0)

            # Per-class improvements
            per_class = metrics.get('improvement', {}).get('per_class', {})
            minority_deltas = {
                cls: per_class.get(cls, {}).get('delta_pct', 0)
                for cls in ['ESTJ', 'ESFP', 'ESFJ', 'ENFJ', 'ISFJ', 'ISTJ', 'ESTP']
            }

            results[exp_name].append({
                'filename': filename,
                'baseline_f1': baseline_f1,
                'augmented_f1': augmented_f1,
                'delta_pct': delta_pct,
                'synthetics': synthetics,
                'minority_deltas': minority_deltas,
                'per_class': per_class
            })

    return dict(results)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate mean, std, min, max, CI for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'ci_low': 0, 'ci_high': 0}

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0

    # 95% CI (using t-distribution approximation)
    t_val = 2.0 if n == 3 else 1.96  # Simplified
    ci = t_val * std / (n ** 0.5) if n > 0 else 0

    return {
        'mean': mean,
        'std': std,
        'min': min(values),
        'max': max(values),
        'ci_low': mean - ci,
        'ci_high': mean + ci,
        'n': n
    }


def analyze_experiments(results: Dict[str, List[Dict]]) -> List[Dict]:
    """Analyze all experiments and return ranked results."""
    analysis = []

    for exp_name, runs in results.items():
        if not runs:
            continue

        # Aggregate metrics across seeds
        deltas = [r['delta_pct'] for r in runs]
        synthetics = [r['synthetics'] for r in runs]

        # Calculate minority class improvements
        minority_deltas = defaultdict(list)
        for r in runs:
            for cls, delta in r['minority_deltas'].items():
                minority_deltas[cls].append(delta)

        # Calculate statistics
        delta_stats = calculate_statistics(deltas)
        synth_stats = calculate_statistics(synthetics)

        # Minority class stats
        minority_stats = {}
        for cls, cls_deltas in minority_deltas.items():
            minority_stats[cls] = calculate_statistics(cls_deltas)

        analysis.append({
            'experiment': exp_name,
            'n_seeds': len(runs),
            'delta_mean': delta_stats['mean'],
            'delta_std': delta_stats['std'],
            'delta_ci_low': delta_stats['ci_low'],
            'delta_ci_high': delta_stats['ci_high'],
            'synthetics_mean': synth_stats['mean'],
            'minority_stats': minority_stats
        })

    # Sort by delta mean (descending)
    analysis.sort(key=lambda x: x['delta_mean'], reverse=True)

    return analysis


def print_results(analysis: List[Dict]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("  PHASE E EXPERIMENT RESULTS - RANKED BY MACRO-F1 IMPROVEMENT")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'Experiment':<25} {'Delta %':<12} {'CI 95%':<15} {'Synth':<8} {'Seeds':<6}")
    print("-" * 100)

    for i, exp in enumerate(analysis, 1):
        delta_str = f"{exp['delta_mean']:+.2f}%"
        ci_str = f"[{exp['delta_ci_low']:+.2f}, {exp['delta_ci_high']:+.2f}]"
        synth_str = f"{exp['synthetics_mean']:.0f}"

        # Highlight top results
        marker = "***" if exp['delta_mean'] > 4.15 else ""

        print(f"{i:<5} {exp['experiment']:<25} {delta_str:<12} {ci_str:<15} {synth_str:<8} {exp['n_seeds']:<6} {marker}")

    # Best result summary
    if analysis:
        best = analysis[0]
        print("\n" + "=" * 100)
        print(f"  BEST EXPERIMENT: {best['experiment']}")
        print(f"  Mean Delta: {best['delta_mean']:+.2f}% (CI: [{best['delta_ci_low']:+.2f}, {best['delta_ci_high']:+.2f}])")
        print(f"  Synthetics: {best['synthetics_mean']:.0f}")

        # Check if beat current best
        if best['delta_mean'] > 4.15:
            print(f"\n  *** BEAT CURRENT BEST (+4.15%)! ***")
        print("=" * 100)

    # Minority class analysis
    print("\n" + "=" * 100)
    print("  MINORITY CLASS IMPROVEMENTS (Top 5 experiments per class)")
    print("=" * 100)

    minority_classes = ['ESTJ', 'ESFP', 'ESFJ', 'ENFJ']

    for cls in minority_classes:
        print(f"\n--- {cls} ---")

        # Get experiments sorted by this class improvement
        cls_ranked = []
        for exp in analysis:
            if cls in exp['minority_stats']:
                cls_ranked.append({
                    'experiment': exp['experiment'],
                    'delta': exp['minority_stats'][cls]['mean'],
                    'std': exp['minority_stats'][cls]['std']
                })

        cls_ranked.sort(key=lambda x: x['delta'], reverse=True)

        for i, exp in enumerate(cls_ranked[:5], 1):
            print(f"  {i}. {exp['experiment']:<25} {exp['delta']:+.2f}% ± {exp['std']:.2f}")


def save_results(analysis: List[Dict], results: Dict[str, List[Dict]]) -> None:
    """Save analysis to JSON files."""
    # Summary
    summary_path = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'ranked_experiments': analysis,
            'generated_at': os.popen('date').read().strip()
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Per-class analysis
    per_class_path = os.path.join(RESULTS_DIR, "per_class_analysis.json")
    per_class_data = {}
    for exp in analysis:
        per_class_data[exp['experiment']] = exp['minority_stats']

    with open(per_class_path, 'w') as f:
        json.dump(per_class_data, f, indent=2)
    print(f"Per-class analysis saved to: {per_class_path}")


def main():
    print("Loading experiment results...")
    results = aggregate_results()

    if not results:
        print("No results found in", RESULTS_DIR)
        print("Run experiments first with: ./run_all_experiments.sh")
        return

    print(f"Found {len(results)} experiments with results")

    # Analyze
    analysis = analyze_experiments(results)

    # Print
    print_results(analysis)

    # Save
    save_results(analysis, results)


if __name__ == "__main__":
    main()
