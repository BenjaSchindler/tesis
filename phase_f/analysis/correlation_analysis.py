#!/usr/bin/env python3
"""
Phase F Analysis: Correlation between #synthetics and F1 delta
"""

import json
import os
from pathlib import Path
import numpy as np
from scipy import stats

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

def load_kfold_results():
    """Load all kfold JSON files and extract key metrics."""
    results = []

    # Load from results/ directory (ensembles)
    for f in RESULTS_DIR.glob("*_kfold_k5.json"):
        if f.stat().st_size == 0:
            continue  # Skip empty files
        try:
            with open(f) as fp:
                data = json.load(fp)
            config = data.get("config", f.stem.replace("_s42_kfold_k5", ""))
            results.append({
                "config": config,
                "n_synthetic": data.get("n_synthetic", 0),
                "delta_mean": data["delta"]["mean"] * 100,  # Convert to %
                "delta_std": data["delta"]["std"] * 100,
                "p_value": data["delta"]["p_value"],
                "win_rate": data["delta"]["win_rate"],
                "type": "ensemble" if config.startswith("ENS_") else "single"
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {f.name}: {e}")
            continue

    # Load from Variance_tests/ directory (singles)
    variance_dir = RESULTS_DIR / "Variance_tests"
    if variance_dir.exists():
        for f in variance_dir.glob("*_kfold.json"):
            if f.stat().st_size == 0:
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
                config = data.get("config", f.stem.replace("_kfold", ""))
                # Skip if already loaded
                if any(r["config"] == config for r in results):
                    continue
                results.append({
                    "config": config,
                    "n_synthetic": data.get("n_synthetic", 0),
                    "delta_mean": data["delta"]["mean"] * 100,
                    "delta_std": data["delta"]["std"] * 100,
                    "p_value": data["delta"]["p_value"],
                    "win_rate": data["delta"]["win_rate"],
                    "type": "single"
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping {f.name}: {e}")
                continue

    return sorted(results, key=lambda x: x["n_synthetic"])

def calculate_correlations(results):
    """Calculate Pearson and Spearman correlations."""
    all_synth = [r["n_synthetic"] for r in results]
    all_delta = [r["delta_mean"] for r in results]

    singles = [r for r in results if r["type"] == "single"]
    ensembles = [r for r in results if r["type"] == "ensemble"]

    analysis = {
        "all": {
            "n": len(results),
            "pearson": stats.pearsonr(all_synth, all_delta) if len(results) > 2 else (0, 1),
            "spearman": stats.spearmanr(all_synth, all_delta) if len(results) > 2 else (0, 1)
        },
        "singles": {
            "n": len(singles),
            "pearson": stats.pearsonr(
                [r["n_synthetic"] for r in singles],
                [r["delta_mean"] for r in singles]
            ) if len(singles) > 2 else (0, 1),
            "spearman": stats.spearmanr(
                [r["n_synthetic"] for r in singles],
                [r["delta_mean"] for r in singles]
            ) if len(singles) > 2 else (0, 1)
        },
        "ensembles": {
            "n": len(ensembles),
            "pearson": stats.pearsonr(
                [r["n_synthetic"] for r in ensembles],
                [r["delta_mean"] for r in ensembles]
            ) if len(ensembles) > 2 else (0, 1),
            "spearman": stats.spearmanr(
                [r["n_synthetic"] for r in ensembles],
                [r["delta_mean"] for r in ensembles]
            ) if len(ensembles) > 2 else (0, 1)
        }
    }

    return analysis

def print_results_table(results):
    """Print sorted results table."""
    print("\n" + "="*80)
    print("PHASE F: CORRELATION ANALYSIS - Synthetics vs F1 Delta")
    print("="*80)

    print("\n### All Configurations (sorted by #synthetics)")
    print("-"*80)
    print(f"{'Config':<25} {'Type':<10} {'Synth':>8} {'Delta%':>10} {'p-value':>12} {'Win%':>8}")
    print("-"*80)

    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['config']:<25} {r['type']:<10} {r['n_synthetic']:>8} {r['delta_mean']:>+9.2f}% {r['p_value']:>11.6f}{sig} {r['win_rate']*100:>7.1f}%")

def print_correlation_analysis(analysis):
    """Print correlation analysis results."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    for group, data in analysis.items():
        print(f"\n### {group.upper()} (n={data['n']})")
        print("-"*40)

        if data['n'] > 2:
            pr, pp = data['pearson']
            sr, sp = data['spearman']

            print(f"  Pearson r:  {pr:+.4f}  (p={pp:.6f})")
            print(f"  Spearman r: {sr:+.4f}  (p={sp:.6f})")
            print(f"  R² (linear): {pr**2:.4f}")

            # Interpretation
            if abs(pr) < 0.3:
                strength = "WEAK"
            elif abs(pr) < 0.6:
                strength = "MODERATE"
            else:
                strength = "STRONG"

            direction = "positive" if pr > 0 else "negative"
            sig = "significant" if pp < 0.05 else "NOT significant"

            print(f"\n  Interpretation: {strength} {direction} correlation, {sig}")
        else:
            print("  Insufficient data points for correlation")

def calculate_marginal_returns(results):
    """Calculate marginal improvement per additional synthetic."""
    print("\n" + "="*80)
    print("MARGINAL RETURNS ANALYSIS")
    print("="*80)

    # Sort by synthetics
    sorted_results = sorted(results, key=lambda x: x["n_synthetic"])

    print("\n### Improvement per Additional Synthetic")
    print("-"*70)
    print(f"{'From → To':<35} {'ΔSynth':>8} {'ΔDelta%':>10} {'Marginal':>12}")
    print("-"*70)

    for i in range(1, len(sorted_results)):
        prev = sorted_results[i-1]
        curr = sorted_results[i]

        d_synth = curr["n_synthetic"] - prev["n_synthetic"]
        d_delta = curr["delta_mean"] - prev["delta_mean"]

        if d_synth > 0:
            marginal = d_delta / d_synth
            print(f"{prev['config'][:15]} → {curr['config'][:15]:<15} {d_synth:>+8} {d_delta:>+9.3f}% {marginal:>+11.4f}%/syn")

def generate_report(results, analysis):
    """Generate markdown report."""
    report = []
    report.append("# Phase F: Correlation Analysis Report")
    report.append(f"\nGenerated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")

    report.append("\n## Summary\n")

    # Key finding
    all_pr = analysis["all"]["pearson"][0]
    singles_pr = analysis["singles"]["pearson"][0]
    ensembles_pr = analysis["ensembles"]["pearson"][0] if analysis["ensembles"]["n"] > 2 else 0

    report.append(f"| Group | N | Pearson r | R² | p-value | Interpretation |")
    report.append(f"|-------|---|-----------|-----|---------|----------------|")

    for group, data in analysis.items():
        if data['n'] > 2:
            pr, pp = data['pearson']
            strength = "Weak" if abs(pr) < 0.3 else "Moderate" if abs(pr) < 0.6 else "Strong"
            sig = "Sig." if pp < 0.05 else "N.S."
            report.append(f"| {group.capitalize()} | {data['n']} | {pr:+.3f} | {pr**2:.3f} | {pp:.4f} | {strength} ({sig}) |")

    report.append("\n## Key Findings\n")

    if abs(singles_pr) < 0.3:
        report.append("1. **Singles show WEAK correlation** between #synthetics and improvement")
        report.append("   - More synthetics ≠ better results for individual configs")
        report.append("   - Quality and filtering strategy matter more than volume")

    if ensembles_pr > 0.6:
        report.append("\n2. **Ensembles show STRONG correlation** with total synthetics")
        report.append("   - But this is DIVERSITY, not raw volume")
        report.append("   - Each config contributes unique samples")

    report.append("\n## Raw Data\n")
    report.append("| Config | Type | Synthetics | Delta% | p-value | Win Rate |")
    report.append("|--------|------|------------|--------|---------|----------|")

    for r in sorted(results, key=lambda x: -x["delta_mean"]):
        report.append(f"| {r['config']} | {r['type']} | {r['n_synthetic']} | {r['delta_mean']:+.2f}% | {r['p_value']:.6f} | {r['win_rate']*100:.1f}% |")

    return "\n".join(report)

def main():
    print("Loading kfold results...")
    results = load_kfold_results()
    print(f"Found {len(results)} configurations with kfold results")

    if not results:
        print("ERROR: No kfold results found!")
        print(f"Searched in: {RESULTS_DIR}")
        return

    # Print results table
    print_results_table(results)

    # Calculate correlations
    analysis = calculate_correlations(results)
    print_correlation_analysis(analysis)

    # Marginal returns
    calculate_marginal_returns(results)

    # Generate report
    report = generate_report(results, analysis)
    report_path = OUTPUT_DIR / "correlation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n\nReport saved to: {report_path}")

    # Summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    singles_pr = analysis["singles"]["pearson"][0] if analysis["singles"]["n"] > 2 else 0
    ensembles_pr = analysis["ensembles"]["pearson"][0] if analysis["ensembles"]["n"] > 2 else 0

    print(f"""
    Singles correlation:   r = {singles_pr:+.3f} (R² = {singles_pr**2:.3f})
    Ensembles correlation: r = {ensembles_pr:+.3f} (R² = {ensembles_pr**2:.3f})

    ANSWER: {"Weak correlation for singles, stronger for ensembles" if abs(singles_pr) < 0.4 and ensembles_pr > 0.5 else "Mixed results"}

    This suggests DIVERSITY (from different configs) matters more than
    raw VOLUME of synthetics for individual configs.
    """)

if __name__ == "__main__":
    main()
