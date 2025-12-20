#!/usr/bin/env python3
"""
Compile and Analyze RARE_MLP Suite Results

This script aggregates all RARE_MLP results and generates:
- Top 10 by macro-F1 improvement
- Top 10 by ESFJ improvement
- Top 10 by ESTJ improvement
- ESFP analysis (likely still irresoluble)
- Comparison of SOTA techniques effectiveness
- Statistical significance summary
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "rare_mlp"
RARE_CLASSES = ["ESFJ", "ESFP", "ESTJ"]


def load_all_results() -> Dict[str, Dict]:
    """Load all RARE_MLP result files."""
    results = {}

    for json_file in RESULTS_DIR.glob("*_kfold.json"):
        if "summary" in json_file.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
                config_name = data.get("config_name", json_file.stem.replace("_kfold", ""))
                results[config_name] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def analyze_sota_techniques(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Analyze effectiveness of each SOTA technique."""
    technique_stats = {
        "focal_loss": {"configs": [], "deltas": [], "esfj_deltas": []},
        "class_weights": {"configs": [], "deltas": [], "esfj_deltas": []},
        "remix_mixup": {"configs": [], "deltas": [], "esfj_deltas": []},
        "intraclass_mixup": {"configs": [], "deltas": [], "esfj_deltas": []},
        "contrastive": {"configs": [], "deltas": [], "esfj_deltas": []},
        "baseline": {"configs": [], "deltas": [], "esfj_deltas": []},
    }

    for name, res in results.items():
        techniques = res.get("sota_techniques", [])
        delta = res.get("delta_pct", 0)
        esfj_delta = res.get("per_class_delta", {}).get("ESFJ", 0)

        if not techniques:
            technique_stats["baseline"]["configs"].append(name)
            technique_stats["baseline"]["deltas"].append(delta)
            technique_stats["baseline"]["esfj_deltas"].append(esfj_delta)
        else:
            for tech in techniques:
                key = tech.split("_g")[0] if "_g" in tech else tech  # Normalize focal_loss_g2 -> focal_loss
                if key in technique_stats:
                    technique_stats[key]["configs"].append(name)
                    technique_stats[key]["deltas"].append(delta)
                    technique_stats[key]["esfj_deltas"].append(esfj_delta)

    # Compute averages
    for tech, stats in technique_stats.items():
        if stats["deltas"]:
            stats["avg_delta"] = np.mean(stats["deltas"])
            stats["avg_esfj"] = np.mean(stats["esfj_deltas"])
            stats["count"] = len(stats["deltas"])
        else:
            stats["avg_delta"] = 0
            stats["avg_esfj"] = 0
            stats["count"] = 0

    return technique_stats


def generate_report(results: Dict[str, Dict]) -> str:
    """Generate markdown report."""
    lines = []

    lines.append("# RARE_MLP Suite Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal configurations tested: {len(results)}")
    lines.append(f"Significant results: {sum(1 for r in results.values() if r.get('significant', False))}")

    # Sort by delta_pct
    sorted_by_delta = sorted(
        results.items(),
        key=lambda x: x[1].get("delta_pct", 0),
        reverse=True
    )

    # Top 10 by Macro-F1
    lines.append("\n## Top 10 by Macro-F1 Improvement")
    lines.append("\n| Rank | Config | Delta | p-value | Sig | SOTA Techniques |")
    lines.append("|------|--------|-------|---------|-----|-----------------|")

    for i, (name, res) in enumerate(sorted_by_delta[:10], 1):
        sig = "Yes" if res.get("significant", False) else "No"
        techniques = ", ".join(res.get("sota_techniques", [])) or "None"
        lines.append(f"| {i} | {name} | {res.get('delta_pct', 0):+.2f}% | {res.get('p_value', 1):.6f} | {sig} | {techniques} |")

    # Top 10 by ESFJ
    sorted_by_esfj = sorted(
        results.items(),
        key=lambda x: x[1].get("per_class_delta", {}).get("ESFJ", 0),
        reverse=True
    )

    lines.append("\n## Top 10 by ESFJ Improvement")
    lines.append("\n| Rank | Config | ESFJ Delta | Macro-F1 Delta | SOTA Techniques |")
    lines.append("|------|--------|------------|----------------|-----------------|")

    for i, (name, res) in enumerate(sorted_by_esfj[:10], 1):
        esfj = res.get("per_class_delta", {}).get("ESFJ", 0)
        techniques = ", ".join(res.get("sota_techniques", [])) or "None"
        lines.append(f"| {i} | {name} | {esfj:+.4f} | {res.get('delta_pct', 0):+.2f}% | {techniques} |")

    # Top 10 by ESTJ
    sorted_by_estj = sorted(
        results.items(),
        key=lambda x: x[1].get("per_class_delta", {}).get("ESTJ", 0),
        reverse=True
    )

    lines.append("\n## Top 10 by ESTJ Improvement")
    lines.append("\n| Rank | Config | ESTJ Delta | Macro-F1 Delta | SOTA Techniques |")
    lines.append("|------|--------|------------|----------------|-----------------|")

    for i, (name, res) in enumerate(sorted_by_estj[:10], 1):
        estj = res.get("per_class_delta", {}).get("ESTJ", 0)
        techniques = ", ".join(res.get("sota_techniques", [])) or "None"
        lines.append(f"| {i} | {name} | {estj:+.4f} | {res.get('delta_pct', 0):+.2f}% | {techniques} |")

    # ESFP Analysis
    sorted_by_esfp = sorted(
        results.items(),
        key=lambda x: x[1].get("per_class_delta", {}).get("ESFP", 0),
        reverse=True
    )

    lines.append("\n## ESFP Analysis (Previously Irresoluble)")

    best_esfp = sorted_by_esfp[0] if sorted_by_esfp else None
    if best_esfp:
        esfp_delta = best_esfp[1].get("per_class_delta", {}).get("ESFP", 0)
        if esfp_delta > 0:
            lines.append(f"\n**BREAKTHROUGH!** ESFP improved by {esfp_delta:+.4f}")
            lines.append(f"\nBest config: {best_esfp[0]}")
            lines.append(f"SOTA techniques: {', '.join(best_esfp[1].get('sota_techniques', []))}")
        else:
            lines.append("\n**ESFP remains irresoluble** - No configuration achieved positive improvement.")
            lines.append("\nBest attempts:")
            for i, (name, res) in enumerate(sorted_by_esfp[:5], 1):
                esfp = res.get("per_class_delta", {}).get("ESFP", 0)
                lines.append(f"  {i}. {name}: {esfp:+.4f}")

    # SOTA Technique Analysis
    tech_stats = analyze_sota_techniques(results)

    lines.append("\n## SOTA Technique Effectiveness")
    lines.append("\n| Technique | Configs | Avg Macro-F1 | Avg ESFJ |")
    lines.append("|-----------|---------|--------------|----------|")

    for tech, stats in sorted(tech_stats.items(), key=lambda x: x[1].get("avg_delta", 0), reverse=True):
        if stats["count"] > 0:
            lines.append(f"| {tech} | {stats['count']} | {stats['avg_delta']:+.2f}% | {stats['avg_esfj']:+.4f} |")

    # Architecture Analysis
    arch_stats = {}
    for name, res in results.items():
        arch = res.get("mlp_architecture", "unknown")
        if arch not in arch_stats:
            arch_stats[arch] = {"deltas": [], "esfj_deltas": []}
        arch_stats[arch]["deltas"].append(res.get("delta_pct", 0))
        arch_stats[arch]["esfj_deltas"].append(res.get("per_class_delta", {}).get("ESFJ", 0))

    lines.append("\n## MLP Architecture Comparison")
    lines.append("\n| Architecture | Configs | Avg Macro-F1 | Avg ESFJ |")
    lines.append("|--------------|---------|--------------|----------|")

    for arch, stats in sorted(arch_stats.items(), key=lambda x: np.mean(x[1]["deltas"]), reverse=True):
        avg_delta = np.mean(stats["deltas"])
        avg_esfj = np.mean(stats["esfj_deltas"])
        lines.append(f"| {arch} | {len(stats['deltas'])} | {avg_delta:+.2f}% | {avg_esfj:+.4f} |")

    # Summary
    lines.append("\n## Summary")

    best_overall = sorted_by_delta[0] if sorted_by_delta else None
    best_esfj_cfg = sorted_by_esfj[0] if sorted_by_esfj else None
    best_estj_cfg = sorted_by_estj[0] if sorted_by_estj else None

    if best_overall:
        lines.append(f"\n**Best Overall (Macro-F1)**: {best_overall[0]} ({best_overall[1].get('delta_pct', 0):+.2f}%)")

    if best_esfj_cfg:
        esfj_val = best_esfj_cfg[1].get('per_class_delta', {}).get('ESFJ', 0)
        lines.append(f"\n**Best for ESFJ**: {best_esfj_cfg[0]} ({esfj_val:+.4f})")

    if best_estj_cfg:
        estj_val = best_estj_cfg[1].get('per_class_delta', {}).get('ESTJ', 0)
        lines.append(f"\n**Best for ESTJ**: {best_estj_cfg[0]} ({estj_val:+.4f})")

    # Comparison with previous best
    lines.append("\n### Comparison with Previous Best")
    lines.append("\n| Metric | Previous Best | RARE_MLP Best | Improvement |")
    lines.append("|--------|---------------|---------------|-------------|")

    prev_macro = 7.88  # TOP_all_common
    prev_esfj = 0.1242  # MLP_512_256_128 individual
    prev_estj = 0.0179  # MLP_512_256_128 individual

    if best_overall:
        new_macro = best_overall[1].get('delta_pct', 0)
        lines.append(f"| Macro-F1 | +{prev_macro:.2f}% | {new_macro:+.2f}% | {new_macro - prev_macro:+.2f} pp |")

    if best_esfj_cfg:
        new_esfj = best_esfj_cfg[1].get('per_class_delta', {}).get('ESFJ', 0)
        lines.append(f"| ESFJ | +{prev_esfj:.4f} | {new_esfj:+.4f} | {new_esfj - prev_esfj:+.4f} |")

    if best_estj_cfg:
        new_estj = best_estj_cfg[1].get('per_class_delta', {}).get('ESTJ', 0)
        lines.append(f"| ESTJ | +{prev_estj:.4f} | {new_estj:+.4f} | {new_estj - prev_estj:+.4f} |")

    return "\n".join(lines)


def main():
    """Main entry point."""
    print("="*70)
    print("Compiling RARE_MLP Suite Results")
    print("="*70)

    # Load results
    results = load_all_results()
    print(f"\nLoaded {len(results)} result files")

    if not results:
        print("No results found! Run exp_rare_mlp_suite.py first.")
        return

    # Generate report
    report = generate_report(results)

    # Save report
    report_file = RESULTS_DIR / "RARE_MLP_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Also save JSON summary
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("delta_pct", 0),
        reverse=True
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(results),
        "significant_count": sum(1 for r in results.values() if r.get("significant", False)),
        "top_10_macro_f1": [
            {
                "name": name,
                "delta_pct": res.get("delta_pct", 0),
                "esfj_delta": res.get("per_class_delta", {}).get("ESFJ", 0),
                "estj_delta": res.get("per_class_delta", {}).get("ESTJ", 0),
                "esfp_delta": res.get("per_class_delta", {}).get("ESFP", 0),
                "significant": res.get("significant", False),
                "sota_techniques": res.get("sota_techniques", []),
            }
            for name, res in sorted_results[:10]
        ],
        "best_per_class": {
            "esfj": max(results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESFJ", 0))[0],
            "estj": max(results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESTJ", 0))[0],
            "esfp": max(results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESFP", 0))[0],
        },
        "technique_effectiveness": analyze_sota_techniques(results),
    }

    summary_file = RESULTS_DIR / "rare_mlp_compiled_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_file}")

    # Print quick summary
    print("\n" + "="*70)
    print("Quick Summary")
    print("="*70)
    print(f"\nTop 5 by Macro-F1:")
    for i, (name, res) in enumerate(sorted_results[:5], 1):
        sig = "sig" if res.get("significant", False) else "ns"
        print(f"  {i}. {name}: {res.get('delta_pct', 0):+.2f}% ({sig})")

    print(f"\nBest ESFJ: {summary['best_per_class']['esfj']}")
    print(f"Best ESTJ: {summary['best_per_class']['estj']}")
    print(f"Best ESFP: {summary['best_per_class']['esfp']}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
