#!/usr/bin/env python3
"""
Synthetic Quality Analyzer for SMOTE-LLM

Analyzes synthetic data quality from enhanced logging output to:
1. Understand rejection patterns
2. Correlate quality metrics with F1 improvement
3. Identify optimal thresholds
4. Diagnose per-class issues

Usage:
    python3 core/synthetic_quality_analyzer.py --metrics metrics.json --csv synthetic.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter

import pandas as pd
import numpy as np


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics JSON file."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_synthetic_csv(csv_path: str) -> pd.DataFrame:
    """Load synthetic CSV with quality metrics."""
    return pd.read_csv(csv_path)


def analyze_rejection_patterns(metrics: Dict) -> Dict:
    """Analyze rejection patterns from metrics."""
    if "rejection_analysis" not in metrics:
        return {"error": "No rejection_analysis found. Did you run with --verbose-logging?"}

    rejection = metrics["rejection_analysis"]

    analysis = {
        "summary": {
            "total_generated": rejection["total_generated"],
            "total_accepted": rejection["total_accepted"],
            "total_rejected": rejection["total_rejected"],
            "overall_acceptance_rate": rejection["overall_acceptance_rate"],
        },
        "rejection_breakdown": {},
        "per_class_analysis": {},
    }

    # Rejection breakdown
    reasons = rejection.get("rejection_reasons", {})
    total_rej = sum(reasons.values()) if reasons else 1

    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total_rej * 100 if total_rej > 0 else 0
        analysis["rejection_breakdown"][reason] = {
            "count": count,
            "percentage": round(pct, 2),
        }

    # Per-class acceptance rates
    per_class_rates = rejection.get("per_class_acceptance_rate", {})
    sorted_classes = sorted(per_class_rates.items(), key=lambda x: x[1])

    analysis["per_class_analysis"]["lowest_acceptance"] = [
        {"class": cls, "rate": round(rate, 4)}
        for cls, rate in sorted_classes[:5]
    ]
    analysis["per_class_analysis"]["highest_acceptance"] = [
        {"class": cls, "rate": round(rate, 4)}
        for cls, rate in sorted_classes[-5:]
    ]

    return analysis


def analyze_quality_metrics(df: pd.DataFrame) -> Dict:
    """Analyze quality metrics from synthetic CSV."""
    quality_cols = ["similarity_to_centroid", "similarity_to_anchor", "similarity_to_neighbor",
                    "knn_similarity", "classifier_confidence"]

    analysis = {
        "columns_found": [],
        "distributions": {},
        "per_class_means": {},
        "correlations": {},
    }

    # Find which quality columns exist
    for col in quality_cols:
        if col in df.columns:
            analysis["columns_found"].append(col)

    if not analysis["columns_found"]:
        return {"error": "No quality metrics columns found. Did you run with --verbose-logging?"}

    # Distribution stats
    for col in analysis["columns_found"]:
        values = df[col].dropna()
        if len(values) > 0:
            analysis["distributions"][col] = {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "q25": round(float(values.quantile(0.25)), 4),
                "median": round(float(values.quantile(0.50)), 4),
                "q75": round(float(values.quantile(0.75)), 4),
            }

    # Per-class means
    if "label" in df.columns and len(analysis["columns_found"]) > 0:
        for cls in df["label"].unique():
            cls_df = df[df["label"] == cls]
            analysis["per_class_means"][cls] = {}
            for col in analysis["columns_found"]:
                values = cls_df[col].dropna()
                if len(values) > 0:
                    analysis["per_class_means"][cls][col] = round(float(values.mean()), 4)

    return analysis


def analyze_per_class_quality(metrics: Dict) -> Dict:
    """Analyze per-class quality metrics from JSON."""
    if "per_class_quality" not in metrics:
        return {"error": "No per_class_quality found. Did you run with --verbose-logging?"}

    pcq = metrics["per_class_quality"]

    analysis = {
        "classes": {},
        "summary": {
            "classes_with_data": len(pcq),
            "total_candidates": sum(c.get("total_candidates", 0) for c in pcq.values()),
            "total_accepted": sum(c.get("total_accepted", 0) for c in pcq.values()),
        }
    }

    for cls, quality in pcq.items():
        analysis["classes"][cls] = {
            "total_candidates": quality.get("total_candidates", 0),
            "total_accepted": quality.get("total_accepted", 0),
            "acceptance_rate": round(quality.get("acceptance_rate", 0), 4),
            "rejection_reasons": quality.get("rejection_stats", {}),
        }

    return analysis


def generate_recommendations(rejection_analysis: Dict, quality_analysis: Dict) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Check overall acceptance rate
    if "summary" in rejection_analysis:
        rate = rejection_analysis["summary"].get("overall_acceptance_rate", 0)
        if rate < 0.05:
            recommendations.append(
                f"Very low acceptance rate ({rate:.1%}). Consider relaxing thresholds or "
                "improving generation quality."
            )
        elif rate > 0.50:
            recommendations.append(
                f"High acceptance rate ({rate:.1%}). Consider tightening filters to "
                "improve synthetic quality."
            )

    # Check rejection reasons
    if "rejection_breakdown" in rejection_analysis:
        reasons = rejection_analysis["rejection_breakdown"]

        # Check for dominant rejection reason
        for reason, data in reasons.items():
            if data["percentage"] > 40:
                if reason == "length":
                    recommendations.append(
                        f"Length filter rejecting {data['percentage']:.0f}% - "
                        "consider adjusting --min-tokens/--max-tokens."
                    )
                elif reason == "similarity":
                    recommendations.append(
                        f"Similarity filter rejecting {data['percentage']:.0f}% - "
                        "consider lowering --min-similarity threshold."
                    )
                elif reason == "classifier":
                    recommendations.append(
                        f"Classifier filter rejecting {data['percentage']:.0f}% - "
                        "consider lowering --min-classifier-confidence."
                    )
                elif reason == "knn":
                    recommendations.append(
                        f"KNN filter rejecting {data['percentage']:.0f}% - "
                        "consider lowering --filter-knn-threshold."
                    )

    # Check per-class issues
    if "per_class_analysis" in rejection_analysis:
        lowest = rejection_analysis["per_class_analysis"].get("lowest_acceptance", [])
        for cls_data in lowest:
            if cls_data["rate"] < 0.01:
                recommendations.append(
                    f"Class '{cls_data['class']}' has near-zero acceptance ({cls_data['rate']:.2%}). "
                    "May have low purity or incompatible class characteristics."
                )

    # Quality distribution recommendations
    if "distributions" in quality_analysis:
        for metric, stats in quality_analysis["distributions"].items():
            if metric == "similarity_to_centroid" and stats.get("mean", 0) < 0.30:
                recommendations.append(
                    f"Low mean similarity to centroid ({stats['mean']:.2f}). "
                    "Synthetics may be off-distribution."
                )
            if metric == "classifier_confidence" and stats.get("mean", 0) < 0.50:
                recommendations.append(
                    f"Low mean classifier confidence ({stats['mean']:.2f}). "
                    "Classifier struggles to identify synthetics as target class."
                )

    if not recommendations:
        recommendations.append("No immediate issues detected. Consider running with different seeds.")

    return recommendations


def print_analysis_report(
    rejection_analysis: Dict,
    quality_analysis: Dict,
    per_class_analysis: Dict,
    recommendations: List[str],
    metrics: Dict
):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("SYNTHETIC QUALITY ANALYSIS REPORT")
    print("=" * 70)

    # F1 improvement summary
    if "improvement" in metrics:
        imp = metrics["improvement"]
        print(f"\nF1 Improvement: {imp.get('f1_delta_pct', 0):+.3f}%")
        print(f"Accepted Synthetics: {metrics.get('synthetic_data', {}).get('accepted_count', 0)}")

    # Rejection Analysis
    print("\n" + "-" * 40)
    print("REJECTION ANALYSIS")
    print("-" * 40)

    if "summary" in rejection_analysis:
        s = rejection_analysis["summary"]
        print(f"Total Generated:    {s.get('total_generated', 0):,}")
        print(f"Total Accepted:     {s.get('total_accepted', 0):,}")
        print(f"Total Rejected:     {s.get('total_rejected', 0):,}")
        print(f"Acceptance Rate:    {s.get('overall_acceptance_rate', 0):.2%}")

    if "rejection_breakdown" in rejection_analysis:
        print("\nRejection Reasons:")
        for reason, data in rejection_analysis["rejection_breakdown"].items():
            print(f"  {reason:20s}: {data['count']:5d} ({data['percentage']:5.1f}%)")

    # Quality Metrics
    print("\n" + "-" * 40)
    print("QUALITY METRICS DISTRIBUTIONS")
    print("-" * 40)

    if "distributions" in quality_analysis:
        for metric, stats in quality_analysis["distributions"].items():
            print(f"\n{metric}:")
            print(f"  Mean: {stats.get('mean', 0):.4f} (std: {stats.get('std', 0):.4f})")
            print(f"  Range: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]")
            print(f"  Quartiles: {stats.get('q25', 0):.4f} / {stats.get('median', 0):.4f} / {stats.get('q75', 0):.4f}")

    # Per-class summary
    print("\n" + "-" * 40)
    print("PER-CLASS ACCEPTANCE RATES")
    print("-" * 40)

    if "per_class_analysis" in rejection_analysis:
        pca = rejection_analysis["per_class_analysis"]

        print("\nLowest Acceptance (problematic classes):")
        for cls_data in pca.get("lowest_acceptance", []):
            print(f"  {cls_data['class']:8s}: {cls_data['rate']:.2%}")

        print("\nHighest Acceptance:")
        for cls_data in pca.get("highest_acceptance", []):
            print(f"  {cls_data['class']:8s}: {cls_data['rate']:.2%}")

    # Recommendations
    print("\n" + "-" * 40)
    print("RECOMMENDATIONS")
    print("-" * 40)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic data quality")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--csv", default=None, help="Path to synthetic CSV file (optional)")
    parser.add_argument("--output", default=None, help="Output JSON file for analysis results")
    args = parser.parse_args()

    # Load data
    print(f"Loading metrics from: {args.metrics}")
    metrics = load_metrics(args.metrics)

    df = None
    if args.csv and Path(args.csv).exists():
        print(f"Loading synthetic CSV from: {args.csv}")
        df = load_synthetic_csv(args.csv)

    # Run analyses
    rejection_analysis = analyze_rejection_patterns(metrics)
    per_class_analysis = analyze_per_class_quality(metrics)

    if df is not None:
        quality_analysis = analyze_quality_metrics(df)
    else:
        quality_analysis = {"note": "No CSV provided, skipping per-sample analysis"}

    # Generate recommendations
    recommendations = generate_recommendations(rejection_analysis, quality_analysis)

    # Print report
    print_analysis_report(
        rejection_analysis,
        quality_analysis,
        per_class_analysis,
        recommendations,
        metrics
    )

    # Save output if requested
    if args.output:
        output = {
            "rejection_analysis": rejection_analysis,
            "quality_analysis": quality_analysis,
            "per_class_analysis": per_class_analysis,
            "recommendations": recommendations,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
