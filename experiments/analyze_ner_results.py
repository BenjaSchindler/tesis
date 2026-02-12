#!/usr/bin/env python3
"""
NER Experiment Results Analyzer

Analyzes results from exp_ner_filter_comparison.py and produces:
1. Filter ranking for NER (does cascade L1 still win?)
2. Cross-task comparison (classification vs NER filter rankings)
3. Per-entity-type analysis
4. N-shot impact analysis
5. Filtered vs unfiltered LLM comparison

Usage:
    python experiments/analyze_ner_results.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
NER_RESULTS_DIR = PROJECT_ROOT / "results" / "ner_filter_comparison"
CLASSIFICATION_RESULTS_DIR = PROJECT_ROOT / "results" / "fixed_output_count"


def load_ner_results():
    """Load NER experiment results."""
    path = NER_RESULTS_DIR / "experiment_results.json"
    if not path.exists():
        print(f"No NER results found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_classification_results():
    """Load classification experiment results for cross-task comparison."""
    path = CLASSIFICATION_RESULTS_DIR / "experiment_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================================
# ANALYSIS 1: NER FILTER RANKING
# ============================================================================

def analyze_filter_ranking(data):
    """Rank filters by NER performance."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: NER FILTER RANKING")
    print("=" * 70)

    filter_stats = defaultdict(lambda: {"deltas": [], "f1s": [], "calls": [], "accepts": []})

    for dr in data["results"]:
        for fr in dr["filter_results"]:
            name = fr["filter_name"]
            filter_stats[name]["deltas"].append(fr["f1_delta_vs_baseline"])
            filter_stats[name]["f1s"].append(fr["mean_f1"])
            filter_stats[name]["calls"].append(fr["total_llm_calls"])
            filter_stats[name]["accepts"].append(fr["avg_acceptance_rate"])

    print(f"\n{'Rank':<6} {'Filter':<16} {'Mean Delta':>12} {'Mean F1':>10} {'Win Rate':>10} {'Avg Calls':>10}")
    print("-" * 66)

    # Sort by mean delta
    sorted_filters = sorted(
        filter_stats.items(),
        key=lambda x: np.mean(x[1]["deltas"]),
        reverse=True
    )

    for rank, (name, stats) in enumerate(sorted_filters, 1):
        mean_delta = np.mean(stats["deltas"])
        mean_f1 = np.mean(stats["f1s"])
        win_rate = sum(1 for d in stats["deltas"] if d > 0) / len(stats["deltas"]) * 100
        avg_calls = np.mean(stats["calls"])

        print(f"{rank:<6} {name:<16} {mean_delta:>+12.2f}pp {mean_f1:>10.4f} {win_rate:>9.0f}% {avg_calls:>10.0f}")

    return sorted_filters


# ============================================================================
# ANALYSIS 2: CROSS-TASK COMPARISON
# ============================================================================

def analyze_cross_task(ner_data, cls_data):
    """Compare filter rankings between classification and NER."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CROSS-TASK COMPARISON (Classification vs NER)")
    print("=" * 70)

    if cls_data is None:
        print("  No classification results available for comparison.")
        return

    # Get classification filter ranking
    cls_summary = cls_data.get("summary", {})
    cls_ranking = sorted(
        cls_summary.items(),
        key=lambda x: x[1].get("mean_f1_delta", 0),
        reverse=True
    )

    # Get NER filter ranking
    ner_summary = ner_data.get("summary", {})
    ner_ranking = sorted(
        ner_summary.items(),
        key=lambda x: x[1].get("mean_f1_delta", 0),
        reverse=True
    )

    print(f"\n{'Rank':<6} {'Classification':<20} {'Delta':>10} {'NER':<20} {'Delta':>10}")
    print("-" * 68)

    max_len = max(len(cls_ranking), len(ner_ranking))
    for i in range(max_len):
        cls_name = cls_ranking[i][0] if i < len(cls_ranking) else "-"
        cls_delta = cls_ranking[i][1].get("mean_f1_delta", 0) if i < len(cls_ranking) else 0
        ner_name = ner_ranking[i][0] if i < len(ner_ranking) else "-"
        ner_delta = ner_ranking[i][1].get("mean_f1_delta", 0) if i < len(ner_ranking) else 0

        print(f"{i+1:<6} {cls_name:<20} {cls_delta:>+10.2f}pp {ner_name:<20} {ner_delta:>+10.2f}pp")

    # Check if top filter is the same
    if cls_ranking and ner_ranking:
        cls_top = cls_ranking[0][0]
        ner_top = ner_ranking[0][0]
        if cls_top == ner_top:
            print(f"\n  >> SAME TOP FILTER: {cls_top} wins in both tasks!")
        else:
            print(f"\n  >> DIFFERENT TOP FILTERS: Classification={cls_top}, NER={ner_top}")


# ============================================================================
# ANALYSIS 3: PER-ENTITY-TYPE BREAKDOWN
# ============================================================================

def analyze_per_entity_type(data):
    """Analyze which entity types benefit most from augmentation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PER-ENTITY-TYPE ANALYSIS")
    print("=" * 70)

    # Collect per-type F1 across datasets and filters
    type_improvements = defaultdict(lambda: {"baseline": [], "best_augmented": [], "best_filter": []})

    for dr in data["results"]:
        baseline_per_type = dr.get("baseline_per_type_f1", {})

        # Find best filter for each entity type
        best_per_type = {}
        for fr in dr["filter_results"]:
            for etype, f1 in fr.get("per_type_f1", {}).items():
                if etype not in best_per_type or f1 > best_per_type[etype]["f1"]:
                    best_per_type[etype] = {"f1": f1, "filter": fr["filter_name"]}

        for etype in set(list(baseline_per_type.keys()) + list(best_per_type.keys())):
            if etype in baseline_per_type:
                type_improvements[etype]["baseline"].append(baseline_per_type[etype])
            if etype in best_per_type:
                type_improvements[etype]["best_augmented"].append(best_per_type[etype]["f1"])
                type_improvements[etype]["best_filter"].append(best_per_type[etype]["filter"])

    if not type_improvements:
        print("  No per-entity-type data available.")
        return

    print(f"\n{'Entity Type':<15} {'Baseline F1':>12} {'Best Aug F1':>12} {'Delta':>10} {'Best Filter':<15}")
    print("-" * 66)

    for etype, stats in sorted(type_improvements.items()):
        if stats["baseline"] and stats["best_augmented"]:
            bl = np.mean(stats["baseline"])
            aug = np.mean(stats["best_augmented"])
            delta = (aug - bl) * 100
            # Most common best filter
            from collections import Counter
            filter_counts = Counter(stats["best_filter"])
            best_filter = filter_counts.most_common(1)[0][0]
            print(f"{etype:<15} {bl:>12.4f} {aug:>12.4f} {delta:>+10.2f}pp {best_filter:<15}")


# ============================================================================
# ANALYSIS 4: N-SHOT IMPACT
# ============================================================================

def analyze_nshot_impact(data):
    """Analyze how n-shot value affects filtering benefit."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: N-SHOT IMPACT")
    print("=" * 70)

    nshot_stats = defaultdict(lambda: {"deltas": [], "best_deltas": []})

    for dr in data["results"]:
        dataset = dr["dataset"]
        # Extract n-shot from name
        for n in [10, 25, 50]:
            if f"_{n}shot" in dataset:
                nshot = n
                break
        else:
            continue

        # Best filter delta
        best_delta = max(
            (fr["f1_delta_vs_baseline"] for fr in dr["filter_results"]),
            default=0
        )
        # None filter delta (unfiltered LLM baseline)
        none_delta = next(
            (fr["f1_delta_vs_baseline"] for fr in dr["filter_results"]
             if fr["filter_name"] == "none"),
            0
        )

        nshot_stats[nshot]["deltas"].append(none_delta)
        nshot_stats[nshot]["best_deltas"].append(best_delta)

    print(f"\n{'N-shot':<10} {'Unfiltered LLM':>16} {'Best Filter':>14} {'Filtering Gain':>16}")
    print("-" * 58)

    for nshot in sorted(nshot_stats.keys()):
        stats = nshot_stats[nshot]
        mean_none = np.mean(stats["deltas"])
        mean_best = np.mean(stats["best_deltas"])
        gain = mean_best - mean_none

        print(f"{nshot:<10} {mean_none:>+16.2f}pp {mean_best:>+14.2f}pp {gain:>+16.2f}pp")


# ============================================================================
# ANALYSIS 5: FILTERED VS UNFILTERED
# ============================================================================

def analyze_filtered_vs_unfiltered(data):
    """Compare filtered vs unfiltered LLM augmentation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: FILTERED vs UNFILTERED LLM AUGMENTATION")
    print("=" * 70)

    comparisons = []

    for dr in data["results"]:
        none_result = next(
            (fr for fr in dr["filter_results"] if fr["filter_name"] == "none"),
            None
        )
        if none_result is None:
            continue

        for fr in dr["filter_results"]:
            if fr["filter_name"] == "none":
                continue
            delta_vs_none = fr["mean_f1"] - none_result["mean_f1"]
            comparisons.append({
                "dataset": dr["dataset"],
                "filter": fr["filter_name"],
                "f1": fr["mean_f1"],
                "none_f1": none_result["mean_f1"],
                "delta_vs_none": delta_vs_none * 100,
            })

    if not comparisons:
        print("  No comparison data available.")
        return

    # Aggregate by filter
    filter_vs_none = defaultdict(list)
    for c in comparisons:
        filter_vs_none[c["filter"]].append(c["delta_vs_none"])

    print(f"\n{'Filter':<16} {'Mean vs None':>14} {'Win Rate':>10}")
    print("-" * 42)

    sorted_f = sorted(filter_vs_none.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for name, deltas in sorted_f:
        mean_d = np.mean(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas) * 100
        print(f"{name:<16} {mean_d:>+14.2f}pp {win_rate:>9.0f}%")

    # Overall: does any filter consistently beat unfiltered?
    any_wins = any(np.mean(deltas) > 0.5 for deltas in filter_vs_none.values())
    if any_wins:
        print("\n  >> CONCLUSION: Geometric filtering improves NER augmentation")
    else:
        print("\n  >> CONCLUSION: Filtering does not consistently help for NER")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("NER EXPERIMENT ANALYSIS")
    print("=" * 70)

    ner_data = load_ner_results()
    if ner_data is None:
        print("No results to analyze. Run exp_ner_filter_comparison.py first.")
        return

    cls_data = load_classification_results()

    # Run all analyses
    analyze_filter_ranking(ner_data)
    analyze_cross_task(ner_data, cls_data)
    analyze_per_entity_type(ner_data)
    analyze_nshot_impact(ner_data)
    analyze_filtered_vs_unfiltered(ner_data)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
