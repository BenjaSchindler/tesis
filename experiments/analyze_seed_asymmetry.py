#!/usr/bin/env python3
"""
Seed Asymmetry Analysis (MUST-RUN A2)

Analyzes the cross-seed variance issue in thesis_final results:
- LLM-based methods (binary_filter, soft_weighted) use cached generations
  that don't vary across seeds, so linear classifiers produce IDENTICAL F1.
- Only stochastic classifiers (MLP, RandomForest) show genuine seed variance.

Produces:
1. Per-(method, classifier) cross-seed standard deviation
2. Breakdown showing which method×classifier combos have zero variance
3. Suggested limitations paragraph for the thesis

Usage:
    python experiments/analyze_seed_asymmetry.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "thesis_final" / "final_results.json"

# Methods that use LLM cache (seed-invariant generation)
LLM_METHODS = {"binary_filter", "soft_weighted", "back_translation"}
# Methods that genuinely vary across seeds
SEED_VARYING_METHODS = {"smote", "random_oversample", "eda", "no_augmentation"}
# Deterministic classifiers (same input → same output regardless of random_state)
DETERMINISTIC_CLASSIFIERS = {"logistic_regression", "svc_linear", "ridge"}
# Stochastic classifiers
STOCHASTIC_CLASSIFIERS = {"random_forest", "mlp"}


def load_results():
    if not RESULTS_PATH.exists():
        print(f"Results not found at {RESULTS_PATH}")
        return None
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data["results"]


# ============================================================================
# ANALYSIS 1: Cross-seed standard deviation per (dataset, classifier, method)
# ============================================================================

def analyze_cross_seed_variance(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: CROSS-SEED VARIANCE BY METHOD AND CLASSIFIER")
    print("=" * 70)

    # Group by (dataset, classifier, method) -> list of F1 across seeds
    grouped = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["classifier"], r["augmentation_method"])
        grouped[key].append(r["f1_macro"])

    # Compute stats per (classifier, method)
    clf_method_stds = defaultdict(list)
    zero_var_count = defaultdict(int)
    total_count = defaultdict(int)

    for (dataset, clf, method), f1s in grouped.items():
        if len(f1s) < 2:
            continue
        std = np.std(f1s, ddof=1)
        key = (clf, method)
        clf_method_stds[key].append(std)
        total_count[key] += 1
        if std == 0.0:
            zero_var_count[key] += 1

    # Print table
    classifiers = sorted(set(k[0] for k in clf_method_stds))
    methods = sorted(set(k[1] for k in clf_method_stds))

    print(f"\nMean cross-seed std (×10⁴) per (classifier, method):")
    print(f"\n{'Classifier':<25}", end="")
    for m in methods:
        print(f" {m[:12]:>12}", end="")
    print()
    print("-" * (25 + 13 * len(methods)))

    for clf in classifiers:
        print(f"{clf:<25}", end="")
        for m in methods:
            key = (clf, m)
            if key in clf_method_stds:
                mean_std = np.mean(clf_method_stds[key]) * 10000
                print(f" {mean_std:>12.2f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    return clf_method_stds, zero_var_count, total_count


# ============================================================================
# ANALYSIS 2: Zero-variance breakdown
# ============================================================================

def analyze_zero_variance(clf_method_stds, zero_var_count, total_count):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ZERO-VARIANCE GROUPS (IDENTICAL F1 ACROSS ALL 5 SEEDS)")
    print("=" * 70)

    classifiers = sorted(set(k[0] for k in clf_method_stds))
    methods = sorted(set(k[1] for k in clf_method_stds))

    print(f"\n% of (dataset) groups with ZERO variance across 5 seeds:")
    print(f"\n{'Classifier':<25}", end="")
    for m in methods:
        print(f" {m[:12]:>12}", end="")
    print()
    print("-" * (25 + 13 * len(methods)))

    for clf in classifiers:
        print(f"{clf:<25}", end="")
        for m in methods:
            key = (clf, m)
            if key in total_count and total_count[key] > 0:
                pct = zero_var_count[key] / total_count[key] * 100
                print(f" {pct:>11.0f}%", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # Summary
    llm_det = sum(zero_var_count[(c, m)]
                  for c in DETERMINISTIC_CLASSIFIERS
                  for m in LLM_METHODS
                  if (c, m) in zero_var_count)
    llm_det_total = sum(total_count[(c, m)]
                        for c in DETERMINISTIC_CLASSIFIERS
                        for m in LLM_METHODS
                        if (c, m) in total_count)

    llm_stoch = sum(zero_var_count[(c, m)]
                    for c in STOCHASTIC_CLASSIFIERS
                    for m in LLM_METHODS
                    if (c, m) in zero_var_count)
    llm_stoch_total = sum(total_count[(c, m)]
                          for c in STOCHASTIC_CLASSIFIERS
                          for m in LLM_METHODS
                          if (c, m) in total_count)

    trad_det = sum(zero_var_count[(c, m)]
                   for c in DETERMINISTIC_CLASSIFIERS
                   for m in SEED_VARYING_METHODS
                   if (c, m) in zero_var_count)
    trad_det_total = sum(total_count[(c, m)]
                         for c in DETERMINISTIC_CLASSIFIERS
                         for m in SEED_VARYING_METHODS
                         if (c, m) in total_count)

    print(f"\n  Summary:")
    if llm_det_total > 0:
        print(f"  LLM methods + deterministic classifiers: {llm_det}/{llm_det_total} "
              f"({llm_det/llm_det_total*100:.0f}%) have zero variance")
    if llm_stoch_total > 0:
        print(f"  LLM methods + stochastic classifiers:    {llm_stoch}/{llm_stoch_total} "
              f"({llm_stoch/llm_stoch_total*100:.0f}%) have zero variance")
    if trad_det_total > 0:
        print(f"  Traditional + deterministic classifiers:  {trad_det}/{trad_det_total} "
              f"({trad_det/trad_det_total*100:.0f}%) have zero variance")


# ============================================================================
# ANALYSIS 3: Impact on statistical tests
# ============================================================================

def analyze_stat_impact(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: IMPACT ON STATISTICAL TESTS")
    print("=" * 70)
    print("\nComparing significance when including vs excluding zero-variance combos:")

    # Paired test: soft_weighted vs smote, grouped by (dataset, classifier)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["dataset"], r["classifier"])
        grouped[key][r["augmentation_method"]].append(r["f1_macro"])

    for label, clf_filter in [("ALL classifiers", None),
                              ("Stochastic only (MLP+RF)", STOCHASTIC_CLASSIFIERS),
                              ("Deterministic only (LR+SVC+Ridge)", DETERMINISTIC_CLASSIFIERS)]:
        smote_means = []
        soft_means = []
        for (dataset, clf), methods in grouped.items():
            if clf_filter and clf not in clf_filter:
                continue
            if "smote" not in methods or "soft_weighted" not in methods:
                continue
            smote_means.append(np.mean(methods["smote"]))
            soft_means.append(np.mean(methods["soft_weighted"]))

        if len(smote_means) < 3:
            print(f"\n  {label}: insufficient data")
            continue

        smote_arr = np.array(smote_means)
        soft_arr = np.array(soft_means)
        diffs = soft_arr - smote_arr
        t_stat, p_value = stats.ttest_rel(soft_arr, smote_arr)
        mean_delta = np.mean(diffs) * 100
        d_std = np.std(diffs, ddof=1)
        cohens_d = np.mean(diffs) / d_std if d_std > 0 else float('inf')
        win_rate = np.mean(diffs > 0) * 100

        print(f"\n  {label} (N={len(smote_means)} pairs):")
        print(f"    Mean delta: {mean_delta:+.2f}pp")
        print(f"    t={t_stat:.3f}, p={p_value:.6f}")
        print(f"    Cohen's d={cohens_d:.3f}")
        print(f"    Win rate={win_rate:.1f}%")


# ============================================================================
# ANALYSIS 4: What the variance actually captures
# ============================================================================

def analyze_variance_sources(results):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: WHAT CROSS-SEED VARIANCE ACTUALLY MEASURES")
    print("=" * 70)

    grouped = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["classifier"], r["augmentation_method"])
        grouped[key].append(r["f1_macro"])

    # For each classifier, compute mean std across datasets for LLM vs traditional
    for clf_type, clf_set in [("Deterministic", DETERMINISTIC_CLASSIFIERS),
                              ("Stochastic", STOCHASTIC_CLASSIFIERS)]:
        print(f"\n  {clf_type} classifiers:")
        for method_type, method_set in [("LLM-based", LLM_METHODS),
                                        ("Traditional", SEED_VARYING_METHODS)]:
            stds = []
            for (dataset, clf, method), f1s in grouped.items():
                if clf in clf_set and method in method_set and len(f1s) >= 2:
                    stds.append(np.std(f1s, ddof=1))
            if stds:
                mean_std = np.mean(stds) * 100
                max_std = np.max(stds) * 100
                pct_zero = np.mean(np.array(stds) == 0) * 100
                print(f"    {method_type:<15}: mean_std={mean_std:.4f}pp, "
                      f"max_std={max_std:.4f}pp, %zero={pct_zero:.0f}%")


# ============================================================================
# SUGGESTED LIMITATIONS TEXT
# ============================================================================

def print_limitations_text():
    print("\n" + "=" * 70)
    print("SUGGESTED LIMITATIONS PARAGRAPH")
    print("=" * 70)
    print("""
A methodological note concerns seed variation in our experiments. While we
report results averaged over 5 random seeds, the LLM-based augmentation
methods (binary filtering and soft weighting) share the same cached LLM
generations across seeds. Consequently, the cross-seed variation for these
methods reflects only classifier initialization sensitivity, not generation
stochasticity. For deterministic classifiers (Logistic Regression, SVC, Ridge),
this results in identical F1 scores across seeds. The observed variance
therefore comes exclusively from stochastic classifiers (MLP, Random Forest).
Our paired statistical tests remain valid because they compare methods within
the same (dataset, classifier) pair, where both the LLM-based and baseline
methods share the same train/test split. However, we acknowledge that the
reported confidence intervals for LLM-based methods underestimate the true
variance that would be observed with independent LLM generations per seed.
The embedding ablation experiment (Section X.X), which tests 4 different
embedding models, provides complementary evidence of robustness to input
variation.
""")


def main():
    print("=" * 70)
    print("SEED ASYMMETRY ANALYSIS — THESIS FINAL EXPERIMENT")
    print("=" * 70)

    results = load_results()
    if results is None:
        return

    n_results = len(results)
    seeds = sorted(set(r["seed"] for r in results))
    methods = sorted(set(r["augmentation_method"] for r in results))
    classifiers = sorted(set(r["classifier"] for r in results))

    print(f"\nLoaded {n_results} results")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"Classifiers: {classifiers}")

    stds, zero_var, total = analyze_cross_seed_variance(results)
    analyze_zero_variance(stds, zero_var, total)
    analyze_stat_impact(results)
    analyze_variance_sources(results)
    print_limitations_text()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
