#!/usr/bin/env python3
"""
New Datasets Analysis Script

Analyzes results from exp_new_datasets.py covering:
  - banking77 (77 classes), clinc150 (150 classes), trec6 (6 classes)

Produces:
1. Delta vs SMOTE per method/dataset/classifier
2. Significance tests (paired t-test, effect size)
3. N-classes pattern comparison with thesis_final results
4. N-shot impact analysis
5. Seed variance check (cross-reference with thesis_final seed issue)

Usage:
    python experiments/analyze_new_datasets.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
NEW_RESULTS_PATH = PROJECT_ROOT / "results" / "new_datasets" / "results.json"
THESIS_RESULTS_PATH = PROJECT_ROOT / "results" / "thesis_final" / "final_results.json"

DATASET_N_CLASSES = {
    "clinc150": 150,
    "banking77": 77,
    "trec6": 6,
    # From thesis_final for cross-reference
    "sms_spam": 2,
    "hate_speech_davidson": 3,
    "20newsgroups": 4,
    "ag_news": 4,
    "emotion": 6,
    "dbpedia14": 14,
    "20newsgroups_20class": 20,
}


def load_results(path):
    """Load experiment results from JSON."""
    if not path.exists():
        print(f"Results not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def get_dataset_base(name):
    """Extract base dataset name from e.g. 'banking77_10shot'."""
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if name.startswith(base + "_"):
            return base
    return name


def get_n_shot(name):
    """Extract n-shot value from dataset name."""
    for n in [10, 25, 50]:
        if f"_{n}shot" in name:
            return n
    return 0


# ============================================================================
# ANALYSIS 1: OVERALL METHOD RANKING (Delta vs SMOTE)
# ============================================================================

def analyze_method_ranking(results):
    """Rank augmentation methods by delta vs SMOTE."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: METHOD RANKING (Delta vs SMOTE)")
    print("=" * 70)

    # Group by (dataset, classifier, seed) -> {method: f1}
    grouped = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        grouped[key][r["augmentation_method"]] = r["f1_macro"]

    # Compute deltas vs SMOTE
    method_deltas = defaultdict(list)
    for key, methods in grouped.items():
        if "smote" not in methods:
            continue
        smote_f1 = methods["smote"]
        for method, f1 in methods.items():
            if method == "smote":
                continue
            delta = (f1 - smote_f1) * 100  # in pp
            method_deltas[method].append(delta)

    print(f"\n{'Method':<20} {'Mean Delta':>12} {'Std':>8} {'Win Rate':>10} {'N':>6}")
    print("-" * 58)

    for method in sorted(method_deltas.keys(),
                         key=lambda m: np.mean(method_deltas[m]), reverse=True):
        deltas = method_deltas[method]
        mean_d = np.mean(deltas)
        std_d = np.std(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas) * 100
        print(f"{method:<20} {mean_d:>+12.2f}pp {std_d:>8.2f} {win_rate:>9.1f}% {len(deltas):>6}")

    return method_deltas


# ============================================================================
# ANALYSIS 2: SIGNIFICANCE TESTS
# ============================================================================

def analyze_significance(results):
    """Paired t-tests and effect sizes for each method vs SMOTE."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: STATISTICAL SIGNIFICANCE (vs SMOTE)")
    print("=" * 70)

    # Group by (dataset, classifier) -> average across seeds -> paired values
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["dataset"], r["classifier"])
        grouped[key][r["augmentation_method"]].append(r["f1_macro"])

    # Average across seeds for each (dataset, classifier)
    pair_data = defaultdict(lambda: {"smote": [], "other": []})
    for key, methods in grouped.items():
        if "smote" not in methods:
            continue
        smote_mean = np.mean(methods["smote"])

        for method, f1_list in methods.items():
            if method == "smote":
                continue
            method_mean = np.mean(f1_list)
            pair_key = method
            pair_data[pair_key]["smote"].append(smote_mean)
            pair_data[pair_key]["other"].append(method_mean)

    print(f"\n{'Method':<20} {'Mean Delta':>12} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10} {'Sig':>6}")
    print("-" * 72)

    for method in sorted(pair_data.keys()):
        smote_vals = np.array(pair_data[method]["smote"])
        other_vals = np.array(pair_data[method]["other"])
        diffs = other_vals - smote_vals

        mean_diff = np.mean(diffs) * 100
        t_stat, p_value = stats.ttest_rel(other_vals, smote_vals)

        # Cohen's d for paired samples
        d_std = np.std(diffs, ddof=1)
        cohens_d = np.mean(diffs) / d_std if d_std > 0 else 0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

        print(f"{method:<20} {mean_diff:>+12.2f}pp {t_stat:>10.3f} {p_value:>12.6f} {cohens_d:>10.3f} {sig:>6}")

    return pair_data


# ============================================================================
# ANALYSIS 3: PER-DATASET BREAKDOWN
# ============================================================================

def analyze_per_dataset(results):
    """Break down results by dataset."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PER-DATASET BREAKDOWN (soft_weighted vs SMOTE)")
    print("=" * 70)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["dataset"]][r["augmentation_method"]].append(r["f1_macro"])

    print(f"\n{'Dataset':<30} {'N_cls':>6} {'SMOTE':>10} {'Soft_W':>10} {'Delta':>10} {'Binary':>10} {'No_Aug':>10}")
    print("-" * 88)

    dataset_rows = []
    for dataset in sorted(grouped.keys()):
        methods = grouped[dataset]
        base = get_dataset_base(dataset)
        n_classes = DATASET_N_CLASSES.get(base, 0)

        smote_f1 = np.mean(methods.get("smote", [0]))
        soft_f1 = np.mean(methods.get("soft_weighted", [0]))
        binary_f1 = np.mean(methods.get("binary_filter", [0]))
        no_aug_f1 = np.mean(methods.get("no_augmentation", [0]))
        delta = (soft_f1 - smote_f1) * 100

        print(f"{dataset:<30} {n_classes:>6} {smote_f1:>10.4f} {soft_f1:>10.4f} "
              f"{delta:>+10.2f}pp {binary_f1:>10.4f} {no_aug_f1:>10.4f}")

        dataset_rows.append({
            "dataset": dataset,
            "n_classes": n_classes,
            "n_shot": get_n_shot(dataset),
            "delta_soft_vs_smote": delta,
        })

    return dataset_rows


# ============================================================================
# ANALYSIS 4: N-CLASSES PATTERN (cross-reference with thesis_final)
# ============================================================================

def analyze_nclasses_pattern(new_results, thesis_results):
    """Compare n_classes effect across new and thesis_final datasets."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: N-CLASSES PATTERN (New Datasets + Thesis Final)")
    print("=" * 70)

    all_results = []

    # Process new datasets
    grouped = defaultdict(lambda: defaultdict(list))
    for r in new_results:
        key = (r["dataset"], r["classifier"])
        grouped[key][r["augmentation_method"]].append(r["f1_macro"])

    for key, methods in grouped.items():
        dataset, classifier = key
        if "smote" not in methods or "soft_weighted" not in methods:
            continue
        delta = (np.mean(methods["soft_weighted"]) - np.mean(methods["smote"])) * 100
        base = get_dataset_base(dataset)
        all_results.append({
            "source": "new_datasets",
            "dataset_base": base,
            "n_classes": DATASET_N_CLASSES.get(base, 0),
            "n_shot": get_n_shot(dataset),
            "classifier": classifier,
            "delta": delta,
        })

    # Process thesis_final results
    if thesis_results:
        thesis_grouped = defaultdict(lambda: defaultdict(list))
        for r in thesis_results:
            key = (r["dataset"], r["classifier"])
            thesis_grouped[key][r["augmentation_method"]].append(r["f1_macro"])

        for key, methods in thesis_grouped.items():
            dataset, classifier = key
            if "smote" not in methods or "soft_weighted" not in methods:
                continue
            delta = (np.mean(methods["soft_weighted"]) - np.mean(methods["smote"])) * 100
            base = get_dataset_base(dataset)
            all_results.append({
                "source": "thesis_final",
                "dataset_base": base,
                "n_classes": DATASET_N_CLASSES.get(base, 0),
                "n_shot": get_n_shot(dataset),
                "classifier": classifier,
                "delta": delta,
            })

    if not all_results:
        print("  No cross-reference data available.")
        return

    # Aggregate by n_classes
    nclass_groups = defaultdict(list)
    for r in all_results:
        nclass_groups[r["n_classes"]].append(r["delta"])

    print(f"\n{'N_Classes':>10} {'Mean Delta':>12} {'Std':>8} {'Win Rate':>10} {'N Pairs':>10} {'Source Datasets'}")
    print("-" * 80)

    for n_cls in sorted(nclass_groups.keys()):
        deltas = nclass_groups[n_cls]
        mean_d = np.mean(deltas)
        std_d = np.std(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas) * 100

        # Which datasets have this n_classes
        ds_names = set(r["dataset_base"] for r in all_results if r["n_classes"] == n_cls)
        ds_str = ", ".join(sorted(ds_names))

        print(f"{n_cls:>10} {mean_d:>+12.2f}pp {std_d:>8.2f} {win_rate:>9.1f}% {len(deltas):>10} {ds_str}")

    # Correlation test
    xs = [r["n_classes"] for r in all_results]
    ys = [r["delta"] for r in all_results]
    if len(set(xs)) >= 3:
        corr, p_corr = stats.spearmanr(xs, ys)
        print(f"\nSpearman correlation (n_classes vs delta): r={corr:.3f}, p={p_corr:.4f}")
        if p_corr < 0.05:
            direction = "MORE classes = LARGER gains" if corr > 0 else "MORE classes = SMALLER gains"
            print(f"  >> SIGNIFICANT: {direction}")
        else:
            print("  >> Not significant — n_classes effect unclear across this range")


# ============================================================================
# ANALYSIS 5: N-SHOT IMPACT
# ============================================================================

def analyze_nshot(results):
    """Analyze how n-shot affects augmentation benefit."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: N-SHOT IMPACT")
    print("=" * 70)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        grouped[key][r["augmentation_method"]] = r["f1_macro"]

    nshot_deltas = defaultdict(lambda: defaultdict(list))
    for key, methods in grouped.items():
        dataset = key[0]
        n_shot = get_n_shot(dataset)
        if "smote" not in methods:
            continue
        smote_f1 = methods["smote"]
        for method in ["binary_filter", "soft_weighted"]:
            if method in methods:
                delta = (methods[method] - smote_f1) * 100
                nshot_deltas[n_shot][method].append(delta)

    print(f"\n{'N-shot':<10} {'binary_filter':>16} {'soft_weighted':>16} {'N':>6}")
    print("-" * 50)

    for n_shot in sorted(nshot_deltas.keys()):
        bf = nshot_deltas[n_shot].get("binary_filter", [])
        sw = nshot_deltas[n_shot].get("soft_weighted", [])
        bf_mean = np.mean(bf) if bf else 0
        sw_mean = np.mean(sw) if sw else 0
        n = len(sw) if sw else len(bf)
        print(f"{n_shot:<10} {bf_mean:>+16.2f}pp {sw_mean:>+16.2f}pp {n:>6}")


# ============================================================================
# ANALYSIS 6: SEED VARIANCE CHECK
# ============================================================================

def analyze_seed_variance(results):
    """Check if the seed variance issue from thesis_final also affects new datasets."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: SEED VARIANCE CHECK")
    print("=" * 70)
    print("(Checking for identical F1 across seeds — known issue in thesis_final)")

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["dataset"], r["classifier"], r["augmentation_method"])
        grouped[key]["f1s"].append(r["f1_macro"])
        grouped[key]["seeds"].append(r["seed"])

    zero_var_count = 0
    nonzero_var_count = 0
    zero_var_groups = []

    for key, data in grouped.items():
        f1s = data["f1s"]
        if len(f1s) < 2:
            continue
        variance = np.var(f1s)
        if variance == 0.0:
            zero_var_count += 1
            zero_var_groups.append(key)
        else:
            nonzero_var_count += 1

    total = zero_var_count + nonzero_var_count
    print(f"\n  Total (dataset, classifier, method) groups: {total}")
    print(f"  Zero variance across seeds: {zero_var_count} ({zero_var_count/total*100:.0f}%)")
    print(f"  Nonzero variance: {nonzero_var_count} ({nonzero_var_count/total*100:.0f}%)")

    # Break down by method
    method_var = defaultdict(lambda: {"zero": 0, "nonzero": 0})
    for key in grouped:
        dataset, classifier, method = key
        f1s = grouped[key]["f1s"]
        if len(f1s) < 2:
            continue
        if np.var(f1s) == 0.0:
            method_var[method]["zero"] += 1
        else:
            method_var[method]["nonzero"] += 1

    print(f"\n  {'Method':<20} {'Zero Var':>10} {'Nonzero Var':>12} {'% Zero':>8}")
    print("  " + "-" * 52)
    for method in sorted(method_var.keys()):
        z = method_var[method]["zero"]
        nz = method_var[method]["nonzero"]
        pct = z / (z + nz) * 100 if (z + nz) > 0 else 0
        print(f"  {method:<20} {z:>10} {nz:>12} {pct:>7.0f}%")

    # Break down zero-var by classifier for LLM methods
    print("\n  Zero-variance groups by classifier (LLM methods only):")
    clf_var = defaultdict(lambda: {"zero": 0, "nonzero": 0})
    for key in grouped:
        dataset, classifier, method = key
        if method not in ("binary_filter", "soft_weighted"):
            continue
        f1s = grouped[key]["f1s"]
        if len(f1s) < 2:
            continue
        if np.var(f1s) == 0.0:
            clf_var[classifier]["zero"] += 1
        else:
            clf_var[classifier]["nonzero"] += 1

    print(f"  {'Classifier':<25} {'Zero Var':>10} {'Nonzero Var':>12}")
    print("  " + "-" * 49)
    for clf in sorted(clf_var.keys()):
        z = clf_var[clf]["zero"]
        nz = clf_var[clf]["nonzero"]
        print(f"  {clf:<25} {z:>10} {nz:>12}")

    if zero_var_count > 0:
        print("\n  >> WARNING: Seed variance issue CONFIRMED for new datasets.")
        print("     LLM-based methods with deterministic classifiers produce identical")
        print("     results across seeds because cached LLM generations are seed-independent.")
    else:
        print("\n  >> No seed variance issue detected.")


# ============================================================================
# ANALYSIS 7: PER-CLASSIFIER BREAKDOWN
# ============================================================================

def analyze_per_classifier(results):
    """Break down results by classifier."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: PER-CLASSIFIER BREAKDOWN (soft_weighted vs SMOTE)")
    print("=" * 70)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        grouped[key][r["augmentation_method"]] = r["f1_macro"]

    clf_deltas = defaultdict(list)
    for key, methods in grouped.items():
        classifier = key[1]
        if "smote" not in methods or "soft_weighted" not in methods:
            continue
        delta = (methods["soft_weighted"] - methods["smote"]) * 100
        clf_deltas[classifier].append(delta)

    print(f"\n{'Classifier':<25} {'Mean Delta':>12} {'Win Rate':>10} {'N':>6}")
    print("-" * 55)

    for clf in sorted(clf_deltas.keys(),
                      key=lambda c: np.mean(clf_deltas[c]), reverse=True):
        deltas = clf_deltas[clf]
        mean_d = np.mean(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas) * 100
        print(f"{clf:<25} {mean_d:>+12.2f}pp {win_rate:>9.1f}% {len(deltas):>6}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("NEW DATASETS ANALYSIS: BANKING77, CLINC150, TREC6")
    print("=" * 70)

    new_data = load_results(NEW_RESULTS_PATH)
    if new_data is None:
        print("No new dataset results found. Run exp_new_datasets.py first.")
        return

    new_results = new_data["results"]
    n_results = len(new_results)
    datasets = sorted(set(r["dataset"] for r in new_results))
    methods = sorted(set(r["augmentation_method"] for r in new_results))
    classifiers = sorted(set(r["classifier"] for r in new_results))
    seeds = sorted(set(r["seed"] for r in new_results))

    print(f"\nLoaded {n_results} results")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Methods ({len(methods)}): {', '.join(methods)}")
    print(f"Classifiers ({len(classifiers)}): {', '.join(classifiers)}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    # Load thesis_final for cross-reference
    thesis_data = load_results(THESIS_RESULTS_PATH)
    thesis_results = thesis_data["results"] if thesis_data else None

    # Run all analyses
    analyze_method_ranking(new_results)
    analyze_significance(new_results)
    analyze_per_dataset(new_results)
    analyze_nclasses_pattern(new_results, thesis_results)
    analyze_nshot(new_results)
    analyze_seed_variance(new_results)
    analyze_per_classifier(new_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
