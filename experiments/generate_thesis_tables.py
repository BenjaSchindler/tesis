#!/usr/bin/env python3
"""Generate LaTeX tables for thesis from modern baselines JSON results.

Produces per-classifier and per-nshot breakdown tables that require
aggregation from the raw results JSON.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "modern_baselines" / "final_results.json"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"


def load_results():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return data["results"]


def compute_paired_deltas(results, group_keys, method_a="soft_weighted", method_b="smote"):
    """Compute paired deltas for method_a vs method_b, grouped by group_keys."""
    grouped = defaultdict(dict)
    for r in results:
        key = tuple(r[k] for k in group_keys)
        grouped[key][r["augmentation_method"]] = r["f1_macro"]

    deltas = []
    for key, methods in grouped.items():
        if method_a in methods and method_b in methods:
            deltas.append(methods[method_a] - methods[method_b])
    return np.array(deltas)


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def generate_classifier_table(results):
    """Table 2: Per-classifier breakdown for soft_weighted and binary_filter vs SMOTE."""
    classifiers = ["logistic_regression", "svc_linear", "ridge"]
    clf_labels = {
        "logistic_regression": "Logistic Regression",
        "svc_linear": "SVC (linear)",
        "ridge": "Ridge",
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance by Classifier: Geometric Filtering vs SMOTE}")
    lines.append(r"\label{tab:modern_classifier}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{3}{c}{\textbf{Soft Weighted}} & \multicolumn{3}{c}{\textbf{Binary Filter}} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(
        r"\textbf{Classifier} & \textbf{$\Delta$ (pp)} & \textbf{$d$} & \textbf{Win\%} "
        r"& \textbf{$\Delta$ (pp)} & \textbf{$d$} & \textbf{Win\%} \\"
    )
    lines.append(r"\midrule")

    for clf in classifiers:
        clf_results = [r for r in results if r["classifier"] == clf]
        group_keys = ["dataset", "seed"]

        # soft_weighted vs smote
        sw_deltas = compute_paired_deltas(clf_results, group_keys, "soft_weighted", "smote")
        sw_mean = sw_deltas.mean() * 100
        sw_win = (sw_deltas > 0).mean() * 100
        sw_t, sw_p = stats.ttest_1samp(sw_deltas, 0) if len(sw_deltas) > 1 else (0, 1)
        sw_d = sw_deltas.mean() / sw_deltas.std() if sw_deltas.std() > 0 else 0

        # binary_filter vs smote
        bf_deltas = compute_paired_deltas(clf_results, group_keys, "binary_filter", "smote")
        bf_mean = bf_deltas.mean() * 100
        bf_win = (bf_deltas > 0).mean() * 100
        bf_t, bf_p = stats.ttest_1samp(bf_deltas, 0) if len(bf_deltas) > 1 else (0, 1)
        bf_d = bf_deltas.mean() / bf_deltas.std() if bf_deltas.std() > 0 else 0

        label = clf_labels[clf]
        sw_stars = sig_stars(sw_p)
        bf_stars = sig_stars(bf_p)
        lines.append(
            f"{label} & +{sw_mean:.2f}{sw_stars} & {sw_d:.2f} & {sw_win:.1f}\\% "
            f"& +{bf_mean:.2f}{bf_stars} & {bf_d:.2f} & {bf_win:.1f}\\% \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\small Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$ (paired t-test)} \\"
    )
    lines.append(
        r"\multicolumn{7}{l}{\small $n$ = 63 paired comparisons per classifier (21 datasets $\times$ 3 seeds)} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_nshot_table(results):
    """Table 3: Per-nshot breakdown for soft_weighted and binary_filter vs SMOTE."""
    nshots = [10, 25, 50]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance by N-Shot: Geometric Filtering vs SMOTE}")
    lines.append(r"\label{tab:modern_nshot}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{3}{c}{\textbf{Soft Weighted}} & \multicolumn{3}{c}{\textbf{Binary Filter}} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(
        r"\textbf{N-shot} & \textbf{$\Delta$ (pp)} & \textbf{$d$} & \textbf{Win\%} "
        r"& \textbf{$\Delta$ (pp)} & \textbf{$d$} & \textbf{Win\%} \\"
    )
    lines.append(r"\midrule")

    for ns in nshots:
        ns_results = [r for r in results if r["n_shot"] == ns]
        group_keys = ["dataset", "classifier", "seed"]

        # soft_weighted vs smote
        sw_deltas = compute_paired_deltas(ns_results, group_keys, "soft_weighted", "smote")
        sw_mean = sw_deltas.mean() * 100
        sw_win = (sw_deltas > 0).mean() * 100
        sw_t, sw_p = stats.ttest_1samp(sw_deltas, 0) if len(sw_deltas) > 1 else (0, 1)
        sw_d = sw_deltas.mean() / sw_deltas.std() if sw_deltas.std() > 0 else 0

        # binary_filter vs smote
        bf_deltas = compute_paired_deltas(ns_results, group_keys, "binary_filter", "smote")
        bf_mean = bf_deltas.mean() * 100
        bf_win = (bf_deltas > 0).mean() * 100
        bf_t, bf_p = stats.ttest_1samp(bf_deltas, 0) if len(bf_deltas) > 1 else (0, 1)
        bf_d = bf_deltas.mean() / bf_deltas.std() if bf_deltas.std() > 0 else 0

        sw_stars = sig_stars(sw_p)
        bf_stars = sig_stars(bf_p)
        lines.append(
            f"{ns} & +{sw_mean:.2f}{sw_stars} & {sw_d:.2f} & {sw_win:.1f}\\% "
            f"& +{bf_mean:.2f}{bf_stars} & {bf_d:.2f} & {bf_win:.1f}\\% \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\small Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$ (paired t-test)} \\"
    )
    lines.append(
        r"\multicolumn{7}{l}{\small Diminishing returns pattern: largest gains at 10-shot, smallest at 50-shot} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("Loading results...")
    results = load_results()
    print(f"  {len(results)} records loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 2: Per-classifier
    clf_table = generate_classifier_table(results)
    out_path = OUTPUT_DIR / "tab_modern_classifier_breakdown.tex"
    out_path.write_text(clf_table)
    print(f"\n  Written: {out_path}")
    print(clf_table)

    # Table 3: Per-nshot
    nshot_table = generate_nshot_table(results)
    out_path = OUTPUT_DIR / "tab_modern_nshot_breakdown.tex"
    out_path.write_text(nshot_table)
    print(f"\n  Written: {out_path}")
    print(nshot_table)


if __name__ == "__main__":
    main()
