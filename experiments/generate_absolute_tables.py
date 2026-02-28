#!/usr/bin/env python3
"""Generate LaTeX tables with ABSOLUTE macro F1 values (not just deltas).

Reads from thesis_final and modern_baselines JSON results.
Produces 3 tables:
  1. tab_absolute_f1_by_method.tex — Absolute F1 + delta for each method
  2. tab_absolute_f1_by_nshot.tex — Absolute F1 by n-shot level
  3. tab_absolute_f1_by_dataset.tex — Absolute F1 by dataset
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
THESIS_RESULTS = PROJECT_ROOT / "results" / "thesis_final" / "final_results.json"
MODERN_RESULTS = PROJECT_ROOT / "results" / "modern_baselines" / "final_results.json"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    """Bootstrap confidence interval for the mean."""
    if len(data) < 2:
        return (0.0, 0.0)
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return (lo, hi)


def generate_method_table(results):
    """Table: Absolute F1 + delta for each augmentation method."""
    methods_order = [
        "no_augmentation", "smote", "random_oversample", "eda",
        "back_translation", "binary_filter", "soft_weighted",
    ]
    method_labels = {
        "no_augmentation": "Sin aumentación",
        "smote": "SMOTE",
        "random_oversample": "Sobremuestreo aleatorio",
        "eda": "EDA",
        "back_translation": "Traducción inversa",
        "binary_filter": "Filtrado binario",
        "soft_weighted": "Ponderación suave",
    }

    # Collect absolute F1 per method
    method_f1s = defaultdict(list)
    for r in results:
        method_f1s[r["augmentation_method"]].append(r["f1_macro"])

    # Compute paired deltas vs SMOTE
    group_keys = ["dataset", "n_shot", "classifier", "seed"]
    grouped = defaultdict(dict)
    for r in results:
        key = tuple(r[k] for k in group_keys)
        grouped[key][r["augmentation_method"]] = r["f1_macro"]

    method_deltas = defaultdict(list)
    for key, methods in grouped.items():
        if "smote" not in methods:
            continue
        smote_f1 = methods["smote"]
        for m in methods_order:
            if m in methods and m != "smote":
                method_deltas[m].append(methods[m] - smote_f1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Rendimiento absoluto y relativo de cada método de aumentación. "
                 r"F1 absoluto promediado sobre todas las configuraciones (datasets $\times$ "
                 r"clasificadores $\times$ semillas). $\Delta$ F1 calculado respecto a SMOTE.}")
    lines.append(r"\label{tab:absolute_f1_by_method}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Método} & \textbf{F1 Macro} & \textbf{$\pm$ Std} & "
                 r"\textbf{$\Delta$ vs SMOTE} & \textbf{IC 95\%} & \textbf{$d$} & \textbf{Victoria} \\")
    lines.append(r"\midrule")

    for m in methods_order:
        f1s = np.array(method_f1s.get(m, []))
        label = method_labels.get(m, m)
        f1_mean = f1s.mean() * 100
        f1_std = f1s.std() * 100

        if m == "smote":
            lines.append(f"{label} & {f1_mean:.2f} & {f1_std:.2f} & --- & --- & --- & --- \\\\")
        elif m in method_deltas and len(method_deltas[m]) > 0:
            deltas = np.array(method_deltas[m])
            delta_mean = deltas.mean() * 100
            ci_lo, ci_hi = bootstrap_ci(deltas * 100)
            win_rate = (deltas > 0).mean() * 100
            d_cohen = deltas.mean() / deltas.std() if deltas.std() > 0 else 0
            _, p_val = stats.ttest_1samp(deltas, 0) if len(deltas) > 1 else (0, 1)
            n_comparisons = len(methods_order) - 1  # Bonferroni
            p_bonf = min(p_val * n_comparisons, 1.0)
            stars = sig_stars(p_bonf)

            sign = "+" if delta_mean >= 0 else ""
            lines.append(
                f"{label} & {f1_mean:.2f} & {f1_std:.2f} & "
                f"{sign}{delta_mean:.2f}{stars} & [{ci_lo:.2f}, {ci_hi:.2f}] & "
                f"{d_cohen:.2f} & {win_rate:.1f}\\% \\\\"
            )
        else:
            lines.append(f"{label} & {f1_mean:.2f} & {f1_std:.2f} & --- & --- & --- & --- \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\small Significancia: * $p<0.05$, ** $p<0.01$, "
                 r"*** $p<0.001$ (t pareada, Bonferroni)} \\")
    lines.append(r"\multicolumn{7}{l}{\small F1 Macro expresado como porcentaje. "
                 r"$n$ = " + str(len(method_deltas.get("soft_weighted", []))) + r" comparaciones pareadas.} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_nshot_table(results):
    """Table: Absolute F1 by n-shot for key methods."""
    nshots = [10, 25, 50]
    key_methods = ["no_augmentation", "smote", "binary_filter", "soft_weighted"]
    method_labels = {
        "no_augmentation": "Sin aum.",
        "smote": "SMOTE",
        "binary_filter": "Filt. binario",
        "soft_weighted": "Pond. suave",
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Macro F1 absoluto por número de ejemplos por clase. "
                 r"Valores promediados sobre todos los datasets, clasificadores y semillas. "
                 r"La columna $\Delta$ indica la mejora de la ponderación suave respecto a SMOTE.}")
    lines.append(r"\label{tab:absolute_f1_by_nshot}")
    lines.append(r"\begin{tabular}{rccccr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{N-shot} & \textbf{Sin aum.} & \textbf{SMOTE} & "
                 r"\textbf{Filt. binario} & \textbf{Pond. suave} & \textbf{$\Delta$ Pond. vs SMOTE} \\")
    lines.append(r"\midrule")

    for ns in nshots:
        ns_results = [r for r in results if r["n_shot"] == ns]

        method_means = {}
        for m in key_methods:
            f1s = [r["f1_macro"] for r in ns_results if r["augmentation_method"] == m]
            method_means[m] = np.mean(f1s) * 100 if f1s else 0

        # Compute paired delta for soft_weighted vs smote at this nshot
        grouped = defaultdict(dict)
        for r in ns_results:
            key = (r["dataset"], r["classifier"], r["seed"])
            grouped[key][r["augmentation_method"]] = r["f1_macro"]

        deltas = []
        for key, methods in grouped.items():
            if "soft_weighted" in methods and "smote" in methods:
                deltas.append(methods["soft_weighted"] - methods["smote"])
        deltas = np.array(deltas)
        delta_mean = deltas.mean() * 100 if len(deltas) > 0 else 0

        _, p_val = stats.ttest_1samp(deltas, 0) if len(deltas) > 1 else (0, 1)
        stars = sig_stars(p_val)
        sign = "+" if delta_mean >= 0 else ""

        lines.append(
            f"{ns} & {method_means['no_augmentation']:.2f} & "
            f"{method_means['smote']:.2f} & "
            f"{method_means['binary_filter']:.2f} & "
            f"{method_means['soft_weighted']:.2f} & "
            f"{sign}{delta_mean:.2f}{stars} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_dataset_table(results):
    """Table: Absolute F1 by dataset for key methods."""
    dataset_bases = [
        "sms_spam", "hate_speech_davidson", "20newsgroups", "ag_news",
        "emotion", "dbpedia14", "20newsgroups_20class",
    ]
    dataset_labels = {
        "sms_spam": "SMS Spam",
        "hate_speech_davidson": "Hate Speech",
        "20newsgroups": "20 Newsgroups (4)",
        "ag_news": "AG News",
        "emotion": "Emotion",
        "dbpedia14": "DBpedia (14)",
        "20newsgroups_20class": "20 Newsgroups (20)",
    }
    nshots = [10, 25, 50]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Macro F1 absoluto por conjunto de datos y número de ejemplos. "
                 r"Se muestran los valores para SMOTE y ponderación suave, promediados sobre "
                 r"clasificadores y semillas. $\Delta$ indica la mejora de la ponderación suave.}")
    lines.append(r"\label{tab:absolute_f1_by_dataset}")
    lines.append(r"\begin{tabular}{llcccr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{N-shot} & \textbf{Sin aum.} & "
                 r"\textbf{SMOTE} & \textbf{Pond. suave} & \textbf{$\Delta$} \\")
    lines.append(r"\midrule")

    for i, db in enumerate(dataset_bases):
        for j, ns in enumerate(nshots):
            ds_results = [r for r in results
                          if r["dataset_base"] == db and r["n_shot"] == ns]

            f1_noaug = np.mean([r["f1_macro"] for r in ds_results
                                if r["augmentation_method"] == "no_augmentation"]) * 100
            f1_smote = np.mean([r["f1_macro"] for r in ds_results
                                if r["augmentation_method"] == "smote"]) * 100
            f1_soft = np.mean([r["f1_macro"] for r in ds_results
                               if r["augmentation_method"] == "soft_weighted"]) * 100
            delta = f1_soft - f1_smote

            label = dataset_labels[db] if j == 0 else ""
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"{label} & {ns} & {f1_noaug:.2f} & {f1_smote:.2f} & "
                f"{f1_soft:.2f} & {sign}{delta:.2f} \\\\"
            )

        if i < len(dataset_bases) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Absolute F1 Tables for Thesis")
    print("=" * 60)

    # Use thesis_final for comprehensive results (5 classifiers, 5 seeds)
    print("\nLoading thesis_final results...")
    results = load_results(THESIS_RESULTS)
    print(f"  {len(results)} records loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: Absolute F1 by method
    print("\n--- Table 1: Absolute F1 by Method ---")
    table1 = generate_method_table(results)
    path1 = OUTPUT_DIR / "tab_absolute_f1_by_method.tex"
    path1.write_text(table1)
    print(f"  Written: {path1}")
    print(table1)

    # Table 2: Absolute F1 by n-shot
    print("\n--- Table 2: Absolute F1 by N-Shot ---")
    table2 = generate_nshot_table(results)
    path2 = OUTPUT_DIR / "tab_absolute_f1_by_nshot.tex"
    path2.write_text(table2)
    print(f"  Written: {path2}")
    print(table2)

    # Table 3: Absolute F1 by dataset
    print("\n--- Table 3: Absolute F1 by Dataset ---")
    table3 = generate_dataset_table(results)
    path3 = OUTPUT_DIR / "tab_absolute_f1_by_dataset.tex"
    path3.write_text(table3)
    print(f"  Written: {path3}")
    print(table3)

    print("\n" + "=" * 60)
    print("Done! 3 tables generated.")


if __name__ == "__main__":
    main()
