#!/usr/bin/env python3
"""
LaTeX Table Generator

Combines all experiment results into formatted LaTeX tables
ready for inclusion in Metodologia.tex

Usage:
    python latex_generator.py  # Regenerate all tables from results/
"""

import json
from pathlib import Path
from base_config import RESULTS_DIR, LATEX_DIR


def load_results(experiment: str) -> list:
    """Load results from experiment directory."""
    exp_dir = RESULTS_DIR / experiment
    if not exp_dir.exists():
        print(f"  Warning: {exp_dir} not found")
        return []

    results = []
    for f in sorted(exp_dir.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def generate_all_tables():
    """Generate all LaTeX tables from results."""
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating LaTeX tables...")

    # 1. Clustering
    print("  - tab_clustering_validation.tex")
    results = load_results("clustering")
    if results:
        with open(LATEX_DIR / "tab_clustering_validation.tex", 'w') as f:
            f.write(generate_clustering_table(results))

    # 2. Anchor strategies
    print("  - tab_anchor_strategies.tex")
    results = load_results("anchor_strategies")
    if results:
        with open(LATEX_DIR / "tab_anchor_strategies.tex", 'w') as f:
            f.write(generate_anchor_table(results))

    # 3. K neighbors
    print("  - tab_k_neighbors.tex")
    results = load_results("k_neighbors")
    if results:
        with open(LATEX_DIR / "tab_k_neighbors.tex", 'w') as f:
            f.write(generate_kneighbors_table(results))

    # 4. Filter cascade
    print("  - tab_filter_cascade.tex")
    results = load_results("filter_cascade")
    if results:
        with open(LATEX_DIR / "tab_filter_cascade.tex", 'w') as f:
            f.write(generate_filter_table(results))

    # 5. Adaptive thresholds
    print("  - tab_adaptive_validation.tex")
    results = load_results("adaptive_thresholds")
    if results:
        with open(LATEX_DIR / "tab_adaptive_validation.tex", 'w') as f:
            f.write(generate_threshold_table(results))

    # 6. Tier impact
    print("  - tab_tier_impact.tex")
    tier_file = RESULTS_DIR / "tier_impact" / "tier_impact_results.json"
    if tier_file.exists():
        with open(tier_file) as f:
            tier_data = json.load(f)
        with open(LATEX_DIR / "tab_tier_impact.tex", 'w') as f:
            f.write(generate_tier_table(tier_data))

    print(f"\nAll tables saved to: {LATEX_DIR}")


def generate_clustering_table(results: list) -> str:
    """Generate clustering validation table."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Validacion preliminar para seleccion del numero maximo de clusters}
\label{tab:clustering_validation}
\begin{tabular}{lcccc}
\hline
$K_{max}$ & Silhouette & Coherencia & Macro F1 & $\Delta$ vs Linea base \\
\hline
"""
    # Sort by k_max
    results = sorted(results, key=lambda x: x.get('k_max', 0))

    # Find best
    deltas = [r.get('delta_pct', 0) for r in results]
    best_idx = deltas.index(max(deltas)) if deltas else -1

    for i, r in enumerate(results):
        k = r.get('k_max', 'N/A')
        sil = r.get('silhouette', -1)
        sil_str = "N/A" if sil < 0 else f"{sil:.2f}"
        coh = r.get('coherence', 0)
        f1 = r.get('macro_f1', 0)
        delta = r.get('delta_pct', 0)

        label = f"{k}" + (" (vanilla)" if k == 1 else "")

        if i == best_idx:
            latex += f"\\textbf{{{label}}} & \\textbf{{{sil_str}}} & \\textbf{{{coh*100:.0f}\\%}} & \\textbf{{{f1:.4f}}} & \\textbf{{{delta:+.2f}\\%}} \\\\\n"
        else:
            latex += f"{label} & {sil_str} & {coh*100:.0f}\\% & {f1:.4f} & {delta:+.2f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_anchor_table(results: list) -> str:
    """Generate anchor strategies table."""
    name_map = {
        "random": "Random",
        "nearest_neighbor": "Nearest Neighbor",
        "medoid": "Medoid",
        "quality_gated": "Quality-gated",
        "diverse": "Diverse",
        "ensemble": "Ensemble"
    }

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparacion de estrategias de seleccion de anclas}
\label{tab:anchor_strategies}
\begin{tabular}{lcccc}
\hline
Estrategia & Macro F1 & Quality & Diversity & $\Delta$ \\
\hline
"""
    deltas = [r.get('delta_pct', 0) for r in results]
    best_idx = deltas.index(max(deltas)) if deltas else -1

    for i, r in enumerate(results):
        strategy = r.get('strategy', 'unknown')
        name = name_map.get(strategy, strategy)
        f1 = r.get('macro_f1', 0)
        quality = r.get('quality', 0)
        diversity = r.get('diversity', 0)
        delta = r.get('delta_pct', 0)

        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{f1:.4f}}} & \\textbf{{{quality:.2f}}} & \\textbf{{{diversity:.2f}}} & \\textbf{{{delta:+.1f}\\%}} \\\\\n"
        else:
            latex += f"{name} & {f1:.4f} & {quality:.2f} & {diversity:.2f} & {delta:+.1f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_kneighbors_table(results: list) -> str:
    """Generate K neighbors table."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto del numero de vecinos K en rendimiento}
\label{tab:k_neighbors}
\begin{tabular}{lccc}
\hline
K vecinos & Macro F1 & Acceptance Rate & Contexto \\
\hline
"""
    results = sorted(results, key=lambda x: x.get('k_value', 0))
    deltas = [r.get('delta_pct', 0) for r in results]
    best_idx = deltas.index(max(deltas)) if deltas else -1

    for i, r in enumerate(results):
        k = r.get('k_value', 0)
        f1 = r.get('macro_f1', 0)
        acc = r.get('acceptance_rate', 0)
        context = r.get('context_quality', '')

        if i == best_idx:
            latex += f"\\textbf{{{k}}} & \\textbf{{{f1:.4f}}} & \\textbf{{{acc*100:.0f}\\%}} & \\textbf{{{context}}} \\\\\n"
        else:
            latex += f"{k} & {f1:.4f} & {acc*100:.0f}\\% & {context} \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_filter_table(results: list) -> str:
    """Generate filter cascade table."""
    name_map = {
        "length_only": "Solo longitud",
        "length_similarity": "Longitud + Similaridad",
        "three_partial": "Tres filtros parciales",
        "full_cascade": "Cascada completa"
    }

    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto del numero de filtros en calidad y aceptacion}
\label{tab:filter_cascade}
\begin{tabular}{lcccc}
\hline
Configuracion & Acceptance & Quality & Macro F1 & $\Delta$ \\
\hline
"""
    deltas = [r.get('delta_pct', 0) for r in results]
    best_idx = deltas.index(max(deltas)) if deltas else -1

    for i, r in enumerate(results):
        config = r.get('config', '')
        name = name_map.get(config, config)
        acc = r.get('acceptance_rate', 0)
        quality = r.get('quality', 0)
        f1 = r.get('macro_f1', 0)
        delta = r.get('delta_pct', 0)

        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{acc*100:.0f}\\%}} & \\textbf{{{quality:.2f}}} & \\textbf{{{f1:.4f}}} & \\textbf{{{delta:+.1f}\\%}} \\\\\n"
        else:
            latex += f"{name} & {acc*100:.0f}\\% & {quality:.2f} & {f1:.4f} & {delta:+.1f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_threshold_table(results: list) -> str:
    """Generate adaptive thresholds table."""
    name_map = {
        "fixed_permissive": "Fijo permisivo (0.60)",
        "fixed_medium": "Fijo medio (0.70)",
        "fixed_strict": "Fijo estricto (0.90)",
        "adaptive": "Adaptativo"
    }

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparacion de estrategias de umbralizacion}
\label{tab:adaptive_validation}
\begin{tabular}{lcccc}
\hline
Estrategia & Macro F1 & Acceptance & Quality & Contam. \\
\hline
"""
    deltas = [r.get('delta_pct', 0) for r in results]
    best_idx = deltas.index(max(deltas)) if deltas else -1

    for i, r in enumerate(results):
        config = r.get('config', '')
        name = name_map.get(config, config)
        f1 = r.get('macro_f1', 0)
        acc = r.get('acceptance_rate', 0)
        quality = r.get('quality', 0)
        contam = r.get('contamination', 0)

        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{f1:.4f}}} & \\textbf{{{acc*100:.0f}\\%}} & \\textbf{{{quality:.2f}}} & \\textbf{{{contam*100:.1f}\\%}} \\\\\n"
        else:
            latex += f"{name} & {f1:.4f} & {acc*100:.0f}\\% & {quality:.2f} & {contam*100:.1f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_tier_table(data: dict) -> str:
    """Generate tier impact table."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto de ponderacion uniforme por nivel de rendimiento}
\label{tab:tier_impact}
\begin{tabular}{lccc}
\hline
Tier & Rango F1 & Clases & $\Delta$F1 Promedio \\
\hline
"""
    for r in data.get('tier_results', []):
        tier = r.get('tier', '')
        f1_range = r.get('f1_range', '')
        n_classes = r.get('n_classes', 0)
        delta = r.get('delta_f1_avg', 0)

        latex += f"{tier} & ${f1_range}$ & {n_classes} & {delta:+.2f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    generate_all_tables()
