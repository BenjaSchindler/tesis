#!/usr/bin/env python3
"""
Publication-Quality Visualizations for Thesis

Generates all key figures from existing experimental results:
  1. t-SNE scatter: Real vs Kept vs Rejected samples in embedding space
  2. Heatmap: Methods × Datasets performance matrix
  3. F1 vs N-shot curves with error bars
  4. Box plots: Distribution of gains by method
  5. Bar chart: Performance by classifier
  6. Score distribution histogram: Kept vs Rejected

Data sources:
  - results/thesis_final/final_results.json (main experiment, 3,675 configs)
  - Live LLM cache for t-SNE scatter (re-embeds cached generations)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "thesis_final"
VIZ_DIR = PROJECT_ROOT / "results" / "visualizations"
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

METHOD_COLORS = {
    "no_augmentation": "#999999",
    "smote": "#4477AA",
    "random_oversample": "#AA7744",
    "eda": "#CC6677",
    "back_translation": "#882255",
    "binary_filter": "#44AA99",
    "soft_weighted": "#228833",
}

METHOD_LABELS = {
    "no_augmentation": "Sin Aumentaci\u00f3n",
    "smote": "SMOTE",
    "random_oversample": "Sobremuestreo Aleatorio",
    "eda": "EDA",
    "back_translation": "Retrotraducci\u00f3n",
    "binary_filter": "Filtro Binario",
    "soft_weighted": "Ponderaci\u00f3n Suave",
}

DATASET_LABELS = {
    "sms_spam": "SMS Spam (2)",
    "hate_speech_davidson": "Discurso de Odio (3)",
    "20newsgroups": "20News-4 (4)",
    "ag_news": "AG News (4)",
    "emotion": "Emotion (6)",
    "dbpedia14": "DBpedia (14)",
    "20newsgroups_20class": "20News-20 (20)",
}


# ============================================================================
# LOAD DATA
# ============================================================================

def load_results():
    with open(RESULTS_DIR / "final_results.json") as f:
        data = json.load(f)
    return data["results"]


def get_dataset_base(dataset_name):
    bases = sorted([
        "sms_spam", "hate_speech_davidson", "20newsgroups_20class",
        "20newsgroups", "ag_news", "emotion", "dbpedia14",
    ], key=len, reverse=True)
    for base in bases:
        if dataset_name.startswith(base + "_"):
            return base
    return dataset_name


# ============================================================================
# FIG 1: HEATMAP — Methods × Datasets
# ============================================================================

def plot_heatmap(results):
    """Heatmap of F1 delta vs SMOTE for each method × dataset_base."""
    methods = ["no_augmentation", "eda", "back_translation",
               "random_oversample", "binary_filter", "soft_weighted"]
    dataset_bases = ["sms_spam", "hate_speech_davidson", "20newsgroups",
                     "ag_news", "emotion", "dbpedia14", "20newsgroups_20class"]

    matrix = np.zeros((len(methods), len(dataset_bases)))

    for j, db in enumerate(dataset_bases):
        # Get SMOTE mean for this dataset base
        smote_f1s = [r["f1_macro"] for r in results if r["dataset_base"] == db
                     and r["augmentation_method"] == "smote"]
        smote_mean = np.mean(smote_f1s) if smote_f1s else 0

        for i, method in enumerate(methods):
            method_f1s = [r["f1_macro"] for r in results if r["dataset_base"] == db
                          and r["augmentation_method"] == method]
            if method_f1s:
                matrix[i, j] = (np.mean(method_f1s) - smote_mean) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    im = sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        xticklabels=[DATASET_LABELS.get(db, db) for db in dataset_bases],
        yticklabels=[METHOD_LABELS.get(m, m) for m in methods],
        cbar_kws={"label": r"$\Delta$ F1 vs SMOTE (pp)"},
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Rendimiento por M\u00e9todo de Aumentaci\u00f3n vs L\u00ednea Base SMOTE")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")

    path = VIZ_DIR / "fig1_heatmap_methods_datasets.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 2: F1 vs N-SHOT (line plot with error bars)
# ============================================================================

def plot_f1_vs_nshot(results):
    """Performance curves across n-shot values."""
    methods = ["smote", "binary_filter", "soft_weighted"]
    nshots = sorted(set(r["n_shot"] for r in results))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method in methods:
        means, ci_low, ci_high = [], [], []
        for ns in nshots:
            f1s = [r["f1_macro"] for r in results
                   if r["augmentation_method"] == method and r["n_shot"] == ns]
            m = np.mean(f1s)
            se = np.std(f1s, ddof=1) / np.sqrt(len(f1s)) if len(f1s) > 1 else 0
            means.append(m)
            ci_low.append(m - 1.96 * se)
            ci_high.append(m + 1.96 * se)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        ax.plot(nshots, means, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(nshots, ci_low, ci_high, alpha=0.15, color=color)

    ax.set_xlabel("Muestras por Clase (N-shot)")
    ax.set_ylabel("Puntuaci\u00f3n Macro F1")
    ax.set_title("Rendimiento vs Tama\u00f1o del Conjunto de Entrenamiento")
    ax.set_xticks(nshots)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = VIZ_DIR / "fig2_f1_vs_nshot.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 3: BOX PLOT — Delta distribution by method
# ============================================================================

def plot_boxplot_deltas(results):
    """Box plot showing distribution of F1 deltas vs SMOTE by method."""
    methods = ["no_augmentation", "eda", "back_translation",
               "random_oversample", "binary_filter", "soft_weighted"]

    classifiers = sorted(set(r["classifier"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))

    # Compute paired deltas: for each (dataset, classifier), avg over seeds
    deltas_by_method = {m: [] for m in methods}

    for ds in datasets:
        for clf in classifiers:
            smote_f1s = [r["f1_macro"] for r in results
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == "smote"]
            if not smote_f1s:
                continue
            smote_mean = np.mean(smote_f1s)

            for method in methods:
                method_f1s = [r["f1_macro"] for r in results
                              if r["dataset"] == ds and r["classifier"] == clf
                              and r["augmentation_method"] == method]
                if method_f1s:
                    delta = (np.mean(method_f1s) - smote_mean) * 100
                    deltas_by_method[method].append(delta)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    box_data = [deltas_by_method[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="L\u00ednea base SMOTE")
    ax.set_ylabel(r"$\Delta$ F1 vs SMOTE (pp)")
    ax.set_title("Distribuci\u00f3n de Ganancias de Rendimiento")
    ax.legend(loc="upper left")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    path = VIZ_DIR / "fig3_boxplot_deltas.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 4: BAR CHART — Performance by classifier
# ============================================================================

def plot_classifier_comparison(results):
    """Grouped bar chart: delta vs SMOTE per classifier for key methods."""
    classifiers = ["logistic_regression", "svc_linear", "ridge", "mlp", "random_forest"]
    methods = ["binary_filter", "soft_weighted"]
    clf_labels = {
        "logistic_regression": "LogReg",
        "svc_linear": "SVC",
        "ridge": "Ridge",
        "mlp": "MLP",
        "random_forest": "RF",
    }

    datasets = sorted(set(r["dataset"] for r in results))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(classifiers))
    width = 0.35

    for k, method in enumerate(methods):
        deltas = []
        errs = []
        for clf in classifiers:
            method_means, smote_means = [], []
            for ds in datasets:
                m_f1s = [r["f1_macro"] for r in results
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == method]
                s_f1s = [r["f1_macro"] for r in results
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == "smote"]
                if m_f1s and s_f1s:
                    method_means.append(np.mean(m_f1s))
                    smote_means.append(np.mean(s_f1s))
            d = (np.mean(method_means) - np.mean(smote_means)) * 100 if method_means else 0
            se = np.std(np.array(method_means) - np.array(smote_means)) / np.sqrt(len(method_means)) * 100 if len(method_means) > 1 else 0
            deltas.append(d)
            errs.append(1.96 * se)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        offset = -width / 2 + k * width
        bars = ax.bar(x + offset, deltas, width, yerr=errs,
                      label=label, color=color, alpha=0.8, capsize=4)

    ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Clasificador")
    ax.set_ylabel(r"$\Delta$ F1 vs SMOTE (pp)")
    ax.set_title("Ganancia de Rendimiento por Clasificador")
    ax.set_xticks(x)
    ax.set_xticklabels([clf_labels[c] for c in classifiers])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = VIZ_DIR / "fig4_classifier_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 5: DELTA BY N-SHOT (bar chart showing where LLM wins)
# ============================================================================

def plot_delta_by_nshot(results):
    """Bar chart: delta vs SMOTE at each n-shot for soft_weighted."""
    nshots = sorted(set(r["n_shot"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))
    classifiers = sorted(set(r["classifier"] for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))

    deltas = []
    errs = []
    win_rates = []
    for ns in nshots:
        method_means, smote_means = [], []
        for ds in datasets:
            if int(ds.split("_")[-1].replace("shot", "")) != ns:
                continue
            for clf in classifiers:
                m_f1s = [r["f1_macro"] for r in results
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == "soft_weighted"]
                s_f1s = [r["f1_macro"] for r in results
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == "smote"]
                if m_f1s and s_f1s:
                    method_means.append(np.mean(m_f1s))
                    smote_means.append(np.mean(s_f1s))

        d_arr = np.array(method_means) - np.array(smote_means)
        deltas.append(np.mean(d_arr) * 100)
        errs.append(1.96 * np.std(d_arr, ddof=1) / np.sqrt(len(d_arr)) * 100 if len(d_arr) > 1 else 0)
        win_rates.append(np.mean(d_arr > 0) * 100)

    colors = ["#228833" if d > 0 else "#CC3311" for d in deltas]
    bars = ax.bar(range(len(nshots)), deltas, yerr=errs, color=colors, alpha=0.8, capsize=5)

    # Add win rate labels on bars
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + errs[i] + 0.15,
                f"{wr:.0f}% vic.", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Muestras por Clase (N-shot)")
    ax.set_ylabel(r"$\Delta$ F1 vs SMOTE (pp)")
    ax.set_title("Mejora de Ponderaci\u00f3n Suave por Tama\u00f1o de Entrenamiento")
    ax.set_xticks(range(len(nshots)))
    ax.set_xticklabels([f"{ns} muestras/clase" for ns in nshots])
    ax.grid(axis="y", alpha=0.3)

    path = VIZ_DIR / "fig5_delta_by_nshot.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 6: t-SNE SCATTER (Real vs Kept vs Rejected)
# ============================================================================

def plot_tsne_scatter(dataset_name="20newsgroups_10shot"):
    """t-SNE visualization of real, kept, and rejected synthetic samples."""
    from sklearn.manifold import TSNE
    from sentence_transformers import SentenceTransformer
    from core.filter_cascade import FilterCascade

    print(f"  Generating t-SNE for {dataset_name}...")

    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)
    train_texts = data["train_texts"]
    train_labels = data["train_labels"]

    # Embed
    model = SentenceTransformer("all-mpnet-base-v2")
    train_emb = model.encode(train_texts, show_progress_bar=False)

    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)

    # Load cached LLM generations and compute scores
    cascade = FilterCascade(filter_level=1, k_neighbors=10)
    all_real_emb, all_real_cls = [], []
    all_kept_emb, all_kept_cls = [], []
    all_rejected_emb, all_rejected_cls = [], []

    import hashlib
    n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

    for cls in unique_classes:
        cls_emb = train_emb[labels_arr == cls]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        all_real_emb.append(cls_emb)
        all_real_cls.extend([cls] * len(cls_emb))

        # Try to load cached generations
        n_gen = 150  # 3x oversample
        cache_key = hashlib.md5(f"{dataset_name}_{cls}_{n_shot}_{n_gen}".encode()).hexdigest()[:16]
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if not cache_file.exists():
            print(f"    No cache for {cls}, skipping...")
            continue

        with open(cache_file) as f:
            cached = json.load(f)
        if not cached.get("texts"):
            continue

        gen_emb = model.encode(cached["texts"], show_progress_bar=False)

        # Score with cascade
        anchor = cls_emb.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)

        # Split into kept (top 50) and rejected
        target_n = min(50, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        bottom_mask = np.ones(len(gen_emb), dtype=bool)
        bottom_mask[top_idx] = False

        all_kept_emb.append(gen_emb[top_idx])
        all_kept_cls.extend([cls] * len(gen_emb[top_idx]))
        all_rejected_emb.append(gen_emb[bottom_mask])
        all_rejected_cls.extend([cls] * int(bottom_mask.sum()))

    if not all_kept_emb:
        print(f"    No cached generations found for {dataset_name}, skipping t-SNE")
        return

    real_emb = np.vstack(all_real_emb)
    kept_emb = np.vstack(all_kept_emb)
    rejected_emb = np.vstack(all_rejected_emb) if all_rejected_emb else np.zeros((0, real_emb.shape[1]))

    # Combine for t-SNE
    all_emb = np.vstack([e for e in [real_emb, kept_emb, rejected_emb] if len(e) > 0])
    n_real = len(real_emb)
    n_kept = len(kept_emb)

    print(f"    Real: {n_real}, Kept: {n_kept}, Rejected: {len(rejected_emb)}")

    # t-SNE
    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(all_emb)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Rejected first (background)
    if len(rejected_emb) > 0:
        start = n_real + n_kept
        ax.scatter(emb_2d[start:, 0], emb_2d[start:, 1],
                   c="#CC3311", marker="x", s=25, alpha=0.35, label=f"Rechazados ({len(rejected_emb)})", zorder=1)

    # Kept
    ax.scatter(emb_2d[n_real:n_real + n_kept, 0], emb_2d[n_real:n_real + n_kept, 1],
               c="#228833", marker="^", s=40, alpha=0.6, label=f"Aceptados ({n_kept})", zorder=2)

    # Real (foreground)
    ax.scatter(emb_2d[:n_real, 0], emb_2d[:n_real, 1],
               c="#4477AA", marker="o", s=50, alpha=0.8, label=f"Reales ({n_real})", zorder=3)

    ax.legend(loc="best", framealpha=0.9)
    ds_label = DATASET_LABELS.get(get_dataset_base(dataset_name), dataset_name)
    ax.set_title(f"t-SNE: Muestras Reales vs Sint\u00e9ticas Filtradas\n{ds_label}, {n_shot}-shot")
    ax.set_xlabel("t-SNE dim. 1")
    ax.set_ylabel("t-SNE dim. 2")

    path = VIZ_DIR / f"fig6_tsne_{dataset_name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 7: SCORE DISTRIBUTION (Kept vs Rejected)
# ============================================================================

def plot_score_distribution(dataset_name="20newsgroups_10shot"):
    """Histogram of cascade L1 scores for kept vs rejected samples."""
    from sentence_transformers import SentenceTransformer
    from core.filter_cascade import FilterCascade
    import hashlib

    print(f"  Generating score distribution for {dataset_name}...")

    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)
    train_texts = data["train_texts"]
    train_labels = data["train_labels"]

    model = SentenceTransformer("all-mpnet-base-v2")
    train_emb = model.encode(train_texts, show_progress_bar=False)
    labels_arr = np.array(train_labels)

    cascade = FilterCascade(filter_level=1, k_neighbors=10)
    n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

    all_kept_scores = []
    all_rejected_scores = []

    for cls in sorted(set(train_labels)):
        cls_emb = train_emb[labels_arr == cls]
        n_gen = 150
        cache_key = hashlib.md5(f"{dataset_name}_{cls}_{n_shot}_{n_gen}".encode()).hexdigest()[:16]
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if not cache_file.exists():
            continue

        with open(cache_file) as f:
            cached = json.load(f)
        if not cached.get("texts"):
            continue

        gen_emb = model.encode(cached["texts"], show_progress_bar=False)
        anchor = cls_emb.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)

        target_n = min(50, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        bottom_mask = np.ones(len(gen_emb), dtype=bool)
        bottom_mask[top_idx] = False

        all_kept_scores.extend(scores[top_idx])
        all_rejected_scores.extend(scores[bottom_mask])

    if not all_kept_scores:
        print(f"    No data for {dataset_name}, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 35)

    ax.hist(all_rejected_scores, bins=bins, alpha=0.6, color="#CC3311",
            label=f"Rechazados (n={len(all_rejected_scores)})", edgecolor="white")
    ax.hist(all_kept_scores, bins=bins, alpha=0.7, color="#228833",
            label=f"Aceptados (n={len(all_kept_scores)})", edgecolor="white")

    ax.axvline(np.mean(all_kept_scores), color="#228833", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(np.mean(all_rejected_scores), color="#CC3311", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Puntaje Cascade L1 (basado en distancia)")
    ax.set_ylabel("Frecuencia")
    ds_label = DATASET_LABELS.get(get_dataset_base(dataset_name), dataset_name)
    ax.set_title(f"Distribuci\u00f3n de Puntajes: Aceptados vs Rechazados\n{ds_label}, {n_shot}-shot")
    ax.legend()

    path = VIZ_DIR / f"fig7_score_dist_{dataset_name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIG 8: MULTI-LLM COMPARISON (horizontal bar chart)
# ============================================================================

def plot_multi_llm_comparison():
    """Horizontal bar chart: delta vs SMOTE per LLM (adjusted results)."""
    analysis_path = PROJECT_ROOT / "results" / "multi_llm" / "analysis.json"
    if not analysis_path.exists():
        print("  Skipping multi-LLM figure (analysis.json not found)")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    adjusted = analysis["adjusted_results"]

    # Sort by delta descending
    llms = sorted(adjusted.keys(), key=lambda k: adjusted[k]["delta_pp"], reverse=True)

    deltas = [adjusted[k]["delta_pp"] for k in llms]
    ci_lows = [adjusted[k]["ci_95"][0] for k in llms]
    ci_highs = [adjusted[k]["ci_95"][1] for k in llms]
    win_rates = [adjusted[k]["win_rate"] * 100 for k in llms]
    bonf_ps = [adjusted[k]["bonf_p"] for k in llms]

    # Error bars (asymmetric)
    xerr_low = [d - cl for d, cl in zip(deltas, ci_lows)]
    xerr_high = [ch - d for d, ch in zip(deltas, ci_highs)]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(llms))

    colors = ["#228833" if p < 0.05 else "#999999" for p in bonf_ps]

    bars = ax.barh(y, deltas, xerr=[xerr_low, xerr_high],
                   color=colors, alpha=0.8, capsize=4, edgecolor="white", height=0.6)

    ax.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="L\u00ednea base SMOTE")

    # Add win rate annotations
    for i, (d, wr, p) in enumerate(zip(deltas, win_rates, bonf_ps)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(d + xerr_high[i] + 0.08, i,
                f"{wr:.0f}% vic. {sig}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(llms)
    ax.set_xlabel(r"Mejora F1 sobre SMOTE (pp)")
    ax.set_title("Ganancia de Rendimiento por Modelo de Lenguaje")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    path = VIZ_DIR / "fig8_multi_llm_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("GENERATING PUBLICATION VISUALIZATIONS")
    print("=" * 70)

    # Load thesis results
    print("\nLoading thesis_final results...")
    results = load_results()
    print(f"  Loaded {len(results)} experiment results")

    # Figures from aggregated results (fast, no models needed)
    print("\n--- Generating aggregate figures ---")
    plot_heatmap(results)
    plot_f1_vs_nshot(results)
    plot_boxplot_deltas(results)
    plot_classifier_comparison(results)
    plot_delta_by_nshot(results)
    plot_multi_llm_comparison()

    # Figures requiring embedding model (slower)
    # Set CUDA_VISIBLE_DEVICES="" to force CPU if CUDA errors occur
    print("\n--- Generating embedding-space figures ---")
    for ds in ["20newsgroups_10shot", "emotion_10shot", "hate_speech_davidson_10shot"]:
        ds_path = DATA_DIR / f"{ds}.json"
        if ds_path.exists():
            try:
                plot_tsne_scatter(ds)
            except Exception as e:
                print(f"  Error in t-SNE for {ds}: {e}")
            try:
                plot_score_distribution(ds)
            except Exception as e:
                print(f"  Error in score dist for {ds}: {e}")
        else:
            print(f"  Skipping {ds} (dataset not found)")

    print(f"\nAll figures saved to: {VIZ_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
