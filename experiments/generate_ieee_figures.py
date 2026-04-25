#!/usr/bin/env python3
"""
Generate English figures for IEEE LACCI 2026 paper.
Optimized for two-column IEEE format (3.5in column width).

Figure 1: t-SNE before/after filtering (emotion dataset) - explains the mechanism
Figure 2: Delta by n-shot bar chart - shows when method is most valuable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import hashlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "thesis_final"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis_IEEE" / "Figures"
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-friendly style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
})

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}


# ============================================================================
# FIGURE 1: t-SNE before/after filtering
# ============================================================================

def get_cache_key(dataset, class_name, n_shot, n_generate):
    return hashlib.md5(f"{dataset}_{class_name}_{n_shot}_{n_generate}".encode()).hexdigest()[:16]


def compute_all_data(ds_name, model):
    """Load all data for a dataset: real, test, LLM candidates, filtered."""
    from core.filter_cascade import FilterCascade

    train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(test_texts, show_progress_bar=False)

    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    n_shot = int(ds_name.split("_")[-1].replace("shot", ""))
    n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)

    cascade = FilterCascade(**FILTER_CONFIG)

    data = {
        "real_emb": [], "real_labels": [],
        "test_emb": test_emb, "test_labels": np.array(test_labels),
        "candidate_emb": [], "candidate_labels": [],
        "kept_emb": [], "kept_labels": [],
        "centroids": {},
        "unique_classes": unique_classes,
    }

    for cls in unique_classes:
        cls_mask = labels_arr == cls
        cls_emb = train_emb[cls_mask]

        data["real_emb"].append(cls_emb)
        data["real_labels"].extend([cls] * len(cls_emb))
        data["centroids"][cls] = cls_emb.mean(axis=0)

        cache_key = get_cache_key(ds_name, cls, n_shot, n_gen)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if not cache_file.exists():
            print(f"    No cache for {cls}, skipping")
            continue
        with open(cache_file) as f:
            cached = json.load(f)
        if not cached.get("texts"):
            continue

        gen_emb = model.encode(cached["texts"], show_progress_bar=False)
        data["candidate_emb"].append(gen_emb)
        data["candidate_labels"].extend([cls] * len(gen_emb))

        anchor = cls_emb.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]

        data["kept_emb"].append(gen_emb[top_idx])
        data["kept_labels"].extend([cls] * len(top_idx))

    data["real_emb"] = np.vstack(data["real_emb"])
    data["real_labels"] = np.array(data["real_labels"])
    data["candidate_emb"] = np.vstack(data["candidate_emb"]) if data["candidate_emb"] else np.zeros((0, 768))
    data["candidate_labels"] = np.array(data["candidate_labels"])
    data["kept_emb"] = np.vstack(data["kept_emb"]) if data["kept_emb"] else np.zeros((0, 768))
    data["kept_labels"] = np.array(data["kept_labels"])

    return data


def load_dataset(name):
    with open(DATA_DIR / f"{name}.json") as f:
        d = json.load(f)
    return d["train_texts"], d["train_labels"], d["test_texts"], d["test_labels"]


def plot_tsne_before_after(ds_name="emotion_10shot"):
    """t-SNE before/after filtering, English labels, IEEE column width."""
    from sklearn.manifold import TSNE
    from sentence_transformers import SentenceTransformer

    print(f"  Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    print(f"  Computing data for {ds_name}...")
    data = compute_all_data(ds_name, model)

    # Fit shared t-SNE
    arrays = []
    offsets = {}
    pos = 0
    for key in ["test_emb", "candidate_emb", "kept_emb", "real_emb"]:
        emb = data[key]
        if len(emb) > 0:
            arrays.append(emb)
            offsets[key] = (pos, pos + len(emb))
            pos += len(emb)
        else:
            offsets[key] = (pos, pos)

    centroid_emb = np.array([data["centroids"][cls] for cls in data["unique_classes"]])
    arrays.append(centroid_emb)
    offsets["centroids"] = (pos, pos + len(centroid_emb))

    all_emb = np.vstack(arrays)
    print(f"  Fitting t-SNE on {len(all_emb)} points...")
    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    coords_2d = tsne.fit_transform(all_emb)

    unique_classes = data["unique_classes"]
    n = len(unique_classes)
    palette = sns.color_palette("tab10", n)
    class_colors = {cls: palette[i] for i, cls in enumerate(unique_classes)}

    # Create figure - full column width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    for ax, title, show_candidates in [
        (ax1, "Before filtering", True),
        (ax2, "After filtering", False),
    ]:
        # Test data (background)
        s, e = offsets["test_emb"]
        if s < e:
            for cls in unique_classes:
                cls_mask = data["test_labels"] == cls
                tc = coords_2d[s:e][cls_mask]
                ax.scatter(tc[:, 0], tc[:, 1], c=[class_colors[cls]], marker=".",
                           s=4, alpha=0.08, zorder=0, rasterized=True)

        if show_candidates:
            # Left: all candidates
            s, e = offsets["candidate_emb"]
            if s < e:
                for cls in unique_classes:
                    cls_mask = data["candidate_labels"] == cls
                    cc = coords_2d[s:e][cls_mask]
                    ax.scatter(cc[:, 0], cc[:, 1], c=[class_colors[cls]], marker="x",
                               s=12, alpha=0.22, zorder=1)
        else:
            # Right: kept only
            s, e = offsets["kept_emb"]
            if s < e:
                for cls in unique_classes:
                    cls_mask = data["kept_labels"] == cls
                    kc = coords_2d[s:e][cls_mask]
                    ax.scatter(kc[:, 0], kc[:, 1], c=[class_colors[cls]], marker="^",
                               s=25, alpha=0.55, zorder=3)

        # Real data
        s, e = offsets["real_emb"]
        for cls in unique_classes:
            cls_mask = data["real_labels"] == cls
            rc = coords_2d[s:e][cls_mask]
            ax.scatter(rc[:, 0], rc[:, 1], c=[class_colors[cls]], marker="o",
                       s=40, alpha=0.85, zorder=4, edgecolors="white", linewidths=0.4)

        # Centroids
        s, e = offsets["centroids"]
        for i, cls in enumerate(unique_classes):
            cx, cy = coords_2d[s + i]
            ax.scatter(cx, cy, c=[class_colors[cls]], marker="*",
                       s=180, alpha=1.0, zorder=5, edgecolors="black", linewidths=0.8)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE dim. 1", fontsize=8)
        ax.set_ylabel("t-SNE dim. 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = []
    for cls in unique_classes:
        legend_elements.append(mpatches.Patch(color=class_colors[cls], label=cls))
    legend_elements.extend([
        plt.Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=5, linestyle="None", label="Real"),
        plt.Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray",
                   markersize=5, linestyle="None", label="Filtered"),
        plt.Line2D([0], [0], marker="x", color="gray", markerfacecolor="gray",
                   markersize=5, linestyle="None", label="Candidates"),
        plt.Line2D([0], [0], marker="*", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="Centroid"),
    ])
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(unique_classes) + 4, 10),
               fontsize=6.5, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.3, columnspacing=0.8)

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])

    path = OUTPUT_DIR / "fig_tsne_before_after_emotion.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 2: Delta by n-shot (fixed text collisions)
# ============================================================================

def load_results():
    with open(RESULTS_DIR / "final_results.json") as f:
        data = json.load(f)
    return data["results"]


def plot_delta_by_nshot(results):
    """Bar chart of delta vs SMOTE at each n-shot, with fixed text positioning."""
    nshots = sorted(set(r["n_shot"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))
    classifiers = sorted(set(r["classifier"] for r in results))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    deltas = []
    errs = []
    win_rates = []
    for ns in nshots:
        method_means, smote_means = [], []
        for ds in datasets:
            try:
                ds_nshot = int(ds.split("_")[-1].replace("shot", ""))
            except ValueError:
                continue
            if ds_nshot != ns:
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

        if method_means:
            d_arr = np.array(method_means) - np.array(smote_means)
            deltas.append(np.mean(d_arr) * 100)
            errs.append(1.96 * np.std(d_arr, ddof=1) / np.sqrt(len(d_arr)) * 100 if len(d_arr) > 1 else 0)
            win_rates.append(np.mean(d_arr > 0) * 100)

    colors = ["#228833" if d > 0 else "#CC3311" for d in deltas]
    bars = ax.bar(range(len(nshots)), deltas, yerr=errs, color=colors,
                  alpha=0.8, capsize=4, edgecolor="black", linewidth=0.5,
                  width=0.6)

    # Fixed text positioning: place win rate labels inside bars if needed
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        y = bar.get_height()
        label_y = y + errs[i] + 0.25
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{wr:.0f}% win", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")

    ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Training Examples per Class", fontsize=9)
    ax.set_ylabel(r"$\Delta$ F1 vs SMOTE (pp)", fontsize=9)
    ax.set_xticks(range(len(nshots)))
    ax.set_xticklabels([str(ns) for ns in nshots])
    ax.grid(axis="y", alpha=0.3)
    # Add more headroom for labels
    ymax = max(d + e for d, e in zip(deltas, errs)) + 1.5
    ax.set_ylim(bottom=-0.5, top=ymax)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_delta_by_nshot.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating IEEE LACCI 2026 figures (English)")
    print("=" * 60)

    print("\n[1/2] t-SNE before/after filtering (emotion, 10-shot)...")
    plot_tsne_before_after("emotion_10shot")

    print("\n[2/2] Delta by n-shot bar chart...")
    results = load_results()
    print(f"  Loaded {len(results)} results")
    plot_delta_by_nshot(results)

    print("\nDone! Figures saved to:", OUTPUT_DIR)
