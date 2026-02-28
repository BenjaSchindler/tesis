#!/usr/bin/env python3
"""Enhanced t-SNE visualizations of the embedding space.

Generates publication-quality figures showing:
1. Before vs After filtering (side-by-side, class-colored)
2. Filtered LLM vs SMOTE (side-by-side, class-colored)

Both include test data as transparent background and class centroids as stars.
Uses a shared t-SNE projection for consistent spatial positions across panels.

Datasets: emotion_10shot (6 classes), 20newsgroups_10shot (4 classes)
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
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
FIGURE_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

DATASETS = ["emotion_10shot", "20newsgroups_10shot"]

DATASET_LABELS = {
    "emotion": "Emotion (6 clases)",
    "20newsgroups": "20 Newsgroups (4 clases)",
}

# Visual styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def get_dataset_base(name):
    bases = ["20newsgroups_20class", "20newsgroups", "hate_speech_davidson",
             "sms_spam", "ag_news", "emotion", "dbpedia14"]
    for base in bases:
        if name.startswith(base + "_"):
            return base
    return name


def load_dataset(name):
    with open(DATA_DIR / f"{name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_cache_key(dataset, class_name, n_shot, n_generate):
    return hashlib.md5(f"{dataset}_{class_name}_{n_shot}_{n_generate}".encode()).hexdigest()[:16]


def generate_smote_samples(real_embeddings, n_generate, seed=42, k_neighbors=5):
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    rng = np.random.RandomState(seed)
    X = np.vstack([real_embeddings, rng.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res[np.where(y_res == 0)[0][n_base:]][:n_generate]
    except Exception:
        return np.array([]).reshape(0, real_embeddings.shape[1])


def compute_all_data(ds_name, model):
    """Load all data for a dataset: real, test, LLM candidates, filtered, SMOTE."""
    print(f"  Loading data for {ds_name}...", flush=True)

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
        "rejected_emb": [], "rejected_labels": [],
        "smote_emb": [], "smote_labels": [],
        "centroids": {},
        "unique_classes": unique_classes,
    }

    for cls in unique_classes:
        cls_mask = labels_arr == cls
        cls_emb = train_emb[cls_mask]

        data["real_emb"].append(cls_emb)
        data["real_labels"].extend([cls] * len(cls_emb))
        data["centroids"][cls] = cls_emb.mean(axis=0)

        # LLM candidates
        cache_key = get_cache_key(ds_name, cls, n_shot, n_gen)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if not cache_file.exists():
            print(f"    No cache for {cls}, skipping LLM data")
            continue
        with open(cache_file) as f:
            cached = json.load(f)
        if not cached.get("texts"):
            continue

        gen_emb = model.encode(cached["texts"], show_progress_bar=False)
        data["candidate_emb"].append(gen_emb)
        data["candidate_labels"].extend([cls] * len(gen_emb))

        # Score and filter
        anchor = cls_emb.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        bottom_mask = np.ones(len(gen_emb), dtype=bool)
        bottom_mask[top_idx] = False

        data["kept_emb"].append(gen_emb[top_idx])
        data["kept_labels"].extend([cls] * len(top_idx))
        data["rejected_emb"].append(gen_emb[bottom_mask])
        data["rejected_labels"].extend([cls] * int(bottom_mask.sum()))

        # SMOTE
        smote_emb = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=42)
        if len(smote_emb) > 0:
            data["smote_emb"].append(smote_emb)
            data["smote_labels"].extend([cls] * len(smote_emb))

    # Stack arrays
    data["real_emb"] = np.vstack(data["real_emb"])
    data["real_labels"] = np.array(data["real_labels"])
    data["candidate_emb"] = np.vstack(data["candidate_emb"]) if data["candidate_emb"] else np.zeros((0, 768))
    data["candidate_labels"] = np.array(data["candidate_labels"])
    data["kept_emb"] = np.vstack(data["kept_emb"]) if data["kept_emb"] else np.zeros((0, 768))
    data["kept_labels"] = np.array(data["kept_labels"])
    data["rejected_emb"] = np.vstack(data["rejected_emb"]) if data["rejected_emb"] else np.zeros((0, 768))
    data["rejected_labels"] = np.array(data["rejected_labels"])
    data["smote_emb"] = np.vstack(data["smote_emb"]) if data["smote_emb"] else np.zeros((0, 768))
    data["smote_labels"] = np.array(data["smote_labels"])

    print(f"    Real: {len(data['real_emb'])}, Test: {len(test_emb)}, "
          f"Candidates: {len(data['candidate_emb'])}, Kept: {len(data['kept_emb'])}, "
          f"Rejected: {len(data['rejected_emb'])}, SMOTE: {len(data['smote_emb'])}", flush=True)

    return data


def fit_tsne(data):
    """Fit t-SNE on the union of all point types. Returns 2D coords and offset dict."""
    arrays = []
    offsets = {}
    pos = 0

    for key in ["test_emb", "rejected_emb", "smote_emb", "candidate_emb", "kept_emb", "real_emb"]:
        emb = data[key]
        if len(emb) > 0:
            arrays.append(emb)
            offsets[key] = (pos, pos + len(emb))
            pos += len(emb)
        else:
            offsets[key] = (pos, pos)

    # Also add centroids
    centroid_emb = np.array([data["centroids"][cls] for cls in data["unique_classes"]])
    arrays.append(centroid_emb)
    offsets["centroids"] = (pos, pos + len(centroid_emb))

    all_emb = np.vstack(arrays)
    print(f"  Fitting t-SNE on {len(all_emb)} points...", flush=True)

    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    coords_2d = tsne.fit_transform(all_emb)
    print(f"  t-SNE complete.", flush=True)

    return coords_2d, offsets


def get_class_colors(unique_classes):
    """Get colorblind-friendly class color palette."""
    n = len(unique_classes)
    if n <= 10:
        palette = sns.color_palette("tab10", n)
    else:
        palette = sns.color_palette("husl", n)
    return {cls: palette[i] for i, cls in enumerate(unique_classes)}


def plot_before_after(ds_name, data, coords_2d, offsets, class_colors):
    """Side-by-side: Left=Real+All Candidates+Test, Right=Real+Kept+Test."""
    unique_classes = data["unique_classes"]
    base = get_dataset_base(ds_name)
    n_shot = int(ds_name.split("_")[-1].replace("shot", ""))
    ds_label = DATASET_LABELS.get(base, base)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, title, show_rejected, show_all_candidates in [
        (ax1, "Antes del filtrado", True, True),
        (ax2, "Después del filtrado", False, False),
    ]:
        # Test data (background)
        s, e = offsets["test_emb"]
        if s < e:
            test_labels = data["test_labels"]
            for cls in unique_classes:
                cls_mask = test_labels == cls
                tc = coords_2d[s:e][cls_mask]
                ax.scatter(tc[:, 0], tc[:, 1], c=[class_colors[cls]], marker=".",
                           s=6, alpha=0.10, zorder=0, rasterized=True)

        # Rejected/All candidates (left panel only)
        if show_all_candidates:
            s, e = offsets["candidate_emb"]
            if s < e:
                cand_labels = data["candidate_labels"]
                for cls in unique_classes:
                    cls_mask = cand_labels == cls
                    cc = coords_2d[s:e][cls_mask]
                    ax.scatter(cc[:, 0], cc[:, 1], c=[class_colors[cls]], marker="x",
                               s=18, alpha=0.25, zorder=1)
        else:
            # Right panel: show kept only
            s, e = offsets["kept_emb"]
            if s < e:
                kept_labels = data["kept_labels"]
                for cls in unique_classes:
                    cls_mask = kept_labels == cls
                    kc = coords_2d[s:e][cls_mask]
                    ax.scatter(kc[:, 0], kc[:, 1], c=[class_colors[cls]], marker="^",
                               s=35, alpha=0.60, zorder=3)

        # Real data (foreground)
        s, e = offsets["real_emb"]
        real_labels = data["real_labels"]
        for cls in unique_classes:
            cls_mask = real_labels == cls
            rc = coords_2d[s:e][cls_mask]
            ax.scatter(rc[:, 0], rc[:, 1], c=[class_colors[cls]], marker="o",
                       s=55, alpha=0.85, zorder=4, edgecolors="white", linewidths=0.5)

        # Centroids
        s, e = offsets["centroids"]
        for i, cls in enumerate(unique_classes):
            cx, cy = coords_2d[s + i]
            ax.scatter(cx, cy, c=[class_colors[cls]], marker="*",
                       s=250, alpha=1.0, zorder=5, edgecolors="black", linewidths=1.0)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE dim. 1")
        ax.set_ylabel("t-SNE dim. 2")
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = []
    for cls in unique_classes:
        legend_elements.append(mpatches.Patch(color=class_colors[cls], label=cls))
    legend_elements.extend([
        plt.Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="Real"),
        plt.Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="Filtrado (kept)"),
        plt.Line2D([0], [0], marker="x", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="Candidatos"),
        plt.Line2D([0], [0], marker=".", color="gray", markerfacecolor="gray",
                   markersize=6, linestyle="None", label="Test"),
        plt.Line2D([0], [0], marker="*", color="gray", markerfacecolor="gray",
                   markersize=12, linestyle="None", label="Centroide"),
    ])
    fig.legend(handles=legend_elements, loc="lower center", ncol=min(len(unique_classes) + 5, 8),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Espacio de embeddings — {ds_label}, {n_shot}-shot", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    for ext in ["png", "pdf"]:
        path = FIGURE_DIR / f"fig_tsne_before_after_{base}.{ext}"
        plt.savefig(path)
        print(f"  Saved: {path}", flush=True)
    plt.close()


def plot_filtered_vs_smote(ds_name, data, coords_2d, offsets, class_colors):
    """Side-by-side: Left=Real+Filtered LLM, Right=Real+SMOTE."""
    unique_classes = data["unique_classes"]
    base = get_dataset_base(ds_name)
    n_shot = int(ds_name.split("_")[-1].replace("shot", ""))
    ds_label = DATASET_LABELS.get(base, base)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, title, syn_key, syn_label_key, marker, label_name in [
        (ax1, "LLM filtrado geométricamente", "kept_emb", "kept_labels", "^", "LLM filtrado"),
        (ax2, "SMOTE", "smote_emb", "smote_labels", "s", "SMOTE"),
    ]:
        # Test data (background)
        s, e = offsets["test_emb"]
        if s < e:
            test_labels = data["test_labels"]
            for cls in unique_classes:
                cls_mask = test_labels == cls
                tc = coords_2d[s:e][cls_mask]
                ax.scatter(tc[:, 0], tc[:, 1], c=[class_colors[cls]], marker=".",
                           s=6, alpha=0.10, zorder=0, rasterized=True)

        # Synthetic data
        s, e = offsets[syn_key]
        if s < e:
            syn_labels = data[syn_label_key]
            for cls in unique_classes:
                cls_mask = syn_labels == cls
                sc = coords_2d[s:e][cls_mask]
                ax.scatter(sc[:, 0], sc[:, 1], c=[class_colors[cls]], marker=marker,
                           s=35, alpha=0.55, zorder=2)

        # Real data (foreground)
        s, e = offsets["real_emb"]
        real_labels = data["real_labels"]
        for cls in unique_classes:
            cls_mask = real_labels == cls
            rc = coords_2d[s:e][cls_mask]
            ax.scatter(rc[:, 0], rc[:, 1], c=[class_colors[cls]], marker="o",
                       s=55, alpha=0.85, zorder=4, edgecolors="white", linewidths=0.5)

        # Centroids
        s, e = offsets["centroids"]
        for i, cls in enumerate(unique_classes):
            cx, cy = coords_2d[s + i]
            ax.scatter(cx, cy, c=[class_colors[cls]], marker="*",
                       s=250, alpha=1.0, zorder=5, edgecolors="black", linewidths=1.0)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE dim. 1")
        ax.set_ylabel("t-SNE dim. 2")
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = []
    for cls in unique_classes:
        legend_elements.append(mpatches.Patch(color=class_colors[cls], label=cls))
    legend_elements.extend([
        plt.Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="Real"),
        plt.Line2D([0], [0], marker="^", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="LLM filtrado"),
        plt.Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
                   markersize=8, linestyle="None", label="SMOTE"),
        plt.Line2D([0], [0], marker=".", color="gray", markerfacecolor="gray",
                   markersize=6, linestyle="None", label="Test"),
        plt.Line2D([0], [0], marker="*", color="gray", markerfacecolor="gray",
                   markersize=12, linestyle="None", label="Centroide"),
    ])
    fig.legend(handles=legend_elements, loc="lower center", ncol=min(len(unique_classes) + 5, 8),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Comparación en espacio de embeddings — {ds_label}, {n_shot}-shot",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    for ext in ["png", "pdf"]:
        path = FIGURE_DIR / f"fig_tsne_filtered_vs_smote_{base}.{ext}"
        plt.savefig(path)
        print(f"  Saved: {path}", flush=True)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ENHANCED EMBEDDING SPACE VISUALIZATIONS")
    print("=" * 70)

    model = SentenceTransformer("all-mpnet-base-v2")

    for ds_name in DATASETS:
        print(f"\nProcessing {ds_name}...", flush=True)
        base = get_dataset_base(ds_name)

        # Compute all data
        data = compute_all_data(ds_name, model)

        if len(data["kept_emb"]) == 0:
            print(f"  No LLM data for {ds_name}, skipping.")
            continue

        # Fit shared t-SNE
        coords_2d, offsets = fit_tsne(data)

        # Class colors
        class_colors = get_class_colors(data["unique_classes"])

        # Generate both figure types
        plot_before_after(ds_name, data, coords_2d, offsets, class_colors)
        plot_filtered_vs_smote(ds_name, data, coords_2d, offsets, class_colors)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
