#!/usr/bin/env python3
"""
Visualize progressive waves in embedding space using t-SNE.

Generates figures OUTSIDE the thesis folder (results/progressive_waves/figures/)
to understand wave behavior:

1. Wave-by-wave t-SNE: 4 panels showing each wave's accepted samples colored by wave
2. Cumulative t-SNE: shows how pools grow as waves are added
3. Distance-to-centroid distribution per wave (histogram)
4. Wave composition in final selection (which waves contribute to top-50)

Uses cached wave generations — no LLM calls needed.
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

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
FIG_DIR = PROJECT_ROOT / "results" / "progressive_waves" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTHETIC_PER_CLASS = 50
CANDIDATES_PER_WAVE = 75
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

WAVE_CONFIGS = [
    {"name": "exploratory", "threshold_k": -2.0},
    {"name": "moderate", "threshold_k": -1.0},
    {"name": "focused", "threshold_k": 0.0},
    {"name": "ultra_precise", "threshold_k": 0.5},
]

WAVE_COLORS = ["#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
WAVE_MARKERS = ["o", "s", "D", "^"]
WAVE_LABELS = ["W0: exploratory", "W1: moderate", "W2: focused", "W3: ultra-precise"]

# Datasets to visualize (fewer classes = cleaner plots)
VIZ_DATASETS = ["20newsgroups_10shot", "emotion_10shot"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(name):
    with open(DATA_DIR / f"{name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_dataset_base(name):
    bases = ["20newsgroups_20class", "20newsgroups", "hate_speech_davidson",
             "sms_spam", "ag_news", "emotion", "dbpedia14"]
    for base in bases:
        if name.startswith(base + "_"):
            return base
    return name


def get_wave_cache_key(dataset_name, class_name, n_shot, n_generate, wave_index):
    raw = f"progressive_waves_v1_{dataset_name}_{class_name}_{n_shot}_{n_generate}_wave{wave_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def compute_wave_threshold(real_embeddings, real_labels, target_class, cascade, threshold_k):
    labels_arr = np.array(real_labels)
    class_mask = labels_arr == target_class
    class_embs = real_embeddings[class_mask]
    if len(class_embs) < 2:
        return 0.0, 0.0, 0.0
    anchor = class_embs.mean(axis=0)
    scores, _ = cascade.compute_quality_scores(
        class_embs, anchor, real_embeddings, labels_arr, target_class
    )
    return float(np.mean(scores)) + threshold_k * float(np.std(scores)), float(np.mean(scores)), float(np.std(scores))


def load_wave_data(ds_name, model, cascade):
    """Load real data + all wave pools for visualization."""
    train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(test_texts, show_progress_bar=False)

    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    n_shot = int(ds_name.split("_")[-1].replace("shot", ""))

    wave_data = {w: {"emb": [], "scores": [], "labels": []} for w in range(4)}
    centroids = {}

    for cls in unique_classes:
        cls_mask = labels_arr == cls
        cls_emb = train_emb[cls_mask]
        centroids[cls] = cls_emb.mean(axis=0)

        for w_idx in range(4):
            cache_key = get_wave_cache_key(ds_name, cls, n_shot, CANDIDATES_PER_WAVE, w_idx)
            cache_file = CACHE_DIR / f"{cache_key}.json"

            if not cache_file.exists():
                continue
            with open(cache_file) as f:
                cached = json.load(f)
            gen_texts = cached.get("texts", [])
            if not gen_texts:
                continue

            gen_emb = model.encode(gen_texts, show_progress_bar=False)
            anchor = cls_emb.mean(axis=0)
            scores, _ = cascade.compute_quality_scores(
                gen_emb, anchor, train_emb, labels_arr, cls
            )

            # Apply threshold
            threshold, mean_real, std_real = compute_wave_threshold(
                train_emb, train_labels, cls, cascade, WAVE_CONFIGS[w_idx]["threshold_k"]
            )
            accepted_mask = scores >= threshold
            if not accepted_mask.any() and len(gen_texts) > 0 and std_real > 0:
                accepted_mask = scores >= (threshold - 0.5 * std_real)

            wave_data[w_idx]["emb"].append(gen_emb[accepted_mask])
            wave_data[w_idx]["scores"].append(scores[accepted_mask])
            wave_data[w_idx]["labels"].extend([cls] * int(accepted_mask.sum()))

    # Stack
    for w in range(4):
        if wave_data[w]["emb"]:
            wave_data[w]["emb"] = np.vstack(wave_data[w]["emb"])
            wave_data[w]["scores"] = np.concatenate(wave_data[w]["scores"])
        else:
            wave_data[w]["emb"] = np.zeros((0, 768))
            wave_data[w]["scores"] = np.array([])
        wave_data[w]["labels"] = np.array(wave_data[w]["labels"])

    return {
        "train_emb": train_emb, "train_labels": train_labels,
        "test_emb": test_emb, "test_labels": test_labels,
        "unique_classes": unique_classes, "centroids": centroids,
        "wave_data": wave_data,
    }


# ============================================================================
# FIGURE 1: 4-panel wave-by-wave t-SNE
# ============================================================================

def plot_waves_tsne(ds_name, data):
    """4 panels, one per wave. Real data + test as background, wave samples highlighted."""
    base = get_dataset_base(ds_name)
    unique_classes = data["unique_classes"]
    class_colors = {cls: sns.color_palette("tab10")[i] for i, cls in enumerate(unique_classes)}

    # Fit t-SNE on all data combined
    arrays = [data["test_emb"], data["train_emb"]]
    offset_test = (0, len(data["test_emb"]))
    offset_real = (offset_test[1], offset_test[1] + len(data["train_emb"]))
    pos = offset_real[1]

    wave_offsets = {}
    for w in range(4):
        emb = data["wave_data"][w]["emb"]
        if len(emb) > 0:
            arrays.append(emb)
            wave_offsets[w] = (pos, pos + len(emb))
            pos += len(emb)
        else:
            wave_offsets[w] = (pos, pos)

    # Centroids
    centroid_emb = np.array([data["centroids"][cls] for cls in unique_classes])
    arrays.append(centroid_emb)
    offset_centroids = (pos, pos + len(centroid_emb))

    all_emb = np.vstack(arrays)
    print(f"  Fitting t-SNE on {len(all_emb)} points for wave panels...", flush=True)
    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    coords = tsne.fit_transform(all_emb)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for w_idx, ax in enumerate(axes.flat):
        # Test background
        s, e = offset_test
        test_labels = np.array(data["test_labels"])
        for cls in unique_classes:
            mask = test_labels == cls
            tc = coords[s:e][mask]
            ax.scatter(tc[:, 0], tc[:, 1], c=[class_colors[cls]], marker=".",
                       s=4, alpha=0.08, zorder=0, rasterized=True)

        # Real data
        s, e = offset_real
        real_labels = np.array(data["train_labels"])
        for cls in unique_classes:
            mask = real_labels == cls
            rc = coords[s:e][mask]
            ax.scatter(rc[:, 0], rc[:, 1], c=[class_colors[cls]], marker="o",
                       s=50, alpha=0.8, zorder=4, edgecolors="white", linewidths=0.5)

        # Wave samples
        s, e = wave_offsets[w_idx]
        if s < e:
            wave_labels = data["wave_data"][w_idx]["labels"]
            for cls in unique_classes:
                mask = wave_labels == cls
                wc = coords[s:e][mask]
                ax.scatter(wc[:, 0], wc[:, 1], c=[class_colors[cls]],
                           marker=WAVE_MARKERS[w_idx], s=30, alpha=0.55, zorder=2,
                           edgecolors=WAVE_COLORS[w_idx], linewidths=1.2)

        # Centroids
        s, e = offset_centroids
        for i, cls in enumerate(unique_classes):
            ax.scatter(coords[s+i, 0], coords[s+i, 1], c=[class_colors[cls]],
                       marker="*", s=200, zorder=5, edgecolors="black", linewidths=0.8)

        n_accepted = len(data["wave_data"][w_idx]["emb"])
        ax.set_title(f"{WAVE_LABELS[w_idx]} ({n_accepted} accepted)",
                     fontsize=12, fontweight="bold", color=WAVE_COLORS[w_idx])
        ax.set_xlabel("t-SNE dim. 1", fontsize=9)
        ax.set_ylabel("t-SNE dim. 2", fontsize=9)
        ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [mpatches.Patch(color=class_colors[cls], label=cls) for cls in unique_classes]
    legend_elements.extend([
        plt.Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=7, linestyle="None", label="Real"),
        plt.Line2D([0], [0], marker="*", color="gray", markerfacecolor="gray",
                   markersize=10, linestyle="None", label="Centroid"),
    ])
    for w in range(4):
        legend_elements.append(
            plt.Line2D([0], [0], marker=WAVE_MARKERS[w], color=WAVE_COLORS[w],
                       markerfacecolor="gray", markersize=7, linestyle="None",
                       label=WAVE_LABELS[w])
        )

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(unique_classes) + 6, 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Progressive Waves — {base} 10-shot", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    path = FIG_DIR / f"waves_tsne_{base}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 2: All waves overlaid (single panel, colored by wave)
# ============================================================================

def plot_waves_overlay(ds_name, data):
    """Single panel with all waves overlaid, colored by wave index (not by class)."""
    base = get_dataset_base(ds_name)
    unique_classes = data["unique_classes"]

    # Fit t-SNE
    arrays = [data["train_emb"]]
    offset_real = (0, len(data["train_emb"]))
    pos = offset_real[1]

    wave_offsets = {}
    for w in range(4):
        emb = data["wave_data"][w]["emb"]
        if len(emb) > 0:
            arrays.append(emb)
            wave_offsets[w] = (pos, pos + len(emb))
            pos += len(emb)
        else:
            wave_offsets[w] = (pos, pos)

    centroid_emb = np.array([data["centroids"][cls] for cls in unique_classes])
    arrays.append(centroid_emb)
    offset_centroids = (pos, pos + len(centroid_emb))

    all_emb = np.vstack(arrays)
    print(f"  Fitting t-SNE on {len(all_emb)} points for overlay...", flush=True)
    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    coords = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Real data (gray)
    s, e = offset_real
    ax.scatter(coords[s:e, 0], coords[s:e, 1], c="black", marker="o",
               s=60, alpha=0.9, zorder=5, edgecolors="white", linewidths=0.5, label="Real data")

    # Waves (colored by wave, not class)
    for w in range(4):
        s, e = wave_offsets[w]
        if s < e:
            ax.scatter(coords[s:e, 0], coords[s:e, 1], c=WAVE_COLORS[w],
                       marker=WAVE_MARKERS[w], s=25, alpha=0.5, zorder=3-w*0.5,
                       edgecolors=WAVE_COLORS[w], linewidths=0.5,
                       label=f"{WAVE_LABELS[w]} ({e-s})")

    # Centroids
    class_colors = {cls: sns.color_palette("tab10")[i] for i, cls in enumerate(unique_classes)}
    s, e = offset_centroids
    for i, cls in enumerate(unique_classes):
        ax.scatter(coords[s+i, 0], coords[s+i, 1], c=[class_colors[cls]],
                   marker="*", s=300, zorder=6, edgecolors="black", linewidths=1.0)

    ax.set_xlabel("t-SNE dim. 1")
    ax.set_ylabel("t-SNE dim. 2")
    ax.set_title(f"All waves overlaid — {base} 10-shot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = FIG_DIR / f"waves_overlay_{base}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 3: Distance-to-centroid distributions per wave
# ============================================================================

def plot_distance_distributions(ds_name, data):
    """Histogram/KDE of distance-to-centroid for each wave vs real data."""
    base = get_dataset_base(ds_name)
    unique_classes = data["unique_classes"]
    train_labels = np.array(data["train_labels"])

    # Compute distances to centroid for real data
    real_dists = []
    for cls in unique_classes:
        mask = train_labels == cls
        cls_emb = data["train_emb"][mask]
        centroid = data["centroids"][cls]
        dists = np.linalg.norm(cls_emb - centroid, axis=1)
        real_dists.extend(dists)

    # Compute distances for each wave
    wave_dists = {}
    for w in range(4):
        wd = data["wave_data"][w]
        if len(wd["emb"]) == 0:
            wave_dists[w] = []
            continue
        dists = []
        for cls in unique_classes:
            mask = wd["labels"] == cls
            if not mask.any():
                continue
            emb = wd["emb"][mask]
            centroid = data["centroids"][cls]
            d = np.linalg.norm(emb - centroid, axis=1)
            dists.extend(d)
        wave_dists[w] = dists

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Real data distribution
    ax.hist(real_dists, bins=30, alpha=0.3, color="black", density=True, label="Real data")

    # Wave distributions
    for w in range(4):
        if wave_dists[w]:
            ax.hist(wave_dists[w], bins=30, alpha=0.25, color=WAVE_COLORS[w],
                    density=True, label=WAVE_LABELS[w])
            # Also add KDE line
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(wave_dists[w])
            x = np.linspace(min(wave_dists[w]), max(wave_dists[w]), 200)
            ax.plot(x, kde(x), color=WAVE_COLORS[w], linewidth=2)

    # Real KDE
    from scipy.stats import gaussian_kde
    kde_real = gaussian_kde(real_dists)
    x = np.linspace(min(real_dists), max(real_dists), 200)
    ax.plot(x, kde_real(x), color="black", linewidth=2.5, linestyle="--")

    ax.set_xlabel("Euclidean distance to class centroid", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distance distributions by wave — {base} 10-shot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = FIG_DIR / f"waves_distance_dist_{base}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 4: Wave composition in top-50 selection
# ============================================================================

def plot_wave_composition(ds_name, data):
    """Stacked bar showing which waves contribute to the top-50 selection per class."""
    base = get_dataset_base(ds_name)
    unique_classes = data["unique_classes"]

    # Pool all waves, select top-50 per class by raw score
    class_compositions = {}
    for cls in unique_classes:
        all_emb, all_scores, all_wave_idx = [], [], []
        for w in range(4):
            wd = data["wave_data"][w]
            mask = wd["labels"] == cls
            if not mask.any():
                continue
            all_emb.append(wd["emb"][mask])
            all_scores.append(wd["scores"][mask])
            all_wave_idx.extend([w] * int(mask.sum()))

        if not all_scores:
            class_compositions[cls] = {0: 0, 1: 0, 2: 0, 3: 0}
            continue

        scores = np.concatenate(all_scores)
        n = min(N_SYNTHETIC_PER_CLASS, len(scores))
        top_idx = np.argsort(scores)[-n:]

        wave_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in top_idx:
            wave_counts[all_wave_idx[i]] += 1
        class_compositions[cls] = wave_counts

    # Plot stacked bar
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(unique_classes) * 1.2), 6))

    x = np.arange(len(unique_classes))
    width = 0.6
    bottoms = np.zeros(len(unique_classes))

    for w in range(4):
        counts = [class_compositions[cls][w] for cls in unique_classes]
        ax.bar(x, counts, width, bottom=bottoms, color=WAVE_COLORS[w],
               label=WAVE_LABELS[w], edgecolor="white", linewidth=0.5)
        # Add count labels
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i, bottoms[i] + c/2, str(c), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(unique_classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of samples in top-50")
    ax.set_title(f"Wave composition in final selection (all_equal_weight) — {base} 10-shot",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    path = FIG_DIR / f"waves_composition_{base}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 5: Score vs wave (violin plot)
# ============================================================================

def plot_score_violin(ds_name, data):
    """Violin plot showing score distributions per wave."""
    base = get_dataset_base(ds_name)

    plot_data = []
    for w in range(4):
        scores = data["wave_data"][w]["scores"]
        if len(scores) > 0:
            for s in scores:
                plot_data.append({"Wave": WAVE_LABELS[w], "Score": float(s), "wave_idx": w})

    if not plot_data:
        return

    import pandas as pd
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    parts = ax.violinplot(
        [df[df["wave_idx"] == w]["Score"].values for w in range(4) if len(df[df["wave_idx"] == w]) > 0],
        showmeans=True, showmedians=True
    )

    # Color violins
    available_waves = [w for w in range(4) if len(df[df["wave_idx"] == w]) > 0]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(WAVE_COLORS[available_waves[i]])
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(available_waves) + 1))
    ax.set_xticklabels([WAVE_LABELS[w] for w in available_waves], fontsize=9)
    ax.set_ylabel("Cascade L1 Score")
    ax.set_title(f"Quality score distributions by wave — {base} 10-shot",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = FIG_DIR / f"waves_score_violin_{base}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PROGRESSIVE WAVES — VISUALIZATION")
    print("=" * 70)

    import torch
    torch.backends.cuda.preferred_blas_library("cublaslt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    cascade = FilterCascade(**FILTER_CONFIG)

    for ds_name in VIZ_DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {ds_name}")
        print(f"{'='*60}")

        data = load_wave_data(ds_name, model, cascade)

        for w in range(4):
            n = len(data["wave_data"][w]["emb"])
            print(f"  {WAVE_LABELS[w]}: {n} accepted samples")

        plot_waves_tsne(ds_name, data)
        plot_waves_overlay(ds_name, data)
        plot_distance_distributions(ds_name, data)
        plot_wave_composition(ds_name, data)
        plot_score_violin(ds_name, data)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
