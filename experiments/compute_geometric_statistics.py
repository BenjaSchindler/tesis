#!/usr/bin/env python3
"""Compute geometric statistics of filtering effect.

Measures 6 geometric metrics for 4 augmentation conditions across 21 dataset configs:
  1. real_only: just training data
  2. real_plus_unfiltered: real + ALL LLM candidates
  3. real_plus_filtered: real + top-N filtered LLM
  4. real_plus_smote: real + SMOTE samples

Metrics: intra-class distance, inter-class separation, silhouette score,
         Davies-Bouldin index, coverage, density ratio.

Uses cached LLM generations — no new API calls needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "geometric_statistics"
TABLE_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

DATASETS = [
    "sms_spam_10shot", "sms_spam_25shot", "sms_spam_50shot",
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "20newsgroups_10shot", "20newsgroups_25shot", "20newsgroups_50shot",
    "ag_news_10shot", "ag_news_25shot", "ag_news_50shot",
    "emotion_10shot", "emotion_25shot", "emotion_50shot",
    "dbpedia14_10shot", "dbpedia14_25shot", "dbpedia14_50shot",
    "20newsgroups_20class_10shot", "20newsgroups_20class_25shot", "20newsgroups_20class_50shot",
]

DATASET_N_CLASSES = {
    "sms_spam": 2, "hate_speech_davidson": 3, "20newsgroups": 4,
    "ag_news": 4, "emotion": 6, "dbpedia14": 14, "20newsgroups_20class": 20,
}


def get_dataset_base(name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
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


# ============================================================================
# GEOMETRIC METRICS
# ============================================================================

def compute_intra_class_distance(embeddings, labels, unique_classes):
    """Mean Euclidean distance of samples to their class centroid, averaged across classes."""
    distances = []
    for cls in unique_classes:
        mask = labels == cls
        cls_emb = embeddings[mask]
        if len(cls_emb) < 2:
            continue
        centroid = cls_emb.mean(axis=0)
        dists = np.linalg.norm(cls_emb - centroid, axis=1)
        distances.append(dists.mean())
    return np.mean(distances) if distances else 0.0


def compute_inter_class_separation(embeddings, labels, unique_classes):
    """Mean Euclidean distance between all pairs of class centroids."""
    centroids = {}
    for cls in unique_classes:
        mask = labels == cls
        cls_emb = embeddings[mask]
        if len(cls_emb) > 0:
            centroids[cls] = cls_emb.mean(axis=0)
    if len(centroids) < 2:
        return 0.0
    dists = []
    classes = list(centroids.keys())
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            d = np.linalg.norm(centroids[classes[i]] - centroids[classes[j]])
            dists.append(d)
    return np.mean(dists)


def compute_coverage(real_emb, synthetic_emb, real_labels, unique_classes, k=5):
    """Fraction of real samples that have at least one synthetic neighbor within a class-adaptive radius."""
    if len(synthetic_emb) == 0:
        return 0.0
    covered = 0
    total = 0
    for cls in unique_classes:
        real_mask = real_labels == cls
        cls_real = real_emb[real_mask]
        if len(cls_real) < 2:
            continue
        # Radius: 90th percentile of intra-class real distances
        centroid = cls_real.mean(axis=0)
        real_dists = np.linalg.norm(cls_real - centroid, axis=1)
        radius = np.percentile(real_dists, 90)
        if radius < 1e-8:
            radius = 1.0
        # Check coverage
        for r in cls_real:
            dists_to_syn = np.linalg.norm(synthetic_emb - r, axis=1)
            if np.min(dists_to_syn) <= radius:
                covered += 1
            total += 1
    return covered / total if total > 0 else 0.0


def compute_density_ratio(real_emb, synthetic_emb, real_labels, syn_labels, unique_classes, k=5):
    """Ratio of synthetic density to real density near class regions. Values near 1.0 = well-matched."""
    ratios = []
    for cls in unique_classes:
        real_mask = real_labels == cls
        syn_mask = syn_labels == cls
        cls_real = real_emb[real_mask]
        cls_syn = synthetic_emb[syn_mask]
        if len(cls_real) < k + 1 or len(cls_syn) < k + 1:
            continue
        # Mean k-NN distance as density proxy (lower distance = higher density)
        nn_real = NearestNeighbors(n_neighbors=min(k, len(cls_real) - 1)).fit(cls_real)
        real_dists, _ = nn_real.kneighbors(cls_real)
        mean_real_dist = real_dists[:, -1].mean()

        nn_syn = NearestNeighbors(n_neighbors=min(k, len(cls_syn) - 1)).fit(cls_syn)
        syn_dists, _ = nn_syn.kneighbors(cls_syn)
        mean_syn_dist = syn_dists[:, -1].mean()

        if mean_real_dist > 1e-8:
            ratios.append(mean_syn_dist / mean_real_dist)
    return np.mean(ratios) if ratios else 1.0


def compute_all_metrics(combined_emb, combined_labels, real_emb, synthetic_emb,
                        real_labels, syn_labels, unique_classes):
    """Compute all 6 geometric metrics for one augmentation condition."""
    label_ints = np.array([list(unique_classes).index(l) for l in combined_labels])

    metrics = {
        "intra_class_dist": compute_intra_class_distance(combined_emb, combined_labels, unique_classes),
        "inter_class_sep": compute_inter_class_separation(combined_emb, combined_labels, unique_classes),
        "coverage": compute_coverage(real_emb, synthetic_emb, real_labels, unique_classes),
        "density_ratio": compute_density_ratio(real_emb, synthetic_emb, real_labels, syn_labels, unique_classes),
    }

    # Silhouette and Davies-Bouldin need at least 2 classes with 2+ samples
    class_counts = defaultdict(int)
    for l in label_ints:
        class_counts[l] += 1
    valid_classes = sum(1 for c in class_counts.values() if c >= 2)

    if valid_classes >= 2 and len(combined_emb) > valid_classes:
        try:
            metrics["silhouette"] = silhouette_score(combined_emb, label_ints)
        except Exception:
            metrics["silhouette"] = 0.0
        try:
            metrics["davies_bouldin"] = davies_bouldin_score(combined_emb, label_ints)
        except Exception:
            metrics["davies_bouldin"] = 0.0
    else:
        metrics["silhouette"] = 0.0
        metrics["davies_bouldin"] = 0.0

    return metrics


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis():
    print("=" * 70)
    print("GEOMETRIC STATISTICS OF FILTERING EFFECT")
    print("=" * 70)

    model = SentenceTransformer("all-mpnet-base-v2")
    cascade = FilterCascade(**FILTER_CONFIG)
    all_results = []

    for ds_idx, ds_name in enumerate(DATASETS):
        print(f"\n[{ds_idx+1}/{len(DATASETS)}] Processing {ds_name}...", flush=True)

        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)

        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)
        n_shot = int(ds_name.split("_")[-1].replace("shot", ""))
        n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)

        # Collect per-class data
        all_unfiltered_emb, all_unfiltered_labels = [], []
        all_filtered_emb, all_filtered_labels = [], []
        all_smote_emb, all_smote_labels = [], []

        has_cache = True
        for cls in unique_classes:
            cls_mask = labels_arr == cls
            cls_emb = train_emb[cls_mask]

            # Load cached LLM generations
            cache_key = get_cache_key(ds_name, cls, n_shot, n_gen)
            cache_file = CACHE_DIR / f"{cache_key}.json"
            if not cache_file.exists():
                has_cache = False
                break
            with open(cache_file) as f:
                cached = json.load(f)
            if not cached.get("texts"):
                has_cache = False
                break

            gen_emb = model.encode(cached["texts"], show_progress_bar=False)

            # All unfiltered candidates
            all_unfiltered_emb.append(gen_emb)
            all_unfiltered_labels.extend([cls] * len(gen_emb))

            # Score and filter
            anchor = cls_emb.mean(axis=0)
            scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
            target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
            top_idx = np.argsort(scores)[-target_n:]
            all_filtered_emb.append(gen_emb[top_idx])
            all_filtered_labels.extend([cls] * len(top_idx))

            # SMOTE
            smote_emb = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=42)
            if len(smote_emb) > 0:
                all_smote_emb.append(smote_emb)
                all_smote_labels.extend([cls] * len(smote_emb))

        if not has_cache:
            print(f"  Skipping {ds_name} (missing cache)")
            continue

        # Build condition arrays
        unfiltered_emb = np.vstack(all_unfiltered_emb) if all_unfiltered_emb else np.zeros((0, 768))
        filtered_emb = np.vstack(all_filtered_emb) if all_filtered_emb else np.zeros((0, 768))
        smote_emb = np.vstack(all_smote_emb) if all_smote_emb else np.zeros((0, 768))
        unfiltered_labels = np.array(all_unfiltered_labels)
        filtered_labels = np.array(all_filtered_labels)
        smote_labels = np.array(all_smote_labels)

        # Condition 1: real only
        m1 = compute_all_metrics(
            train_emb, labels_arr, train_emb, np.zeros((0, 768)),
            labels_arr, np.array([]), unique_classes)

        # Condition 2: real + unfiltered LLM
        comb_emb2 = np.vstack([train_emb, unfiltered_emb])
        comb_labels2 = np.concatenate([labels_arr, unfiltered_labels])
        m2 = compute_all_metrics(
            comb_emb2, comb_labels2, train_emb, unfiltered_emb,
            labels_arr, unfiltered_labels, unique_classes)

        # Condition 3: real + filtered LLM
        comb_emb3 = np.vstack([train_emb, filtered_emb])
        comb_labels3 = np.concatenate([labels_arr, filtered_labels])
        m3 = compute_all_metrics(
            comb_emb3, comb_labels3, train_emb, filtered_emb,
            labels_arr, filtered_labels, unique_classes)

        # Condition 4: real + SMOTE
        if len(smote_emb) > 0:
            comb_emb4 = np.vstack([train_emb, smote_emb])
            comb_labels4 = np.concatenate([labels_arr, smote_labels])
            m4 = compute_all_metrics(
                comb_emb4, comb_labels4, train_emb, smote_emb,
                labels_arr, smote_labels, unique_classes)
        else:
            m4 = {k: 0.0 for k in m1}

        result = {
            "dataset": ds_name,
            "dataset_base": get_dataset_base(ds_name),
            "n_shot": n_shot,
            "n_classes": len(unique_classes),
            "n_real": len(train_emb),
            "n_unfiltered": len(unfiltered_emb),
            "n_filtered": len(filtered_emb),
            "n_smote": len(smote_emb),
            "real_only": m1,
            "real_plus_unfiltered": m2,
            "real_plus_filtered": m3,
            "real_plus_smote": m4,
        }
        all_results.append(result)

        print(f"  Silhouette: real={m1['silhouette']:.3f}, +unfilt={m2['silhouette']:.3f}, "
              f"+filt={m3['silhouette']:.3f}, +SMOTE={m4['silhouette']:.3f}", flush=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return convert_numpy(d)

    all_results = deep_convert(all_results)

    # Save results
    output = {"timestamp": datetime.now().isoformat(), "n_datasets": len(all_results), "results": all_results}
    with open(RESULTS_DIR / "geometric_stats.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR / 'geometric_stats.json'}")

    return all_results


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_main_table(results):
    """Generate main comparison table: 4 conditions × 6 metrics, averaged across datasets."""
    conditions = ["real_only", "real_plus_unfiltered", "real_plus_filtered", "real_plus_smote"]
    cond_labels = {
        "real_only": "Solo datos reales",
        "real_plus_unfiltered": "+ LLM sin filtrar",
        "real_plus_filtered": "+ LLM filtrado",
        "real_plus_smote": "+ SMOTE",
    }
    metrics = ["intra_class_dist", "inter_class_sep", "silhouette", "davies_bouldin", "coverage", "density_ratio"]
    metric_headers = [
        "Dist. Intra",
        "Sep. Inter",
        "Silhouette",
        "Davies-B.",
        "Cobertura",
        "Ratio Dens.",
    ]

    # Aggregate means and stds
    agg = {}
    for cond in conditions:
        agg[cond] = defaultdict(list)
        for r in results:
            for m in metrics:
                agg[cond][m].append(r[cond][m])

    # Paired t-tests: filtered vs unfiltered, filtered vs SMOTE
    ttests = {}
    for comparison, c1, c2 in [
        ("filt_vs_unfilt", "real_plus_filtered", "real_plus_unfiltered"),
        ("filt_vs_smote", "real_plus_filtered", "real_plus_smote"),
    ]:
        ttests[comparison] = {}
        for m in metrics:
            v1 = [r[c1][m] for r in results]
            v2 = [r[c2][m] for r in results]
            diffs = [a - b for a, b in zip(v1, v2)]
            if np.std(diffs) > 1e-10:
                t_stat, p_val = stats.ttest_1samp(diffs, 0)
                ttests[comparison][m] = p_val
            else:
                ttests[comparison][m] = 1.0

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Estadísticas geométricas del espacio de embeddings para 4 condiciones de aumentación. "
                 r"Los valores son promedios sobre las 21 configuraciones de dataset. "
                 r"Dist. Intra: distancia media al centroide de clase (menor = clústeres más compactos). "
                 r"Sep. Inter: distancia media entre centroides (mayor = mejor separación). "
                 r"Silhouette: cohesión + separación combinada (mayor = mejor). "
                 r"Davies-B.: ratio intra/inter (menor = mejor). "
                 r"Cobertura: fracción de datos reales cubiertos por sintéticos. "
                 r"Ratio Dens.: ratio de densidad sintética/real (cercano a 1.0 = mejor).}")
    lines.append(r"\label{tab:geometric_statistics}")
    lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Condición} & " + " & ".join(f"\\textbf{{{h}}}" for h in metric_headers) + r" \\")
    lines.append(r"\midrule")

    for cond in conditions:
        label = cond_labels[cond]
        vals = []
        for m in metrics:
            mean_v = np.mean(agg[cond][m])
            # Format based on metric type
            if m == "coverage":
                vals.append(f"{mean_v:.1%}")
            elif m == "density_ratio":
                vals.append(f"{mean_v:.2f}")
            else:
                vals.append(f"{mean_v:.3f}")
        row = f"{label} & " + " & ".join(vals) + r" \\"
        if cond == "real_plus_filtered":
            row = r"\rowcolor{green!10}" + row
        lines.append(row)

    lines.append(r"\midrule")

    # Add significance notes
    sig_notes = []
    for m_idx, m in enumerate(metrics):
        p_unfilt = ttests["filt_vs_unfilt"].get(m, 1.0)
        p_smote = ttests["filt_vs_smote"].get(m, 1.0)
        if p_unfilt < 0.001:
            sig_notes.append(f"{metric_headers[m_idx]}: filtrado vs sin filtrar $p < 0.001$")
        elif p_unfilt < 0.05:
            sig_notes.append(f"{metric_headers[m_idx]}: filtrado vs sin filtrar $p = {p_unfilt:.3f}$")

    if sig_notes:
        note = "; ".join(sig_notes[:3])  # Limit to 3 notes
        lines.append(r"\multicolumn{" + str(len(metrics) + 1) + r"}{l}{\small " + note + r"} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(TABLE_DIR / "tab_geometric_statistics.tex", "w") as f:
        f.write(tex)
    print(f"Saved: {TABLE_DIR / 'tab_geometric_statistics.tex'}")
    return tex


def generate_per_dataset_table(results):
    """Generate per-dataset breakdown for silhouette and Davies-Bouldin."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Estadísticas geométricas por conjunto de datos. Se reportan silhouette score "
                 r"y Davies-Bouldin index para las condiciones con y sin filtrado. "
                 r"$\Delta$ Silh. = diferencia filtrado vs sin filtrar (positivo = mejora).}")
    lines.append(r"\label{tab:geometric_per_dataset}")
    lines.append(r"\begin{tabular}{lrccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{N-shot} & "
                 r"\textbf{Silh. Real} & \textbf{Silh. +Unfilt} & \textbf{Silh. +Filt} & "
                 r"\textbf{$\Delta$ Silh.} & \textbf{DB +Filt} \\")
    lines.append(r"\midrule")

    prev_base = None
    for r in sorted(results, key=lambda x: (x["dataset_base"], x["n_shot"])):
        base = r["dataset_base"]
        if prev_base is not None and base != prev_base:
            lines.append(r"\midrule")
        prev_base = base

        s_real = r["real_only"]["silhouette"]
        s_unfilt = r["real_plus_unfiltered"]["silhouette"]
        s_filt = r["real_plus_filtered"]["silhouette"]
        delta_s = s_filt - s_unfilt
        db_filt = r["real_plus_filtered"]["davies_bouldin"]

        delta_str = f"+{delta_s:.3f}" if delta_s >= 0 else f"{delta_s:.3f}"

        lines.append(f"{base} & {r['n_shot']} & {s_real:.3f} & {s_unfilt:.3f} & "
                     f"{s_filt:.3f} & {delta_str} & {db_filt:.2f} \\\\")

    # Summary row
    mean_delta = np.mean([r["real_plus_filtered"]["silhouette"] - r["real_plus_unfiltered"]["silhouette"]
                          for r in results])
    filt_wins = sum(1 for r in results
                    if r["real_plus_filtered"]["silhouette"] > r["real_plus_unfiltered"]["silhouette"])
    win_pct = filt_wins / len(results) * 100

    lines.append(r"\midrule")
    delta_str = f"+{mean_delta:.3f}" if mean_delta >= 0 else f"{mean_delta:.3f}"
    lines.append(r"\multicolumn{7}{l}{\small Promedio $\Delta$ Silhouette (filtrado vs sin filtrar): "
                 f"{delta_str}, victoria filtrado: {win_pct:.0f}\\% ({filt_wins}/{len(results)})" + r"} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(TABLE_DIR / "tab_geometric_per_dataset.tex", "w") as f:
        f.write(tex)
    print(f"Saved: {TABLE_DIR / 'tab_geometric_per_dataset.tex'}")
    return tex


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_analysis()
    if results:
        print("\n" + "=" * 70)
        print("GENERATING TABLES")
        print("=" * 70)
        generate_main_table(results)
        generate_per_dataset_table(results)
        print("\nDone!")
    else:
        print("No results to generate tables from.")
