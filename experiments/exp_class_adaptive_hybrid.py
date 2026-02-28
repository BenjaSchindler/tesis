#!/usr/bin/env python3
"""Class-Adaptive Hybrid Augmentation Experiment.

Tests a hybrid strategy where geometric filtering is applied to hard classes
and SMOTE to easy ones, motivated by the finding that hard classes (F1<30%)
gain +10.44pp from filtering while easy classes (F1>80%) gain only +0.55pp.

Three difficulty estimation methods (no test data leakage):
1. CV-based: Stratified 5-fold CV on training data → per-class F1 estimate
2. Geometric: intra-class spread / nearest inter-class distance
3. Oracle: uses test F1 (upper bound, not practical)

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

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "class_adaptive_hybrid"
TABLE_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

SEEDS = [42, 123, 456]

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

CLASSIFIERS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "svc_linear": lambda seed: SVC(kernel="linear", random_state=seed),
    "ridge": lambda seed: RidgeClassifier(alpha=1.0),
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


def normalize_scores(raw_scores, method="minmax", temperature=1.0, min_weight=0.0):
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)
    weight_range = 1.0 - min_weight
    if method == "minmax":
        s_min, s_max = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    else:
        normalized = raw_scores
    if temperature != 1.0:
        normalized = np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / temperature)
    return min_weight + weight_range * normalized


# ============================================================================
# DIFFICULTY ESTIMATION
# ============================================================================

def estimate_difficulty_geometric(train_emb, train_labels, unique_classes):
    """Geometric proxy: intra_spread / nearest_inter_distance. Higher = harder."""
    labels_arr = np.array(train_labels)
    centroids = {}
    spreads = {}
    for cls in unique_classes:
        cls_emb = train_emb[labels_arr == cls]
        if len(cls_emb) == 0:
            continue
        centroid = cls_emb.mean(axis=0)
        centroids[cls] = centroid
        spreads[cls] = np.mean(np.linalg.norm(cls_emb - centroid, axis=1))

    difficulties = {}
    for cls in unique_classes:
        if cls not in centroids:
            difficulties[cls] = 1.0
            continue
        # Nearest other centroid
        min_inter = float("inf")
        for other in unique_classes:
            if other == cls or other not in centroids:
                continue
            d = np.linalg.norm(centroids[cls] - centroids[other])
            if d < min_inter:
                min_inter = d
        if min_inter == float("inf") or min_inter < 1e-8:
            difficulties[cls] = 1.0
        else:
            difficulties[cls] = spreads[cls] / min_inter
    return difficulties


def estimate_difficulty_cv(train_emb, train_labels, unique_classes, seed=42):
    """Stratified 5-fold CV per-class F1 estimate. Lower F1 = harder class."""
    labels_arr = np.array(train_labels)
    n = len(train_emb)

    if n < 4:
        return {cls: 0.5 for cls in unique_classes}

    # Use StratifiedKFold — much faster than LOOCV for large datasets
    n_splits = min(5, min(np.bincount(np.unique(labels_arr, return_inverse=True)[1])))
    if n_splits < 2:
        return {cls: 0.5 for cls in unique_classes}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(train_emb, labels_arr):
        X_tr, X_te = train_emb[train_idx], train_emb[test_idx]
        y_tr, y_te = labels_arr[train_idx], labels_arr[test_idx]
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        try:
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
        except Exception:
            pred = y_te  # fallback
        all_true.extend(y_te)
        all_pred.extend(pred)

    per_class_f1 = f1_score(all_true, all_pred, labels=unique_classes, average=None, zero_division=0)
    return {cls: f1 for cls, f1 in zip(unique_classes, per_class_f1)}


def estimate_difficulty_oracle(train_emb, train_labels, test_emb, test_labels, unique_classes, seed=42):
    """Oracle: train on real, evaluate on test. Returns per-class F1."""
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_emb, train_labels)
    pred = clf.predict(test_emb)
    per_class_f1 = f1_score(test_labels, pred, labels=unique_classes, average=None, zero_division=0)
    return {cls: f1 for cls, f1 in zip(unique_classes, per_class_f1)}


# ============================================================================
# HYBRID AUGMENTATION
# ============================================================================

def augment_pure_smote(train_emb, train_labels, unique_classes, seed):
    """SMOTE for all classes."""
    labels_arr = np.array(train_labels)
    all_syn_emb, all_syn_labels, all_weights = [], [], []
    for cls in unique_classes:
        cls_emb = train_emb[labels_arr == cls]
        smote_emb = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
        if len(smote_emb) > 0:
            all_syn_emb.append(smote_emb)
            all_syn_labels.extend([cls] * len(smote_emb))
            all_weights.extend([1.0] * len(smote_emb))
    if not all_syn_emb:
        return np.zeros((0, train_emb.shape[1])), [], []
    return np.vstack(all_syn_emb), all_syn_labels, all_weights


def augment_pure_geo(train_emb, train_labels, unique_classes, llm_data, seed):
    """LLM + geometric filtering + soft weights for all classes."""
    labels_arr = np.array(train_labels)
    cascade = FilterCascade(**FILTER_CONFIG)
    all_syn_emb, all_syn_labels, all_weights = [], [], []

    for cls in unique_classes:
        if cls not in llm_data or len(llm_data[cls]) == 0:
            # Fallback to SMOTE
            cls_emb = train_emb[labels_arr == cls]
            smote_emb = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
            if len(smote_emb) > 0:
                all_syn_emb.append(smote_emb)
                all_syn_labels.extend([cls] * len(smote_emb))
                all_weights.extend([1.0] * len(smote_emb))
            continue

        gen_emb = llm_data[cls]
        cls_emb = train_emb[labels_arr == cls]
        anchor = cls_emb.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        kept_emb = gen_emb[top_idx]
        kept_scores = scores[top_idx]
        weights = normalize_scores(kept_scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)

        all_syn_emb.append(kept_emb)
        all_syn_labels.extend([cls] * len(kept_emb))
        all_weights.extend(weights.tolist())

    if not all_syn_emb:
        return np.zeros((0, train_emb.shape[1])), [], []
    return np.vstack(all_syn_emb), all_syn_labels, all_weights


def augment_hybrid(train_emb, train_labels, unique_classes, llm_data, class_difficulties,
                   threshold, threshold_type, seed):
    """Hybrid: LLM+filtering for hard classes, SMOTE for easy ones.

    threshold_type: 'f1' (lower = harder) or 'geometric' (higher = harder)
    """
    labels_arr = np.array(train_labels)
    cascade = FilterCascade(**FILTER_CONFIG)
    all_syn_emb, all_syn_labels, all_weights = [], [], []
    n_llm_classes = 0
    n_smote_classes = 0

    for cls in unique_classes:
        difficulty = class_difficulties.get(cls, 0.5)
        cls_emb = train_emb[labels_arr == cls]

        # Determine if class is hard
        if threshold_type == "f1":
            is_hard = difficulty < threshold  # Low F1 = hard
        elif threshold_type == "geometric":
            is_hard = difficulty > threshold  # High ratio = hard
        else:
            is_hard = True

        if is_hard and cls in llm_data and len(llm_data[cls]) > 0:
            # LLM + geometric filtering + soft weights
            gen_emb = llm_data[cls]
            anchor = cls_emb.mean(axis=0)
            scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
            target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
            top_idx = np.argsort(scores)[-target_n:]
            kept_emb = gen_emb[top_idx]
            kept_scores = scores[top_idx]
            weights = normalize_scores(kept_scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)

            all_syn_emb.append(kept_emb)
            all_syn_labels.extend([cls] * len(kept_emb))
            all_weights.extend(weights.tolist())
            n_llm_classes += 1
        else:
            # SMOTE
            smote_emb = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
            if len(smote_emb) > 0:
                all_syn_emb.append(smote_emb)
                all_syn_labels.extend([cls] * len(smote_emb))
                all_weights.extend([1.0] * len(smote_emb))
            n_smote_classes += 1

    if not all_syn_emb:
        return np.zeros((0, train_emb.shape[1])), [], [], 0, 0
    return np.vstack(all_syn_emb), all_syn_labels, all_weights, n_llm_classes, n_smote_classes


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 70)
    print("CLASS-ADAPTIVE HYBRID AUGMENTATION EXPERIMENT")
    print("=" * 70)

    model = SentenceTransformer("all-mpnet-base-v2")
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

        # Pre-load all LLM data for this dataset
        llm_data = {}
        has_all_cache = True
        for cls in unique_classes:
            cache_key = get_cache_key(ds_name, cls, n_shot, n_gen)
            cache_file = CACHE_DIR / f"{cache_key}.json"
            if not cache_file.exists():
                has_all_cache = False
                break
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("texts"):
                llm_data[cls] = model.encode(cached["texts"], show_progress_bar=False)
            else:
                llm_data[cls] = np.zeros((0, 768))

        if not has_all_cache:
            print(f"  Skipping {ds_name} (missing cache)", flush=True)
            continue

        for seed in SEEDS:
            # Compute difficulties once per seed (CV uses seed)
            diff_cv = estimate_difficulty_cv(train_emb, train_labels, unique_classes, seed)
            diff_geo = estimate_difficulty_geometric(train_emb, train_labels, unique_classes)
            diff_oracle = estimate_difficulty_oracle(train_emb, train_labels, test_emb, test_labels,
                                                     unique_classes, seed)

            # Geometric threshold = median difficulty
            geo_vals = list(diff_geo.values())
            geo_median = np.median(geo_vals) if geo_vals else 0.5

            # Methods to evaluate
            methods = {
                "pure_smote": ("smote", None, None, None),
                "pure_geo": ("geo", None, None, None),
                "hybrid_cv_30": ("hybrid", diff_cv, 0.30, "f1"),
                "hybrid_cv_50": ("hybrid", diff_cv, 0.50, "f1"),
                "hybrid_cv_70": ("hybrid", diff_cv, 0.70, "f1"),
                "hybrid_geo_median": ("hybrid", diff_geo, geo_median, "geometric"),
                "hybrid_oracle_50": ("hybrid", diff_oracle, 0.50, "f1"),
            }

            for clf_name, clf_factory in CLASSIFIERS.items():
                for method_name, (method_type, difficulties, threshold, thr_type) in methods.items():
                    clf = clf_factory(seed)

                    if method_type == "smote":
                        syn_emb, syn_labels, syn_weights = augment_pure_smote(
                            train_emb, train_labels, unique_classes, seed)
                        n_llm_cls, n_smote_cls = 0, len(unique_classes)
                    elif method_type == "geo":
                        syn_emb, syn_labels, syn_weights = augment_pure_geo(
                            train_emb, train_labels, unique_classes, llm_data, seed)
                        n_llm_cls, n_smote_cls = len(unique_classes), 0
                    else:
                        result_tuple = augment_hybrid(
                            train_emb, train_labels, unique_classes, llm_data,
                            difficulties, threshold, thr_type, seed)
                        syn_emb, syn_labels, syn_weights, n_llm_cls, n_smote_cls = result_tuple

                    # Combine real + synthetic
                    if len(syn_emb) > 0:
                        X = np.vstack([train_emb, syn_emb])
                        y = list(train_labels) + list(syn_labels)
                        w = [1.0] * len(train_emb) + list(syn_weights)
                    else:
                        X = train_emb
                        y = list(train_labels)
                        w = [1.0] * len(train_emb)

                    # Train and evaluate
                    try:
                        if hasattr(clf, 'sample_weight') or clf_name in ["logistic_regression", "ridge"]:
                            clf.fit(X, y, sample_weight=np.array(w))
                        else:
                            clf.fit(X, y)
                    except Exception:
                        clf.fit(X, y)

                    pred = clf.predict(test_emb)
                    f1 = f1_score(test_labels, pred, average="macro")

                    all_results.append({
                        "dataset": ds_name,
                        "dataset_base": get_dataset_base(ds_name),
                        "n_shot": n_shot,
                        "n_classes": len(unique_classes),
                        "classifier": clf_name,
                        "method": method_name,
                        "seed": seed,
                        "f1_macro": float(f1),
                        "n_llm_classes": n_llm_cls,
                        "n_smote_classes": n_smote_cls,
                    })

        # Quick summary for this dataset
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        for method_name in ["pure_smote", "pure_geo", "hybrid_cv_50", "hybrid_geo_median", "hybrid_oracle_50"]:
            method_f1s = [r["f1_macro"] for r in ds_results if r["method"] == method_name]
            if method_f1s:
                print(f"  {method_name}: F1={np.mean(method_f1s):.4f}", flush=True)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(all_results),
        "results": all_results,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {RESULTS_DIR / 'results.json'}")

    return all_results


# ============================================================================
# ANALYSIS AND TABLE GENERATION
# ============================================================================

def analyze_results(results):
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Group by method
    method_f1s = defaultdict(list)
    smote_f1_map = {}  # (dataset, classifier, seed) -> f1

    for r in results:
        method_f1s[r["method"]].append(r["f1_macro"])
        if r["method"] == "pure_smote":
            key = (r["dataset"], r["classifier"], r["seed"])
            smote_f1_map[key] = r["f1_macro"]

    # Compute deltas vs SMOTE
    method_deltas = defaultdict(list)
    for r in results:
        if r["method"] == "pure_smote":
            continue
        key = (r["dataset"], r["classifier"], r["seed"])
        smote_f1 = smote_f1_map.get(key)
        if smote_f1 is not None:
            method_deltas[r["method"]].append(r["f1_macro"] - smote_f1)

    # Print summary
    print(f"\n{'Method':<25} {'F1 Mean':>8} {'Δ SMOTE':>8} {'Win%':>6} {'p-val':>10} {'d':>6}")
    print("-" * 70)

    stats_rows = []
    for method in sorted(method_deltas.keys()):
        deltas = method_deltas[method]
        mean_delta = np.mean(deltas) * 100
        wins = sum(1 for d in deltas if d > 0) / len(deltas) * 100
        mean_f1 = np.mean(method_f1s[method]) * 100

        if np.std(deltas) > 1e-10:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d_cohen = np.mean(deltas) / np.std(deltas)
        else:
            p_val, d_cohen = 1.0, 0.0

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_means = [np.mean(rng.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"{method:<25} {mean_f1:>7.2f} {mean_delta:>+7.2f}{sig} {wins:>5.1f}% {p_val:>10.4f} {d_cohen:>5.2f}")

        stats_rows.append({
            "method": method,
            "f1_mean": mean_f1,
            "delta_pp": mean_delta,
            "win_pct": wins,
            "p_val": p_val,
            "d_cohen": d_cohen,
            "ci_low": ci_low * 100,
            "ci_high": ci_high * 100,
            "sig": sig,
        })

    # SMOTE baseline
    smote_f1 = np.mean(method_f1s["pure_smote"]) * 100
    print(f"{'pure_smote':<25} {smote_f1:>7.2f}    ---     ---        ---    ---")

    return stats_rows


def generate_comparison_table(stats_rows, results):
    """Generate main methods comparison table."""
    method_labels = {
        "pure_geo": "Geométrico puro",
        "hybrid_cv_30": "Híbrido CV ($\\theta=0.30$)",
        "hybrid_cv_50": "Híbrido CV ($\\theta=0.50$)",
        "hybrid_cv_70": "Híbrido CV ($\\theta=0.70$)",
        "hybrid_geo_median": "Híbrido geométrico (mediana)",
        "hybrid_oracle_50": "Híbrido oráculo ($\\theta=0.50$)",
    }

    # Compute SMOTE mean F1
    smote_f1s = [r["f1_macro"] for r in results if r["method"] == "pure_smote"]
    smote_mean = np.mean(smote_f1s) * 100

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Comparación de estrategias de aumentación híbrida adaptativa por clase. "
                 r"Los métodos híbridos aplican filtrado geométrico a clases difíciles y SMOTE a clases fáciles. "
                 r"CV: dificultad estimada por validación cruzada leave-one-out. "
                 r"Geométrico: dificultad estimada por ratio dispersión/separación. "
                 r"Oráculo: dificultad conocida (cota superior, no práctico). "
                 r"$\theta$: umbral de F1 bajo el cual una clase se considera difícil.}")
    lines.append(r"\label{tab:hybrid_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Método} & \textbf{F1 Macro} & \textbf{$\Delta$ vs SMOTE} & "
                 r"\textbf{IC 95\%} & \textbf{$d$} & \textbf{Victoria} & \textbf{$p$} \\")
    lines.append(r"\midrule")

    # SMOTE baseline
    lines.append(f"SMOTE & {smote_mean:.2f} & --- & --- & --- & --- & --- \\\\")

    # Sort by delta descending
    sorted_rows = sorted(stats_rows, key=lambda x: x["delta_pp"], reverse=True)

    for row in sorted_rows:
        label = method_labels.get(row["method"], row["method"])
        f1 = row["f1_mean"]
        delta = row["delta_pp"]
        ci = f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"
        d = row["d_cohen"]
        win = row["win_pct"]
        p = row["p_val"]

        p_str = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        sig = row["sig"]

        row_str = f"{label} & {f1:.2f} & {delta:+.2f}{sig} & {ci} & {d:.2f} & {win:.1f}\\% & {p_str} \\\\"

        # Highlight best hybrid
        if row == sorted_rows[0]:
            row_str = r"\rowcolor{green!10}" + row_str

        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(TABLE_DIR / "tab_hybrid_comparison.tex", "w") as f:
        f.write(tex)
    print(f"\nSaved: {TABLE_DIR / 'tab_hybrid_comparison.tex'}")


def generate_threshold_table(results):
    """Generate threshold sensitivity table: for CV-based hybrids, show how threshold affects routing."""
    cv_methods = ["hybrid_cv_30", "hybrid_cv_50", "hybrid_cv_70"]
    threshold_labels = {"hybrid_cv_30": "0.30", "hybrid_cv_50": "0.50", "hybrid_cv_70": "0.70"}

    smote_f1_map = {}
    for r in results:
        if r["method"] == "pure_smote":
            smote_f1_map[(r["dataset"], r["classifier"], r["seed"])] = r["f1_macro"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Sensibilidad al umbral de dificultad para la estrategia híbrida CV. "
                 r"$\theta$ controla el umbral de F1 por clase bajo el cual se usa LLM+filtrado "
                 r"en lugar de SMOTE. Clases LLM y SMOTE son promedios sobre todas las configuraciones.}")
    lines.append(r"\label{tab:hybrid_threshold}")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$\theta$} & \textbf{Clases LLM} & \textbf{Clases SMOTE} & "
                 r"\textbf{F1 Macro} & \textbf{$\Delta$ vs SMOTE} & \textbf{Victoria} & \textbf{$p$} \\")
    lines.append(r"\midrule")

    for method in cv_methods:
        method_results = [r for r in results if r["method"] == method]
        if not method_results:
            continue

        mean_f1 = np.mean([r["f1_macro"] for r in method_results]) * 100
        mean_llm_cls = np.mean([r["n_llm_classes"] for r in method_results])
        mean_smote_cls = np.mean([r["n_smote_classes"] for r in method_results])

        deltas = []
        for r in method_results:
            key = (r["dataset"], r["classifier"], r["seed"])
            if key in smote_f1_map:
                deltas.append(r["f1_macro"] - smote_f1_map[key])

        mean_delta = np.mean(deltas) * 100
        wins = sum(1 for d in deltas if d > 0) / len(deltas) * 100 if deltas else 0

        if deltas and np.std(deltas) > 1e-10:
            _, p_val = stats.ttest_1samp(deltas, 0)
        else:
            p_val = 1.0

        p_str = "$<$0.001" if p_val < 0.001 else f"{p_val:.3f}"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        lines.append(f"{threshold_labels[method]} & {mean_llm_cls:.1f} & {mean_smote_cls:.1f} & "
                     f"{mean_f1:.2f} & {mean_delta:+.2f}{sig} & {wins:.1f}\\% & {p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(TABLE_DIR / "tab_hybrid_threshold.tex", "w") as f:
        f.write(tex)
    print(f"Saved: {TABLE_DIR / 'tab_hybrid_threshold.tex'}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_experiment()
    if results:
        stats_rows = analyze_results(results)
        generate_comparison_table(stats_rows, results)
        generate_threshold_table(results)
        print("\nDone!")
