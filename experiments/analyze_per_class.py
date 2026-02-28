#!/usr/bin/env python3
"""Per-class F1 analysis: Which classes benefit most from geometric filtering?

Re-computes per-class F1 scores for 3 representative datasets (emotion, hate_speech, 20news)
at 10-shot with 3 classifiers and 1 seed to understand class-level impact.

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
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "per_class"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration — matches exp_thesis_final.py
N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0
SEED = 42

DATASETS = ["emotion_10shot", "hate_speech_davidson_10shot", "20newsgroups_10shot"]
CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "svc_linear": lambda: SVC(kernel="linear", random_state=SEED),
    "ridge": lambda: RidgeClassifier(alpha=1.0),
}


def load_dataset(name):
    with open(DATA_DIR / f"{name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_cache_key(dataset, class_name, n_shot, n_generate):
    return hashlib.md5(f"{dataset}_{class_name}_{n_shot}_{n_generate}".encode()).hexdigest()[:16]


def load_cached_texts(dataset, class_name, n_shot, n_generate):
    cache_key = get_cache_key(dataset, class_name, n_shot, n_generate)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        return cached.get("texts", [])
    return []


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


def run_per_class_analysis():
    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    all_results = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        n_shot = int(ds_name.split("_")[-1].replace("shot", "")) if "shot" in ds_name else 10
        # Parse n_shot from dataset name
        for part in ds_name.split("_"):
            if "shot" in part:
                n_shot = int(part.replace("shot", ""))
                break

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)

        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)

        # --- Prepare SMOTE data ---
        smote_embs, smote_labels = [], []
        for cls in unique_classes:
            cls_emb = train_emb[labels_arr == cls]
            s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=SEED)
            if len(s) > 0:
                smote_embs.append(s)
                smote_labels.extend([cls] * len(s))
        smote_all_emb = np.vstack(smote_embs) if smote_embs else np.zeros((0, 768))

        # --- Prepare LLM + filter data ---
        cascade = FilterCascade(**FILTER_CONFIG)
        filtered_embs, filtered_labels, filtered_weights = [], [], []
        for cls in unique_classes:
            cls_emb = train_emb[labels_arr == cls]
            cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
            n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
            gen_texts = load_cached_texts(ds_name, cls, n_shot, n_gen)
            if not gen_texts:
                print(f"  WARNING: No cached texts for class '{cls}' — skipping")
                continue
            gen_emb = model.encode(gen_texts, show_progress_bar=False)
            anchor = cls_emb.mean(axis=0)
            scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
            weights = normalize_scores(scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)
            target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
            top_idx = np.argsort(scores)[-target_n:]
            filtered_embs.append(gen_emb[top_idx])
            filtered_labels.extend([cls] * len(top_idx))
            filtered_weights.append(weights[top_idx])

        if not filtered_embs:
            print("  No LLM data available, skipping dataset")
            continue

        syn_emb = np.vstack(filtered_embs)
        syn_weights = np.concatenate(filtered_weights)

        for clf_name, clf_factory in CLASSIFIERS.items():
            print(f"\n  Classifier: {clf_name}")

            # --- No augmentation ---
            clf = clf_factory()
            clf.fit(train_emb, train_labels)
            pred_noaug = clf.predict(test_emb)
            f1_noaug_per = f1_score(test_labels, pred_noaug, average=None, labels=unique_classes)
            f1_noaug_macro = f1_score(test_labels, pred_noaug, average="macro")

            # --- SMOTE ---
            if len(smote_labels) > 0:
                aug_emb_smote = np.vstack([train_emb, smote_all_emb])
                aug_labels_smote = list(train_labels) + smote_labels
                clf = clf_factory()
                clf.fit(aug_emb_smote, aug_labels_smote)
                pred_smote = clf.predict(test_emb)
                f1_smote_per = f1_score(test_labels, pred_smote, average=None, labels=unique_classes)
                f1_smote_macro = f1_score(test_labels, pred_smote, average="macro")
            else:
                f1_smote_per = f1_noaug_per
                f1_smote_macro = f1_noaug_macro

            # --- Soft weighted ---
            aug_emb_sw = np.vstack([train_emb, syn_emb])
            aug_labels_sw = list(train_labels) + filtered_labels
            sample_w = np.concatenate([np.ones(len(train_emb)), syn_weights])
            clf = clf_factory()
            clf.fit(aug_emb_sw, aug_labels_sw, sample_weight=sample_w)
            pred_sw = clf.predict(test_emb)
            f1_sw_per = f1_score(test_labels, pred_sw, average=None, labels=unique_classes)
            f1_sw_macro = f1_score(test_labels, pred_sw, average="macro")

            for i, cls in enumerate(unique_classes):
                result = {
                    "dataset": ds_name,
                    "classifier": clf_name,
                    "class": str(cls),
                    "n_test_samples": int(np.sum(np.array(test_labels) == cls)),
                    "f1_noaug": float(f1_noaug_per[i]),
                    "f1_smote": float(f1_smote_per[i]),
                    "f1_soft_weighted": float(f1_sw_per[i]),
                    "delta_sw_vs_smote": float(f1_sw_per[i] - f1_smote_per[i]),
                    "delta_sw_vs_noaug": float(f1_sw_per[i] - f1_noaug_per[i]),
                }
                all_results.append(result)
                print(f"    Class '{cls}': noaug={f1_noaug_per[i]:.3f}, "
                      f"smote={f1_smote_per[i]:.3f}, sw={f1_sw_per[i]:.3f}, "
                      f"Δ={f1_sw_per[i]-f1_smote_per[i]:+.3f}")

            print(f"    MACRO: noaug={f1_noaug_macro:.3f}, smote={f1_smote_macro:.3f}, "
                  f"sw={f1_sw_macro:.3f}")

    # Save results
    out_path = RESULTS_DIR / "per_class_results.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def generate_per_class_table(results):
    """Generate LaTeX table showing per-class F1 for each dataset."""
    datasets = sorted(set(r["dataset"] for r in results))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Macro F1 por clase individual para conjuntos representativos con 10 "
                 r"ejemplos por clase. Valores promediados sobre 3 clasificadores lineales "
                 r"(regresión logística, SVC, Ridge). $\Delta$ indica mejora de la ponderación "
                 r"suave respecto a SMOTE.}")
    lines.append(r"\label{tab:per_class_analysis}")
    lines.append(r"\begin{tabular}{llrccr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Clase} & \textbf{N test} & "
                 r"\textbf{SMOTE} & \textbf{Pond. suave} & \textbf{$\Delta$} \\")
    lines.append(r"\midrule")

    for i, ds in enumerate(datasets):
        ds_results = [r for r in results if r["dataset"] == ds]
        classes = sorted(set(r["class"] for r in ds_results))
        ds_label = ds.replace("_10shot", "").replace("_", " ").title()

        for j, cls in enumerate(classes):
            cls_results = [r for r in ds_results if r["class"] == cls]
            n_test = cls_results[0]["n_test_samples"]
            f1_smote = np.mean([r["f1_smote"] for r in cls_results]) * 100
            f1_sw = np.mean([r["f1_soft_weighted"] for r in cls_results]) * 100
            delta = f1_sw - f1_smote
            sign = "+" if delta >= 0 else ""

            label = ds_label if j == 0 else ""
            cls_display = cls[:15] if len(cls) > 15 else cls
            lines.append(
                f"{label} & {cls_display} & {n_test} & {f1_smote:.1f} & "
                f"{f1_sw:.1f} & {sign}{delta:.1f} \\\\"
            )

        if i < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_rare_class_table(results):
    """Generate table showing correlation between base F1 and delta from filtering."""
    # For each class, compute base F1 (SMOTE) and delta
    class_data = defaultdict(lambda: {"f1_base": [], "delta": []})
    for r in results:
        key = (r["dataset"], r["class"])
        class_data[key]["f1_base"].append(r["f1_smote"])
        class_data[key]["delta"].append(r["delta_sw_vs_smote"])

    f1_bases = []
    deltas = []
    for key, data in class_data.items():
        f1_bases.append(np.mean(data["f1_base"]))
        deltas.append(np.mean(data["delta"]))

    f1_bases = np.array(f1_bases)
    deltas = np.array(deltas) * 100  # to pp

    # Spearman correlation
    from scipy import stats
    rho, p_val = stats.spearmanr(f1_bases, deltas)

    # Bin by difficulty
    bins = [(0, 0.3, "Difícil (F1$<$0.30)"), (0.3, 0.6, "Media (0.30--0.60)"),
            (0.6, 0.8, "Fácil (0.60--0.80)"), (0.8, 1.01, "Muy fácil (F1$>$0.80)")]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Impacto del filtrado geométrico según la dificultad de la clase. "
                 r"Las clases se agrupan por su F1 base (con SMOTE). Correlación de Spearman "
                 rf"entre F1 base y $\Delta$: $\rho = {rho:.3f}$ ($p = {p_val:.4f}$).}}")
    lines.append(r"\label{tab:rare_class_benefit}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dificultad de clase} & \textbf{N clases} & "
                 r"\textbf{$\Delta$ medio (pp)} & \textbf{$\Delta$ mediana (pp)} \\")
    lines.append(r"\midrule")

    for lo, hi, label in bins:
        mask = (f1_bases >= lo) & (f1_bases < hi)
        n = mask.sum()
        if n > 0:
            mean_d = deltas[mask].mean()
            med_d = np.median(deltas[mask])
            sign_mean = "+" if mean_d >= 0 else ""
            sign_med = "+" if med_d >= 0 else ""
            lines.append(f"{label} & {n} & {sign_mean}{mean_d:.2f} & {sign_med}{med_d:.2f} \\\\")
        else:
            lines.append(f"{label} & 0 & --- & --- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Per-Class F1 Analysis")
    print("=" * 60)

    results = run_per_class_analysis()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: Per-class breakdown
    print("\n--- Generating per-class table ---")
    table1 = generate_per_class_table(results)
    path1 = OUTPUT_DIR / "tab_per_class_analysis.tex"
    path1.write_text(table1)
    print(f"Written: {path1}")
    print(table1)

    # Table 2: Rare class benefit
    print("\n--- Generating rare class benefit table ---")
    table2 = generate_rare_class_table(results)
    path2 = OUTPUT_DIR / "tab_rare_class_benefit.tex"
    path2.write_text(table2)
    print(f"Written: {path2}")
    print(table2)


if __name__ == "__main__":
    main()
