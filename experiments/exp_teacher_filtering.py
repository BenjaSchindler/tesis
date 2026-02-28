#!/usr/bin/env python3
"""Teacher-Model Filtering Comparison.

Compares geometric filtering (Euclidean distance via cascade_l1) vs
model-based filtering (LogReg teacher confidence) for selecting synthetic samples.

The teacher is always LogReg trained on real data only.
Evaluation classifiers: SVC(linear) and Ridge (to avoid circular evaluation).

Uses cached LLM generations — no new API calls needed.

Configuration: 21 datasets × 2 eval classifiers × 4 methods × 3 seeds = 504 evaluations
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
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.filter_cascade import FilterCascade

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "teacher_comparison"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
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

# Eval classifiers (NOT LogReg, to avoid circularity with teacher)
EVAL_CLASSIFIERS = {
    "svc_linear": lambda seed: SVC(kernel="linear", random_state=seed),
    "ridge": lambda seed: RidgeClassifier(alpha=1.0),
}

METHODS = ["smote", "geometric_binary", "geometric_soft", "teacher_binary", "teacher_soft"]


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
            return json.load(f).get("texts", [])
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


def normalize_scores(raw_scores):
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    return np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / TEMPERATURE)


def get_dataset_base(name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if name.startswith(base + "_"):
            return base
    return name


def main():
    print("=" * 60)
    print("Teacher-Model vs Geometric Filtering Comparison")
    print("=" * 60)

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    all_results = []
    n_total = len(DATASETS) * len(EVAL_CLASSIFIERS) * len(SEEDS)
    n_done = 0

    for ds_idx, ds_name in enumerate(DATASETS):
        print(f"\n[{ds_idx+1}/{len(DATASETS)}] Processing {ds_name}...", flush=True)
        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        ds_base = get_dataset_base(ds_name)
        n_classes = DATASET_N_CLASSES.get(ds_base, len(set(train_labels)))

        n_shot = 10
        for part in ds_name.split("_"):
            if "shot" in part:
                n_shot = int(part.replace("shot", ""))
                break

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)

        # --- Train teacher (LogReg on real data only) ---
        teacher = LogisticRegression(max_iter=1000, random_state=42)
        teacher.fit(train_emb, train_labels)

        # --- Prepare candidates per class ---
        cascade = FilterCascade(**FILTER_CONFIG)
        class_data = {}

        for cls in unique_classes:
            cls_emb = train_emb[labels_arr == cls]
            n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
            gen_texts = load_cached_texts(ds_name, cls, n_shot, n_gen)
            if not gen_texts:
                continue
            gen_emb = model.encode(gen_texts, show_progress_bar=False)

            # Geometric scores (cascade_l1)
            anchor = cls_emb.mean(axis=0)
            geo_scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)

            # Teacher scores: P(correct class | sample)
            teacher_probs = teacher.predict_proba(gen_emb)
            cls_idx = list(teacher.classes_).index(cls)
            teacher_scores = teacher_probs[:, cls_idx]

            class_data[cls] = {
                "embeddings": gen_emb,
                "geo_scores": geo_scores,
                "teacher_scores": teacher_scores,
            }

        if not class_data:
            print(f"  {ds_name}: No LLM data, skipping")
            continue

        for seed in SEEDS:
            # SMOTE
            smote_embs, smote_labels = [], []
            for cls in unique_classes:
                cls_emb = train_emb[labels_arr == cls]
                s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                if len(s) > 0:
                    smote_embs.append(s)
                    smote_labels.extend([cls] * len(s))

            for clf_name, clf_factory in EVAL_CLASSIFIERS.items():
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  Progress: {n_done}/{n_total} ({100*n_done/n_total:.0f}%)")

                results_by_method = {}

                # --- SMOTE ---
                if smote_labels:
                    aug_emb = np.vstack([train_emb, np.vstack(smote_embs)])
                    aug_lab = list(train_labels) + smote_labels
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab)
                    f1 = f1_score(test_labels, clf.predict(test_emb), average="macro")
                else:
                    clf = clf_factory(seed)
                    clf.fit(train_emb, train_labels)
                    f1 = f1_score(test_labels, clf.predict(test_emb), average="macro")
                results_by_method["smote"] = float(f1)

                # Helper to select top-N and train
                def select_and_train(score_key, use_weights, method_name):
                    syn_embs, syn_labels, syn_weights = [], [], []
                    for cls in unique_classes:
                        if cls not in class_data:
                            continue
                        cd = class_data[cls]
                        scores = cd[score_key]
                        target_n = min(N_SYNTHETIC_PER_CLASS, len(cd["embeddings"]))
                        top_idx = np.argsort(scores)[-target_n:]
                        syn_embs.append(cd["embeddings"][top_idx])
                        syn_labels.extend([cls] * len(top_idx))
                        if use_weights:
                            w = normalize_scores(scores[top_idx])
                            syn_weights.append(w)
                        else:
                            syn_weights.append(np.ones(len(top_idx)))

                    if not syn_embs:
                        clf = clf_factory(seed)
                        clf.fit(train_emb, train_labels)
                        return float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                    s_emb = np.vstack(syn_embs)
                    s_w = np.concatenate(syn_weights)
                    aug_e = np.vstack([train_emb, s_emb])
                    aug_l = list(train_labels) + syn_labels
                    sw = np.concatenate([np.ones(len(train_emb)), s_w])
                    clf = clf_factory(seed)
                    clf.fit(aug_e, aug_l, sample_weight=sw)
                    return float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                # --- Geometric binary (top-N, uniform weight) ---
                results_by_method["geometric_binary"] = select_and_train("geo_scores", False, "geo_bin")

                # --- Geometric soft (top-N, quality weight) ---
                results_by_method["geometric_soft"] = select_and_train("geo_scores", True, "geo_soft")

                # --- Teacher binary (top-N by confidence, uniform weight) ---
                results_by_method["teacher_binary"] = select_and_train("teacher_scores", False, "teach_bin")

                # --- Teacher soft (top-N by confidence, confidence weight) ---
                results_by_method["teacher_soft"] = select_and_train("teacher_scores", True, "teach_soft")

                for method, f1_val in results_by_method.items():
                    all_results.append({
                        "dataset": ds_name,
                        "dataset_base": ds_base,
                        "n_classes": n_classes,
                        "n_shot": n_shot,
                        "classifier": clf_name,
                        "seed": seed,
                        "method": method,
                        "f1_macro": f1_val,
                    })

    # Save results
    out_path = RESULTS_DIR / "teacher_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                    "n_results": len(all_results),
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total evaluations: {len(all_results)}")

    generate_comparison_table(all_results)


def generate_comparison_table(results):
    """Generate table comparing geometric vs teacher filtering."""
    methods_order = ["smote", "geometric_binary", "geometric_soft", "teacher_binary", "teacher_soft"]
    method_labels = {
        "smote": "SMOTE",
        "geometric_binary": "Geométrico (binario)",
        "geometric_soft": "Geométrico (ponderado)",
        "teacher_binary": "Teacher (binario)",
        "teacher_soft": "Teacher (ponderado)",
    }

    # Compute paired deltas vs SMOTE
    grouped = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        grouped[key][r["method"]] = r["f1_macro"]

    method_deltas = defaultdict(list)
    method_f1s = defaultdict(list)
    for key, methods in grouped.items():
        if "smote" not in methods:
            continue
        smote_f1 = methods["smote"]
        for m in methods_order:
            if m in methods:
                method_f1s[m].append(methods[m])
                if m != "smote":
                    method_deltas[m].append(methods[m] - smote_f1)

    n_comparisons = len(methods_order) - 1  # for Bonferroni

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Comparación entre filtrado geométrico y filtrado basado en modelo teacher. "
                 r"El teacher es una regresión logística entrenada exclusivamente con datos reales. "
                 r"Los clasificadores de evaluación son SVC lineal y Ridge (distintos del teacher).}")
    lines.append(r"\label{tab:teacher_vs_geometric}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Método} & \textbf{F1 Macro} & \textbf{$\Delta$ vs SMOTE} & "
                 r"\textbf{IC 95\%} & \textbf{$d$} & \textbf{Victoria} & \textbf{$p$} \\")
    lines.append(r"\midrule")

    for m in methods_order:
        f1s = np.array(method_f1s.get(m, []))
        f1_mean = f1s.mean() * 100

        if m == "smote":
            lines.append(f"{method_labels[m]} & {f1_mean:.2f} & --- & --- & --- & --- & --- \\\\")
        elif m in method_deltas and len(method_deltas[m]) > 0:
            deltas = np.array(method_deltas[m])
            delta_mean = deltas.mean() * 100
            win_rate = (deltas > 0).mean() * 100
            d_cohen = deltas.mean() / deltas.std() if deltas.std() > 0 else 0
            _, p_val = stats.ttest_1samp(deltas, 0) if len(deltas) > 1 else (0, 1)
            p_bonf = min(p_val * n_comparisons, 1.0)

            # Bootstrap CI
            rng = np.random.RandomState(42)
            boot_means = [np.mean(rng.choice(deltas*100, len(deltas), replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

            stars = ""
            if p_bonf < 0.001: stars = "***"
            elif p_bonf < 0.01: stars = "**"
            elif p_bonf < 0.05: stars = "*"

            sign = "+" if delta_mean >= 0 else ""
            p_str = f"{p_bonf:.4f}" if p_bonf >= 0.001 else "$<$0.001"
            lines.append(
                f"{method_labels[m]} & {f1_mean:.2f} & {sign}{delta_mean:.2f}{stars} & "
                f"[{ci_lo:.2f}, {ci_hi:.2f}] & {d_cohen:.2f} & {win_rate:.1f}\\% & {p_str} \\\\"
            )

    # Add head-to-head: geometric_soft vs teacher_soft
    geo_soft_f1 = defaultdict(float)
    teach_soft_f1 = defaultdict(float)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "geometric_soft":
            geo_soft_f1[key] = r["f1_macro"]
        elif r["method"] == "teacher_soft":
            teach_soft_f1[key] = r["f1_macro"]

    h2h_deltas = []
    for key in geo_soft_f1:
        if key in teach_soft_f1:
            h2h_deltas.append(geo_soft_f1[key] - teach_soft_f1[key])
    h2h_deltas = np.array(h2h_deltas)

    if len(h2h_deltas) > 1:
        h2h_mean = h2h_deltas.mean() * 100
        h2h_win = (h2h_deltas > 0).mean() * 100
        _, h2h_p = stats.ttest_1samp(h2h_deltas, 0)

        lines.append(r"\midrule")
        sign = "+" if h2h_mean >= 0 else ""
        lines.append(
            f"\\multicolumn{{7}}{{l}}{{\\small Geométrico vs Teacher (ponderado): "
            f"$\\Delta$ = {sign}{h2h_mean:.2f}pp, victoria geométrico = {h2h_win:.1f}\\%, "
            f"$p$ = {h2h_p:.4f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "tab_teacher_vs_geometric.tex"
    path.write_text(table)
    print(f"\nTable written: {path}")
    print(table)


if __name__ == "__main__":
    main()
