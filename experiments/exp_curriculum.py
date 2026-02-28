#!/usr/bin/env python3
"""Curriculum Learning with Geometric Scores.

Instead of adding all synthetic samples at once, introduce them ordered by
geometric quality score (highest first). Evaluate F1 at each increment to
find the optimal proportion of candidates to include.

Uses cached LLM generations — no new API calls needed.

Configuration: 7 datasets × 3 n-shot × 3 classifiers × 3 seeds × 10 steps = 1,890 evaluations
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
RESULTS_DIR = PROJECT_ROOT / "results" / "curriculum"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
FIGURE_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    print("Curriculum Learning Experiment")
    print("=" * 60)

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    all_results = []
    n_total = len(DATASETS) * len(CLASSIFIERS) * len(SEEDS)
    n_done = 0

    for ds_idx, ds_name in enumerate(DATASETS):
        print(f"\n[{ds_idx+1}/{len(DATASETS)}] Processing {ds_name}...", flush=True)
        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        ds_base = get_dataset_base(ds_name)
        n_classes = DATASET_N_CLASSES.get(ds_base, len(set(train_labels)))

        # Parse n_shot
        n_shot = 10
        for part in ds_name.split("_"):
            if "shot" in part:
                n_shot = int(part.replace("shot", ""))
                break

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)

        # Prepare ALL LLM candidates with scores (not yet filtered)
        cascade = FilterCascade(**FILTER_CONFIG)
        class_candidates = {}  # cls -> (embeddings, scores, weights)

        for cls in unique_classes:
            cls_emb = train_emb[labels_arr == cls]
            cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
            n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
            gen_texts = load_cached_texts(ds_name, cls, n_shot, n_gen)
            if not gen_texts:
                continue
            gen_emb = model.encode(gen_texts, show_progress_bar=False)
            anchor = cls_emb.mean(axis=0)
            scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_emb, labels_arr, cls)
            weights = normalize_scores(scores)
            # Sort by score descending
            order = np.argsort(scores)[::-1]
            class_candidates[cls] = {
                "embeddings": gen_emb[order],
                "scores": scores[order],
                "weights": weights[order],
            }

        if not class_candidates:
            print(f"  {ds_name}: No LLM data, skipping")
            continue

        for seed in SEEDS:
            # SMOTE baseline (computed once per seed)
            smote_embs, smote_labels = [], []
            for cls in unique_classes:
                cls_emb = train_emb[labels_arr == cls]
                s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                if len(s) > 0:
                    smote_embs.append(s)
                    smote_labels.extend([cls] * len(s))

            for clf_name, clf_factory in CLASSIFIERS.items():
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  Progress: {n_done}/{n_total} ({100*n_done/n_total:.0f}%)")

                # --- No augmentation ---
                clf = clf_factory(seed)
                clf.fit(train_emb, train_labels)
                f1_noaug = f1_score(test_labels, clf.predict(test_emb), average="macro")

                # --- SMOTE ---
                if smote_labels:
                    aug_emb = np.vstack([train_emb, np.vstack(smote_embs)])
                    aug_lab = list(train_labels) + smote_labels
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab)
                    f1_smote = f1_score(test_labels, clf.predict(test_emb), average="macro")
                else:
                    f1_smote = f1_noaug

                # --- All-at-once (standard soft_weighted) ---
                all_syn_embs, all_syn_labels, all_syn_weights = [], [], []
                for cls in unique_classes:
                    if cls not in class_candidates:
                        continue
                    cd = class_candidates[cls]
                    target_n = min(N_SYNTHETIC_PER_CLASS, len(cd["embeddings"]))
                    all_syn_embs.append(cd["embeddings"][:target_n])
                    all_syn_labels.extend([cls] * target_n)
                    all_syn_weights.append(cd["weights"][:target_n])

                if all_syn_embs:
                    syn_all = np.vstack(all_syn_embs)
                    w_all = np.concatenate(all_syn_weights)
                    aug_emb = np.vstack([train_emb, syn_all])
                    aug_lab = list(train_labels) + all_syn_labels
                    sw = np.concatenate([np.ones(len(train_emb)), w_all])
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab, sample_weight=sw)
                    f1_all_at_once = f1_score(test_labels, clf.predict(test_emb), average="macro")
                else:
                    f1_all_at_once = f1_noaug

                # --- Curriculum: add incrementally by quality ---
                for step in STEPS:
                    curr_embs, curr_labels, curr_weights = [], [], []
                    for cls in unique_classes:
                        if cls not in class_candidates:
                            continue
                        cd = class_candidates[cls]
                        total = len(cd["embeddings"])
                        n_take = max(1, int(total * step))
                        curr_embs.append(cd["embeddings"][:n_take])
                        curr_labels.extend([cls] * n_take)
                        curr_weights.append(cd["weights"][:n_take])

                    if curr_embs:
                        syn_curr = np.vstack(curr_embs)
                        w_curr = np.concatenate(curr_weights)
                        aug_emb = np.vstack([train_emb, syn_curr])
                        aug_lab = list(train_labels) + curr_labels
                        sw = np.concatenate([np.ones(len(train_emb)), w_curr])
                        clf = clf_factory(seed)
                        clf.fit(aug_emb, aug_lab, sample_weight=sw)
                        f1_curr = f1_score(test_labels, clf.predict(test_emb), average="macro")
                    else:
                        f1_curr = f1_noaug

                    all_results.append({
                        "dataset": ds_name,
                        "dataset_base": ds_base,
                        "n_classes": n_classes,
                        "n_shot": n_shot,
                        "classifier": clf_name,
                        "seed": seed,
                        "step": step,
                        "n_synthetic": len(curr_labels) if curr_embs else 0,
                        "f1_curriculum": float(f1_curr),
                        "f1_all_at_once": float(f1_all_at_once),
                        "f1_smote": float(f1_smote),
                        "f1_noaug": float(f1_noaug),
                    })

    # Save results
    out_path = RESULTS_DIR / "curriculum_results.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                    "n_results": len(all_results),
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total evaluations: {len(all_results)}")

    # Generate table and figure
    generate_curriculum_table(all_results)
    generate_curriculum_figure(all_results)


def generate_curriculum_table(results):
    """Table: Best curriculum step vs all-at-once vs SMOTE."""
    # For each config (dataset, classifier, seed), find the best step
    config_keys = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        config_keys[key].append(r)

    best_steps = []
    f1_improvements = []  # curriculum_best vs all_at_once

    for key, runs in config_keys.items():
        best_run = max(runs, key=lambda x: x["f1_curriculum"])
        best_steps.append(best_run["step"])
        f1_improvements.append(best_run["f1_curriculum"] - best_run["f1_all_at_once"])

    best_steps = np.array(best_steps)
    f1_improvements = np.array(f1_improvements) * 100

    # Aggregate by step
    step_f1s = defaultdict(list)
    step_deltas_smote = defaultdict(list)
    for r in results:
        step_f1s[r["step"]].append(r["f1_curriculum"])
        step_deltas_smote[r["step"]].append(r["f1_curriculum"] - r["f1_smote"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Rendimiento del aprendizaje curricular por proporción de candidatos. "
                 r"Las muestras sintéticas se agregan ordenadas por score geométrico (mejores primero). "
                 r"F1 y $\Delta$ vs SMOTE promediados sobre todas las configuraciones.}")
    lines.append(r"\label{tab:curriculum_results}")
    lines.append(r"\begin{tabular}{rcccr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Proporción} & \textbf{N muestras} & \textbf{F1 Macro} & "
                 r"\textbf{$\Delta$ vs SMOTE} & \textbf{Victoria} \\")
    lines.append(r"\midrule")

    for step in STEPS:
        f1s = np.array(step_f1s[step])
        deltas = np.array(step_deltas_smote[step])
        mean_f1 = f1s.mean() * 100
        mean_delta = deltas.mean() * 100
        win_rate = (deltas > 0).mean() * 100
        # Avg n_synthetic
        n_syns = [r["n_synthetic"] for r in results if r["step"] == step]
        avg_n = np.mean(n_syns)
        sign = "+" if mean_delta >= 0 else ""
        lines.append(
            f"{step:.0%} & {avg_n:.0f} & {mean_f1:.2f} & "
            f"{sign}{mean_delta:.2f} & {win_rate:.1f}\\% \\\\"
        )

    # Add baselines for comparison
    lines.append(r"\midrule")
    all_smote = [r["f1_smote"] for r in results if r["step"] == 1.0]
    all_noaug = [r["f1_noaug"] for r in results if r["step"] == 1.0]
    all_aao = [r["f1_all_at_once"] for r in results if r["step"] == 1.0]
    lines.append(f"SMOTE & --- & {np.mean(all_smote)*100:.2f} & --- & --- \\\\")
    lines.append(f"Estándar (top-N) & {np.mean([r['n_synthetic'] for r in results if r['step']==STEPS[-1]]):.0f} & "
                 f"{np.mean(all_aao)*100:.2f} & "
                 f"+{(np.mean(all_aao)-np.mean(all_smote))*100:.2f} & --- \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\small Paso óptimo medio: {best_steps.mean():.0%} "
                 f"(mediana: {np.median(best_steps):.0%}). "
                 f"Mejora curriculum vs estándar: {f1_improvements.mean():+.2f}pp.}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    path = OUTPUT_DIR / "tab_curriculum_results.tex"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(table)
    print(f"\nTable written: {path}")
    print(table)


def generate_curriculum_figure(results):
    """Generate matplotlib figure: F1 vs proportion of candidates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure generation")
        return

    step_f1s = defaultdict(list)
    for r in results:
        step_f1s[r["step"]].append(r["f1_curriculum"])

    smote_f1 = np.mean([r["f1_smote"] for r in results if r["step"] == 1.0])

    steps_x = sorted(step_f1s.keys())
    means = [np.mean(step_f1s[s]) for s in steps_x]
    stds = [np.std(step_f1s[s]) / np.sqrt(len(step_f1s[s])) for s in steps_x]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar([s*100 for s in steps_x], [m*100 for m in means],
                yerr=[s*100 for s in stds], marker='o', linewidth=2,
                capsize=4, label='Curriculum (por score)', color='#2196F3')
    ax.axhline(y=smote_f1*100, color='#FF5722', linestyle='--', linewidth=1.5,
               label=f'SMOTE ({smote_f1*100:.2f}%)')

    best_step_idx = np.argmax(means)
    ax.scatter([steps_x[best_step_idx]*100], [means[best_step_idx]*100],
               s=200, zorder=5, color='#4CAF50', marker='*',
               label=f'Óptimo ({steps_x[best_step_idx]:.0%})')

    ax.set_xlabel('Proporción de candidatos incluidos (%)', fontsize=12)
    ax.set_ylabel('Macro F1 (%)', fontsize=12)
    ax.set_title('Aprendizaje Curricular: F1 vs. Proporción de Candidatos', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fig_curriculum_curve.pdf"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
