#!/usr/bin/env python3
"""
Wave Ablation Analysis — Post-Run

Re-evaluates progressive waves results by systematically removing waves
to measure each wave's marginal contribution. No new LLM calls needed;
uses cached wave pools stored during exp_progressive_waves.py.

Tests:
1. Leave-one-out: {0,1,2,3} minus each wave → marginal contribution
2. Individual waves: {0}, {1}, {2}, {3} → standalone performance
3. Cumulative addition: {3}, {3,2}, {3,2,1}, {3,2,1,0} → incremental gain
4. Wave 0 concern: {1,2,3} vs {0,1,2,3} → is wave 0 just noise?

Uses the best combination strategy from the main experiment (all_equal_weight)
plus progressive_replace for comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import hashlib
import numpy as np
from collections import defaultdict
from itertools import combinations

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.filter_cascade import FilterCascade
from exp_fixed_output_count import get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "progressive_waves"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
FIGURES_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Figures"

# ============================================================================
# CONFIG — must match exp_progressive_waves.py
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
CANDIDATES_PER_WAVE = 75
N_WAVES = 4
SEEDS = [42, 123, 456]
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

WAVE_CONFIGS = [
    {"name": "exploratory", "llm_temperature": 0.9, "threshold_k": -2.0, "example_strategy": "random"},
    {"name": "moderate", "llm_temperature": 0.7, "threshold_k": -1.0, "example_strategy": "mixed"},
    {"name": "focused", "llm_temperature": 0.5, "threshold_k": 0.0, "example_strategy": "centroid_nearest"},
    {"name": "ultra_precise", "llm_temperature": 0.3, "threshold_k": 0.5, "example_strategy": "top5_nearest"},
]

DATASETS = [
    "sms_spam_10shot", "sms_spam_25shot", "sms_spam_50shot",
    "20newsgroups_10shot", "20newsgroups_25shot", "20newsgroups_50shot",
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "ag_news_10shot", "ag_news_25shot", "ag_news_50shot",
    "emotion_10shot", "emotion_25shot", "emotion_50shot",
    "dbpedia14_10shot", "dbpedia14_25shot", "dbpedia14_50shot",
    "20newsgroups_20class_10shot", "20newsgroups_20class_25shot", "20newsgroups_20class_50shot",
]

CLASSIFIERS = {
    "logistic_regression": lambda s: LogisticRegression(max_iter=1000, random_state=s),
    "svc_linear": lambda s: SVC(kernel="linear", random_state=s),
    "ridge": lambda s: RidgeClassifier(random_state=s),
}


# ============================================================================
# HELPERS — copied from exp_progressive_waves.py
# ============================================================================

def load_dataset(ds_name):
    ds_file = DATA_DIR / f"{ds_name}.json"
    with open(ds_file) as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_dataset_base(ds_name):
    for suffix in ["_10shot", "_25shot", "_50shot"]:
        if ds_name.endswith(suffix):
            return ds_name[:-len(suffix)]
    return ds_name


def parse_n_shot(ds_name):
    for n in [10, 25, 50]:
        if f"_{n}shot" in ds_name:
            return n
    return 10


def get_wave_cache_key(dataset_name, class_name, n_shot, n_generate, wave_index):
    raw = f"progressive_waves_v1_{dataset_name}_{class_name}_{n_shot}_{n_generate}_wave{wave_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def normalize_scores(scores):
    if len(scores) == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.ones_like(scores)
    return (scores - mn) / (mx - mn)


def generate_smote_samples(class_embeddings, n_target, seed=42):
    n_real = len(class_embeddings)
    if n_real < 2:
        return np.array([]).reshape(0, class_embeddings.shape[1])
    dummy_other = np.random.RandomState(seed).randn(max(n_real, n_target + 1), class_embeddings.shape[1])
    X = np.vstack([class_embeddings, dummy_other])
    y = np.array([1]*n_real + [0]*len(dummy_other))
    target_count = n_real + n_target
    try:
        sm = SMOTE(random_state=seed, k_neighbors=min(5, n_real - 1),
                   sampling_strategy={1: target_count})
        X_res, y_res = sm.fit_resample(X, y)
        new_samples = X_res[len(X):]
        new_labels = y_res[len(X):]
        return new_samples[new_labels == 1][:n_target]
    except Exception:
        return np.array([]).reshape(0, class_embeddings.shape[1])


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
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    threshold = mean_score + threshold_k * std_score
    return threshold, mean_score, std_score


def compute_paired_statistics(method_f1s, smote_f1s, n_comparisons=1):
    deltas = np.array(method_f1s) - np.array(smote_f1s)
    n = len(deltas)
    if n < 2:
        return {"delta_mean_pp": float(np.mean(deltas)) * 100, "n": n,
                "p_value": 1.0, "bonferroni_p": 1.0, "cohen_d": 0.0,
                "win_rate": float(np.mean(deltas > 0)),
                "significant_005": False, "significant_bonferroni": False}
    t_stat, p_val = stats.ttest_rel(method_f1s, smote_f1s)
    bonf_p = min(1.0, p_val * n_comparisons)
    std_delta = np.std(deltas, ddof=1)
    d = float(np.mean(deltas) / std_delta) if std_delta > 0 else 0.0
    return {
        "delta_mean_pp": float(np.mean(deltas)) * 100,
        "n": n,
        "p_value": float(p_val),
        "bonferroni_p": float(bonf_p),
        "cohen_d": d,
        "win_rate": float(np.mean(deltas > 0)),
        "significant_005": p_val < 0.05,
        "significant_bonferroni": bonf_p < 0.05,
    }


# ============================================================================
# WAVE POOL LOADING — reads cached wave generations
# ============================================================================

def load_wave_pool(dataset_name, class_name, n_shot, wave_index,
                   model, cascade, real_embeddings, real_labels, labels_arr):
    """Load a single wave's cached generation, embed, and filter by threshold."""
    cache_key = get_wave_cache_key(dataset_name, class_name, n_shot,
                                   CANDIDATES_PER_WAVE, wave_index)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return np.array([]).reshape(0, 768), np.array([])

    with open(cache_file) as f:
        cached = json.load(f)
    gen_texts = cached.get("texts", [])
    if not gen_texts:
        return np.array([]).reshape(0, 768), np.array([])

    gen_emb = model.encode(gen_texts, show_progress_bar=False)

    class_mask = labels_arr == class_name
    class_embs = real_embeddings[class_mask]
    if len(class_embs) == 0:
        return gen_emb, np.ones(len(gen_emb))

    anchor = class_embs.mean(axis=0)
    scores, _ = cascade.compute_quality_scores(
        gen_emb, anchor, real_embeddings, labels_arr, class_name
    )

    # Apply threshold
    threshold_k = WAVE_CONFIGS[wave_index]["threshold_k"]
    threshold, mean_real, std_real = compute_wave_threshold(
        real_embeddings, real_labels, class_name, cascade, threshold_k
    )

    accepted_mask = scores >= threshold

    # Adaptive fallback
    if not accepted_mask.any() and len(gen_texts) > 0 and std_real > 0:
        fallback_threshold = threshold - 0.5 * std_real
        accepted_mask = scores >= fallback_threshold

    return gen_emb[accepted_mask], scores[accepted_mask]


# ============================================================================
# COMBINATION — pool from subset of waves, top-N by score
# ============================================================================

def combine_wave_subset(wave_pools, wave_indices, target_n, use_soft_weight=False):
    """Combine samples from a subset of waves. Top-N by raw score."""
    all_emb, all_scores = [], []
    for w_idx in sorted(wave_indices):
        if w_idx not in wave_pools:
            continue
        emb, scores = wave_pools[w_idx]
        if len(emb) == 0:
            continue
        all_emb.append(emb)
        all_scores.append(scores)

    if not all_emb:
        return np.array([]).reshape(0, 768), np.array([])

    all_emb = np.vstack(all_emb)
    all_scores = np.concatenate(all_scores)
    n = min(target_n, len(all_emb))
    top_idx = np.argsort(all_scores)[-n:]

    selected_emb = all_emb[top_idx]
    if use_soft_weight:
        weights = normalize_scores(all_scores[top_idx])
    else:
        weights = np.ones(n)

    return selected_emb, weights


# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

def get_ablation_configs():
    """Define all wave subset configurations to test."""
    configs = {}

    # Full set (reference)
    configs["all_waves"] = {0, 1, 2, 3}

    # Leave-one-out
    for w in range(4):
        name = f"without_wave{w}_{WAVE_CONFIGS[w]['name']}"
        configs[name] = {0, 1, 2, 3} - {w}

    # Individual waves
    for w in range(4):
        name = f"only_wave{w}_{WAVE_CONFIGS[w]['name']}"
        configs[name] = {w}

    # Cumulative (precision-first: start with wave 3, add less precise)
    configs["cumul_w3"] = {3}
    configs["cumul_w3_w2"] = {3, 2}
    configs["cumul_w3_w2_w1"] = {3, 2, 1}
    configs["cumul_w3_w2_w1_w0"] = {3, 2, 1, 0}

    # Pairs
    configs["early_waves_0_1"] = {0, 1}
    configs["late_waves_2_3"] = {2, 3}

    return configs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("WAVE ABLATION ANALYSIS")
    print("=" * 70)

    ablation_configs = get_ablation_configs()
    print(f"\nAblation configs: {len(ablation_configs)}")
    for name, waves in ablation_configs.items():
        print(f"  {name}: waves {sorted(waves)}")

    print("\nLoading embedding model...")
    import torch
    # Fix cublas strided batched matmul bug (PyTorch 2.10 + CUDA 12.8)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    cascade = FilterCascade(**FILTER_CONFIG)

    all_results = []
    n_total = len(DATASETS) * (len(ablation_configs) + 2) * len(CLASSIFIERS) * len(SEEDS)
    n_done = 0

    for ds_idx, ds_name in enumerate(DATASETS):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(DATASETS)}] {ds_name}")
        print(f"{'='*60}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        ds_base = get_dataset_base(ds_name)
        n_shot = parse_n_shot(ds_name)

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)

        # Load all wave pools for all classes
        print(f"  Loading wave pools for {len(unique_classes)} classes...")
        wave_pools_per_class = {}
        for cls in unique_classes:
            pools = {}
            for w_idx in range(N_WAVES):
                emb, scores = load_wave_pool(
                    ds_name, cls, n_shot, w_idx,
                    model, cascade, train_emb, train_labels, labels_arr
                )
                pools[w_idx] = (emb, scores)
                n_loaded = len(emb)
            wave_pools_per_class[cls] = pools

        # Print pool sizes
        for cls in unique_classes:
            sizes = [len(wave_pools_per_class[cls][w][0]) for w in range(N_WAVES)]
            print(f"    {cls}: waves = {sizes}, total = {sum(sizes)}")

        # Evaluate
        for seed in SEEDS:
            # Baseline: SMOTE
            smote_embs, smote_labels = [], []
            for cls in unique_classes:
                cls_emb = train_emb[labels_arr == cls]
                s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                if len(s) > 0:
                    smote_embs.append(s)
                    smote_labels.extend([cls] * len(s))

            for clf_name, clf_factory in CLASSIFIERS.items():
                if smote_labels:
                    aug_emb = np.vstack([train_emb, np.vstack(smote_embs)])
                    aug_lab = list(train_labels) + smote_labels
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab)
                    f1_smote = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                else:
                    clf = clf_factory(seed)
                    clf.fit(train_emb, train_labels)
                    f1_smote = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                all_results.append({
                    "dataset": ds_name, "dataset_base": ds_base, "n_shot": n_shot,
                    "method": "smote", "wave_subset": "none",
                    "classifier": clf_name, "seed": seed, "f1": f1_smote,
                })

            # Baseline: no augmentation
            for clf_name, clf_factory in CLASSIFIERS.items():
                clf = clf_factory(seed)
                clf.fit(train_emb, train_labels)
                f1_noaug = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                all_results.append({
                    "dataset": ds_name, "dataset_base": ds_base, "n_shot": n_shot,
                    "method": "no_augmentation", "wave_subset": "none",
                    "classifier": clf_name, "seed": seed, "f1": f1_noaug,
                })

            # Ablation configs
            for config_name, wave_set in ablation_configs.items():
                combined_emb_all, combined_labels, combined_weights = [], [], []
                for cls in unique_classes:
                    emb, weights = combine_wave_subset(
                        wave_pools_per_class[cls], wave_set, N_SYNTHETIC_PER_CLASS
                    )
                    if len(emb) > 0:
                        combined_emb_all.append(emb)
                        combined_labels.extend([cls] * len(emb))
                        combined_weights.append(weights)

                for clf_name, clf_factory in CLASSIFIERS.items():
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"    Progress: {n_done}/{n_total} "
                              f"({100*n_done/n_total:.0f}%)")

                    if combined_emb_all:
                        syn_emb = np.vstack(combined_emb_all)
                        syn_w = np.concatenate(combined_weights)
                        aug_emb = np.vstack([train_emb, syn_emb])
                        aug_lab = list(train_labels) + combined_labels
                        sw = np.concatenate([np.ones(len(train_emb)), syn_w])
                        clf = clf_factory(seed)
                        clf.fit(aug_emb, aug_lab, sample_weight=sw)
                        f1_val = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                    else:
                        clf = clf_factory(seed)
                        clf.fit(train_emb, train_labels)
                        f1_val = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                    all_results.append({
                        "dataset": ds_name, "dataset_base": ds_base, "n_shot": n_shot,
                        "method": "ablation", "wave_subset": config_name,
                        "waves_used": sorted(wave_set),
                        "classifier": clf_name, "seed": seed, "f1": f1_val,
                        "n_synthetic": len(combined_labels) if combined_emb_all else 0,
                    })

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "wave_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "ablation_configs": {k: sorted(v) for k, v in ablation_configs.items()}}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate report
    generate_ablation_report(all_results, ablation_configs)
    generate_ablation_table(all_results, ablation_configs)
    generate_ablation_figure(all_results, ablation_configs)


# ============================================================================
# REPORT
# ============================================================================

def generate_ablation_report(results, ablation_configs):
    """Statistical report for wave ablation."""
    print("\n" + "=" * 80)
    print("WAVE ABLATION — STATISTICAL REPORT")
    print("=" * 80)

    # Collect F1s
    smote_f1s = {}
    config_f1s = defaultdict(dict)

    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "smote":
            smote_f1s[key] = r["f1"]
        elif r["method"] == "ablation":
            config_f1s[r["wave_subset"]][key] = r["f1"]

    n_comparisons = len(ablation_configs)

    # ---- Section 1: All configs vs SMOTE ----
    print(f"\n{'Config':<35} {'Waves':>12} {'Delta':>10} {'Win%':>8} {'p':>10} {'d':>6}")
    print("-" * 85)

    all_stats = {}
    for config_name in ablation_configs:
        if config_name not in config_f1s:
            continue
        common = sorted(set(smote_f1s.keys()) & set(config_f1s[config_name].keys()))
        if not common:
            continue
        m = [config_f1s[config_name][k] for k in common]
        s = [smote_f1s[k] for k in common]
        st = compute_paired_statistics(m, s, n_comparisons)
        all_stats[config_name] = st

        waves_str = str(sorted(ablation_configs[config_name]))
        sig = "***" if st["significant_bonferroni"] else ("*" if st["significant_005"] else "")
        print(f"  {config_name:<33} {waves_str:>12} {st['delta_mean_pp']:>+8.2f}pp "
              f"{st['win_rate']:>7.1%}  {st['p_value']:>9.4f} {st['cohen_d']:>6.2f} {sig}")

    # ---- Section 2: Marginal contribution (leave-one-out) ----
    print(f"\n\n--- MARGINAL CONTRIBUTION (leave-one-out) ---")
    print(f"{'Removed wave':<25} {'Full':>8} {'Without':>8} {'Marginal':>10}")
    print("-" * 55)

    full_key = "all_waves"
    if full_key in all_stats:
        full_delta = all_stats[full_key]["delta_mean_pp"]
        for w in range(N_WAVES):
            loo_key = f"without_wave{w}_{WAVE_CONFIGS[w]['name']}"
            if loo_key in all_stats:
                loo_delta = all_stats[loo_key]["delta_mean_pp"]
                marginal = full_delta - loo_delta
                direction = "+" if marginal > 0 else "-"
                assessment = "HELPS" if marginal > 0.1 else ("HURTS" if marginal < -0.1 else "NEUTRAL")
                print(f"  Wave {w} ({WAVE_CONFIGS[w]['name']:<13}) "
                      f"{full_delta:>+7.2f}pp {loo_delta:>+7.2f}pp "
                      f"{marginal:>+9.2f}pp  [{assessment}]")

    # ---- Section 3: Individual wave performance ----
    print(f"\n\n--- INDIVIDUAL WAVE PERFORMANCE ---")
    print(f"{'Wave':<30} {'Delta vs SMOTE':>15} {'Win Rate':>10}")
    print("-" * 60)

    for w in range(N_WAVES):
        solo_key = f"only_wave{w}_{WAVE_CONFIGS[w]['name']}"
        if solo_key in all_stats:
            st = all_stats[solo_key]
            sig = "***" if st["significant_bonferroni"] else ("*" if st["significant_005"] else "")
            print(f"  Wave {w} ({WAVE_CONFIGS[w]['name']:<13})    "
                  f"{st['delta_mean_pp']:>+8.2f}pp     {st['win_rate']:>7.1%}  {sig}")

    # ---- Section 4: Cumulative (precision-first) ----
    print(f"\n\n--- CUMULATIVE (precision-first) ---")
    print(f"{'Config':<25} {'Delta':>10} {'Win%':>8} {'Marginal':>10}")
    print("-" * 55)

    cumul_keys = ["cumul_w3", "cumul_w3_w2", "cumul_w3_w2_w1", "cumul_w3_w2_w1_w0"]
    prev_delta = None
    for ck in cumul_keys:
        if ck in all_stats:
            delta = all_stats[ck]["delta_mean_pp"]
            marginal = f"{delta - prev_delta:>+8.2f}pp" if prev_delta is not None else "   base"
            print(f"  {ck:<23} {delta:>+8.2f}pp  {all_stats[ck]['win_rate']:>7.1%}  {marginal}")
            prev_delta = delta

    # ---- Section 5: Early vs Late ----
    print(f"\n\n--- EARLY vs LATE WAVES ---")
    for pair_key in ["early_waves_0_1", "late_waves_2_3"]:
        if pair_key in all_stats:
            st = all_stats[pair_key]
            print(f"  {pair_key:<25} {st['delta_mean_pp']:>+8.2f}pp  {st['win_rate']:>7.1%}")

    # ---- Section 6: Per n-shot breakdown for all_waves ----
    print(f"\n\n--- PER N-SHOT (all_waves) ---")
    for nshot in [10, 25, 50]:
        shot_keys = [k for k in smote_f1s if f"{nshot}shot" in k[0]]
        if full_key in config_f1s:
            common = sorted(set(shot_keys) & set(config_f1s[full_key].keys()))
            if common:
                m = [config_f1s[full_key][k] for k in common]
                s = [smote_f1s[k] for k in common]
                delta = (np.mean(m) - np.mean(s)) * 100
                win = np.mean(np.array(m) > np.array(s))
                print(f"  {nshot}-shot: {delta:>+7.2f}pp ({win*100:.0f}% win)")


def generate_ablation_table(results, ablation_configs):
    """LaTeX table of ablation results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    smote_f1s = {}
    config_f1s = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "smote":
            smote_f1s[key] = r["f1"]
        elif r["method"] == "ablation":
            config_f1s[r["wave_subset"]][key] = r["f1"]

    n_comparisons = len(ablation_configs)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Ablaci\'on de oleadas: contribuci\'on marginal de cada oleada "
                 r"y rendimiento individual. $\Delta$ es respecto a SMOTE.}")
    lines.append(r"\label{tab:wave_ablation}")
    lines.append(r"\begin{tabular}{llrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuraci\'on} & \textbf{Oleadas} & "
                 r"\textbf{$\Delta$ F1 (pp)} & \textbf{Win \%} & \textbf{Sig.} \\")
    lines.append(r"\midrule")

    # Groups
    groups = [
        ("Referencia", ["all_waves"]),
        ("Leave-one-out", [f"without_wave{w}_{WAVE_CONFIGS[w]['name']}" for w in range(4)]),
        ("Individual", [f"only_wave{w}_{WAVE_CONFIGS[w]['name']}" for w in range(4)]),
        ("Acumulativa", ["cumul_w3", "cumul_w3_w2", "cumul_w3_w2_w1", "cumul_w3_w2_w1_w0"]),
        ("Pares", ["early_waves_0_1", "late_waves_2_3"]),
    ]

    display_names = {
        "all_waves": "Todas (0-3)",
        "without_wave0_exploratory": "Sin oleada 0 (exploratoria)",
        "without_wave1_moderate": "Sin oleada 1 (moderada)",
        "without_wave2_focused": "Sin oleada 2 (enfocada)",
        "without_wave3_ultra_precise": "Sin oleada 3 (ultra-precisa)",
        "only_wave0_exploratory": "Solo oleada 0",
        "only_wave1_moderate": "Solo oleada 1",
        "only_wave2_focused": "Solo oleada 2",
        "only_wave3_ultra_precise": "Solo oleada 3",
        "cumul_w3": "Oleada 3",
        "cumul_w3_w2": "Oleadas 3+2",
        "cumul_w3_w2_w1": "Oleadas 3+2+1",
        "cumul_w3_w2_w1_w0": "Oleadas 3+2+1+0",
        "early_waves_0_1": "Tempranas (0+1)",
        "late_waves_2_3": "Tard\\'{\\i}as (2+3)",
    }

    for group_name, keys in groups:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for config_key in keys:
            if config_key not in config_f1s:
                continue
            common = sorted(set(smote_f1s.keys()) & set(config_f1s[config_key].keys()))
            if not common:
                continue
            m = [config_f1s[config_key][k] for k in common]
            s = [smote_f1s[k] for k in common]
            st = compute_paired_statistics(m, s, n_comparisons)

            sig = "***" if st["significant_bonferroni"] else ("**" if st["significant_005"] else "")
            name = display_names.get(config_key, config_key)
            waves_str = str(sorted(ablation_configs[config_key]))

            # Bold if best
            delta_str = f"{st['delta_mean_pp']:+.2f}"
            if config_key == "all_waves":
                delta_str = f"\\textbf{{{delta_str}}}"

            lines.append(f"  {name} & {waves_str} & {delta_str} & "
                        f"{st['win_rate']*100:.1f}\\% & {sig} \\\\")
        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_path = OUTPUT_DIR / "tab_wave_ablation.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nTable saved: {table_path}")


def generate_ablation_figure(results, ablation_configs):
    """Bar chart comparing wave subsets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping figure")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    smote_f1s = {}
    config_f1s = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "smote":
            smote_f1s[key] = r["f1"]
        elif r["method"] == "ablation":
            config_f1s[r["wave_subset"]][key] = r["f1"]

    # Leave-one-out marginal contribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Marginal contribution (leave-one-out)
    ax = axes[0]
    full_key = "all_waves"
    if full_key in config_f1s:
        full_common = sorted(set(smote_f1s.keys()) & set(config_f1s[full_key].keys()))
        full_deltas = np.array([config_f1s[full_key][k] - smote_f1s[k] for k in full_common])
        full_mean = full_deltas.mean() * 100

        wave_names = []
        marginals = []
        for w in range(N_WAVES):
            loo_key = f"without_wave{w}_{WAVE_CONFIGS[w]['name']}"
            if loo_key in config_f1s:
                loo_common = sorted(set(smote_f1s.keys()) & set(config_f1s[loo_key].keys()))
                loo_deltas = np.array([config_f1s[loo_key][k] - smote_f1s[k] for k in loo_common])
                loo_mean = loo_deltas.mean() * 100
                marginals.append(full_mean - loo_mean)
                wave_names.append(f"W{w}\n({WAVE_CONFIGS[w]['name'][:6]})")

        colors = ['#e74c3c' if m < 0 else '#2ecc71' for m in marginals]
        ax.bar(wave_names, marginals, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel("Marginal Contribution (pp)")
        ax.set_title("Leave-One-Out: Marginal Contribution")
        ax.set_xlabel("Removed Wave")

        for i, v in enumerate(marginals):
            ax.text(i, v + (0.05 if v >= 0 else -0.15),
                    f"{v:+.2f}", ha='center', fontsize=9, fontweight='bold')

    # Panel 2: Individual wave vs SMOTE
    ax = axes[1]
    wave_names = []
    wave_deltas = []
    wave_cis = []
    for w in range(N_WAVES):
        solo_key = f"only_wave{w}_{WAVE_CONFIGS[w]['name']}"
        if solo_key in config_f1s:
            common = sorted(set(smote_f1s.keys()) & set(config_f1s[solo_key].keys()))
            if common:
                m = np.array([config_f1s[solo_key][k] for k in common])
                s = np.array([smote_f1s[k] for k in common])
                deltas = (m - s) * 100
                wave_deltas.append(deltas.mean())
                wave_cis.append(1.96 * deltas.std() / np.sqrt(len(deltas)))
                wave_names.append(f"W{w}\n({WAVE_CONFIGS[w]['name'][:6]})")

    colors = ['#3498db', '#f39c12', '#e74c3c', '#9b59b6'][:len(wave_names)]
    ax.bar(wave_names, wave_deltas, yerr=wave_cis, color=colors,
           edgecolor='black', linewidth=0.5, capsize=5)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel("Delta vs SMOTE (pp)")
    ax.set_title("Individual Wave Performance")
    ax.set_xlabel("Wave (standalone)")

    for i, (v, ci) in enumerate(zip(wave_deltas, wave_cis)):
        ax.text(i, v + ci + 0.1, f"{v:+.2f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_wave_ablation.pdf"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
