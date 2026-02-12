#!/usr/bin/env python3
"""
Test geometric filtering on 3 NEW datasets: CLINC150, BANKING77, TREC-6.

Reuses the full pipeline from exp_thesis_final.py but focused on:
- 3 new datasets × 3 shots × 4 methods × 3 classifiers × 3 seeds = 324 configs
- Key methods: no_augmentation, smote, binary_filter, soft_weighted
- Key classifiers: ridge, svc_linear, logistic_regression (excluding RF and MLP for speed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import numpy as np
from datetime import datetime
from collections import Counter
from dataclasses import asdict

# Import everything from thesis_final
import exp_thesis_final as etf

# ============================================================================
# OVERRIDE CONFIGURATION
# ============================================================================

# New datasets only
NEW_DATASETS = [
    "clinc150_10shot", "clinc150_25shot", "clinc150_50shot",
    "banking77_10shot", "banking77_25shot", "banking77_50shot",
    "trec6_10shot", "trec6_25shot", "trec6_50shot",
]

NEW_DATASET_N_CLASSES = {
    "clinc150": 150,
    "banking77": 77,
    "trec6": 6,
}

# Focused methods (skip EDA, back_translation, random_oversample for speed)
METHODS = ["no_augmentation", "smote", "binary_filter", "soft_weighted"]

# Focused classifiers (skip RF which doesn't benefit, skip MLP for speed)
CLASSIFIERS = ["ridge", "svc_linear", "logistic_regression"]

# 3 seeds (sufficient for significance with many dataset×shot pairs)
SEEDS = [42, 123, 456]

# Results directory
RESULTS_DIR = etf.PROJECT_ROOT / "results" / "new_datasets"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_base(dataset_name):
    """Get base name for new datasets."""
    for base in sorted(NEW_DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if dataset_name.startswith(base + "_"):
            return base
    # Fallback to original
    return etf.get_dataset_base(dataset_name)


def get_n_shot(dataset_name):
    """Extract n_shot from dataset name."""
    for shot in [10, 25, 50]:
        if f"_{shot}shot" in dataset_name:
            return shot
    return 0


def main():
    start_time = time.time()
    total_configs = len(NEW_DATASETS) * len(SEEDS) * len(CLASSIFIERS) * len(METHODS)

    print("=" * 90)
    print("NEW DATASETS EXPERIMENT: CLINC150, BANKING77, TREC-6")
    print(f"Datasets: {len(NEW_DATASETS)} | Methods: {len(METHODS)} | "
          f"Classifiers: {len(CLASSIFIERS)} | Seeds: {len(SEEDS)}")
    print(f"Total configs: {total_configs}")
    print("=" * 90)

    # Initialize models
    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    from core.llm_providers import create_provider
    provider = create_provider("google", "gemini-3-flash-preview")

    results = []
    config_idx = 0

    for ds_idx, dataset_name in enumerate(NEW_DATASETS):
        dataset_base = get_dataset_base(dataset_name)
        n_classes = NEW_DATASET_N_CLASSES.get(dataset_base, 0)
        n_shot = get_n_shot(dataset_name)

        print(f"\n{'='*90}")
        print(f"DATASET {ds_idx+1}/{len(NEW_DATASETS)}: {dataset_name} "
              f"({n_classes} classes, {n_shot}-shot)")
        print(f"{'='*90}")

        # Load data
        try:
            train_texts, train_labels, test_texts, test_labels = etf.load_dataset(dataset_name)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        label_counts = Counter(train_labels)
        print(f"  Classes: {len(label_counts)}, samples/class: "
              f"min={min(label_counts.values())}, max={max(label_counts.values())}")

        # Embed
        print("  Embedding train/test...")
        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        train_labels_arr = np.array(train_labels)
        test_labels_arr = np.array(test_labels)

        # Pre-compute back-translation placeholder (not used in our methods)
        bt_data_cache = None

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  SEED {seed} ({seed_idx+1}/{len(SEEDS)})")

            # Pre-compute augmentations
            # We need a BackTranslationAugmenter placeholder
            class DummyBT:
                pass
            dummy_bt = DummyBT()

            aug_data = etf.precompute_augmentations(
                dataset_name, train_texts, train_labels, train_emb,
                n_shot, model, provider, dummy_bt, seed
            )

            # Back-translation: set empty (we skip it)
            aug_data["back_translation"] = {"embeddings": np.zeros((0, 768)), "labels": []}

            for clf_name in CLASSIFIERS:
                for method in METHODS:
                    config_idx += 1

                    result = etf.run_single_config(
                        dataset_name=dataset_name,
                        dataset_base=dataset_base,
                        n_classes=n_classes,
                        n_shot=n_shot,
                        classifier_name=clf_name,
                        aug_method=method,
                        seed=seed,
                        train_embeddings=train_emb,
                        train_labels=train_labels_arr,
                        test_embeddings=test_emb,
                        test_labels=test_labels_arr,
                        aug_data=aug_data,
                    )

                    results.append(result)

                    # Progress
                    if method == "soft_weighted":
                        smote_f1 = next(
                            (r.f1_macro for r in results
                             if r.dataset == dataset_name and r.classifier == clf_name
                             and r.seed == seed and r.augmentation_method == "smote"),
                            None
                        )
                        delta = (result.f1_macro - smote_f1) * 100 if smote_f1 else 0
                        print(f"    [{config_idx}/{total_configs}] {clf_name}/{method}: "
                              f"F1={result.f1_macro:.4f} ({delta:+.2f}pp vs SMOTE)")

        # Save incrementally
        save_results(results)

    # Generate report
    generate_report(results)

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Total configs: {len(results)}")


def save_results(results):
    """Save results to JSON."""
    output = {
        "experiment": "new_datasets",
        "datasets": list(set(r.dataset for r in results)),
        "methods": METHODS,
        "classifiers": CLASSIFIERS,
        "seeds": SEEDS,
        "n_results": len(results),
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


def generate_report(results):
    """Generate summary tables."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 90)
    print("REPORT: NEW DATASETS")
    print("=" * 90)

    # ---- Table 1: Overall per dataset ----
    print("\n--- TABLE 1: Soft Weighted vs SMOTE by Dataset ---")
    print(f"{'Dataset':<25s} {'N cls':>5s} {'Shot':>5s} {'SMOTE F1':>10s} "
          f"{'Soft F1':>10s} {'Δ':>8s} {'Win%':>6s}")
    print("-" * 75)

    for ds_name in NEW_DATASETS:
        base = get_dataset_base(ds_name)
        n_cls = NEW_DATASET_N_CLASSES.get(base, 0)
        n_shot = get_n_shot(ds_name)

        smote_f1s = [r.f1_macro for r in results
                     if r.dataset == ds_name and r.augmentation_method == "smote"]
        soft_f1s = [r.f1_macro for r in results
                    if r.dataset == ds_name and r.augmentation_method == "soft_weighted"]

        if not smote_f1s or not soft_f1s:
            continue

        smote_mean = np.mean(smote_f1s)
        soft_mean = np.mean(soft_f1s)
        delta = (soft_mean - smote_mean) * 100

        # Paired win rate
        wins = sum(1 for s, m in zip(soft_f1s, smote_f1s) if s > m)
        win_pct = wins / len(soft_f1s) * 100

        print(f"  {ds_name:<23s} {n_cls:>5d} {n_shot:>5d} {smote_mean*100:>9.2f}% "
              f"{soft_mean*100:>9.2f}% {delta:>+7.2f}pp {win_pct:>5.1f}%")

    # ---- Table 2: Aggregate per dataset base ----
    print("\n--- TABLE 2: Aggregate by Dataset (all shots pooled) ---")
    print(f"{'Dataset':<15s} {'N cls':>5s} {'Δ SMOTE':>10s} {'p-value':>10s} "
          f"{'Cohen d':>8s} {'Win%':>6s}")
    print("-" * 60)

    for base in sorted(NEW_DATASET_N_CLASSES.keys()):
        n_cls = NEW_DATASET_N_CLASSES[base]
        smote_f1s = [r.f1_macro for r in results
                     if r.dataset_base == base and r.augmentation_method == "smote"]
        soft_f1s = [r.f1_macro for r in results
                    if r.dataset_base == base and r.augmentation_method == "soft_weighted"]

        if len(smote_f1s) < 2:
            continue

        deltas = np.array(soft_f1s) - np.array(smote_f1s)
        mean_d = np.mean(deltas)
        std_d = np.std(deltas, ddof=1)

        if std_d > 0 and len(deltas) > 1:
            t_stat, p_val = sp_stats.ttest_rel(soft_f1s, smote_f1s)
            d = mean_d / std_d
        else:
            p_val, d = 1.0, 0.0

        win_pct = np.mean(deltas > 0) * 100
        sig = "*" if p_val < 0.05 else ""

        print(f"  {base:<13s} {n_cls:>5d} {mean_d*100:>+9.2f}pp {p_val:>9.4f}{sig} "
              f"{d:>7.2f} {win_pct:>5.1f}%")

    # ---- Table 3: By N-shot ----
    print("\n--- TABLE 3: Soft Weighted vs SMOTE by N-Shot ---")
    print(f"{'N-Shot':>8s} {'Δ SMOTE':>10s} {'Win%':>6s} {'N configs':>10s}")
    print("-" * 40)

    for shot in [10, 25, 50]:
        smote_f1s = [r.f1_macro for r in results
                     if r.n_shot == shot and r.augmentation_method == "smote"]
        soft_f1s = [r.f1_macro for r in results
                    if r.n_shot == shot and r.augmentation_method == "soft_weighted"]
        if not smote_f1s:
            continue
        deltas = np.array(soft_f1s) - np.array(smote_f1s)
        print(f"  {shot:>6d} {np.mean(deltas)*100:>+9.2f}pp {np.mean(deltas>0)*100:>5.1f}% "
              f"{len(deltas):>10d}")

    # ---- Table 4: All methods ----
    print("\n--- TABLE 4: All Methods Comparison ---")
    print(f"{'Method':<20s} {'Mean F1':>10s} {'Δ SMOTE':>10s} {'Win%':>6s}")
    print("-" * 50)

    smote_all = np.array([r.f1_macro for r in results if r.augmentation_method == "smote"])

    for method in METHODS:
        method_f1s = np.array([r.f1_macro for r in results if r.augmentation_method == method])
        if len(method_f1s) == 0:
            continue
        mean_f1 = np.mean(method_f1s)
        if method == "smote":
            print(f"  {method:<18s} {mean_f1*100:>9.2f}%    (reference)")
        else:
            delta = np.mean(method_f1s - smote_all) * 100 if len(method_f1s) == len(smote_all) else 0
            wins = np.mean(method_f1s > smote_all) * 100 if len(method_f1s) == len(smote_all) else 0
            print(f"  {method:<18s} {mean_f1*100:>9.2f}% {delta:>+9.2f}pp {wins:>5.1f}%")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
