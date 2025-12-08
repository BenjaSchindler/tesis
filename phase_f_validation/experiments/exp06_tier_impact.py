#!/usr/bin/env python3
"""
Experiment 06: Tier Impact Analysis

Analyzes impact by performance tier:
- LOW: F1 < 0.20 (6 classes)
- MID: 0.20 <= F1 < 0.45 (4 classes)
- HIGH: F1 >= 0.45 (6 classes)

Metrics: Delta F1 per tier, number of classes

Output: tab:tier_impact for Metodologia.tex
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from base_config import RESULTS_DIR, LATEX_DIR, BASE_PARAMS, KFOLD_CONFIG, EXPERIMENT_PARAMS, MBTI_CLASSES, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from validation_runner import load_data, EmbeddingCache, KFoldEvaluator, LLMSyntheticGenerator
from typing import List, Tuple


TIERS = EXPERIMENT_PARAMS["tier_impact"]["TIERS"]
EXPERIMENT_NAME = "tier_impact"


@dataclass
class TierResult:
    tier: str
    f1_range: str
    n_classes: int
    classes: list
    delta_f1_avg: float
    delta_f1_std: float


def compute_baseline_f1_per_class(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """Compute baseline F1 for each class using K-fold CV."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    class_f1s = defaultdict(list)
    unique_labels = np.unique(labels)

    for train_idx, test_idx in kfold.split(embeddings, labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Per-class F1
        for i, cls in enumerate(unique_labels):
            cls_mask = y_test == cls
            if cls_mask.sum() > 0:
                cls_pred_mask = y_pred == cls
                # Compute F1 for this class
                tp = ((y_test == cls) & (y_pred == cls)).sum()
                fp = ((y_test != cls) & (y_pred == cls)).sum()
                fn = ((y_test == cls) & (y_pred != cls)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                class_f1s[cls].append(f1)

    # Average across folds
    return {cls: np.mean(f1s) for cls, f1s in class_f1s.items()}


def classify_classes_by_tier(baseline_f1s: dict) -> dict:
    """Classify classes into tiers based on baseline F1."""
    tier_classes = {"LOW": [], "MID": [], "HIGH": []}

    for cls, f1 in baseline_f1s.items():
        if f1 < TIERS["LOW"]["max"]:
            tier_classes["LOW"].append((cls, f1))
        elif f1 < TIERS["MID"]["max"]:
            tier_classes["MID"].append((cls, f1))
        else:
            tier_classes["HIGH"].append((cls, f1))

    return tier_classes


def generate_synthetic_simple(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples using LLM and base configuration."""
    generator = LLMSyntheticGenerator(cache)
    return generator.generate_all(
        embeddings, labels, texts,
        k_clusters=BASE_PARAMS["max_clusters"],
        samples_per_cluster=max(1, BASE_PARAMS["samples_per_prompt"] // 2)
    )


def compute_augmented_f1_per_class(
    embeddings: np.ndarray,
    labels: np.ndarray,
    synth_embeddings: np.ndarray,
    synth_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """Compute augmented F1 for each class using K-fold CV."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    class_f1s = defaultdict(list)
    unique_labels = np.unique(labels)

    for train_idx, test_idx in kfold.split(embeddings, labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Add synthetics to training
        if len(synth_embeddings) > 0:
            X_train_aug = np.vstack([X_train, synth_embeddings])
            y_train_aug = np.concatenate([y_train, synth_labels])
            weights = np.concatenate([
                np.ones(len(X_train)),
                np.full(len(synth_embeddings), BASE_PARAMS["synthetic_weight"])
            ])
        else:
            X_train_aug = X_train
            y_train_aug = y_train
            weights = None

        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X_train_aug, y_train_aug, sample_weight=weights)
        y_pred = clf.predict(X_test)

        # Per-class F1
        for cls in unique_labels:
            tp = ((y_test == cls) & (y_pred == cls)).sum()
            fp = ((y_test != cls) & (y_pred == cls)).sum()
            fn = ((y_test == cls) & (y_pred != cls)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_f1s[cls].append(f1)

    return {cls: np.mean(f1s) for cls, f1s in class_f1s.items()}


def run_tier_experiment() -> list:
    """Run tier impact analysis experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 06: TIER IMPACT ANALYSIS")
    print("="*60)
    print(f"  Tiers: LOW (<0.20), MID (0.20-0.45), HIGH (>=0.45)")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    # Compute baseline F1 per class
    print("\n  Computing baseline F1 per class...")
    baseline_f1s = compute_baseline_f1_per_class(embeddings, labels)

    print("\n  Baseline F1 per class:")
    for cls in sorted(baseline_f1s.keys()):
        print(f"    {cls}: {baseline_f1s[cls]:.4f}")

    # Classify into tiers
    tier_classes = classify_classes_by_tier(baseline_f1s)

    print("\n  Tier classification:")
    for tier, classes in tier_classes.items():
        print(f"    {tier}: {len(classes)} classes - {[c for c, _ in classes]}")

    # Generate synthetic data using LLM
    print("\n  Generating synthetic data with LLM...")
    synth_embeddings, synth_labels = generate_synthetic_simple(embeddings, labels, texts, cache)
    print(f"  Generated {len(synth_embeddings)} synthetic samples")

    # Compute augmented F1 per class
    print("\n  Computing augmented F1 per class...")
    augmented_f1s = compute_augmented_f1_per_class(
        embeddings, labels, synth_embeddings, synth_labels
    )

    # Compute delta per class
    deltas = {}
    for cls in baseline_f1s:
        baseline = baseline_f1s[cls]
        augmented = augmented_f1s.get(cls, baseline)
        if baseline > 0:
            delta_pct = ((augmented - baseline) / baseline) * 100
        else:
            delta_pct = 0 if augmented == 0 else 100
        deltas[cls] = delta_pct

    print("\n  Delta per class:")
    for cls in sorted(deltas.keys()):
        print(f"    {cls}: {deltas[cls]:+.2f}%")

    # Aggregate by tier
    results = []
    tier_ranges = {"LOW": "< 0.20", "MID": "0.20 - 0.45", "HIGH": ">= 0.45"}

    for tier in ["LOW", "MID", "HIGH"]:
        classes = [c for c, _ in tier_classes[tier]]
        if not classes:
            continue

        tier_deltas = [deltas[c] for c in classes if c in deltas]

        result = TierResult(
            tier=tier,
            f1_range=tier_ranges[tier],
            n_classes=len(classes),
            classes=classes,
            delta_f1_avg=np.mean(tier_deltas) if tier_deltas else 0.0,
            delta_f1_std=np.std(tier_deltas) if len(tier_deltas) > 1 else 0.0
        )
        results.append(result)

        print(f"\n{'─'*60}")
        print(f"  Tier {tier} ({tier_ranges[tier]})")
        print(f"{'─'*60}")
        print(f"  Classes: {classes}")
        print(f"  Delta F1 avg: {result.delta_f1_avg:+.2f}%")

    # Save results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tier_impact_results.json", 'w') as f:
        json.dump({
            "baseline_f1s": {str(k): v for k, v in baseline_f1s.items()},
            "augmented_f1s": {str(k): v for k, v in augmented_f1s.items()},
            "deltas": {str(k): v for k, v in deltas.items()},
            "tier_results": [asdict(r) for r in results]
        }, f, indent=2)

    return results


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table for tier impact."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto de ponderacion uniforme por nivel de rendimiento}
\label{tab:tier_impact}
\begin{tabular}{lccc}
\hline
Tier & Rango F1 & Clases & $\Delta$F1 Promedio \\
\hline
"""

    for r in results:
        latex += f"{r.tier} & ${r.f1_range}$ & {r.n_classes} & {r.delta_f1_avg:+.2f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_tier_experiment()

    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_tier_impact.tex", 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("  SUMMARY: Tier Impact Results")
    print("="*60)
    for r in results:
        print(f"  {r.tier:>5} | Range: {r.f1_range:>12} | Classes: {r.n_classes} | Delta: {r.delta_f1_avg:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
