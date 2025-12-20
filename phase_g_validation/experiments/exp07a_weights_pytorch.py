#!/usr/bin/env python3
"""
Experiment 07a: Synthetic Weights validation with PyTorch MLP

Tests: weight = 0.3, 0.5, 0.7, 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import asdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from pytorch_mlp_classifier import train_pytorch_mlp, DEVICE
from base_config import RESULTS_DIR

print(f"Using device: {DEVICE}")

# Weight values to test
WEIGHT_VALUES = [0.3, 0.5, 0.7, 1.0]

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def run_weighted_kfold(X_orig, y_orig, X_synth, y_synth, unique_labels, weight, n_splits=5, n_repeats=3):
    """Run K-fold CV with weighted synthetic samples."""
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    y_encoded = np.array([label_to_idx[l] for l in y_orig])
    y_synth_encoded = np.array([label_to_idx[l] for l in y_synth]) if len(y_synth) > 0 else np.array([])

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_folds = n_splits * n_repeats

    baseline_f1s = []
    augmented_f1s = []
    per_class_baseline = {l: [] for l in unique_labels}
    per_class_augmented = {l: [] for l in unique_labels}

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_encoded)):
        X_train, y_train = X_orig[train_idx], y_encoded[train_idx]
        X_test, y_test = X_orig[test_idx], y_encoded[test_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # Baseline
        y_pred_base = train_pytorch_mlp(X_tr_scaled, y_train, X_te_scaled, num_classes)
        base_macro = f1_score(y_test, y_pred_base, average="macro")
        base_per_class = f1_score(y_test, y_pred_base, average=None, labels=range(num_classes), zero_division=0)
        baseline_f1s.append(base_macro)
        for i, l in enumerate(unique_labels):
            per_class_baseline[l].append(base_per_class[i])

        # Augmented with weights (duplicate weighted samples)
        if len(X_synth) > 0:
            # For simplicity with PyTorch, we duplicate samples based on weight
            if weight < 1.0:
                # Sample a fraction of synthetics
                n_keep = max(1, int(len(X_synth) * weight))
                idx = np.random.choice(len(X_synth), n_keep, replace=False)
                X_synth_use = X_synth[idx]
                y_synth_use = y_synth_encoded[idx]
            else:
                X_synth_use = X_synth
                y_synth_use = y_synth_encoded

            X_train_aug = np.vstack([X_train, X_synth_use])
            y_train_aug = np.concatenate([y_train, y_synth_use])

            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_te_aug_scaled = scaler_aug.transform(X_test)

            y_pred_aug = train_pytorch_mlp(X_tr_aug_scaled, y_train_aug, X_te_aug_scaled, num_classes)
        else:
            y_pred_aug = y_pred_base

        aug_macro = f1_score(y_test, y_pred_aug, average="macro")
        aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=range(num_classes), zero_division=0)
        augmented_f1s.append(aug_macro)
        for i, l in enumerate(unique_labels):
            per_class_augmented[l].append(aug_per_class[i])

        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{total_folds}: base={base_macro:.4f}, aug={aug_macro:.4f}")

    # Compute statistics
    base_arr = np.array(baseline_f1s)
    aug_arr = np.array(augmented_f1s)
    deltas = aug_arr - base_arr

    base_mean = np.mean(base_arr)
    aug_mean = np.mean(aug_arr)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    delta_pp = delta_mean * 100

    n = len(deltas)
    se = delta_std / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    win_rate = np.mean(deltas > 0)

    # Per-class results
    per_class_f1 = {}
    for l in unique_labels:
        base_class = np.mean(per_class_baseline[l])
        aug_class = np.mean(per_class_augmented[l])
        per_class_f1[l] = {
            "baseline_f1": float(base_class),
            "augmented_f1": float(aug_class),
            "delta_pp": float((aug_class - base_class) * 100)
        }

    return {
        "baseline_mean": float(base_mean),
        "augmented_mean": float(aug_mean),
        "delta_pp": float(delta_pp),
        "delta_std": float(delta_std),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "win_rate": float(win_rate),
        "ci_95": [float(ci_95[0]), float(ci_95[1])],
        "per_class_f1": per_class_f1
    }


def main():
    print("=" * 70)
    print("Experiment 07a: Synthetic Weights Validation (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    # Generate synthetics once
    print("\nGenerating synthetics...")
    generator = SyntheticGenerator(cache, BASE_CONFIG)

    all_synth_emb = []
    all_synth_labels = []

    for label in unique_labels:
        class_mask = np.array(labels) == label
        class_texts = texts[class_mask]
        class_emb = embeddings[class_mask]

        try:
            synth_texts, _ = generator.generate_for_class(class_texts, class_emb, label)
            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label] * len(synth_emb))
                print(f"  {label}: +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"  {label}: Error - {e}")

    if all_synth_emb:
        X_synth = np.vstack(all_synth_emb)
        y_synth = np.array(all_synth_labels)
    else:
        X_synth = np.array([]).reshape(0, embeddings.shape[1])
        y_synth = np.array([])

    print(f"Total synthetic: {len(X_synth)}")

    results = []

    for weight in WEIGHT_VALUES:
        print(f"\n{'='*70}")
        print(f"Testing weight = {weight}")
        print("=" * 70)

        result = run_weighted_kfold(embeddings, labels, X_synth, y_synth, unique_labels, weight)
        result["config_name"] = "weight"
        result["config_value"] = weight
        result["n_synthetic"] = len(X_synth)
        results.append(result)

        sig = "*" if result["significant"] else ""
        print(f"  Result: Δ = {result['delta_pp']:+.2f} pp (p={result['p_value']:.4f}){sig}")

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "synthetic_weights",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "values_tested": WEIGHT_VALUES,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp07a_weights.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Synthetic Weights Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_weights}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Weight & $\Delta$ (pp) & p-value & Win\% \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r["significant"] else ""
        latex_lines.append(
            f"{r['config_value']} & {r['delta_pp']:+.2f}{sig} & {r['p_value']:.4f} & {r['win_rate']*100:.0f}\\% \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_weights.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Weights Experiment")
    print("=" * 70)
    print(f"{'Weight':<8} {'Δ (pp)':<10} {'p-value':<10} {'Sig':<6}")
    print("-" * 36)
    for r in results:
        sig = "YES" if r["significant"] else "no"
        print(f"{r['config_value']:<8} {r['delta_pp']:+.2f}      {r['p_value']:.4f}     {sig}")

    print(f"\nResults saved to {output_dir / 'exp07a_weights.json'}")


if __name__ == "__main__":
    main()
