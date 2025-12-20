#!/usr/bin/env python3
"""
Experiment 06: Tier Impact Analysis with PyTorch MLP

Analyzes improvement by performance tier:
- LOW: F1 < 0.20 (difficult classes)
- MID: 0.20 ≤ F1 < 0.45
- HIGH: F1 ≥ 0.45
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from pytorch_mlp_classifier import train_pytorch_mlp, DEVICE
from base_config import RESULTS_DIR

print(f"Using device: {DEVICE}")

# Tier thresholds
TIER_THRESHOLDS = {
    "LOW": (0, 0.20),
    "MID": (0.20, 0.45),
    "HIGH": (0.45, 1.0),
}

# Base config (best from previous experiments)
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def main():
    print("=" * 70)
    print("Experiment 06: Tier Impact Analysis (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    # Generate synthetics
    print("\nGenerating synthetics...")
    generator = SyntheticGenerator(cache, BASE_CONFIG)

    all_synth_emb = []
    all_synth_labels = []
    synth_per_class = {}

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
                synth_per_class[label] = len(synth_emb)
                print(f"  {label}: +{len(synth_emb)} synthetic")
        except Exception as e:
            synth_per_class[label] = 0
            print(f"  {label}: Error - {e}")

    if all_synth_emb:
        X_synth = np.vstack(all_synth_emb)
        y_synth = np.array(all_synth_labels)
    else:
        X_synth = np.array([]).reshape(0, embeddings.shape[1])
        y_synth = np.array([])

    print(f"Total synthetic: {len(X_synth)}")

    # Run K-fold with detailed per-class tracking
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    y_encoded = np.array([label_to_idx[l] for l in labels])
    y_synth_encoded = np.array([label_to_idx[l] for l in y_synth]) if len(y_synth) > 0 else np.array([])

    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    per_class_baseline = {l: [] for l in unique_labels}
    per_class_augmented = {l: [] for l in unique_labels}
    macro_baseline = []
    macro_augmented = []

    print(f"\nRunning 15-fold CV with per-class tracking...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(embeddings, y_encoded)):
        X_train, y_train = embeddings[train_idx], y_encoded[train_idx]
        X_test, y_test = embeddings[test_idx], y_encoded[test_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # Baseline
        y_pred_base = train_pytorch_mlp(X_tr_scaled, y_train, X_te_scaled, num_classes)
        base_macro = f1_score(y_test, y_pred_base, average="macro")
        base_per_class = f1_score(y_test, y_pred_base, average=None, labels=range(num_classes), zero_division=0)
        macro_baseline.append(base_macro)
        for i, l in enumerate(unique_labels):
            per_class_baseline[l].append(base_per_class[i])

        # Augmented
        if len(X_synth) > 0:
            X_train_aug = np.vstack([X_train, X_synth])
            y_train_aug = np.concatenate([y_train, y_synth_encoded])

            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_te_aug_scaled = scaler_aug.transform(X_test)

            y_pred_aug = train_pytorch_mlp(X_tr_aug_scaled, y_train_aug, X_te_aug_scaled, num_classes)
        else:
            y_pred_aug = y_pred_base

        aug_macro = f1_score(y_test, y_pred_aug, average="macro")
        aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=range(num_classes), zero_division=0)
        macro_augmented.append(aug_macro)
        for i, l in enumerate(unique_labels):
            per_class_augmented[l].append(aug_per_class[i])

        if (fold_idx + 1) % 5 == 0:
            print(f"  Fold {fold_idx + 1}/15: base={base_macro:.4f}, aug={aug_macro:.4f}")

    # Compute per-class results and assign tiers
    per_class_results = {}
    tier_classes = {"LOW": [], "MID": [], "HIGH": []}

    for l in unique_labels:
        base_mean = np.mean(per_class_baseline[l])
        aug_mean = np.mean(per_class_augmented[l])
        delta_pp = (aug_mean - base_mean) * 100

        # Assign tier based on baseline F1
        if base_mean < TIER_THRESHOLDS["LOW"][1]:
            tier = "LOW"
        elif base_mean < TIER_THRESHOLDS["MID"][1]:
            tier = "MID"
        else:
            tier = "HIGH"

        tier_classes[tier].append(l)

        per_class_results[l] = {
            "baseline_f1": float(base_mean),
            "augmented_f1": float(aug_mean),
            "delta_pp": float(delta_pp),
            "tier": tier,
            "n_synthetic": synth_per_class.get(l, 0)
        }

    # Compute tier-level statistics
    tier_results = {}
    for tier in ["LOW", "MID", "HIGH"]:
        tier_deltas = [per_class_results[l]["delta_pp"] for l in tier_classes[tier]]

        if tier_deltas:
            tier_results[tier] = {
                "classes": tier_classes[tier],
                "n_classes": len(tier_classes[tier]),
                "mean_delta_pp": float(np.mean(tier_deltas)),
                "std_delta_pp": float(np.std(tier_deltas)),
                "min_delta_pp": float(np.min(tier_deltas)),
                "max_delta_pp": float(np.max(tier_deltas)),
                "improved": sum(1 for d in tier_deltas if d > 0),
            }
        else:
            tier_results[tier] = {
                "classes": [],
                "n_classes": 0,
                "mean_delta_pp": 0.0,
            }

    # Macro-level statistics
    base_arr = np.array(macro_baseline)
    aug_arr = np.array(macro_augmented)
    deltas = aug_arr - base_arr

    delta_mean = np.mean(deltas) * 100
    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS: Tier Impact Analysis")
    print("=" * 70)

    print(f"\nMacro F1:")
    print(f"  Baseline:  {np.mean(base_arr):.4f}")
    print(f"  Augmented: {np.mean(aug_arr):.4f}")
    print(f"  Delta:     {delta_mean:+.2f} pp (p={p_value:.4f})")

    print(f"\nTier Summary:")
    print(f"  {'Tier':<6} {'Classes':<8} {'Mean Δ (pp)':<12} {'Improved':<10}")
    print(f"  {'-'*40}")
    for tier in ["LOW", "MID", "HIGH"]:
        tr = tier_results[tier]
        print(f"  {tier:<6} {tr['n_classes']:<8} {tr.get('mean_delta_pp', 0):+.2f}         "
              f"{tr.get('improved', 0)}/{tr['n_classes']}")

    print(f"\nPer-Class Detail (sorted by tier and delta):")
    print(f"  {'Class':<6} {'Tier':<5} {'Base F1':<8} {'Aug F1':<8} {'Δ (pp)':<8} {'Synth':<6}")
    print(f"  {'-'*48}")

    sorted_classes = sorted(per_class_results.items(),
                           key=lambda x: (x[1]['tier'], -x[1]['delta_pp']))

    for l, r in sorted_classes:
        print(f"  {l:<6} {r['tier']:<5} {r['baseline_f1']:.4f}   {r['augmented_f1']:.4f}   "
              f"{r['delta_pp']:+.2f}     {r['n_synthetic']}")

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "tier_impact",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "tier_thresholds": TIER_THRESHOLDS,
        "macro_f1": {
            "baseline": float(np.mean(base_arr)),
            "augmented": float(np.mean(aug_arr)),
            "delta_pp": float(delta_mean),
            "p_value": float(p_value),
        },
        "tier_results": tier_results,
        "per_class_results": per_class_results,
        "n_synthetic": len(X_synth),
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp06_tier.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tier Impact Analysis (PyTorch MLP)}",
        r"\label{tab:pytorch_tier}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Tier & Classes & Mean $\Delta$ (pp) & Std & Improved \\",
        r"\midrule",
    ]

    for tier in ["LOW", "MID", "HIGH"]:
        tr = tier_results[tier]
        if tr['n_classes'] > 0:
            latex_lines.append(
                f"{tier} & {tr['n_classes']} & {tr['mean_delta_pp']:+.2f} & "
                f"{tr.get('std_delta_pp', 0):.2f} & {tr.get('improved', 0)}/{tr['n_classes']} \\\\"
            )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_tier.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\nResults saved to {output_dir / 'exp06_tier.json'}")
    print(f"LaTeX saved to {latex_path}")


if __name__ == "__main__":
    main()
