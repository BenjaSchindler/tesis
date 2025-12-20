#!/usr/bin/env python3
"""
Experiment 07b: Weight by Tier validation with PyTorch MLP

Tests:
- uniform: LOW=1.0, MID=1.0, HIGH=1.0
- tier_boost_low: LOW=1.5, MID=1.0, HIGH=0.5
- tier_boost_extreme: LOW=2.0, MID=0.8, HIGH=0.3
- only_low: LOW=1.0, MID=0.0, HIGH=0.0
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

# Weight by tier configurations
TIER_CONFIGS = {
    "uniform": {"LOW": 1.0, "MID": 1.0, "HIGH": 1.0},
    "tier_boost_low": {"LOW": 1.5, "MID": 1.0, "HIGH": 0.5},
    "tier_boost_extreme": {"LOW": 2.0, "MID": 0.8, "HIGH": 0.3},
    "only_low": {"LOW": 1.0, "MID": 0.0, "HIGH": 0.0},
}

# Tier thresholds (baseline F1)
TIER_THRESHOLDS = {
    "LOW": (0, 0.20),
    "MID": (0.20, 0.45),
    "HIGH": (0.45, 1.0),
}

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def compute_class_tiers(embeddings, labels, unique_labels):
    """Compute tier for each class based on baseline F1."""
    # Quick baseline evaluation
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in labels])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    clf = LogisticRegression(max_iter=1000)
    y_pred = cross_val_predict(clf, embeddings, y_encoded, cv=3)

    per_class_f1 = f1_score(y_encoded, y_pred, average=None, labels=range(len(unique_labels)), zero_division=0)

    class_tiers = {}
    for i, label in enumerate(unique_labels):
        f1 = per_class_f1[i]
        if f1 < TIER_THRESHOLDS["LOW"][1]:
            class_tiers[label] = "LOW"
        elif f1 < TIER_THRESHOLDS["MID"][1]:
            class_tiers[label] = "MID"
        else:
            class_tiers[label] = "HIGH"

    return class_tiers


def main():
    print("=" * 70)
    print("Experiment 07b: Weight by Tier Validation (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    # Compute class tiers
    print("\nComputing class tiers...")
    class_tiers = compute_class_tiers(embeddings, labels, unique_labels)
    tier_counts = {"LOW": 0, "MID": 0, "HIGH": 0}
    for t in class_tiers.values():
        tier_counts[t] += 1
    print(f"Tiers: LOW={tier_counts['LOW']}, MID={tier_counts['MID']}, HIGH={tier_counts['HIGH']}")

    # Generate synthetics once
    print("\nGenerating synthetics...")
    generator = SyntheticGenerator(cache, BASE_CONFIG)

    synth_by_class = {}

    for label in unique_labels:
        class_mask = np.array(labels) == label
        class_texts = texts[class_mask]
        class_emb = embeddings[class_mask]

        try:
            synth_texts, _ = generator.generate_for_class(class_texts, class_emb, label)
            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                synth_by_class[label] = {"texts": synth_texts, "emb": synth_emb}
                print(f"  {label} ({class_tiers[label]}): +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"  {label}: Error - {e}")

    results = []

    for config_name, tier_weights in TIER_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Testing tier config: {config_name}")
        print(f"Weights: {tier_weights}")
        print("=" * 70)

        # Build synthetic dataset with tier weights
        all_synth_emb = []
        all_synth_labels = []

        for label in unique_labels:
            if label not in synth_by_class:
                continue

            tier = class_tiers[label]
            weight = tier_weights[tier]

            if weight <= 0:
                continue

            synth_emb = synth_by_class[label]["emb"]

            if weight > 1:
                # Oversample
                n_copies = int(weight)
                for _ in range(n_copies):
                    all_synth_emb.append(synth_emb)
                    all_synth_labels.extend([label] * len(synth_emb))
            elif weight < 1:
                # Subsample
                n_keep = max(1, int(len(synth_emb) * weight))
                idx = np.random.choice(len(synth_emb), n_keep, replace=False)
                all_synth_emb.append(synth_emb[idx])
                all_synth_labels.extend([label] * n_keep)
            else:
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label] * len(synth_emb))

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        print(f"Total synthetic (weighted): {len(X_synth)}")

        # Run K-fold
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_encoded = np.array([label_to_idx[l] for l in labels])
        y_synth_encoded = np.array([label_to_idx[l] for l in y_synth]) if len(y_synth) > 0 else np.array([])

        kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        num_classes = len(unique_labels)

        baseline_f1s = []
        augmented_f1s = []
        tier_deltas = {"LOW": [], "MID": [], "HIGH": []}

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
            baseline_f1s.append(base_macro)

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
            augmented_f1s.append(aug_macro)

            # Track tier deltas
            for i, label in enumerate(unique_labels):
                tier = class_tiers[label]
                delta = aug_per_class[i] - base_per_class[i]
                tier_deltas[tier].append(delta)

            if (fold_idx + 1) % 5 == 0:
                print(f"    Fold {fold_idx + 1}/15: base={base_macro:.4f}, aug={aug_macro:.4f}")

        # Compute statistics
        base_arr = np.array(baseline_f1s)
        aug_arr = np.array(augmented_f1s)
        deltas = aug_arr - base_arr

        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas, ddof=1)
        delta_pp = delta_mean * 100

        t_stat, p_value = stats.ttest_1samp(deltas, 0)

        tier_avg_deltas = {
            tier: float(np.mean(vals) * 100) if vals else 0.0
            for tier, vals in tier_deltas.items()
        }

        result = {
            "config_name": config_name,
            "tier_weights": tier_weights,
            "baseline_mean": float(np.mean(base_arr)),
            "augmented_mean": float(np.mean(aug_arr)),
            "delta_pp": float(delta_pp),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "n_synthetic": len(X_synth),
            "tier_deltas": tier_avg_deltas,
        }
        results.append(result)

        sig = "*" if result["significant"] else ""
        print(f"\n  Result: Δ = {delta_pp:+.2f} pp (p={p_value:.4f}){sig}")
        print(f"  Tier Δ: LOW={tier_avg_deltas['LOW']:+.2f}, MID={tier_avg_deltas['MID']:+.2f}, HIGH={tier_avg_deltas['HIGH']:+.2f}")

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "weight_by_tier",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "tier_configs": TIER_CONFIGS,
        "class_tiers": class_tiers,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp07b_weight_tier.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Weight by Tier Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_weight_tier}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Config & $\Delta$ (pp) & LOW $\Delta$ & MID $\Delta$ & HIGH $\Delta$ \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r["significant"] else ""
        td = r["tier_deltas"]
        latex_lines.append(
            f"{r['config_name']} & {r['delta_pp']:+.2f}{sig} & {td['LOW']:+.2f} & {td['MID']:+.2f} & {td['HIGH']:+.2f} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_weight_tier.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Weight by Tier Experiment")
    print("=" * 70)
    print(f"{'Config':<20} {'Δ (pp)':<10} {'LOW':<8} {'MID':<8} {'HIGH':<8}")
    print("-" * 56)
    for r in results:
        td = r["tier_deltas"]
        print(f"{r['config_name']:<20} {r['delta_pp']:+.2f}      {td['LOW']:+.2f}    {td['MID']:+.2f}    {td['HIGH']:+.2f}")

    print(f"\nResults saved to {output_dir / 'exp07b_weight_tier.json'}")


if __name__ == "__main__":
    main()
