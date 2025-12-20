#!/usr/bin/env python3
"""
Detailed experiment for high_quality_few config with per-class F1 tracking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from base_config import RESULTS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def train_pytorch_mlp(X_train, y_train, X_test, num_classes,
                      epochs=100, batch_size=64, lr=0.001, early_stop_patience=10):
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], num_classes, dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        outputs = model(X_te)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


def main():
    print("="*70)
    print("High Quality Few - Detailed Per-Class Analysis")
    print("="*70)

    # Config
    params = {
        "n_shot": 60,
        "temperature": 0.2,
        "max_clusters": 3,
        "prompts_per_cluster": 2,
        "samples_per_prompt": 3,
    }

    # Load data
    texts, labels = load_data()
    texts = np.array(texts)
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[l] for l in labels])
    num_classes = len(unique_labels)

    print(f"\nClasses: {unique_labels}")
    print(f"Samples: {len(texts)}")

    # Generate synthetics
    print("\nGenerating synthetics...")
    generator = SyntheticGenerator(cache, params)

    all_synth_emb = []
    all_synth_labels = []
    synth_per_class = {}

    for label in unique_labels:
        class_mask = np.array(labels) == label
        class_texts = texts[class_mask]
        class_emb = embeddings[class_mask]

        try:
            synth_texts, _ = generator.generate_for_class(
                class_texts, class_emb, label
            )
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
        y_synth = np.array([label_to_idx[l] for l in all_synth_labels])
    else:
        X_synth = np.array([]).reshape(0, embeddings.shape[1])
        y_synth = np.array([])

    print(f"\nTotal synthetic: {len(X_synth)}")

    # K-Fold with per-class tracking
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
        base_per_class = f1_score(y_test, y_pred_base, average=None,
                                   labels=range(num_classes), zero_division=0)
        macro_baseline.append(base_macro)
        for i, l in enumerate(unique_labels):
            per_class_baseline[l].append(base_per_class[i])

        # Augmented
        if len(X_synth) > 0:
            X_train_aug = np.vstack([X_train, X_synth])
            y_train_aug = np.concatenate([y_train, y_synth])

            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_te_aug_scaled = scaler_aug.transform(X_test)

            y_pred_aug = train_pytorch_mlp(X_tr_aug_scaled, y_train_aug, X_te_aug_scaled, num_classes)
        else:
            y_pred_aug = y_pred_base

        aug_macro = f1_score(y_test, y_pred_aug, average="macro")
        aug_per_class = f1_score(y_test, y_pred_aug, average=None,
                                  labels=range(num_classes), zero_division=0)
        macro_augmented.append(aug_macro)
        for i, l in enumerate(unique_labels):
            per_class_augmented[l].append(aug_per_class[i])

        if (fold_idx + 1) % 5 == 0:
            print(f"  Fold {fold_idx + 1}/15: base={base_macro:.4f}, aug={aug_macro:.4f}, "
                  f"Δ={aug_macro - base_macro:+.4f}")

    # Compute statistics
    base_arr = np.array(macro_baseline)
    aug_arr = np.array(macro_augmented)
    deltas = aug_arr - base_arr

    base_mean = np.mean(base_arr)
    aug_mean = np.mean(aug_arr)
    delta_mean = np.mean(deltas)
    delta_pp = delta_mean * 100  # percentage points

    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    # Per-class results
    per_class_results = {}
    for l in unique_labels:
        base_class = np.mean(per_class_baseline[l])
        aug_class = np.mean(per_class_augmented[l])
        delta_class = aug_class - base_class
        delta_class_pp = delta_class * 100

        per_class_results[l] = {
            "baseline_f1": base_class,
            "augmented_f1": aug_class,
            "delta_pp": delta_class_pp,
            "n_synthetic": synth_per_class.get(l, 0)
        }

    # Print results
    print("\n" + "="*70)
    print("RESULTS: high_quality_few with PyTorch MLP")
    print("="*70)

    print(f"\nMacro F1:")
    print(f"  Baseline:  {base_mean:.4f}")
    print(f"  Augmented: {aug_mean:.4f}")
    print(f"  Delta:     {delta_mean:+.4f} ({delta_pp:+.2f} pp)")
    print(f"  p-value:   {p_value:.6f} {'*' if p_value < 0.05 else ''}")

    print(f"\nPer-Class F1:")
    print(f"  {'Class':<6} {'Base F1':<10} {'Aug F1':<10} {'Δ (pp)':<10} {'Synth':<6}")
    print(f"  {'-'*42}")

    # Sort by delta
    sorted_classes = sorted(per_class_results.items(),
                           key=lambda x: x[1]['delta_pp'], reverse=True)

    for l, r in sorted_classes:
        marker = "**" if l in ["ESFJ", "ESFP", "ESTJ"] else ""
        print(f"  {l:<6} {r['baseline_f1']:.4f}     {r['augmented_f1']:.4f}     "
              f"{r['delta_pp']:+.2f}      {r['n_synthetic']:<6} {marker}")

    # Problem classes summary
    print(f"\nProblem Classes (ESFJ, ESFP, ESTJ):")
    problem_deltas = []
    for l in ["ESFJ", "ESFP", "ESTJ"]:
        r = per_class_results[l]
        problem_deltas.append(r['delta_pp'])
        print(f"  {l}: {r['baseline_f1']:.4f} -> {r['augmented_f1']:.4f} ({r['delta_pp']:+.2f} pp)")
    print(f"  Average problem class Δ: {np.mean(problem_deltas):+.2f} pp")

    # Save results
    results = {
        "config": "high_quality_few",
        "config_params": params,
        "n_synthetic_total": len(X_synth),
        "macro_f1": {
            "baseline": base_mean,
            "augmented": aug_mean,
            "delta_pp": delta_pp,
            "p_value": p_value,
            "significant": p_value < 0.05
        },
        "per_class_f1": per_class_results,
        "problem_class_avg_delta_pp": np.mean(problem_deltas),
        "timestamp": datetime.now().isoformat()
    }

    output_dir = RESULTS_DIR / "pytorch_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "high_quality_few_detailed.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'high_quality_few_detailed.json'}")


if __name__ == "__main__":
    main()
