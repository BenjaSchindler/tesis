#!/usr/bin/env python3
"""
RARE_MLP with PyTorch GPU acceleration.

Compares:
1. Original RARE_massive_oversample (default n_shot)
2. RARE_massive_oversample + n_shot=60

Uses RTX 3090 for fast training.
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

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class MLP(nn.Module):
    """MLP 512-256-128 matching sklearn architecture."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(X_train, y_train, X_test, y_test, num_classes,
              epochs=100, batch_size=64, lr=0.001, early_stop_patience=10):
    """Train MLP on GPU with early stopping."""

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = torch.LongTensor(y_test).to(DEVICE)

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(X_train.shape[1], num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping
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

        # Early stopping check
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_te)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()

    return y_pred


RARE_CLASSES = ["ESFJ", "ESFP", "ESTJ"]


def get_params(n_shot=None):
    """Get RARE_massive_oversample params with optional n_shot."""
    params = {
        "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],
        "min_synthetic_per_class": 100,
        "rare_class_boost": 5.0,
        "samples_per_prompt": 15,
        "prompts_per_cluster": 15,
        "max_clusters": 3,
        "disable_quality_gate": True,
        "similarity_threshold": 0.99,
    }
    if n_shot is not None:
        params["n_shot"] = n_shot
    return params


def generate_synthetics(cache, texts, labels, embeddings, params):
    """Generate synthetics for rare classes."""
    generator = SyntheticGenerator(cache, params)
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    all_synth_emb = []
    all_synth_labels = []
    target_classes = params.get("force_generation_classes", RARE_CLASSES)

    for label in unique_labels:
        if label not in target_classes:
            continue

        class_mask = np.array(labels) == label
        class_texts = [t for t, m in zip(texts, class_mask) if m]
        class_emb = embeddings[class_mask]
        n_original = len(class_texts)

        try:
            synth_texts, _ = generator.generate_for_class(
                np.array(class_texts), class_emb, label
            )
            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label_to_idx[label]] * len(synth_emb))
                print(f"    {label}: {n_original} -> +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"    {label}: Error - {e}")

    if all_synth_emb:
        return np.vstack(all_synth_emb), np.array(all_synth_labels)
    return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


def run_kfold(X_orig, y_orig, X_synth, y_synth, unique_labels,
              n_splits=5, n_repeats=3, seed=42):
    """Run K-fold CV with PyTorch MLP on GPU."""

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    num_classes = len(unique_labels)

    baseline_f1s, augmented_f1s, deltas = [], [], []
    per_class_base = {l: [] for l in unique_labels}
    per_class_aug = {l: [] for l in unique_labels}

    total_folds = n_splits * n_repeats
    print(f"\n  Running {total_folds}-fold CV with PyTorch MLP on {DEVICE}...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_orig)):
        X_train, y_train = X_orig[train_idx], y_orig[train_idx]
        X_test, y_test = X_orig[test_idx], y_orig[test_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # === BASELINE ===
        y_pred_base = train_mlp(X_tr_scaled, y_train, X_te_scaled, y_test, num_classes)
        base_f1 = f1_score(y_test, y_pred_base, average="macro")
        baseline_f1s.append(base_f1)

        base_pc = f1_score(y_test, y_pred_base, average=None, labels=range(num_classes))
        for i, l in enumerate(unique_labels):
            per_class_base[l].append(base_pc[i])

        # === AUGMENTED ===
        if len(X_synth) > 0:
            X_train_aug = np.vstack([X_train, X_synth])
            y_train_aug = np.concatenate([y_train, y_synth])
        else:
            X_train_aug, y_train_aug = X_train, y_train

        scaler_aug = StandardScaler()
        X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
        X_te_aug_scaled = scaler_aug.transform(X_test)

        y_pred_aug = train_mlp(X_tr_aug_scaled, y_train_aug, X_te_aug_scaled, y_test, num_classes)
        aug_f1 = f1_score(y_test, y_pred_aug, average="macro")
        augmented_f1s.append(aug_f1)

        aug_pc = f1_score(y_test, y_pred_aug, average=None, labels=range(num_classes))
        for i, l in enumerate(unique_labels):
            per_class_aug[l].append(aug_pc[i])

        delta = aug_f1 - base_f1
        deltas.append(delta)

        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{total_folds}: base={base_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}")

    # Statistics
    base_mean, base_std = np.mean(baseline_f1s), np.std(baseline_f1s, ddof=1)
    aug_mean, aug_std = np.mean(augmented_f1s), np.std(augmented_f1s, ddof=1)
    delta_mean, delta_std = np.mean(deltas), np.std(deltas, ddof=1)
    delta_pct = (delta_mean / base_mean) * 100 if base_mean > 0 else 0

    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    win_rate = sum(1 for d in deltas if d > 0) / n
    per_class_delta = {l: np.mean(per_class_aug[l]) - np.mean(per_class_base[l]) for l in unique_labels}

    return {
        "n_folds": total_folds,
        "baseline_mean": base_mean, "baseline_std": base_std,
        "augmented_mean": aug_mean, "augmented_std": aug_std,
        "delta_mean": delta_mean, "delta_std": delta_std,
        "delta_pct": delta_pct,
        "ci_95_lower": ci_95[0], "ci_95_upper": ci_95[1],
        "t_statistic": t_stat, "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": win_rate,
        "per_class_delta": per_class_delta,
        "n_synthetic": len(X_synth),
    }


def run_experiment(name, cache, texts, labels, embeddings, unique_labels, y_encoded, n_shot=None):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"n_shot: {n_shot if n_shot else 'DEFAULT'}")
    print(f"{'='*70}")

    params = get_params(n_shot=n_shot)

    print("\nGenerating synthetics...")
    X_synth, y_synth = generate_synthetics(cache, texts, labels, embeddings, params)
    print(f"Total synthetic: {len(X_synth)}")

    results = run_kfold(embeddings, y_encoded, X_synth, y_synth, unique_labels)
    results["config_name"] = name
    results["n_shot"] = n_shot if n_shot else "default"
    results["timestamp"] = datetime.now().isoformat()

    print(f"\n  Results:")
    print(f"    Baseline:  {results['baseline_mean']:.4f}")
    print(f"    Augmented: {results['augmented_mean']:.4f}")
    print(f"    Delta:     {results['delta_pct']:+.2f}% (p={results['p_value']:.6f})")

    print(f"\n  Rare class deltas:")
    for cls in RARE_CLASSES:
        d = results['per_class_delta'].get(cls, 0)
        print(f"    {cls}: {d*100:+.2f}%")

    return results


def main():
    print("="*70)
    print("RARE_MLP PyTorch GPU Comparison")
    print("="*70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in labels])

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    # Rare class counts
    labels_list = list(labels)
    for cls in RARE_CLASSES:
        print(f"  {cls}: {labels_list.count(cls)} samples")

    results_dir = RESULTS_DIR / "rare_mlp_pytorch"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Original (default n_shot)
    r1 = run_experiment("RARE_MLP_original", cache, texts, labels, embeddings,
                        unique_labels, y_encoded, n_shot=None)
    all_results["original"] = r1
    with open(results_dir / "RARE_MLP_original_pytorch.json", 'w') as f:
        json.dump(r1, f, indent=2, default=str)

    # 2. With n_shot=60
    r2 = run_experiment("RARE_MLP_nshot60", cache, texts, labels, embeddings,
                        unique_labels, y_encoded, n_shot=60)
    all_results["nshot60"] = r2
    with open(results_dir / "RARE_MLP_nshot60_pytorch.json", 'w') as f:
        json.dump(r2, f, indent=2, default=str)

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Config':<25} {'Baseline':<10} {'Augmented':<10} {'Delta %':<10} {'p-value':<10}")
    print("-"*65)
    for name, r in all_results.items():
        print(f"{name:<25} {r['baseline_mean']:.4f}     {r['augmented_mean']:.4f}     {r['delta_pct']:+.2f}%     {r['p_value']:.6f}")

    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
