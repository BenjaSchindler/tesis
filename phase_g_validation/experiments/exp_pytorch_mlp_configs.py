#!/usr/bin/env python3
"""
Experiment: Test best configs with PyTorch MLP classifier.

Compares LogisticRegression vs PyTorch MLP across key configurations
to understand why sklearn MLP shows +20% improvement but PyTorch shows -1.6%.

Key configs to test:
1. W5 (best overall with LogReg)
2. W5b_temp03_n60 (best temperature)
3. RARE_massive_oversample (rare class focus)
4. baseline (no augmentation)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from config_definitions import get_config_params
from base_config import RESULTS_DIR

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class MLP(nn.Module):
    """MLP 512-256-128 matching sklearn architecture."""

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


def train_pytorch_mlp(X_train, y_train, X_test, y_test, num_classes,
                      epochs=100, batch_size=64, lr=0.001, early_stop_patience=10,
                      dropout=0.2):
    """Train MLP on GPU with early stopping."""

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(X_train.shape[1], num_classes, dropout=dropout).to(DEVICE)
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

        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    # Load best model
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_te)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()

    return y_pred


def run_kfold_comparison(X_orig, y_orig, X_synth, y_synth, unique_labels,
                         n_splits=5, n_repeats=3, seed=42, dropout=0.2):
    """Run K-fold CV comparing LogReg vs PyTorch MLP."""

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    num_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in y_orig])

    # Encode synthetic labels too
    if len(y_synth) > 0:
        y_synth_encoded = np.array([label_to_idx[l] for l in y_synth])
    else:
        y_synth_encoded = np.array([])

    results = {
        "logreg": {"baseline": [], "augmented": [], "deltas": []},
        "pytorch": {"baseline": [], "augmented": [], "deltas": []}
    }

    total_folds = n_splits * n_repeats
    print(f"\n  Running {total_folds}-fold CV comparing LogReg vs PyTorch MLP...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_encoded)):
        X_train, y_train = X_orig[train_idx], y_encoded[train_idx]
        X_test, y_test = X_orig[test_idx], y_encoded[test_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # === LOGREG BASELINE ===
        clf_lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf_lr.fit(X_tr_scaled, y_train)
        y_pred_lr = clf_lr.predict(X_te_scaled)
        lr_base_f1 = f1_score(y_test, y_pred_lr, average="macro")
        results["logreg"]["baseline"].append(lr_base_f1)

        # === PYTORCH BASELINE ===
        y_pred_pt = train_pytorch_mlp(X_tr_scaled, y_train, X_te_scaled, y_test,
                                       num_classes, dropout=dropout)
        pt_base_f1 = f1_score(y_test, y_pred_pt, average="macro")
        results["pytorch"]["baseline"].append(pt_base_f1)

        # === AUGMENTED ===
        if len(X_synth) > 0:
            X_train_aug = np.vstack([X_train, X_synth])
            y_train_aug = np.concatenate([y_train, y_synth_encoded])

            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_te_aug_scaled = scaler_aug.transform(X_test)

            # LogReg augmented
            clf_lr_aug = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf_lr_aug.fit(X_tr_aug_scaled, y_train_aug)
            y_pred_lr_aug = clf_lr_aug.predict(X_te_aug_scaled)
            lr_aug_f1 = f1_score(y_test, y_pred_lr_aug, average="macro")

            # PyTorch augmented
            y_pred_pt_aug = train_pytorch_mlp(X_tr_aug_scaled, y_train_aug, X_te_aug_scaled,
                                               y_test, num_classes, dropout=dropout)
            pt_aug_f1 = f1_score(y_test, y_pred_pt_aug, average="macro")
        else:
            lr_aug_f1 = lr_base_f1
            pt_aug_f1 = pt_base_f1

        results["logreg"]["augmented"].append(lr_aug_f1)
        results["logreg"]["deltas"].append(lr_aug_f1 - lr_base_f1)
        results["pytorch"]["augmented"].append(pt_aug_f1)
        results["pytorch"]["deltas"].append(pt_aug_f1 - pt_base_f1)

        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{total_folds}: "
                  f"LR Δ={lr_aug_f1 - lr_base_f1:+.4f}, "
                  f"PT Δ={pt_aug_f1 - pt_base_f1:+.4f}")

    # Compute stats for each classifier
    output = {"n_folds": total_folds, "n_synthetic": len(X_synth), "dropout": dropout}

    for clf_name in ["logreg", "pytorch"]:
        data = results[clf_name]
        base_arr = np.array(data["baseline"])
        aug_arr = np.array(data["augmented"])
        deltas = np.array(data["deltas"])

        base_mean, base_std = np.mean(base_arr), np.std(base_arr, ddof=1)
        aug_mean, aug_std = np.mean(aug_arr), np.std(aug_arr, ddof=1)
        delta_mean, delta_std = np.mean(deltas), np.std(deltas, ddof=1)
        delta_pct = (delta_mean / base_mean) * 100 if base_mean > 0 else 0

        n = len(deltas)
        ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))
        t_stat, p_value = stats.ttest_1samp(deltas, 0)
        win_rate = sum(1 for d in deltas if d > 0) / n

        output[clf_name] = {
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "augmented_mean": aug_mean,
            "augmented_std": aug_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "delta_pct": delta_pct,
            "ci_95": [ci_95[0], ci_95[1]],
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "win_rate": win_rate
        }

    return output


def generate_synthetics(cache, texts, labels, embeddings, params):
    """Generate synthetics with given params."""
    generator = SyntheticGenerator(cache, params)
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    all_synth_emb = []
    all_synth_labels = []
    target_classes = params.get("force_generation_classes", unique_labels)

    for label in unique_labels:
        if target_classes and label not in target_classes:
            continue

        class_mask = np.array(labels) == label
        class_texts = [t for t, m in zip(texts, class_mask) if m]
        class_emb = embeddings[class_mask]

        try:
            synth_texts, _ = generator.generate_for_class(
                np.array(class_texts), class_emb, label
            )
            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label] * len(synth_emb))
                print(f"    {label}: +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"    {label}: Error - {e}")

    if all_synth_emb:
        return np.vstack(all_synth_emb), np.array(all_synth_labels)
    return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


def run_experiment(name, cache, texts, labels, embeddings, unique_labels, params, dropout=0.2):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"Dropout: {dropout}")
    print(f"{'='*70}")

    print("\nGenerating synthetics...")
    X_synth, y_synth = generate_synthetics(cache, texts, labels, embeddings, params)
    print(f"Total synthetic: {len(X_synth)}")

    results = run_kfold_comparison(embeddings, labels, X_synth, y_synth,
                                    unique_labels, dropout=dropout)
    results["config_name"] = name
    results["config_params"] = {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                                 for k, v in params.items()}
    results["timestamp"] = datetime.now().isoformat()

    # Print comparison
    print(f"\n  {'Classifier':<12} {'Baseline':<10} {'Augmented':<10} {'Delta %':<10} {'p-value':<10}")
    print(f"  {'-'*52}")
    for clf in ["logreg", "pytorch"]:
        r = results[clf]
        sig = "*" if r["significant"] else ""
        print(f"  {clf:<12} {r['baseline_mean']:.4f}     {r['augmented_mean']:.4f}     "
              f"{r['delta_pct']:+.2f}%    {r['p_value']:.4f}{sig}")

    return results


CONFIGS_TO_TEST = {
    "W5_optimal": {
        "description": "Best overall config from Phase G",
        "params": {
            "n_shot": 60,
            "temperature": 0.7,
            "max_clusters": 12,
            "prompts_per_cluster": 9,
            "samples_per_prompt": 5,
        }
    },
    "W5b_temp03_n60": {
        "description": "Best temperature (τ=0.3) with n_shot=60",
        "params": {
            "n_shot": 60,
            "temperature": 0.3,
            "max_clusters": 12,
            "prompts_per_cluster": 9,
            "samples_per_prompt": 5,
        }
    },
    "conservative_low_synth": {
        "description": "Fewer synthetics to reduce noise",
        "params": {
            "n_shot": 30,
            "temperature": 0.5,
            "max_clusters": 6,
            "prompts_per_cluster": 3,
            "samples_per_prompt": 3,
        }
    },
    "high_quality_few": {
        "description": "High quality, very few synthetics",
        "params": {
            "n_shot": 60,
            "temperature": 0.2,
            "max_clusters": 3,
            "prompts_per_cluster": 2,
            "samples_per_prompt": 3,
        }
    }
}


def main():
    print("="*70)
    print("PyTorch MLP vs LogReg Comparison")
    print("="*70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    results_dir = RESULTS_DIR / "pytorch_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Test each config
    for config_name, config_info in CONFIGS_TO_TEST.items():
        params = config_info["params"]

        r = run_experiment(config_name, cache, texts, labels, embeddings,
                           unique_labels, params, dropout=0.2)
        all_results[config_name] = r

        # Save individual result
        with open(results_dir / f"{config_name}_comparison.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)

    # Also test with NO dropout to see if that matches sklearn better
    print("\n" + "="*70)
    print("Testing PyTorch WITHOUT Dropout (to match sklearn)")
    print("="*70)

    params_no_dropout = CONFIGS_TO_TEST["W5_optimal"]["params"].copy()
    r_no_dropout = run_experiment("W5_optimal_no_dropout", cache, texts, labels,
                                   embeddings, unique_labels, params_no_dropout, dropout=0.0)
    all_results["W5_optimal_no_dropout"] = r_no_dropout

    with open(results_dir / "W5_optimal_no_dropout_comparison.json", 'w') as f:
        json.dump(r_no_dropout, f, indent=2, default=str)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: LogReg vs PyTorch MLP")
    print("="*70)
    print(f"\n{'Config':<25} {'LR Δ%':<10} {'PT Δ%':<10} {'LR sig':<8} {'PT sig':<8}")
    print("-"*61)
    for name, r in all_results.items():
        lr = r["logreg"]
        pt = r["pytorch"]
        lr_sig = "YES" if lr["significant"] else "no"
        pt_sig = "YES" if pt["significant"] else "no"
        print(f"{name:<25} {lr['delta_pct']:+.2f}%     {pt['delta_pct']:+.2f}%     {lr_sig:<8} {pt_sig:<8}")

    # Save full summary
    with open(results_dir / "SUMMARY_pytorch_vs_logreg.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}")
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
