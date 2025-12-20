#!/usr/bin/env python3
"""
PyTorch MLP Classifier for Phase G Validation

Architecture: 512-256-128 with dropout=0.2
Matching sklearn MLPClassifier but with GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """MLP 512-256-128 matching sklearn architecture."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
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


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    dropout: float = 0.2,
    early_stop_patience: int = 10,
    device: torch.device = DEVICE
) -> np.ndarray:
    """Train MLP on GPU with early stopping."""

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(X_train.shape[1], num_classes, dropout=dropout).to(device)
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
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_te)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


@dataclass
class PyTorchKFoldResult:
    """Results from K-Fold evaluation with PyTorch MLP."""
    config_name: str
    config_value: Any
    n_folds: int
    baseline_mean: float
    baseline_std: float
    augmented_mean: float
    augmented_std: float
    delta_mean: float
    delta_std: float
    delta_pp: float  # percentage points
    ci_95_lower: float
    ci_95_upper: float
    t_statistic: float
    p_value: float
    significant: bool
    win_rate: float
    n_synthetic: int
    per_class_f1: Optional[Dict[str, Dict[str, float]]] = None
    tier_deltas: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


def run_pytorch_kfold(
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: Optional[np.ndarray],
    y_synthetic: Optional[np.ndarray],
    unique_labels: List[str],
    config_name: str,
    config_value: Any,
    n_splits: int = 5,
    n_repeats: int = 3,
    dropout: float = 0.2,
    verbose: bool = True
) -> PyTorchKFoldResult:
    """Run K-Fold CV with PyTorch MLP classifier."""

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # Encode labels
    y_encoded = np.array([label_to_idx[l] for l in y_original])

    if X_synthetic is not None and len(X_synthetic) > 0:
        y_synth_encoded = np.array([label_to_idx[l] for l in y_synthetic])
    else:
        y_synth_encoded = np.array([])

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_folds = n_splits * n_repeats

    baseline_f1s = []
    augmented_f1s = []
    per_class_baseline = {l: [] for l in unique_labels}
    per_class_augmented = {l: [] for l in unique_labels}

    if verbose:
        print(f"  Running {total_folds}-fold CV with PyTorch MLP...", flush=True)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_encoded)):
        X_train, y_train = X_original[train_idx], y_encoded[train_idx]
        X_test, y_test = X_original[test_idx], y_encoded[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # Baseline
        y_pred_base = train_pytorch_mlp(
            X_tr_scaled, y_train, X_te_scaled, num_classes, dropout=dropout
        )
        base_macro = f1_score(y_test, y_pred_base, average="macro")
        base_per_class = f1_score(
            y_test, y_pred_base, average=None,
            labels=range(num_classes), zero_division=0
        )
        baseline_f1s.append(base_macro)
        for i, l in enumerate(unique_labels):
            per_class_baseline[l].append(base_per_class[i])

        # Augmented
        if X_synthetic is not None and len(X_synthetic) > 0:
            X_train_aug = np.vstack([X_train, X_synthetic])
            y_train_aug = np.concatenate([y_train, y_synth_encoded])

            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_te_aug_scaled = scaler_aug.transform(X_test)

            y_pred_aug = train_pytorch_mlp(
                X_tr_aug_scaled, y_train_aug, X_te_aug_scaled, num_classes, dropout=dropout
            )
        else:
            y_pred_aug = y_pred_base

        aug_macro = f1_score(y_test, y_pred_aug, average="macro")
        aug_per_class = f1_score(
            y_test, y_pred_aug, average=None,
            labels=range(num_classes), zero_division=0
        )
        augmented_f1s.append(aug_macro)
        for i, l in enumerate(unique_labels):
            per_class_augmented[l].append(aug_per_class[i])

        if verbose and (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{total_folds}: "
                  f"base={base_macro:.4f}, aug={aug_macro:.4f}, "
                  f"Δ={aug_macro - base_macro:+.4f}", flush=True)

    # Compute statistics
    base_arr = np.array(baseline_f1s)
    aug_arr = np.array(augmented_f1s)
    deltas = aug_arr - base_arr

    base_mean = np.mean(base_arr)
    base_std = np.std(base_arr, ddof=1)
    aug_mean = np.mean(aug_arr)
    aug_std = np.std(aug_arr, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    delta_pp = delta_mean * 100  # percentage points

    # CI and t-test
    n = len(deltas)
    se = delta_std / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    # Win rate
    win_rate = np.mean(deltas > 0)

    # Per-class results
    per_class_f1 = {}
    for l in unique_labels:
        base_class = np.mean(per_class_baseline[l])
        aug_class = np.mean(per_class_augmented[l])
        delta_class = aug_class - base_class
        per_class_f1[l] = {
            "baseline_f1": float(base_class),
            "augmented_f1": float(aug_class),
            "delta_pp": float(delta_class * 100)
        }

    # Tier deltas (LOW: F1<0.20, MID: 0.20-0.45, HIGH: >0.45)
    tier_deltas = {"LOW": [], "MID": [], "HIGH": []}
    for l, metrics in per_class_f1.items():
        base_f1 = metrics["baseline_f1"]
        delta = metrics["delta_pp"]
        if base_f1 < 0.20:
            tier_deltas["LOW"].append(delta)
        elif base_f1 < 0.45:
            tier_deltas["MID"].append(delta)
        else:
            tier_deltas["HIGH"].append(delta)

    tier_avg = {
        tier: float(np.mean(vals)) if vals else 0.0
        for tier, vals in tier_deltas.items()
    }

    n_synthetic = len(X_synthetic) if X_synthetic is not None else 0

    return PyTorchKFoldResult(
        config_name=config_name,
        config_value=config_value,
        n_folds=total_folds,
        baseline_mean=float(base_mean),
        baseline_std=float(base_std),
        augmented_mean=float(aug_mean),
        augmented_std=float(aug_std),
        delta_mean=float(delta_mean),
        delta_std=float(delta_std),
        delta_pp=float(delta_pp),
        ci_95_lower=float(ci_95[0]),
        ci_95_upper=float(ci_95[1]),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        significant=bool(p_value < 0.05),
        win_rate=float(win_rate),
        n_synthetic=n_synthetic,
        per_class_f1=per_class_f1,
        tier_deltas=tier_avg,
        timestamp=datetime.now().isoformat()
    )


def print_result_summary(result: PyTorchKFoldResult):
    """Print formatted result summary."""
    sig = "*" if result.significant else ""
    print(f"\n  {result.config_name} = {result.config_value}:")
    print(f"    Baseline:  {result.baseline_mean:.4f} ± {result.baseline_std:.4f}")
    print(f"    Augmented: {result.augmented_mean:.4f} ± {result.augmented_std:.4f}")
    print(f"    Delta:     {result.delta_pp:+.2f} pp (p={result.p_value:.4f}){sig}")
    print(f"    Win rate:  {result.win_rate*100:.1f}%")
    print(f"    Synthetics: {result.n_synthetic}")
    if result.tier_deltas:
        print(f"    Tier Δ: LOW={result.tier_deltas.get('LOW', 0):+.2f}, "
              f"MID={result.tier_deltas.get('MID', 0):+.2f}, "
              f"HIGH={result.tier_deltas.get('HIGH', 0):+.2f}")
