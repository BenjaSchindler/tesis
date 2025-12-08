#!/usr/bin/env python3
"""
Phase I - Multi-Model Classifier Comparison (GPU Optimized)

Tests different ML and DL models to see if they can predict
the problematic MBTI classes (ESFJ, ESFP, ESTJ) that have F1=0
with Logistic Regression.

GPU-accelerated models:
- XGBoost with gpu_hist
- LightGBM with GPU
- PyTorch MLP on CUDA
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Check CUDA
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# XGBoost with GPU
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# LightGBM with GPU
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class PyTorchMLP(nn.Module):
    """Simple MLP for classification."""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier:
    """Sklearn-compatible wrapper for PyTorch MLP."""
    def __init__(self, hidden_dims=(256, 128), epochs=50, batch_size=256, lr=0.001, device='cuda'):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        input_dim = X.shape[1]
        num_classes = len(np.unique(y))

        self.model = PyTorchMLP(input_dim, self.hidden_dims, num_classes).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()


def get_models():
    """Return dict of models to test."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, n_jobs=-1),
        'LogisticRegression_balanced': LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
        'RandomForest_balanced': RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42),
    }

    # GPU-accelerated XGBoost
    if HAS_XGBOOST:
        models['XGBoost_GPU'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            tree_method='gpu_hist', gpu_id=0,
            random_state=42, verbosity=0
        )

    # GPU-accelerated LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM_GPU'] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            random_state=42, verbose=-1
        )
        models['LightGBM_GPU_balanced'] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            class_weight='balanced',
            random_state=42, verbose=-1
        )

    # PyTorch MLP on GPU
    if DEVICE == 'cuda':
        models['MLP_GPU_small'] = TorchMLPClassifier(hidden_dims=(256, 128), epochs=50, device=DEVICE)
        models['MLP_GPU_large'] = TorchMLPClassifier(hidden_dims=(512, 256, 128), epochs=100, device=DEVICE)
        models['MLP_GPU_deep'] = TorchMLPClassifier(hidden_dims=(768, 512, 256, 128), epochs=100, device=DEVICE)

    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, n_classes):
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=range(n_classes))

    return {
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1
    }


def run_comparison(X, y, class_names, n_splits=5, seed=42):
    """Run K-Fold comparison across all models."""

    n_classes = len(class_names)
    models = get_models()

    # Initialize results storage
    results = {name: {
        'macro_f1': [],
        'per_class_f1': {c: [] for c in class_names}
    } for name in models}

    # K-Fold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print(f"\nRunning {n_splits}-Fold CV with {len(models)} models...")
    print("=" * 70)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for name, model in models.items():
            try:
                # Clone model for each fold
                if hasattr(model, 'get_params'):
                    model_clone = model.__class__(**model.get_params())
                else:
                    model_clone = model.__class__(
                        hidden_dims=model.hidden_dims,
                        epochs=model.epochs,
                        batch_size=model.batch_size,
                        lr=model.lr,
                        device=model.device
                    )

                metrics = evaluate_model(model_clone, X_train, y_train, X_test, y_test, n_classes)

                results[name]['macro_f1'].append(metrics['macro_f1'])
                for i, c in enumerate(class_names):
                    results[name]['per_class_f1'][c].append(metrics['per_class_f1'][i])

                print(f"  {name}: macro_f1={metrics['macro_f1']:.4f}")

            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                results[name]['macro_f1'].append(0)
                for c in class_names:
                    results[name]['per_class_f1'][c].append(0)

    return results


def print_results(results, class_names, problematic_classes):
    """Print formatted results."""

    print("\n" + "=" * 80)
    print("PHASE I - MULTI-MODEL COMPARISON RESULTS (GPU)")
    print("=" * 80)

    # Macro F1 comparison
    print("\n### MACRO F1 (mean +/- std) ###")
    print("-" * 50)

    sorted_models = sorted(results.keys(),
                          key=lambda x: np.mean(results[x]['macro_f1']),
                          reverse=True)

    for name in sorted_models:
        mean = np.mean(results[name]['macro_f1'])
        std = np.std(results[name]['macro_f1'])
        print(f"{name:30} {mean:.4f} +/- {std:.4f}")

    # Per-class F1 for problematic classes
    print("\n### F1 POR CLASE PROBLEMATICA ###")
    print("-" * 80)

    header = f"{'Model':<25}"
    for c in problematic_classes:
        header += f" | {c:^8}"
    print(header)
    print("-" * 80)

    for name in sorted_models:
        row = f"{name:<25}"
        for c in problematic_classes:
            mean = np.mean(results[name]['per_class_f1'][c])
            if mean > 0:
                row += f" | {mean:^8.4f}"
            else:
                row += f" | {'0.0000':^8}"
        print(row)

    # Find best model for each problematic class
    print("\n### MEJOR MODELO POR CLASE PROBLEMATICA ###")
    print("-" * 50)

    for c in problematic_classes:
        best_model = None
        best_f1 = 0
        for name in results:
            mean = np.mean(results[name]['per_class_f1'][c])
            if mean > best_f1:
                best_f1 = mean
                best_model = name

        if best_f1 > 0:
            print(f"{c}: {best_model} (F1={best_f1:.4f})")
        else:
            print(f"{c}: NINGUN MODELO PUEDE PREDECIR (F1=0 en todos)")


def main():
    # Paths
    PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
    DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
    EMBEDDINGS_PATH = PROJECT_ROOT / "phase_e/embeddings_cache/full_embeddings.npy"
    OUTPUT_PATH = PROJECT_ROOT / "phase_i/results/multi_model_comparison.json"

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    labels = df['type'].tolist()

    class_names = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels])

    print(f"  Samples: {len(y)}")
    print(f"  Classes: {len(class_names)}")

    # Load embeddings
    print("\nLoading embeddings...")
    X = np.load(EMBEDDINGS_PATH)
    print(f"  Shape: {X.shape}")

    # Define problematic classes
    problematic_classes = ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISTJ']

    # Run comparison
    results = run_comparison(X, y, class_names, n_splits=5, seed=42)

    # Print results
    print_results(results, class_names, problematic_classes)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for name in results:
        json_results[name] = {
            'macro_f1': {
                'mean': float(np.mean(results[name]['macro_f1'])),
                'std': float(np.std(results[name]['macro_f1'])),
                'values': [float(v) for v in results[name]['macro_f1']]
            },
            'per_class_f1': {}
        }
        for c in class_names:
            json_results[name]['per_class_f1'][c] = {
                'mean': float(np.mean(results[name]['per_class_f1'][c])),
                'std': float(np.std(results[name]['per_class_f1'][c])),
                'values': [float(v) for v in results[name]['per_class_f1'][c]]
            }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
