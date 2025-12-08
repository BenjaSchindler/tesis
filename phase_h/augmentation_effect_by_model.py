#!/usr/bin/env python3
"""
Phase I - Augmentation Effect by Model

Compares baseline vs augmented performance across different classifiers
to see which models benefit most from synthetic data augmentation.
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Check CUDA
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class PyTorchMLP(nn.Module):
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
    def __init__(self, hidden_dims=(256, 128), epochs=50, batch_size=256, lr=0.001, device='cuda'):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        self.num_classes = None

    def fit(self, X, y, sample_weight=None):
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        if sample_weight is not None:
            weights_tensor = torch.FloatTensor(sample_weight).to(self.device)
        else:
            weights_tensor = None

        input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))

        self.model = PyTorchMLP(input_dim, self.hidden_dims, self.num_classes).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if weights_tensor is not None:
            # Use weighted cross entropy
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (batch_X, batch_y) in enumerate(loader):
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if weights_tensor is not None:
                    # Get corresponding weights for this batch
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + len(batch_y), len(weights_tensor))
                    batch_weights = weights_tensor[start_idx:end_idx]
                    loss = (criterion(outputs, batch_y) * batch_weights[:len(batch_y)]).mean()
                else:
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

    def get_params(self, deep=True):
        return {
            'hidden_dims': self.hidden_dims,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': self.device
        }


def get_models():
    """Return dict of models to test."""
    models = {
        'LogisticRegression': lambda: LogisticRegression(max_iter=2000, n_jobs=-1),
        'LogisticRegression_balanced': lambda: LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1),
    }

    if HAS_XGBOOST:
        models['XGBoost_GPU'] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            tree_method='hist', device='cuda',
            random_state=42, verbosity=0
        )

    if HAS_LIGHTGBM:
        models['LightGBM_GPU'] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            random_state=42, verbose=-1
        )

    if DEVICE == 'cuda':
        models['MLP_GPU_small'] = lambda: TorchMLPClassifier(hidden_dims=(256, 128), epochs=50, device=DEVICE)
        models['MLP_GPU_large'] = lambda: TorchMLPClassifier(hidden_dims=(512, 256, 128), epochs=100, device=DEVICE)

    return models


def run_comparison(X, y, X_synth, y_synth, class_names, n_splits=5, seed=42):
    """Run K-Fold comparison: baseline vs augmented for each model."""

    n_classes = len(class_names)
    model_factories = get_models()

    results = {}

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print(f"\nRunning {n_splits}-Fold CV comparing baseline vs augmented...")
    print(f"Synthetic samples: {len(y_synth)}")
    print("=" * 80)

    for model_name, model_factory in model_factories.items():
        print(f"\n=== {model_name} ===")

        baseline_macro = []
        augmented_macro = []
        baseline_per_class = {c: [] for c in class_names}
        augmented_per_class = {c: [] for c in class_names}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Baseline
            try:
                model_base = model_factory()
                model_base.fit(X_train, y_train)
                y_pred_base = model_base.predict(X_test)

                base_macro = f1_score(y_test, y_pred_base, average='macro')
                base_per_class = f1_score(y_test, y_pred_base, average=None, labels=range(n_classes))

                baseline_macro.append(base_macro)
                for i, c in enumerate(class_names):
                    baseline_per_class[c].append(base_per_class[i])
            except Exception as e:
                print(f"  Fold {fold_idx+1} baseline error: {e}")
                baseline_macro.append(0)
                for c in class_names:
                    baseline_per_class[c].append(0)
                continue

            # Augmented
            try:
                X_aug = np.vstack([X_train, X_synth])
                y_aug = np.concatenate([y_train, y_synth])
                weights = np.concatenate([np.ones(len(y_train)), np.full(len(y_synth), 0.5)])

                model_aug = model_factory()

                # Check if model supports sample_weight
                if hasattr(model_aug, 'fit'):
                    import inspect
                    sig = inspect.signature(model_aug.fit)
                    if 'sample_weight' in sig.parameters:
                        model_aug.fit(X_aug, y_aug, sample_weight=weights)
                    else:
                        model_aug.fit(X_aug, y_aug)

                y_pred_aug = model_aug.predict(X_test)

                aug_macro = f1_score(y_test, y_pred_aug, average='macro')
                aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=range(n_classes))

                augmented_macro.append(aug_macro)
                for i, c in enumerate(class_names):
                    augmented_per_class[c].append(aug_per_class[i])

            except Exception as e:
                print(f"  Fold {fold_idx+1} augmented error: {e}")
                augmented_macro.append(base_macro)
                for i, c in enumerate(class_names):
                    augmented_per_class[c].append(base_per_class[i])

            delta = (aug_macro - base_macro) * 100
            print(f"  Fold {fold_idx+1}: Base={base_macro:.4f}, Aug={aug_macro:.4f}, Delta={delta:+.2f}%")

        # Store results
        results[model_name] = {
            'baseline_macro': {
                'mean': float(np.mean(baseline_macro)),
                'std': float(np.std(baseline_macro)),
                'values': [float(v) for v in baseline_macro]
            },
            'augmented_macro': {
                'mean': float(np.mean(augmented_macro)),
                'std': float(np.std(augmented_macro)),
                'values': [float(v) for v in augmented_macro]
            },
            'delta_pct': {
                'mean': float((np.mean(augmented_macro) - np.mean(baseline_macro)) / np.mean(baseline_macro) * 100),
                'absolute': float((np.mean(augmented_macro) - np.mean(baseline_macro)) * 100)
            },
            'per_class': {}
        }

        for c in class_names:
            base_mean = np.mean(baseline_per_class[c])
            aug_mean = np.mean(augmented_per_class[c])
            results[model_name]['per_class'][c] = {
                'baseline': float(base_mean),
                'augmented': float(aug_mean),
                'delta_pp': float((aug_mean - base_mean) * 100)
            }

    return results


def print_results(results, problematic_classes):
    """Print formatted comparison results."""

    print("\n" + "=" * 90)
    print("PHASE I - AUGMENTATION EFFECT BY MODEL")
    print("=" * 90)

    # Sort by delta
    sorted_models = sorted(results.keys(),
                          key=lambda x: results[x]['delta_pct']['mean'],
                          reverse=True)

    # Macro F1 comparison
    print("\n### MACRO F1: BASELINE vs AUGMENTED ###")
    print("-" * 90)
    print(f"{'Model':<28} | {'Baseline':<12} | {'Augmented':<12} | {'Delta %':<10} | {'Delta pp':<10}")
    print("-" * 90)

    for name in sorted_models:
        r = results[name]
        base = r['baseline_macro']['mean']
        aug = r['augmented_macro']['mean']
        delta_pct = r['delta_pct']['mean']
        delta_pp = r['delta_pct']['absolute']

        marker = " **BEST**" if delta_pct == max(results[m]['delta_pct']['mean'] for m in results) else ""
        print(f"{name:<28} | {base:.4f}       | {aug:.4f}       | {delta_pct:+.2f}%     | {delta_pp:+.2f} pp{marker}")

    # Per-class analysis for problematic classes
    print("\n### PROBLEMATIC CLASSES: DELTA (pp) BY MODEL ###")
    print("-" * 100)

    header = f"{'Model':<28}"
    for c in problematic_classes:
        header += f" | {c:^8}"
    header += " | AVG"
    print(header)
    print("-" * 100)

    for name in sorted_models:
        row = f"{name:<28}"
        deltas = []
        for c in problematic_classes:
            delta = results[name]['per_class'][c]['delta_pp']
            deltas.append(delta)
            if delta > 0:
                row += f" | {delta:+7.2f}"
            else:
                row += f" | {delta:+7.2f}"
        avg_delta = np.mean(deltas)
        row += f" | {avg_delta:+.2f}"
        print(row)

    # Best model for each problematic class
    print("\n### BEST MODEL FOR EACH PROBLEMATIC CLASS (by augmentation delta) ###")
    print("-" * 60)

    for c in problematic_classes:
        best_model = None
        best_delta = -999
        for name in results:
            delta = results[name]['per_class'][c]['delta_pp']
            if delta > best_delta:
                best_delta = delta
                best_model = name

        base = results[best_model]['per_class'][c]['baseline']
        aug = results[best_model]['per_class'][c]['augmented']
        print(f"{c}: {best_model}")
        print(f"       Baseline={base:.4f} -> Augmented={aug:.4f} ({best_delta:+.2f} pp)")


def main():
    PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
    DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
    EMBEDDINGS_PATH = PROJECT_ROOT / "phase_e/embeddings_cache/full_embeddings.npy"
    SYNTH_PATH = PROJECT_ROOT / "phase_g/results/ENS_Top3_G5_s42_synth.csv"
    OUTPUT_PATH = PROJECT_ROOT / "phase_i/results/augmentation_effect_by_model.json"

    # Load original data
    print("Loading original data...")
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

    # Load synthetic data
    print("\nLoading synthetic data...")
    synth_df = pd.read_csv(SYNTH_PATH)
    print(f"  Synthetic samples: {len(synth_df)}")

    # Generate synthetic embeddings
    print("\nGenerating synthetic embeddings...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)

    text_col = 'text' if 'text' in synth_df.columns else 'posts'
    label_col = 'label' if 'label' in synth_df.columns else 'type'

    X_synth = model.encode(synth_df[text_col].tolist(), show_progress_bar=True, batch_size=64)
    y_synth = np.array([label_to_idx[l] for l in synth_df[label_col]])

    print(f"  Synthetic embeddings shape: {X_synth.shape}")

    # Show synthetic distribution
    print("\nSynthetic distribution:")
    for c in class_names:
        count = np.sum(y_synth == label_to_idx[c])
        if count > 0:
            print(f"  {c}: {count}")

    # Problematic classes
    problematic_classes = ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISTJ']

    # Run comparison
    results = run_comparison(X, y, X_synth, y_synth, class_names, n_splits=5, seed=42)

    # Print results
    print_results(results, problematic_classes)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
