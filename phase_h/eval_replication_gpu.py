#!/usr/bin/env python3
"""
Phase H - GPU Multi-Model Evaluation for Replication Results

Evaluates synthetic ensembles with GPU-accelerated models:
- LogisticRegression
- LogisticRegression_balanced
- XGBoost_GPU
- LightGBM_GPU
- MLP_GPU_small
- MLP_GPU_large

Usage:
    python eval_replication_gpu.py --synth path/to/synth.csv
    python eval_replication_gpu.py --synth path/to/synth.csv --k 5
"""

import argparse
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
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")


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

        input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))

        self.model = PyTorchMLP(input_dim, self.hidden_dims, self.num_classes).to(self.device)
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
    """Return dict of GPU-optimized models to test."""
    models = {
        'LogisticRegression': lambda: LogisticRegression(max_iter=2000, n_jobs=-1),
        'LogReg_balanced': lambda: LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1),
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


def run_evaluation(X, y, X_synth, y_synth, class_names, n_splits=5, seed=42):
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

                model_aug = model_factory()
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
        base_mean = np.mean(baseline_macro)
        aug_mean = np.mean(augmented_macro)

        results[model_name] = {
            'baseline_macro': {
                'mean': float(base_mean),
                'std': float(np.std(baseline_macro)),
            },
            'augmented_macro': {
                'mean': float(aug_mean),
                'std': float(np.std(augmented_macro)),
            },
            'delta_pct': float((aug_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0,
            'delta_pp': float((aug_mean - base_mean) * 100),
            'per_class': {}
        }

        for c in class_names:
            base_c = np.mean(baseline_per_class[c])
            aug_c = np.mean(augmented_per_class[c])
            results[model_name]['per_class'][c] = {
                'baseline': float(base_c),
                'augmented': float(aug_c),
                'delta_pp': float((aug_c - base_c) * 100)
            }

    return results


def print_results(results, synth_name):
    """Print formatted comparison results."""

    print("\n" + "=" * 90)
    print(f"GPU MULTI-MODEL EVALUATION: {synth_name}")
    print("=" * 90)

    # Sort by delta
    sorted_models = sorted(results.keys(),
                          key=lambda x: results[x]['delta_pct'],
                          reverse=True)

    # Macro F1 comparison
    print(f"\n{'Model':<20} | {'Baseline':<10} | {'Augmented':<10} | {'Delta %':<10} | {'Delta pp':<10}")
    print("-" * 75)

    for name in sorted_models:
        r = results[name]
        base = r['baseline_macro']['mean']
        aug = r['augmented_macro']['mean']
        delta_pct = r['delta_pct']
        delta_pp = r['delta_pp']

        print(f"{name:<20} | {base:.4f}     | {aug:.4f}     | {delta_pct:+.2f}%     | {delta_pp:+.2f} pp")

    # Best model
    best_name = sorted_models[0]
    best = results[best_name]
    print("-" * 75)
    print(f"\nBEST MODEL: {best_name}")
    print(f"  Baseline: {best['baseline_macro']['mean']:.4f}")
    print(f"  Augmented: {best['augmented_macro']['mean']:.4f}")
    print(f"  Delta: {best['delta_pct']:+.2f}% ({best['delta_pp']:+.2f} pp)")


def main():
    parser = argparse.ArgumentParser(description='GPU Multi-Model Evaluation')
    parser.add_argument('--synth', type=str, required=True, help='Path to synthetic CSV')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, help='Output JSON path')
    args = parser.parse_args()

    PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
    DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
    EMBEDDINGS_PATH = PROJECT_ROOT / "phase_e/embeddings_cache/full_embeddings.npy"

    synth_path = Path(args.synth)
    synth_name = synth_path.stem

    print(f"\n{'='*80}")
    print(f"  PHASE H - GPU Multi-Model Evaluation")
    print(f"{'='*80}")
    print(f"  Synthetic: {synth_name}")
    print(f"  K-Fold: {args.k}")

    # Load original data
    print("\nLoading original data...")
    df = pd.read_csv(DATA_PATH)
    labels = df['type'].tolist()

    class_names = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels])

    print(f"  Samples: {len(y)}, Classes: {len(class_names)}")

    # Load embeddings
    print("Loading embeddings...")
    X = np.load(EMBEDDINGS_PATH)
    print(f"  Shape: {X.shape}")

    # Load synthetic data
    print(f"Loading synthetic data from {synth_path}...")
    synth_df = pd.read_csv(synth_path)
    print(f"  Synthetic samples: {len(synth_df)}")

    # Generate synthetic embeddings
    print("Generating synthetic embeddings (GPU)...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)

    text_col = 'text' if 'text' in synth_df.columns else 'posts'
    label_col = 'label' if 'label' in synth_df.columns else 'type'

    X_synth = model.encode(synth_df[text_col].tolist(), show_progress_bar=True, batch_size=64)
    y_synth = np.array([label_to_idx[l] for l in synth_df[label_col]])

    # Run evaluation
    results = run_evaluation(X, y, X_synth, y_synth, class_names, n_splits=args.k, seed=args.seed)

    # Print results
    print_results(results, synth_name)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "phase_h" / "results" / f"gpu_eval_{synth_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'synth_file': str(synth_path),
            'n_synthetic': len(synth_df),
            'k_folds': args.k,
            'seed': args.seed,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
