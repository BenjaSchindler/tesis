#!/usr/bin/env python3
"""
Phase G Extended - K-Fold Multi-Model Evaluator
Evaluates synthetic data with LogisticRegression, MLP, and XGBoost.
Reports per-class F1 scores with focus on problematic classes (ENFJ, ESFJ, ESFP, ESTJ, ISTJ).
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_mlp(X_train, y_train, input_dim: int, n_classes: int,
              hidden_dims: List[int] = [256, 128], epochs: int = 50,
              lr: float = 0.001, device: str = 'cuda') -> MLP:
    """Train MLP classifier."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    model = MLP(input_dim, hidden_dims, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_mlp(model: MLP, X_test, device: str = 'cuda') -> np.ndarray:
    """Get predictions from MLP."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


def evaluate_config(
    config_name: str,
    synth_path: str,
    data_path: str,
    embeddings_path: str,
    k: int = 5,
    repeats: int = 3,
    models: List[str] = ['LogisticRegression', 'MLP_small'],
    synthetic_weight: float = 0.5,
    random_state: int = 42,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate a synthetic data configuration with multiple models.

    Returns:
        Dictionary with per-model, per-class results
    """
    # Load data
    df = pd.read_csv(data_path)
    labels = df['type'].tolist()
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])

    # Load embeddings
    X = np.load(embeddings_path)

    # Load synthetic
    synth_df = pd.read_csv(synth_path)
    synth_labels = [label_to_idx.get(l, -1) for l in synth_df['label']]
    valid_synth = [i for i, l in enumerate(synth_labels) if l >= 0]
    synth_df = synth_df.iloc[valid_synth]
    y_synth = np.array([synth_labels[i] for i in valid_synth])

    # Compute synthetic embeddings
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        X_synth = st_model.encode(synth_df['text'].tolist(), show_progress_bar=False)
    else:
        raise ImportError("sentence_transformers not available")

    n_classes = len(unique_labels)
    results = {
        'config': config_name,
        'n_synthetic': len(synth_df),
        'synth_distribution': synth_df['label'].value_counts().to_dict(),
        'models': {},
    }

    # K-Fold evaluation for each model
    kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=random_state)

    for model_name in models:
        print(f"  Evaluating {model_name}...")

        baseline_macro = []
        augmented_macro = []
        baseline_per_class = {i: [] for i in range(n_classes)}
        augmented_per_class = {i: [] for i in range(n_classes)}

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Baseline model
            if model_name == 'LogisticRegression':
                clf_base = LogisticRegression(max_iter=2000)
                clf_base.fit(X_train, y_train)
                y_pred_base = clf_base.predict(X_test)
            elif model_name == 'LogisticRegression_balanced':
                clf_base = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf_base.fit(X_train, y_train)
                y_pred_base = clf_base.predict(X_test)
            elif model_name == 'MLP_small' and TORCH_AVAILABLE:
                clf_base = train_mlp(X_train, y_train, X.shape[1], n_classes,
                                    hidden_dims=[256, 128], epochs=50, device=device)
                y_pred_base = predict_mlp(clf_base, X_test, device)
            elif model_name == 'MLP_large' and TORCH_AVAILABLE:
                clf_base = train_mlp(X_train, y_train, X.shape[1], n_classes,
                                    hidden_dims=[512, 256, 128], epochs=100, device=device)
                y_pred_base = predict_mlp(clf_base, X_test, device)
            elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                clf_base = XGBClassifier(tree_method='hist', device='cuda', n_estimators=100, verbosity=0)
                clf_base.fit(X_train, y_train)
                y_pred_base = clf_base.predict(X_test)
            else:
                continue

            # Augmented model
            X_aug = np.vstack([X_train, X_synth])
            y_aug = np.concatenate([y_train, y_synth])
            weights = np.concatenate([np.ones(len(y_train)), np.full(len(y_synth), synthetic_weight)])

            if model_name == 'LogisticRegression':
                clf_aug = LogisticRegression(max_iter=2000)
                clf_aug.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_aug.predict(X_test)
            elif model_name == 'LogisticRegression_balanced':
                clf_aug = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf_aug.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_aug.predict(X_test)
            elif model_name == 'MLP_small' and TORCH_AVAILABLE:
                # For MLP, we need to handle weights differently
                clf_aug = train_mlp(X_aug, y_aug, X.shape[1], n_classes,
                                   hidden_dims=[256, 128], epochs=50, device=device)
                y_pred_aug = predict_mlp(clf_aug, X_test, device)
            elif model_name == 'MLP_large' and TORCH_AVAILABLE:
                clf_aug = train_mlp(X_aug, y_aug, X.shape[1], n_classes,
                                   hidden_dims=[512, 256, 128], epochs=100, device=device)
                y_pred_aug = predict_mlp(clf_aug, X_test, device)
            elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                clf_aug = XGBClassifier(tree_method='hist', device='cuda', n_estimators=100, verbosity=0)
                clf_aug.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_aug.predict(X_test)
            else:
                continue

            # Calculate metrics
            base_f1_macro = f1_score(y_test, y_pred_base, average='macro')
            aug_f1_macro = f1_score(y_test, y_pred_aug, average='macro')
            base_f1_per = f1_score(y_test, y_pred_base, average=None, labels=range(n_classes))
            aug_f1_per = f1_score(y_test, y_pred_aug, average=None, labels=range(n_classes))

            baseline_macro.append(base_f1_macro)
            augmented_macro.append(aug_f1_macro)
            for i in range(n_classes):
                baseline_per_class[i].append(base_f1_per[i])
                augmented_per_class[i].append(aug_f1_per[i])

        # Statistical analysis
        delta_macro = np.array(augmented_macro) - np.array(baseline_macro)
        t_stat, p_value = stats.ttest_rel(augmented_macro, baseline_macro)

        results['models'][model_name] = {
            'baseline_macro_mean': float(np.mean(baseline_macro)),
            'baseline_macro_std': float(np.std(baseline_macro)),
            'augmented_macro_mean': float(np.mean(augmented_macro)),
            'augmented_macro_std': float(np.std(augmented_macro)),
            'delta_mean': float(np.mean(delta_macro)),
            'delta_pct': float(np.mean(delta_macro) / np.mean(baseline_macro) * 100) if np.mean(baseline_macro) > 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'win_rate': float(np.mean(delta_macro > 0)),
            'per_class': {},
        }

        # Per-class results
        for i, label in enumerate(unique_labels):
            base_mean = np.mean(baseline_per_class[i])
            aug_mean = np.mean(augmented_per_class[i])
            delta_pp = (aug_mean - base_mean) * 100
            results['models'][model_name]['per_class'][label] = {
                'baseline': float(base_mean),
                'augmented': float(aug_mean),
                'delta_pp': float(delta_pp),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase G Extended - K-Fold Multi-Model Evaluator')
    parser.add_argument('--config', type=str, help='Config name to evaluate (e.g., W1_no_gate)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for synthetic file')
    parser.add_argument('--all', action='store_true', help='Evaluate all synthetic files in results/')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--repeats', type=int, default=3, help='Number of K-fold repeats')
    parser.add_argument('--models', nargs='+', default=['LogisticRegression', 'MLP_small'],
                       help='Models to evaluate')
    parser.add_argument('--output', type=str, default='kfold_multimodel_results.json',
                       help='Output JSON file')
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    phase_g_dir = Path(__file__).parent
    data_path = project_root / 'mbti_1.csv'
    embeddings_path = project_root / 'phase_e' / 'embeddings_cache' / 'full_embeddings.npy'
    results_dir = phase_g_dir / 'results'

    # Determine device
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    all_results = []

    if args.all:
        # Evaluate all synthetic files
        synth_files = list(results_dir.glob('*_synth.csv'))
        print(f"Found {len(synth_files)} synthetic files")

        for synth_path in sorted(synth_files):
            config_name = synth_path.stem.replace('_synth', '')
            print(f"\nEvaluating {config_name}...")

            try:
                result = evaluate_config(
                    config_name=config_name,
                    synth_path=str(synth_path),
                    data_path=str(data_path),
                    embeddings_path=str(embeddings_path),
                    k=args.k,
                    repeats=args.repeats,
                    models=args.models,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                continue
    else:
        # Evaluate single config
        config_name = f"{args.config}_s{args.seed}"
        synth_path = results_dir / f"{config_name}_synth.csv"

        if not synth_path.exists():
            print(f"Error: {synth_path} not found")
            sys.exit(1)

        print(f"Evaluating {config_name}...")
        result = evaluate_config(
            config_name=config_name,
            synth_path=str(synth_path),
            data_path=str(data_path),
            embeddings_path=str(embeddings_path),
            k=args.k,
            repeats=args.repeats,
            models=args.models,
            device=device,
        )
        all_results.append(result)

    # Save results
    output_path = results_dir / args.output
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Problematic Classes (ENFJ, ESFJ, ESFP, ESTJ, ISTJ)")
    print("="*70)

    problematic = ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISTJ']

    for result in all_results[:5]:  # Show first 5
        print(f"\n{result['config']} ({result['n_synthetic']} synthetics)")
        for model_name, model_results in result['models'].items():
            print(f"  {model_name}: macro_delta={model_results['delta_pct']:+.2f}%")
            for cls in problematic:
                if cls in model_results['per_class']:
                    delta = model_results['per_class'][cls]['delta_pp']
                    print(f"    {cls}: {delta:+.2f} pp")


if __name__ == '__main__':
    main()
