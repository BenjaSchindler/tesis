#!/usr/bin/env python3
"""
Phase H - Small Embedder Evaluation

Tests smaller, faster embedding models for comparison.
"""

import torch
import pandas as pd
import numpy as np
import json
import gc
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
RESULTS_DIR = PROJECT_ROOT / "phase_h" / "results"

# Small embedders to test
SMALL_EMBEDDERS = {
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "params": "22M",
        "dims": 384
    },
    "bge-small-en-v1.5": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "params": "33M",
        "dims": 384
    }
}

# Synth files
SYNTH_FILES = [
    PROJECT_ROOT / "phase_g/replication_run1/results/ENS_SUPER_G5_F7_v2_synth.csv",
    PROJECT_ROOT / "phase_g/replication_run2/results/ENS_SUPER_G5_F7_v2_synth.csv",
    PROJECT_ROOT / "phase_g/replication_run3/results/ENS_SUPER_G5_F7_v2_synth.csv",
]


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_embedder(emb_name, emb_config, texts, labels, synth_data):
    """Evaluate a single embedder across all synth files."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {emb_name}")
    print(f"  Model: {emb_config['model_id']}")
    print(f"  Params: {emb_config['params']}, Dims: {emb_config['dims']}")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model = SentenceTransformer(emb_config['model_id'], device='cuda')

    # Embed original data
    print("Embedding original dataset...")
    X_orig = model.encode(texts, batch_size=128, show_progress_bar=True)
    print(f"  Shape: {X_orig.shape}")

    # Encode labels
    le = LabelEncoder()
    y_orig = le.fit_transform(labels)

    results = {
        'model_id': emb_config['model_id'],
        'params': emb_config['params'],
        'dims': emb_config['dims'],
        'runs': []
    }

    # Evaluate each synth file
    for run_name, (synth_texts, synth_labels) in synth_data.items():
        print(f"\n  --- {run_name} ({len(synth_texts)} synthetics) ---")

        # Embed synthetics
        X_synth = model.encode(synth_texts, batch_size=128, show_progress_bar=False)
        y_synth = le.transform(synth_labels)

        # K-fold evaluation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_scores, aug_scores = [], []

        for train_idx, test_idx in kf.split(X_orig, y_orig):
            X_train, X_test = X_orig[train_idx], X_orig[test_idx]
            y_train, y_test = y_orig[train_idx], y_orig[test_idx]

            # Baseline
            clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            base_scores.append(f1_score(y_test, clf.predict(X_test), average='macro'))

            # Augmented
            X_aug = np.vstack([X_train, X_synth])
            y_aug = np.concatenate([y_train, y_synth])
            clf.fit(X_aug, y_aug)
            aug_scores.append(f1_score(y_test, clf.predict(X_test), average='macro'))

        baseline = float(np.mean(base_scores))
        augmented = float(np.mean(aug_scores))
        delta = (augmented - baseline) / baseline * 100

        results['runs'].append({
            'run': run_name,
            'baseline': baseline,
            'augmented': augmented,
            'delta_pct': float(delta)
        })

        print(f"    Baseline:  {baseline:.4f}")
        print(f"    Augmented: {augmented:.4f}")
        print(f"    Delta:     {delta:+.2f}%")

    # Aggregates
    deltas = [r['delta_pct'] for r in results['runs']]
    baselines = [r['baseline'] for r in results['runs']]
    results['mean_delta'] = float(np.mean(deltas))
    results['std_delta'] = float(np.std(deltas))
    results['mean_baseline'] = float(np.mean(baselines))

    print(f"\n  SUMMARY: Mean Delta = {results['mean_delta']:+.2f}% ± {results['std_delta']:.2f}%")

    # Free GPU
    del model
    free_gpu()

    return results


def main():
    print("="*60)
    print("  PHASE H - Small Embedder Evaluation")
    print("="*60)
    print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load original data
    print("\nLoading original dataset...")
    df = pd.read_csv(DATA_PATH)
    texts = df['posts'].tolist()
    labels = df['type'].tolist()
    print(f"  Samples: {len(texts)}, Classes: {len(set(labels))}")

    # Load synth data
    print("\nLoading synthetic datasets...")
    synth_data = {}
    for synth_path in SYNTH_FILES:
        if synth_path.exists():
            synth_df = pd.read_csv(synth_path)
            run_name = synth_path.parent.parent.name
            synth_data[run_name] = (synth_df['text'].tolist(), synth_df['label'].tolist())
            print(f"  {run_name}: {len(synth_df)} samples")

    # Evaluate each embedder
    all_results = {}
    for emb_name, emb_config in SMALL_EMBEDDERS.items():
        results = evaluate_embedder(emb_name, emb_config, texts, labels, synth_data)
        all_results[emb_name] = results

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Small Embedders")
    print("="*80)
    print(f"\n{'Embedder':<25} | {'Params':<8} | {'Dims':<6} | {'Baseline':<10} | {'Delta %':<12}")
    print("-"*80)

    for name, data in sorted(all_results.items(), key=lambda x: x[1]['mean_delta'], reverse=True):
        print(f"{name:<25} | {data['params']:<8} | {data['dims']:<6} | {data['mean_baseline']:.4f}     | {data['mean_delta']:+.2f}% ± {data['std_delta']:.2f}%")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "small_embedders_results.json"

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
