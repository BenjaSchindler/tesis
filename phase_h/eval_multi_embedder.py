#!/usr/bin/env python3
"""
Phase H - Multi-Embedder Evaluation

Evaluates SMOTE-LLM robustness across different SOTA embedding models (2024-2025).
Tests if augmentation benefits are consistent regardless of the embedder used.

Usage:
    python eval_multi_embedder.py
    python eval_multi_embedder.py --embedders all-mpnet-base-v2 bge-large-en-v1.5
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
RESULTS_DIR = PROJECT_ROOT / "phase_h" / "results"

# Embedders to evaluate (SOTA 2024-2025)
EMBEDDERS = {
    "all-mpnet-base-v2": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "params": "110M",
        "dims": 768,
        "prefix": None,
    },
    "nomic-embed-text-v1.5": {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "params": "137M",
        "dims": 768,
        "prefix": None,
        "trust_remote_code": True,
    },
    "bge-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "params": "335M",
        "dims": 1024,
        "prefix": None,
    },
    "e5-large-v2": {
        "model_id": "intfloat/e5-large-v2",
        "params": "335M",
        "dims": 1024,
        "prefix": "query: ",  # E5 requires prefix
    },
    "gte-large-en-v1.5": {
        "model_id": "Alibaba-NLP/gte-large-en-v1.5",
        "params": "434M",
        "dims": 1024,
        "prefix": None,
        "trust_remote_code": True,
    },
    "stella_en_1.5B_v5": {
        "model_id": "NovaSearch/stella_en_1.5B_v5",
        "params": "1.5B",
        "dims": 1024,
        "prefix": None,
        "trust_remote_code": True,
    },
    "gte-Qwen2-1.5B-instruct": {
        "model_id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "params": "1.5B",
        "dims": 1536,
        "prefix": None,
        "trust_remote_code": True,
    },
}

# Synthetic files to evaluate
SYNTH_FILES = [
    PROJECT_ROOT / "phase_g/replication_run1/results/ENS_SUPER_G5_F7_v2_synth.csv",
    PROJECT_ROOT / "phase_g/replication_run2/results/ENS_SUPER_G5_F7_v2_synth.csv",
    PROJECT_ROOT / "phase_g/replication_run3/results/ENS_SUPER_G5_F7_v2_synth.csv",
]


def load_dataset():
    """Load MBTI dataset."""
    df = pd.read_csv(DATA_PATH)
    texts = df['posts'].tolist()
    labels = df['type'].tolist()
    return texts, labels


def load_synthetics(synth_path):
    """Load synthetic data from CSV."""
    df = pd.read_csv(synth_path)
    text_col = 'text' if 'text' in df.columns else 'posts'
    label_col = 'label' if 'label' in df.columns else 'type'
    return df[text_col].tolist(), df[label_col].tolist()


def add_prefix(texts, prefix):
    """Add prefix to texts if required by the model."""
    if prefix:
        return [prefix + t for t in texts]
    return texts


def compute_embeddings(model, texts, batch_size=64, prefix=None):
    """Compute embeddings with optional prefix."""
    texts_with_prefix = add_prefix(texts, prefix)
    embeddings = model.encode(
        texts_with_prefix,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def run_kfold_evaluation(X_orig, y_orig, X_synth, y_synth, n_splits=5, seed=42):
    """Run K-fold evaluation comparing baseline vs augmented."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)
    y_synth_encoded = le.transform(y_synth)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    baseline_scores = []
    augmented_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_orig, y_encoded)):
        X_train, X_test = X_orig[train_idx], X_orig[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Baseline
        clf_base = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
        clf_base.fit(X_train, y_train)
        y_pred_base = clf_base.predict(X_test)
        f1_base = f1_score(y_test, y_pred_base, average='macro')
        baseline_scores.append(f1_base)

        # Augmented
        X_train_aug = np.vstack([X_train, X_synth])
        y_train_aug = np.concatenate([y_train, y_synth_encoded])

        clf_aug = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
        clf_aug.fit(X_train_aug, y_train_aug)
        y_pred_aug = clf_aug.predict(X_test)
        f1_aug = f1_score(y_test, y_pred_aug, average='macro')
        augmented_scores.append(f1_aug)

    baseline_mean = np.mean(baseline_scores)
    augmented_mean = np.mean(augmented_scores)
    delta_pct = ((augmented_mean - baseline_mean) / baseline_mean) * 100

    return {
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(np.std(baseline_scores)),
        'augmented_mean': float(augmented_mean),
        'augmented_std': float(np.std(augmented_scores)),
        'delta_pct': float(delta_pct),
        'delta_pp': float((augmented_mean - baseline_mean) * 100),
        'fold_scores': {
            'baseline': [float(s) for s in baseline_scores],
            'augmented': [float(s) for s in augmented_scores]
        }
    }


def free_gpu_memory(model):
    """Free GPU memory after using a model."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)  # Give CUDA time to free memory


def print_summary_table(results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: Multi-Embedder Evaluation Results")
    print("=" * 100)
    print(f"\n{'Embedder':<30} | {'Params':<8} | {'Baseline':<10} | {'Augmented':<10} | {'Delta %':<10} | {'Std':<8}")
    print("-" * 100)

    for name, data in sorted(results.items(), key=lambda x: x[1]['mean_delta'], reverse=True):
        print(f"{name:<30} | {data['params']:<8} | {data['mean_baseline']:.4f}     | "
              f"{data['mean_augmented']:.4f}     | {data['mean_delta']:+.2f}%     | {data['std_delta']:.2f}")

    print("-" * 100)

    # Overall statistics
    all_deltas = [data['mean_delta'] for data in results.values()]
    print(f"\nOverall: Mean Delta = {np.mean(all_deltas):+.2f}% ± {np.std(all_deltas):.2f}%")
    print(f"Range: [{min(all_deltas):+.2f}%, {max(all_deltas):+.2f}%]")
    print(f"All positive: {all(d > 0 for d in all_deltas)}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Embedder Evaluation')
    parser.add_argument('--embedders', nargs='+', default=None,
                       help='Specific embedders to evaluate (default: all)')
    parser.add_argument('--synth-files', nargs='+', default=None,
                       help='Specific synth files to use')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding')
    parser.add_argument('--k-folds', type=int, default=5,
                       help='Number of CV folds')
    args = parser.parse_args()

    # Select embedders
    if args.embedders:
        embedders = {k: v for k, v in EMBEDDERS.items() if k in args.embedders}
    else:
        embedders = EMBEDDERS

    # Select synth files
    synth_files = args.synth_files if args.synth_files else SYNTH_FILES
    synth_files = [Path(f) for f in synth_files]

    print("=" * 80)
    print("  PHASE H - Multi-Embedder Robustness Evaluation")
    print("=" * 80)
    print(f"\nEmbedders: {len(embedders)}")
    print(f"Synth files: {len(synth_files)}")
    print(f"Total evaluations: {len(embedders) * len(synth_files)}")
    print(f"K-Folds: {args.k_folds}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load original dataset
    print("\n" + "-" * 80)
    print("Loading original dataset...")
    orig_texts, orig_labels = load_dataset()
    print(f"  Samples: {len(orig_texts)}")
    print(f"  Classes: {len(set(orig_labels))}")

    # Load all synthetic datasets
    print("\nLoading synthetic datasets...")
    synth_data = {}
    for synth_path in synth_files:
        if synth_path.exists():
            synth_texts, synth_labels = load_synthetics(synth_path)
            run_name = synth_path.parent.parent.name  # e.g., "replication_run1"
            synth_data[run_name] = (synth_texts, synth_labels)
            print(f"  {run_name}: {len(synth_texts)} samples")
        else:
            print(f"  WARNING: {synth_path} not found")

    # Results storage
    all_results = {}

    # Evaluate each embedder
    start_time = time.time()

    for emb_idx, (emb_name, emb_config) in enumerate(embedders.items(), 1):
        print("\n" + "=" * 80)
        print(f"[{emb_idx}/{len(embedders)}] Embedder: {emb_name}")
        print(f"  Model: {emb_config['model_id']}")
        print(f"  Params: {emb_config['params']}, Dims: {emb_config['dims']}")
        print("=" * 80)

        # Load model
        print("\nLoading model...")
        load_start = time.time()

        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        if emb_config.get("trust_remote_code"):
            model_kwargs["trust_remote_code"] = True

        try:
            model = SentenceTransformer(emb_config['model_id'], **model_kwargs)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print("  Skipping this embedder...")
            continue

        print(f"  Loaded in {time.time() - load_start:.1f}s")

        # Compute original embeddings
        print("\nComputing original embeddings...")
        embed_start = time.time()
        X_orig = compute_embeddings(
            model, orig_texts,
            batch_size=args.batch_size,
            prefix=emb_config.get('prefix')
        )
        print(f"  Shape: {X_orig.shape}")
        print(f"  Time: {time.time() - embed_start:.1f}s")

        # Evaluate each synthetic dataset
        emb_results = {
            'model_id': emb_config['model_id'],
            'params': emb_config['params'],
            'dims': emb_config['dims'],
            'runs': []
        }

        for run_name, (synth_texts, synth_labels) in synth_data.items():
            print(f"\n  --- {run_name} ({len(synth_texts)} synthetics) ---")

            # Compute synthetic embeddings
            X_synth = compute_embeddings(
                model, synth_texts,
                batch_size=args.batch_size,
                prefix=emb_config.get('prefix')
            )

            # Run evaluation
            result = run_kfold_evaluation(
                X_orig, orig_labels,
                X_synth, synth_labels,
                n_splits=args.k_folds
            )

            result['run'] = run_name
            result['n_synthetic'] = len(synth_texts)
            emb_results['runs'].append(result)

            print(f"    Baseline:  {result['baseline_mean']:.4f}")
            print(f"    Augmented: {result['augmented_mean']:.4f}")
            print(f"    Delta:     {result['delta_pct']:+.2f}%")

        # Compute aggregates
        deltas = [r['delta_pct'] for r in emb_results['runs']]
        baselines = [r['baseline_mean'] for r in emb_results['runs']]
        augmenteds = [r['augmented_mean'] for r in emb_results['runs']]

        emb_results['mean_delta'] = float(np.mean(deltas))
        emb_results['std_delta'] = float(np.std(deltas))
        emb_results['mean_baseline'] = float(np.mean(baselines))
        emb_results['mean_augmented'] = float(np.mean(augmenteds))

        all_results[emb_name] = emb_results

        print(f"\n  SUMMARY for {emb_name}:")
        print(f"    Mean Delta: {emb_results['mean_delta']:+.2f}% ± {emb_results['std_delta']:.2f}%")

        # Free GPU memory
        print("\n  Freeing GPU memory...")
        free_gpu_memory(model)

    # Total time
    total_time = time.time() - start_time
    print(f"\n\nTotal evaluation time: {total_time/60:.1f} minutes")

    # Print summary table
    print_summary_table(all_results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "multi_embedder_results.json"

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'k_folds': args.k_folds,
        'n_embedders': len(all_results),
        'synth_files': [str(f) for f in synth_files],
        'results': all_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
