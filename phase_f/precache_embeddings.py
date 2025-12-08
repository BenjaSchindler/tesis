#!/usr/bin/env python3
"""Pre-cache embeddings for Phase F experiments."""

import sys
import os

# Add phase_f/core to path (isolated from root core)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from embedding_cache import EmbeddingCache, get_or_compute_embeddings


def main():
    parser = argparse.ArgumentParser(description='Pre-cache embeddings for all seeds')
    parser.add_argument('--data-path', default='../mbti_1.csv')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 123])
    parser.add_argument('--cache-dir', default='../phase_e/embeddings_cache')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print("  Pre-caching embeddings for Phase F")
    print("=" * 60)
    print(f"  Data: {args.data_path}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Cache: {args.cache_dir}")
    print("=" * 60)

    # Initialize cache
    cache = EmbeddingCache(args.cache_dir)

    # Load model once
    print(f"\nLoading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model, device=args.device)
    print("Model loaded!")

    # Load full dataset once
    print(f"\nLoading dataset: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Dataset: {len(df)} samples")

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Split data (same logic as runner_phase2.py)
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=seed, stratify=df['type']
        )
        train_df, val_df = train_test_split(
            train_df, test_size=args.val_size, random_state=seed, stratify=train_df['type']
        )

        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Get/compute embeddings for each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            texts = split_df['posts'].tolist()
            emb = get_or_compute_embeddings(
                cache=cache,
                embedder=embedder,
                texts=texts,
                data_path=args.data_path,
                seed=seed,
                split=split_name,
                model_name=args.model,
                batch_size=args.batch_size,
                test_size=args.test_size,
                val_size=args.val_size
            )
            print(f"  {split_name}: {emb.shape}")

    print("\n" + "=" * 60)
    print(f"  Cache stats: {cache.get_stats()}")
    print("  All embeddings cached!")
    print("=" * 60)


if __name__ == '__main__':
    main()
