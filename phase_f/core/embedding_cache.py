#!/usr/bin/env python3
"""
Embedding Cache for Phase E experiments.

Caches embeddings to disk to avoid recomputing for same seed/data combinations.
Key insight: For the same seed, train/test/val splits are identical across configs.

Usage:
    cache = EmbeddingCache(cache_dir="embeddings_cache")

    # Try to load from cache
    embeddings = cache.load(data_path, seed, split="train", model_name="all-mpnet-base-v2")

    if embeddings is None:
        # Compute and save
        embeddings = compute_embeddings(embedder, texts, batch_size)
        cache.save(embeddings, data_path, seed, split="train", model_name="all-mpnet-base-v2")

Estimated savings: ~55% of embedding time (from ~78 min to ~35 min per experiment)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


class EmbeddingCache:
    """Disk-based cache for embeddings."""

    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hits": 0, "misses": 0, "saves": 0}

    def _make_key(
        self,
        data_path: str,
        seed: int,
        split: str,
        model_name: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> str:
        """Create a unique cache key based on parameters."""
        # Hash the data path for shorter filenames
        data_hash = hashlib.md5(data_path.encode()).hexdigest()[:8]

        # Build key components
        components = [
            f"data_{data_hash}",
            f"seed_{seed}",
            f"split_{split}",
            f"test_{test_size}",
            f"model_{model_name.replace('/', '_')}",
        ]
        if val_size is not None:
            components.append(f"val_{val_size}")

        return "_".join(components)

    def _get_cache_path(self, key: str) -> Path:
        """Get the path for a cache file."""
        return self.cache_dir / f"{key}.npz"

    def _get_meta_path(self, key: str) -> Path:
        """Get the path for metadata file."""
        return self.cache_dir / f"{key}_meta.json"

    def load(
        self,
        data_path: str,
        seed: int,
        split: str,
        model_name: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Load embeddings from cache if available.

        Returns:
            np.ndarray if cache hit, None if cache miss
        """
        key = self._make_key(data_path, seed, split, model_name, test_size, val_size)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                data = np.load(cache_path)
                embeddings = data["embeddings"]
                self.stats["hits"] += 1
                print(f"  [Cache HIT] Loaded {split} embeddings: {embeddings.shape}")
                return embeddings
            except Exception as e:
                print(f"  [Cache ERROR] Failed to load {cache_path}: {e}")
                return None

        self.stats["misses"] += 1
        print(f"  [Cache MISS] No cached {split} embeddings for seed {seed}")
        return None

    def save(
        self,
        embeddings: np.ndarray,
        data_path: str,
        seed: int,
        split: str,
        model_name: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> None:
        """Save embeddings to cache."""
        key = self._make_key(data_path, seed, split, model_name, test_size, val_size)
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        try:
            # Save embeddings
            np.savez_compressed(cache_path, embeddings=embeddings)

            # Save metadata
            meta = {
                "data_path": data_path,
                "seed": seed,
                "split": split,
                "model_name": model_name,
                "test_size": test_size,
                "val_size": val_size,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            self.stats["saves"] += 1
            size_mb = cache_path.stat().st_size / 1024 / 1024
            print(f"  [Cache SAVE] Saved {split} embeddings: {embeddings.shape} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [Cache ERROR] Failed to save {cache_path}: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.stats.copy()

    def clear(self) -> int:
        """Clear all cached embeddings. Returns number of files deleted."""
        count = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            count += 1
        for f in self.cache_dir.glob("*_meta.json"):
            f.unlink()
        return count

    def list_cached(self) -> Dict[str, Any]:
        """List all cached embeddings."""
        cached = {}
        for meta_path in self.cache_dir.glob("*_meta.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                key = meta_path.stem.replace("_meta", "")
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    meta["size_mb"] = cache_path.stat().st_size / 1024 / 1024
                    cached[key] = meta
            except Exception:
                pass
        return cached


def get_or_compute_embeddings(
    cache: EmbeddingCache,
    embedder,
    texts,
    data_path: str,
    seed: int,
    split: str,
    model_name: str,
    batch_size: int = 128,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
) -> np.ndarray:
    """
    Get embeddings from cache or compute them.

    This is the main function to use for cached embeddings.
    """
    # Try cache first
    embeddings = cache.load(data_path, seed, split, model_name, test_size, val_size)

    if embeddings is not None:
        return embeddings

    # Compute embeddings
    print(f"  Computing {split} embeddings...")
    embeddings = embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save to cache
    cache.save(embeddings, data_path, seed, split, model_name, test_size, val_size)

    return embeddings


# Demo/test
if __name__ == "__main__":
    print("=" * 60)
    print("  Embedding Cache Demo")
    print("=" * 60)

    cache = EmbeddingCache("test_cache")

    # Simulate saving
    fake_embeddings = np.random.randn(1000, 768).astype(np.float32)
    cache.save(
        fake_embeddings,
        data_path="../MBTI_500.csv",
        seed=42,
        split="train",
        model_name="all-mpnet-base-v2",
    )

    # Load back
    loaded = cache.load(
        data_path="../MBTI_500.csv",
        seed=42,
        split="train",
        model_name="all-mpnet-base-v2",
    )

    if loaded is not None:
        print(f"\nLoaded embeddings match: {np.allclose(fake_embeddings, loaded)}")

    print(f"\nCache stats: {cache.get_stats()}")
    print(f"\nCached files: {list(cache.list_cached().keys())}")

    # Cleanup test
    import shutil
    shutil.rmtree("test_cache")
    print("\nTest cache cleaned up.")
