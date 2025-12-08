#!/usr/bin/env python3
"""
Unified Validation Runner for Phase F Experiments

Features:
- K-Fold CV (5×3 = 15 folds) for all experiments
- Embedding cache (load once, reuse always)
- Parallel API calls (10 concurrent)
- Unbuffered logging support
- Statistical analysis (mean, std, CI, p-value, win rate)
"""

import os
import sys
import json
import pickle
import asyncio
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings('ignore')

# Import base config
from base_config import (
    PROJECT_ROOT, DATA_PATH, CACHE_DIR, RESULTS_DIR,
    EMBEDDING_MODEL, EMBEDDING_CACHE_PATH, LABELS_CACHE_PATH, TEXTS_CACHE_PATH,
    LLM_MODEL, MAX_CONCURRENT_API_CALLS, TEMPERATURE, MAX_TOKENS,
    BASE_PARAMS, KFOLD_CONFIG, MBTI_CLASSES
)


@dataclass
class KFoldResult:
    """Results from K-Fold evaluation."""
    config_name: str
    n_folds: int
    baseline_mean: float
    baseline_std: float
    augmented_mean: float
    augmented_std: float
    delta_mean: float
    delta_std: float
    delta_pct: float
    ci_95_lower: float
    ci_95_upper: float
    t_statistic: float
    p_value: float
    significant: bool
    win_rate: float
    n_synthetic: int
    acceptance_rate: Optional[float] = None
    silhouette: Optional[float] = None
    coherence: Optional[float] = None
    quality: Optional[float] = None
    diversity: Optional[float] = None
    contamination: Optional[float] = None
    per_class_delta: Optional[Dict[str, float]] = None


class EmbeddingCache:
    """Manages embedding cache for fast reuse."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._embeddings = None
        self._labels = None
        self._texts = None

    def load_or_compute(self, texts: List[str], labels: np.ndarray) -> np.ndarray:
        """Load embeddings from cache or compute them."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if EMBEDDING_CACHE_PATH.exists() and LABELS_CACHE_PATH.exists():
            print(f"  Loading cached embeddings from {EMBEDDING_CACHE_PATH}")
            self._embeddings = np.load(EMBEDDING_CACHE_PATH)
            self._labels = np.load(LABELS_CACHE_PATH, allow_pickle=True)
            with open(TEXTS_CACHE_PATH, 'rb') as f:
                self._texts = pickle.load(f)
            print(f"  Loaded {len(self._embeddings)} embeddings from cache")
        else:
            print(f"  Computing embeddings with {self.model_name}...")
            if self.model is None:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            self._embeddings = self.model.encode(
                texts, show_progress_bar=True, convert_to_numpy=True, batch_size=128
            )
            self._labels = labels
            self._texts = texts

            # Save to cache
            np.save(EMBEDDING_CACHE_PATH, self._embeddings)
            np.save(LABELS_CACHE_PATH, self._labels)
            with open(TEXTS_CACHE_PATH, 'wb') as f:
                pickle.dump(self._texts, f)
            print(f"  Saved {len(self._embeddings)} embeddings to cache")

        return self._embeddings

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def texts(self) -> List[str]:
        return self._texts

    def embed_synthetic(self, texts: List[str]) -> np.ndarray:
        """Embed synthetic texts (not cached)."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


class ParallelLLMGenerator:
    """Async parallel LLM generator with rate limiting."""

    def __init__(self, model: str = LLM_MODEL, max_concurrent: int = MAX_CONCURRENT_API_CALLS):
        self.model = model
        self.max_concurrent = max_concurrent
        self.client = None

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts in parallel."""
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_one(prompt: str) -> str:
            async with semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"  API error: {e}")
                    return ""

        tasks = [generate_one(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        return results

    def generate_sync(self, prompts: List[str]) -> List[str]:
        """Synchronous wrapper for async generation."""
        return asyncio.run(self.generate_batch(prompts))


class KFoldEvaluator:
    """K-Fold cross-validation evaluator."""

    def __init__(
        self,
        n_splits: int = KFOLD_CONFIG["n_splits"],
        n_repeats: int = KFOLD_CONFIG["n_repeats"],
        random_state: int = KFOLD_CONFIG["random_state"],
        synthetic_weight: float = BASE_PARAMS["synthetic_weight"]
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.synthetic_weight = synthetic_weight
        self.total_folds = n_splits * n_repeats

    def evaluate(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_synthetic: Optional[np.ndarray] = None,
        y_synthetic: Optional[np.ndarray] = None,
        config_name: str = "unknown"
    ) -> KFoldResult:
        """Run K-Fold evaluation and compute statistics."""

        if self.n_repeats > 1:
            kfold = RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
        else:
            kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )

        baseline_f1s = []
        augmented_f1s = []
        per_class_baselines = {c: [] for c in np.unique(y_original)}
        per_class_augmented = {c: [] for c in np.unique(y_original)}

        print(f"\n  Running {self.n_splits}-fold × {self.n_repeats} repeats = {self.total_folds} folds")

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
            X_train, X_test = X_original[train_idx], X_original[test_idx]
            y_train, y_test = y_original[train_idx], y_original[test_idx]

            # Baseline: train on original only
            clf_baseline = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf_baseline.fit(X_train, y_train)
            y_pred_baseline = clf_baseline.predict(X_test)
            baseline_f1 = f1_score(y_test, y_pred_baseline, average="macro")
            baseline_f1s.append(baseline_f1)

            # Per-class F1 for baseline
            baseline_per_class = f1_score(y_test, y_pred_baseline, average=None, labels=np.unique(y_original))
            for i, c in enumerate(np.unique(y_original)):
                per_class_baselines[c].append(baseline_per_class[i] if i < len(baseline_per_class) else 0)

            # Augmented: train on original + synthetic
            if X_synthetic is not None and len(X_synthetic) > 0:
                # Create sample weights (down-weight synthetics)
                n_train = len(X_train)
                n_synth = len(X_synthetic)
                weights = np.concatenate([
                    np.ones(n_train),
                    np.full(n_synth, self.synthetic_weight)
                ])

                X_aug = np.vstack([X_train, X_synthetic])
                y_aug = np.concatenate([y_train, y_synthetic])

                clf_augmented = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf_augmented.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_augmented.predict(X_test)
                augmented_f1 = f1_score(y_test, y_pred_aug, average="macro")

                # Per-class F1 for augmented
                aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=np.unique(y_original))
                for i, c in enumerate(np.unique(y_original)):
                    per_class_augmented[c].append(aug_per_class[i] if i < len(aug_per_class) else 0)
            else:
                augmented_f1 = baseline_f1
                for c in np.unique(y_original):
                    per_class_augmented[c].append(per_class_baselines[c][-1])

            augmented_f1s.append(augmented_f1)

            # Print progress for every fold
            print(f"    Fold {fold_idx + 1}/{self.total_folds}: "
                  f"baseline={baseline_f1:.4f}, aug={augmented_f1:.4f}, "
                  f"delta={augmented_f1 - baseline_f1:+.4f}", flush=True)

        # Compute statistics
        baseline_arr = np.array(baseline_f1s)
        augmented_arr = np.array(augmented_f1s)
        deltas = augmented_arr - baseline_arr

        baseline_mean = np.mean(baseline_arr)
        baseline_std = np.std(baseline_arr, ddof=1)
        augmented_mean = np.mean(augmented_arr)
        augmented_std = np.std(augmented_arr, ddof=1)
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas, ddof=1)
        delta_pct = (delta_mean / baseline_mean) * 100 if baseline_mean > 0 else 0

        # Confidence interval and t-test
        n = len(deltas)
        se = delta_std / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
        t_stat, p_value = stats.ttest_1samp(deltas, 0)

        # Win rate
        win_rate = np.mean(deltas > 0)

        # Per-class delta
        per_class_delta = {}
        for c in np.unique(y_original):
            baseline_class_mean = np.mean(per_class_baselines[c])
            augmented_class_mean = np.mean(per_class_augmented[c])
            delta_class = augmented_class_mean - baseline_class_mean
            per_class_delta[str(c)] = float(delta_class)

        n_synthetic = len(X_synthetic) if X_synthetic is not None else 0

        return KFoldResult(
            config_name=config_name,
            n_folds=self.total_folds,
            baseline_mean=float(baseline_mean),
            baseline_std=float(baseline_std),
            augmented_mean=float(augmented_mean),
            augmented_std=float(augmented_std),
            delta_mean=float(delta_mean),
            delta_std=float(delta_std),
            delta_pct=float(delta_pct),
            ci_95_lower=float(ci_95[0]),
            ci_95_upper=float(ci_95[1]),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            significant=bool(p_value < 0.05),
            win_rate=float(win_rate),
            n_synthetic=n_synthetic,
            per_class_delta=per_class_delta
        )


def load_data() -> Tuple[List[str], np.ndarray]:
    """Load MBTI dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Handle both column naming conventions
    if 'posts' in df.columns:
        texts = df['posts'].tolist()
        labels = df['type'].values
    else:
        texts = df['text'].tolist()
        labels = df['label'].values

    print(f"  Loaded {len(texts)} samples, {len(np.unique(labels))} classes")
    return texts, labels


def compute_silhouette(embeddings: np.ndarray, k_clusters: int) -> float:
    """Compute silhouette score for K-means clustering."""
    if k_clusters <= 1 or k_clusters >= len(embeddings):
        return float('nan')

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    return silhouette_score(embeddings, cluster_labels)


def compute_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_clusters: int,
    target_class: str
) -> float:
    """
    Compute intra-cluster coherence for a specific class.
    Coherence = % of cluster neighbors that belong to the same class.
    """
    class_mask = labels == target_class
    class_embeddings = embeddings[class_mask]

    if len(class_embeddings) < k_clusters:
        return 0.0

    kmeans = KMeans(n_clusters=max(1, k_clusters), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(class_embeddings)

    # For each cluster, compute purity
    purities = []
    for cluster_id in range(k_clusters):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() == 0:
            continue

        # All points in cluster belong to target class by construction
        # So purity within class is 1.0
        # But we measure coherence as how tight the cluster is
        cluster_points = class_embeddings[cluster_mask]
        if len(cluster_points) > 1:
            centroid = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            # Normalize by max possible distance
            coherence = 1.0 / (1.0 + dists.mean())
            purities.append(coherence)

    return np.mean(purities) if purities else 0.0


def save_result(result: KFoldResult, experiment_name: str):
    """Save result to JSON file."""
    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{result.config_name}_kfold.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"  Saved results to {output_path}")


def print_summary(result: KFoldResult):
    """Print formatted result summary."""
    print(f"\n{'='*60}")
    print(f"  Config: {result.config_name}")
    print(f"  Folds: {result.n_folds}")
    print(f"  Baseline:  {result.baseline_mean:.4f} +/- {result.baseline_std:.4f}")
    print(f"  Augmented: {result.augmented_mean:.4f} +/- {result.augmented_std:.4f}")
    print(f"  Delta:     {result.delta_mean:+.4f} ({result.delta_pct:+.2f}%)")
    print(f"  95% CI:    [{result.ci_95_lower:+.4f}, {result.ci_95_upper:+.4f}]")
    print(f"  p-value:   {result.p_value:.6f} {'*' if result.significant else ''}")
    print(f"  Win rate:  {result.win_rate*100:.1f}%")
    print(f"  Synthetics: {result.n_synthetic}")
    print(f"{'='*60}\n")


class LLMSyntheticGenerator:
    """Generate synthetic samples using LLM."""

    def __init__(self, cache: EmbeddingCache):
        from openai import OpenAI
        self.client = OpenAI()
        self.cache = cache

    def create_prompt(self, examples: List[str], target_class: str, n_samples: int = 5) -> str:
        """Create generation prompt from examples."""
        examples_text = "\n".join([
            f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
            for ex in examples[:5]
        ])

        return f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_samples} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

    def generate_for_class(
        self,
        class_texts: np.ndarray,
        class_embeddings: np.ndarray,
        target_class: str,
        k_clusters: int = 5,
        samples_per_cluster: int = 5
    ) -> Tuple[List[str], List[str]]:
        """Generate synthetic samples for one class using clustering."""

        if len(class_embeddings) < 10:
            return [], []

        # Determine actual number of clusters
        k_actual = min(k_clusters, max(1, len(class_embeddings) // 60))
        if k_actual < 1:
            k_actual = 1

        # Cluster the class
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        synthetic_texts = []
        synthetic_labels = []

        for c_id in range(k_actual):
            c_mask = cluster_labels == c_id
            if c_mask.sum() < 3:
                continue

            cluster_texts = class_texts[c_mask]
            cluster_embs = class_embeddings[c_mask]

            # Select examples closest to centroid
            centroid = kmeans.cluster_centers_[c_id]
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [cluster_texts[i] for i in nearest_idx]

            # Generate
            prompt = self.create_prompt(example_texts, str(target_class), samples_per_cluster)

            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS * samples_per_cluster,
                )

                generated_text = response.choices[0].message.content.strip()
                samples = [s.strip() for s in generated_text.split('\n')
                          if s.strip() and len(s.strip()) > 10]

                for sample in samples[:samples_per_cluster]:
                    synthetic_texts.append(sample)
                    synthetic_labels.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

        return synthetic_texts, synthetic_labels

    def generate_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        k_clusters: int = 5,
        samples_per_cluster: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples for all classes."""

        texts_array = np.array(texts)
        all_synthetic_texts = []
        all_synthetic_labels = []

        for target_class in np.unique(labels):
            class_mask = labels == target_class
            class_embeddings = embeddings[class_mask]
            class_texts = texts_array[class_mask]

            synth_texts, synth_labels = self.generate_for_class(
                class_texts, class_embeddings, target_class,
                k_clusters, samples_per_cluster
            )

            all_synthetic_texts.extend(synth_texts)
            all_synthetic_labels.extend(synth_labels)

        if not all_synthetic_texts:
            return np.array([]).reshape(0, embeddings.shape[1]), np.array([])

        # Embed all synthetic texts
        print(f"    Embedding {len(all_synthetic_texts)} synthetic texts...", flush=True)
        synthetic_embeddings = self.cache.embed_synthetic(all_synthetic_texts)

        return synthetic_embeddings, np.array(all_synthetic_labels)


if __name__ == "__main__":
    print("Phase F Validation Runner")
    print("="*60)
    print(f"  K-Fold config: {KFOLD_CONFIG['n_splits']} splits × {KFOLD_CONFIG['n_repeats']} repeats")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  LLM model: {LLM_MODEL}")
    print(f"  Max parallel API calls: {MAX_CONCURRENT_API_CALLS}")
    print("="*60)

    # Load data and cache embeddings
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    print(f"\nReady for experiments.")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
