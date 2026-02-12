#!/usr/bin/env python3
"""
Phase I Robust Validation Runner

Based on Phase H, with added support for:
- Fixed N samples per class (fair comparisons)
- Oversample-then-select strategy
- Multiple selection strategies (similarity, diverse, random)

Features:
- K-Fold CV (5 splits x 3 repeats = 15 folds)
- High-concurrency API calls (25 concurrent)
- GPU-accelerated embeddings
- N-shot prompting support
- Variable temperature support
- Incremental result saving
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
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# Import base config
from core.base_config import (
    PROJECT_ROOT, DATA_PATH, CACHE_DIR, RESULTS_DIR,
    EMBEDDING_MODEL, EMBEDDING_CACHE_PATH, LABELS_CACHE_PATH, TEXTS_CACHE_PATH,
    LLM_MODEL, LLM_PROVIDER, MAX_CONCURRENT_API_CALLS, MAX_TOKENS,
    BASE_PARAMS, KFOLD_CONFIG, MBTI_CLASSES, PROBLEM_CLASSES,
    DEVICE, EMBEDDING_BATCH_SIZE, TOTAL_FOLDS
)

# ==============================================================================
# Hardware-Optimized Settings
# ==============================================================================
MAX_CONCURRENT_API = 25
EMBEDDING_BATCH = 256


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class KFoldResult:
    """Results from K-Fold evaluation."""
    config_name: str
    config_params: Dict[str, Any]
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
    method: str = "smote_llm"  # Method identifier
    per_class_delta: Optional[Dict[str, float]] = None
    problem_class_delta: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# Embedding Cache
# ==============================================================================
class EmbeddingCache:
    """Manages embedding cache with GPU acceleration."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
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
            print(f"  Computing embeddings with {self.model_name} on {self.device}...")
            if self.model is None:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            self._embeddings = self.model.encode(
                texts, show_progress_bar=True, convert_to_numpy=True,
                batch_size=EMBEDDING_BATCH
            )
            self._labels = labels
            self._texts = texts

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
        if not texts:
            return np.array([]).reshape(0, self._embeddings.shape[1])
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True,
            batch_size=EMBEDDING_BATCH
        )


# ==============================================================================
# Async LLM Generator (Multi-Provider Support)
# ==============================================================================
class AsyncLLMGenerator:
    """
    Async parallel LLM generator with multi-provider support.

    Supports:
    - OpenAI (gpt-4o-mini, etc.) - native async
    - Google Gemini 3 Flash - via asyncio.to_thread
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        max_concurrent: int = MAX_CONCURRENT_API,
        temperature: float = 0.7,
        provider: str = "openai"
    ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.provider = provider.lower()

        # Initialize provider for Gemini (lazy load for OpenAI)
        self._gemini_provider = None
        if self.provider in ["google", "gemini"]:
            from core.llm_providers import create_provider
            self._gemini_provider = create_provider("google", model)

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts in parallel."""

        if self.provider == "openai":
            return await self._generate_openai_batch(prompts)
        else:
            return await self._generate_gemini_batch(prompts)

    async def _generate_openai_batch(self, prompts: List[str]) -> List[str]:
        """Generate using OpenAI with native async."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_one(prompt: str) -> str:
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=MAX_TOKENS,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"  OpenAI API error: {e}")
                    return ""

        try:
            tasks = [generate_one(p) for p in prompts]
            results = await asyncio.gather(*tasks)
        finally:
            await client.close()

        return results

    async def _generate_gemini_batch(self, prompts: List[str]) -> List[str]:
        """Generate using Gemini via asyncio.to_thread."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_one(prompt: str) -> str:
            async with semaphore:
                try:
                    # Run synchronous Gemini call in thread pool
                    result = await asyncio.to_thread(
                        self._gemini_generate_sync,
                        prompt
                    )
                    return result
                except Exception as e:
                    print(f"  Gemini API error: {e}")
                    return ""

        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def _gemini_generate_sync(self, prompt: str) -> str:
        """Synchronous Gemini generation (called in thread)."""
        messages = [{"role": "user", "content": prompt}]
        text, _ = self._gemini_provider.generate(
            messages,
            temperature=self.temperature,
            max_tokens=MAX_TOKENS
        )
        return text.strip() if text else ""

    def generate_sync(self, prompts: List[str]) -> List[str]:
        """Synchronous wrapper for async generation."""
        return asyncio.run(self.generate_batch(prompts))


# ==============================================================================
# Synthetic Generator with Fixed-N Support
# ==============================================================================
class SyntheticGenerator:
    """Generate synthetic samples using LLM with fixed-N support."""

    def __init__(self, cache: EmbeddingCache, config_params: Dict[str, Any]):
        self.cache = cache
        self.params = config_params
        self.llm = AsyncLLMGenerator(
            model=config_params.get("llm_model", LLM_MODEL),
            temperature=config_params.get("temperature", 0.7),
            max_concurrent=MAX_CONCURRENT_API,
            provider=config_params.get("llm_provider", "openai")
        )

    def create_prompt(
        self,
        examples: List[str],
        target_class: str,
        n_samples: int = 5
    ) -> str:
        """Create generation prompt with n-shot support."""
        n_shot = self.params.get("n_shot", 0)

        if n_shot == 0:
            return f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type (MBTI).

Generate {n_samples} new, unique posts. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

        examples_to_use = examples[:n_shot]
        examples_text = "\n".join([
            f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
            for ex in examples_to_use
        ])

        return f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are {len(examples_to_use)} examples of posts from this personality type:
{examples_text}

Generate {n_samples} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

    def generate_for_class(
        self,
        class_texts: np.ndarray,
        class_embeddings: np.ndarray,
        target_class: str
    ) -> Tuple[List[str], List[str]]:
        """Generate synthetic samples for one class."""
        fixed_n = self.params.get("fixed_n_per_class", None)

        if fixed_n is not None:
            return self._generate_fixed_n(
                class_texts, class_embeddings, target_class, target_n=fixed_n
            )
        return self._generate_variable(class_texts, class_embeddings, target_class)

    def _generate_fixed_n(
        self,
        class_texts: np.ndarray,
        class_embeddings: np.ndarray,
        target_class: str,
        target_n: int
    ) -> Tuple[List[str], List[str]]:
        """Generate exactly target_n samples using oversample-then-select strategy."""
        oversample = self.params.get("oversample_factor", 1.5)
        generate_target = int(target_n * oversample)

        # Generate more than needed
        all_synthetic, all_labels = self._generate_batch(
            class_texts, class_embeddings, target_class, min_samples=generate_target
        )

        if len(all_synthetic) < target_n:
            print(f"  Warning: Only generated {len(all_synthetic)}/{target_n} for {target_class}")
            return all_synthetic, all_labels

        # Select exactly target_n using strategy
        return self._select_top_n(all_synthetic, all_labels, class_embeddings, target_n)

    def _select_top_n(
        self,
        synthetic_texts: List[str],
        synthetic_labels: List[str],
        real_embeddings: np.ndarray,
        target_n: int
    ) -> Tuple[List[str], List[str]]:
        """Select top N samples using configured strategy."""
        strategy = self.params.get("fixed_n_selection_strategy", "similarity")

        # Embed synthetic samples
        synth_embeddings = self.cache.embed_synthetic(synthetic_texts)

        if strategy == "random":
            indices = np.random.choice(len(synthetic_texts), target_n, replace=False)

        elif strategy == "similarity":
            # Select samples most similar to real class centroid
            centroid = real_embeddings.mean(axis=0)
            similarities = cosine_similarity(synth_embeddings, [centroid]).flatten()
            indices = np.argsort(similarities)[-target_n:]

        elif strategy == "diverse":
            # Maximin diversity selection (farthest point sampling)
            indices = self._maximin_selection(synth_embeddings, target_n)

        else:
            # Default to similarity
            centroid = real_embeddings.mean(axis=0)
            similarities = cosine_similarity(synth_embeddings, [centroid]).flatten()
            indices = np.argsort(similarities)[-target_n:]

        return [synthetic_texts[i] for i in indices], [synthetic_labels[i] for i in indices]

    def _maximin_selection(self, embeddings: np.ndarray, n: int) -> List[int]:
        """Select n diverse samples using farthest point sampling."""
        if len(embeddings) <= n:
            return list(range(len(embeddings)))

        selected = [0]  # Start with first sample

        for _ in range(n - 1):
            max_min_dist = -1
            best_idx = -1

            for i in range(len(embeddings)):
                if i in selected:
                    continue
                # Minimum distance to any selected point
                min_dist = min(
                    np.linalg.norm(embeddings[i] - embeddings[j])
                    for j in selected
                )
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)

        return selected

    def _generate_batch(
        self,
        class_texts: np.ndarray,
        class_embeddings: np.ndarray,
        target_class: str,
        min_samples: int = None
    ) -> Tuple[List[str], List[str]]:
        """Generate a batch of samples (internal helper)."""
        max_clusters = self.params.get("max_clusters", 12)
        prompts_per_cluster = self.params.get("prompts_per_cluster", 9)
        samples_per_prompt = self.params.get("samples_per_prompt", 5)
        n_shot = self.params.get("n_shot", 0)
        k_neighbors = self.params.get("k_neighbors", 50)
        anchor_strategy = self.params.get("anchor_strategy", "medoid")

        if len(class_embeddings) < 10:
            return [], []

        # Determine actual number of clusters
        k_actual = min(max_clusters, max(1, len(class_embeddings) // 30))
        if len(class_embeddings) < 30:
            k_actual = 1

        # If we need more samples, increase prompts_per_cluster
        if min_samples is not None:
            expected_samples = k_actual * prompts_per_cluster * samples_per_prompt * 0.8
            if expected_samples < min_samples:
                prompts_per_cluster = max(
                    prompts_per_cluster,
                    int(np.ceil(min_samples / (k_actual * samples_per_prompt * 0.8)))
                )

        # Cluster the class
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        all_prompts = []

        for c_id in range(k_actual):
            c_mask = cluster_labels == c_id
            if c_mask.sum() < 3:
                continue

            cluster_texts = class_texts[c_mask]
            cluster_embs = class_embeddings[c_mask]

            # Select examples based on anchor strategy
            if anchor_strategy == "medoid":
                centroid = kmeans.cluster_centers_[c_id]
                dists = np.linalg.norm(cluster_embs - centroid, axis=1)
                n_examples = min(k_neighbors, n_shot if n_shot > 0 else 5, len(cluster_texts))
                nearest_idx = np.argsort(dists)[:n_examples]
                example_texts = [cluster_texts[i] for i in nearest_idx]
            elif anchor_strategy == "random":
                n_examples = min(n_shot if n_shot > 0 else 5, len(cluster_texts))
                idx = np.random.choice(len(cluster_texts), n_examples, replace=False)
                example_texts = [cluster_texts[i] for i in idx]
            elif anchor_strategy == "diverse":
                n_examples = min(n_shot if n_shot > 0 else 5, len(cluster_texts))
                selected = [0]
                for _ in range(n_examples - 1):
                    if len(selected) >= len(cluster_embs):
                        break
                    max_min_dist = -1
                    best_idx = -1
                    for i in range(len(cluster_embs)):
                        if i in selected:
                            continue
                        min_dist = min(np.linalg.norm(cluster_embs[i] - cluster_embs[j])
                                      for j in selected)
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            best_idx = i
                    if best_idx >= 0:
                        selected.append(best_idx)
                example_texts = [cluster_texts[i] for i in selected]
            else:
                centroid = kmeans.cluster_centers_[c_id]
                dists = np.linalg.norm(cluster_embs - centroid, axis=1)
                n_examples = min(k_neighbors, n_shot if n_shot > 0 else 5, len(cluster_texts))
                nearest_idx = np.argsort(dists)[:n_examples]
                example_texts = [cluster_texts[i] for i in nearest_idx]

            for _ in range(prompts_per_cluster):
                prompt = self.create_prompt(example_texts, str(target_class), samples_per_prompt)
                all_prompts.append(prompt)

        if not all_prompts:
            return [], []

        # Generate all prompts in parallel
        responses = self.llm.generate_sync(all_prompts)

        synthetic_texts = []
        synthetic_labels = []

        for response in responses:
            if not response:
                continue
            samples = [s.strip() for s in response.split('\n')
                      if s.strip() and len(s.strip()) > 10]

            for sample in samples[:samples_per_prompt]:
                synthetic_texts.append(sample)
                synthetic_labels.append(target_class)

        return synthetic_texts, synthetic_labels

    def _generate_variable(
        self,
        class_texts: np.ndarray,
        class_embeddings: np.ndarray,
        target_class: str
    ) -> Tuple[List[str], List[str]]:
        """Generate variable number of samples (original behavior)."""
        return self._generate_batch(class_texts, class_embeddings, target_class)

    def generate_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic samples for all classes."""
        texts_array = np.array(texts)
        all_synthetic_texts = []
        all_synthetic_labels = []

        for target_class in np.unique(labels):
            class_mask = labels == target_class
            class_embeddings = embeddings[class_mask]
            class_texts = texts_array[class_mask]

            print(f"    {target_class}: {class_mask.sum()} samples...", end=" ", flush=True)

            synth_texts, synth_labels = self.generate_for_class(
                class_texts, class_embeddings, target_class
            )

            all_synthetic_texts.extend(synth_texts)
            all_synthetic_labels.extend(synth_labels)
            print(f"-> {len(synth_texts)} synthetic", flush=True)

        if not all_synthetic_texts:
            return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), []

        print(f"    Embedding {len(all_synthetic_texts)} synthetic texts...", flush=True)
        synthetic_embeddings = self.cache.embed_synthetic(all_synthetic_texts)

        return synthetic_embeddings, np.array(all_synthetic_labels), all_synthetic_texts


# ==============================================================================
# K-Fold Evaluator
# ==============================================================================
class KFoldEvaluator:
    """K-Fold cross-validation evaluator with statistical analysis."""

    def __init__(
        self,
        n_splits: int = KFOLD_CONFIG["n_splits"],
        n_repeats: int = KFOLD_CONFIG["n_repeats"],
        random_state: int = KFOLD_CONFIG["random_state"]
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.total_folds = n_splits * n_repeats

    def evaluate(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_synthetic: Optional[np.ndarray] = None,
        y_synthetic: Optional[np.ndarray] = None,
        synthetic_weight: float = 1.0,
        config_name: str = "unknown",
        config_params: dict = None,
        method: str = "smote_llm"
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

        print(f"\n  K-Fold: {self.n_splits} splits x {self.n_repeats} repeats = {self.total_folds} folds")

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
            X_train, X_test = X_original[train_idx], X_original[test_idx]
            y_train, y_test = y_original[train_idx], y_original[test_idx]

            # Baseline
            clf_baseline = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
            clf_baseline.fit(X_train, y_train)
            y_pred_baseline = clf_baseline.predict(X_test)
            baseline_f1 = f1_score(y_test, y_pred_baseline, average="macro")
            baseline_f1s.append(baseline_f1)

            unique_classes = np.unique(y_original)
            baseline_per_class = f1_score(y_test, y_pred_baseline, average=None, labels=unique_classes)
            for i, c in enumerate(unique_classes):
                per_class_baselines[c].append(baseline_per_class[i] if i < len(baseline_per_class) else 0)

            # Augmented
            if X_synthetic is not None and len(X_synthetic) > 0:
                n_train = len(X_train)
                n_synth = len(X_synthetic)
                weights = np.concatenate([
                    np.ones(n_train),
                    np.full(n_synth, synthetic_weight)
                ])

                X_aug = np.vstack([X_train, X_synthetic])
                y_aug = np.concatenate([y_train, y_synthetic])

                clf_augmented = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
                clf_augmented.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_augmented.predict(X_test)
                augmented_f1 = f1_score(y_test, y_pred_aug, average="macro")

                aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=unique_classes)
                for i, c in enumerate(unique_classes):
                    per_class_augmented[c].append(aug_per_class[i] if i < len(aug_per_class) else 0)
            else:
                augmented_f1 = baseline_f1
                for c in unique_classes:
                    per_class_augmented[c].append(per_class_baselines[c][-1])

            augmented_f1s.append(augmented_f1)

            print(f"    Fold {fold_idx + 1}/{self.total_folds}: "
                  f"base={baseline_f1:.4f}, aug={augmented_f1:.4f}, "
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

        n = len(deltas)
        se = delta_std / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
        t_stat, p_value = stats.ttest_1samp(deltas, 0)

        win_rate = np.mean(deltas > 0)

        per_class_delta = {}
        problem_class_delta = {}
        for c in np.unique(y_original):
            baseline_class_mean = np.mean(per_class_baselines[c])
            augmented_class_mean = np.mean(per_class_augmented[c])
            delta_class = augmented_class_mean - baseline_class_mean
            per_class_delta[str(c)] = float(delta_class)
            if str(c) in PROBLEM_CLASSES:
                problem_class_delta[str(c)] = float(delta_class)

        n_synthetic = len(X_synthetic) if X_synthetic is not None else 0

        return KFoldResult(
            config_name=config_name,
            config_params=config_params or {},
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
            method=method,
            per_class_delta=per_class_delta,
            problem_class_delta=problem_class_delta,
            timestamp=datetime.now().isoformat()
        )


# ==============================================================================
# Utility Functions
# ==============================================================================
def load_data() -> Tuple[List[str], np.ndarray]:
    """Load MBTI dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    if 'posts' in df.columns:
        texts = df['posts'].tolist()
        labels = df['type'].values
    else:
        texts = df['text'].tolist()
        labels = df['label'].values

    print(f"  Loaded {len(texts)} samples, {len(np.unique(labels))} classes")
    return texts, labels


def save_result(result: KFoldResult, experiment_name: str, config_name: str = None):
    """Save result to JSON file."""
    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = config_name or result.config_name
    output_path = output_dir / f"{filename}.json"

    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"  Saved: {output_path}")
    return output_path


def save_results_all(results: List[KFoldResult], experiment_name: str):
    """Save all results to a single JSON file."""
    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "all_results.json"

    results_dict = [r.to_dict() for r in results]
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"  Saved all results to: {output_path}")


def print_summary(result: KFoldResult):
    """Print formatted result summary."""
    sig_marker = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.significant else ""

    print(f"\n{'='*60}")
    print(f"  Config: {result.config_name}")
    print(f"  Method: {result.method}")
    print(f"  Folds: {result.n_folds}")
    print(f"  Baseline:  {result.baseline_mean:.4f} +/- {result.baseline_std:.4f}")
    print(f"  Augmented: {result.augmented_mean:.4f} +/- {result.augmented_std:.4f}")
    print(f"  Delta:     {result.delta_mean:+.4f} ({result.delta_pct:+.2f}%)")
    print(f"  95% CI:    [{result.ci_95_lower:+.4f}, {result.ci_95_upper:+.4f}]")
    print(f"  p-value:   {result.p_value:.6f} {sig_marker}")
    print(f"  Win rate:  {result.win_rate*100:.1f}%")
    print(f"  Synthetics: {result.n_synthetic}")
    print(f"{'='*60}\n")


def run_experiment(
    config_name: str,
    config_params: dict,
    cache: EmbeddingCache,
    evaluator: KFoldEvaluator,
    experiment_name: str
) -> KFoldResult:
    """Run a single SMOTE-LLM experiment configuration."""

    print(f"\n{'#'*60}")
    print(f"# Running: {config_name}")
    print(f"# Params: n_shot={config_params.get('n_shot', 0)}, "
          f"temp={config_params.get('temperature', 0.7)}, "
          f"fixed_n={config_params.get('fixed_n_per_class', 'variable')}")
    print(f"{'#'*60}")

    generator = SyntheticGenerator(cache, config_params)
    X_synth, y_synth, texts_synth = generator.generate_all(
        cache.embeddings, cache.labels, cache.texts
    )

    synthetic_weight = config_params.get("synthetic_weight", 1.0)
    result = evaluator.evaluate(
        cache.embeddings, cache.labels,
        X_synth, y_synth,
        synthetic_weight=synthetic_weight,
        config_name=config_name,
        config_params=config_params,
        method="smote_llm"
    )

    print_summary(result)
    save_result(result, experiment_name, config_name)

    return result


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print("Phase I Robust Validation Runner")
    print("=" * 60)
    print(f"  K-Fold: {KFOLD_CONFIG['n_splits']} splits x {KFOLD_CONFIG['n_repeats']} repeats")
    print(f"  Embedding: {EMBEDDING_MODEL}")
    print(f"  LLM: {LLM_MODEL}")
    print(f"  Max concurrent API: {MAX_CONCURRENT_API}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    print(f"\nReady for experiments.")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Labels: {labels.shape}")
