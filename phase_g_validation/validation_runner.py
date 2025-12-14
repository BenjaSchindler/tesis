#!/usr/bin/env python3
"""
Phase G Validation Runner

Features:
- K-Fold CV (5 splits x 3 repeats = 15 folds)
- Parallel API calls (25 concurrent - high OpenAI tier)
- GPU-accelerated embeddings (RTX 3090)
- Config-aware parameter handling
- Statistical analysis (mean, std, CI, p-value, win rate)
- Per-class F1 tracking for problem classes
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings('ignore')

# Import configs
from base_config import (
    PROJECT_ROOT, DATA_PATH, CACHE_DIR, RESULTS_DIR, LATEX_DIR,
    EMBEDDING_MODEL, EMBEDDING_CACHE_PATH, LABELS_CACHE_PATH, TEXTS_CACHE_PATH,
    LLM_MODEL, MAX_CONCURRENT_API_CALLS, TEMPERATURE, MAX_TOKENS,
    BASE_PARAMS, KFOLD_CONFIG, MBTI_CLASSES, PROBLEM_CLASSES,
    DEVICE, EMBEDDING_BATCH_SIZE
)
from config_definitions import ALL_CONFIGS, get_config_params, ENSEMBLES


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
    per_class_delta: Optional[Dict[str, float]] = None
    problem_class_delta: Optional[Dict[str, float]] = None
    config_params: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class EmbeddingCache:
    """Manages embedding cache for fast reuse with GPU acceleration."""

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
                batch_size=EMBEDDING_BATCH_SIZE
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
        if not texts:
            return np.array([]).reshape(0, self._embeddings.shape[1])
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True,
            batch_size=EMBEDDING_BATCH_SIZE
        )


class AsyncLLMGenerator:
    """Async parallel LLM generator with high concurrency."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        max_concurrent: int = MAX_CONCURRENT_API_CALLS,
        temperature: float = TEMPERATURE
    ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
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
                        temperature=self.temperature,
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


class SyntheticGenerator:
    """Generate synthetic samples using LLM with config-aware parameters."""

    def __init__(self, cache: EmbeddingCache, config_params: Dict[str, Any]):
        self.cache = cache
        self.params = config_params
        self.llm = AsyncLLMGenerator(
            model=config_params.get("llm_model", LLM_MODEL),
            temperature=config_params.get("temperature", TEMPERATURE)
        )

    def create_prompt(
        self,
        examples: List[str],
        target_class: str,
        n_samples: int = 5
    ) -> str:
        """Create generation prompt from examples."""
        n_shot = self.params.get("n_shot", 5)

        if n_shot == 0:
            # Zero-shot
            return f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type (MBTI).

Generate {n_samples} new, unique posts. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

        # Few-shot with examples
        examples_to_use = examples[:n_shot]
        examples_text = "\n".join([
            f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
            for ex in examples_to_use
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
        target_class: str
    ) -> Tuple[List[str], List[str]]:
        """Generate synthetic samples for one class using clustering."""

        max_clusters = self.params.get("max_clusters", 12)
        prompts_per_cluster = self.params.get("prompts_per_cluster", 9)
        samples_per_prompt = self.params.get("samples_per_prompt", 5)
        min_synthetic = self.params.get("min_synthetic_per_class", 0)
        n_shot = self.params.get("n_shot", 5)
        use_all_examples = self.params.get("use_all_examples", False)
        rare_class_boost = self.params.get("rare_class_boost", 1.0)

        # For rare classes (<50 samples), reduce minimum threshold
        min_class_size = 3 if len(class_embeddings) < 50 else 10
        if len(class_embeddings) < min_class_size:
            return [], []

        # Determine actual number of clusters (fewer for rare classes)
        if len(class_embeddings) < 50:
            k_actual = min(max_clusters, max(1, len(class_embeddings) // 15))
        else:
            k_actual = min(max_clusters, max(1, len(class_embeddings) // 30))

        # For very small classes, use single cluster
        if len(class_embeddings) < 30:
            k_actual = 1

        # Cluster the class
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        all_prompts = []
        prompt_metadata = []  # (cluster_id, cluster_texts)

        # Boost prompts for rare classes
        actual_prompts_per_cluster = prompts_per_cluster
        if len(class_embeddings) < 50:
            actual_prompts_per_cluster = int(prompts_per_cluster * rare_class_boost)

        for c_id in range(k_actual):
            c_mask = cluster_labels == c_id
            # Lower threshold for rare classes
            min_cluster_size = 1 if len(class_embeddings) < 50 else 3
            if c_mask.sum() < min_cluster_size:
                continue

            cluster_texts = class_texts[c_mask]
            cluster_embs = class_embeddings[c_mask]

            # Select examples for prompt
            if use_all_examples:
                # Use ALL examples from the class (for rare classes)
                example_texts = list(class_texts[:n_shot])
            else:
                # Select examples closest to centroid (medoid strategy)
                centroid = kmeans.cluster_centers_[c_id]
                dists = np.linalg.norm(cluster_embs - centroid, axis=1)
                k_neighbors = self.params.get("k_neighbors", 15)
                k_neighbors = min(k_neighbors, n_shot, len(cluster_texts))
                nearest_idx = np.argsort(dists)[:k_neighbors]
                example_texts = [cluster_texts[i] for i in nearest_idx]

            # Create prompts for this cluster (no hard limit for rare classes)
            max_prompts = actual_prompts_per_cluster if len(class_embeddings) < 50 else min(prompts_per_cluster, 3)
            for _ in range(max_prompts):
                prompt = self.create_prompt(example_texts, str(target_class), samples_per_prompt)
                all_prompts.append(prompt)
                prompt_metadata.append((c_id, example_texts))

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

        # If we haven't reached minimum, generate more (for rare classes)
        if min_synthetic > 0 and len(synthetic_texts) < min_synthetic:
            additional_needed = min_synthetic - len(synthetic_texts)
            extra_prompts_needed = (additional_needed // samples_per_prompt) + 1

            # Generate additional prompts using all examples
            extra_prompts = []
            example_texts = list(class_texts[:min(n_shot, len(class_texts))])
            for _ in range(extra_prompts_needed):
                prompt = self.create_prompt(example_texts, str(target_class), samples_per_prompt)
                extra_prompts.append(prompt)

            extra_responses = self.llm.generate_sync(extra_prompts)
            for response in extra_responses:
                if not response:
                    continue
                samples = [s.strip() for s in response.split('\n')
                          if s.strip() and len(s.strip()) > 10]
                for sample in samples[:samples_per_prompt]:
                    synthetic_texts.append(sample)
                    synthetic_labels.append(target_class)
                    if len(synthetic_texts) >= min_synthetic:
                        break
                if len(synthetic_texts) >= min_synthetic:
                    break

        return synthetic_texts, synthetic_labels

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

        # Check for force_generation_classes
        force_classes = self.params.get("force_generation_classes", None)
        classes_to_process = force_classes if force_classes else np.unique(labels)

        for target_class in classes_to_process:
            class_mask = labels == target_class
            if class_mask.sum() == 0:
                continue

            class_embeddings = embeddings[class_mask]
            class_texts = texts_array[class_mask]

            print(f"    Generating for {target_class} ({class_mask.sum()} samples)...", flush=True)

            synth_texts, synth_labels = self.generate_for_class(
                class_texts, class_embeddings, target_class
            )

            all_synthetic_texts.extend(synth_texts)
            all_synthetic_labels.extend(synth_labels)
            print(f"      Generated {len(synth_texts)} synthetic samples", flush=True)

        if not all_synthetic_texts:
            return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), []

        # Embed all synthetic texts
        print(f"    Embedding {len(all_synthetic_texts)} synthetic texts...", flush=True)
        synthetic_embeddings = self.cache.embed_synthetic(all_synthetic_texts)

        return synthetic_embeddings, np.array(all_synthetic_labels), all_synthetic_texts


class KFoldEvaluator:
    """K-Fold cross-validation evaluator with per-class tracking."""

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
        config_name: str = "unknown",
        config_params: Optional[Dict[str, Any]] = None
    ) -> KFoldResult:
        """Run K-Fold evaluation and compute statistics."""

        # Use config-specific weight if provided
        weight = self.synthetic_weight
        if config_params and "synthetic_weight" in config_params:
            weight = config_params["synthetic_weight"]

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
        unique_classes = np.unique(y_original)
        per_class_baselines = {c: [] for c in unique_classes}
        per_class_augmented = {c: [] for c in unique_classes}

        print(f"\n  Running {self.n_splits}-fold x {self.n_repeats} repeats = {self.total_folds} folds")
        print(f"  Synthetic weight: {weight}")

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
            baseline_per_class = f1_score(
                y_test, y_pred_baseline, average=None,
                labels=unique_classes, zero_division=0
            )
            for i, c in enumerate(unique_classes):
                per_class_baselines[c].append(baseline_per_class[i])

            # Augmented: train on original + synthetic
            if X_synthetic is not None and len(X_synthetic) > 0:
                n_train = len(X_train)
                n_synth = len(X_synthetic)
                weights = np.concatenate([
                    np.ones(n_train),
                    np.full(n_synth, weight)
                ])

                X_aug = np.vstack([X_train, X_synthetic])
                y_aug = np.concatenate([y_train, y_synthetic])

                clf_augmented = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf_augmented.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_augmented.predict(X_test)
                augmented_f1 = f1_score(y_test, y_pred_aug, average="macro")

                # Per-class F1 for augmented
                aug_per_class = f1_score(
                    y_test, y_pred_aug, average=None,
                    labels=unique_classes, zero_division=0
                )
                for i, c in enumerate(unique_classes):
                    per_class_augmented[c].append(aug_per_class[i])
            else:
                augmented_f1 = baseline_f1
                for c in unique_classes:
                    per_class_augmented[c].append(per_class_baselines[c][-1])

            augmented_f1s.append(augmented_f1)

            # Print progress
            if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
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
        for c in unique_classes:
            baseline_class_mean = np.mean(per_class_baselines[c])
            augmented_class_mean = np.mean(per_class_augmented[c])
            delta_class = augmented_class_mean - baseline_class_mean
            per_class_delta[str(c)] = float(delta_class)

        # Problem class specific delta
        problem_class_delta = {
            c: per_class_delta.get(c, 0.0)
            for c in PROBLEM_CLASSES
        }

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
            per_class_delta=per_class_delta,
            problem_class_delta=problem_class_delta,
            config_params=config_params,
            timestamp=datetime.now().isoformat()
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


def save_result(result: KFoldResult, wave: str):
    """Save result to JSON file."""
    output_dir = RESULTS_DIR / wave
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{result.config_name}_kfold.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"  Saved results to {output_path}")
    return output_path


def print_summary(result: KFoldResult):
    """Print formatted result summary."""
    print(f"\n{'='*70}")
    print(f"  Config: {result.config_name}")
    print(f"  Folds: {result.n_folds}")
    print(f"  Baseline:  {result.baseline_mean:.4f} +/- {result.baseline_std:.4f}")
    print(f"  Augmented: {result.augmented_mean:.4f} +/- {result.augmented_std:.4f}")
    print(f"  Delta:     {result.delta_mean:+.4f} ({result.delta_pct:+.2f}%)")
    print(f"  95% CI:    [{result.ci_95_lower:+.4f}, {result.ci_95_upper:+.4f}]")
    print(f"  p-value:   {result.p_value:.6f} {'*' if result.significant else ''}")
    print(f"  Win rate:  {result.win_rate*100:.1f}%")
    print(f"  Synthetics: {result.n_synthetic}")

    if result.problem_class_delta:
        print(f"\n  Problem class deltas:")
        for cls, delta in result.problem_class_delta.items():
            print(f"    {cls}: {delta:+.4f}")

    print(f"{'='*70}\n")


def run_config_validation(
    config_name: str,
    cache: EmbeddingCache,
    verbose: bool = True
) -> KFoldResult:
    """Run validation for a single config."""

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Validating: {config_name}")
        print(f"{'#'*70}")

    # Get config parameters
    params = get_config_params(config_name)
    config_info = ALL_CONFIGS[config_name]

    if verbose:
        print(f"\n  Description: {config_info['description']}")
        print(f"  Wave: {config_info['wave']}")
        print(f"  Crucial params: {config_info['crucial_params']}")

    # Generate synthetic data
    generator = SyntheticGenerator(cache, params)

    if verbose:
        print(f"\n  Generating synthetic data...")

    X_synthetic, y_synthetic, synthetic_texts = generator.generate_all(
        cache.embeddings, cache.labels, cache.texts
    )

    if verbose:
        print(f"  Generated {len(X_synthetic)} synthetic samples")

    # Evaluate with K-fold
    evaluator = KFoldEvaluator(synthetic_weight=params.get("synthetic_weight", 1.0))
    result = evaluator.evaluate(
        cache.embeddings,
        cache.labels,
        X_synthetic,
        y_synthetic,
        config_name=config_name,
        config_params=params
    )

    # Save result
    wave = config_info['wave']
    save_result(result, wave)

    if verbose:
        print_summary(result)

    return result


def run_ensemble_validation(
    ensemble_name: str,
    component_results: Dict[str, KFoldResult],
    cache: EmbeddingCache
) -> KFoldResult:
    """Run validation for an ensemble by combining synthetics from components."""

    print(f"\n{'#'*70}")
    print(f"# Validating Ensemble: {ensemble_name}")
    print(f"{'#'*70}")

    ensemble_info = ENSEMBLES[ensemble_name]
    components = ensemble_info["components"]

    print(f"\n  Description: {ensemble_info['description']}")
    print(f"  Components: {components}")

    # For ensemble, we need to regenerate and combine synthetics
    # This is a simplified version - in production, you'd load cached synthetics
    all_synthetic_emb = []
    all_synthetic_labels = []

    for comp_name in components:
        if comp_name in component_results:
            # In a real implementation, load cached synthetics
            # For now, regenerate
            params = get_config_params(comp_name)
            generator = SyntheticGenerator(cache, params)
            X_synth, y_synth, _ = generator.generate_all(
                cache.embeddings, cache.labels, cache.texts
            )
            if len(X_synth) > 0:
                all_synthetic_emb.append(X_synth)
                all_synthetic_labels.append(y_synth)

    if all_synthetic_emb:
        X_ensemble = np.vstack(all_synthetic_emb)
        y_ensemble = np.concatenate(all_synthetic_labels)
    else:
        X_ensemble = np.array([]).reshape(0, cache.embeddings.shape[1])
        y_ensemble = np.array([])

    print(f"  Combined {len(X_ensemble)} synthetic samples from {len(components)} components")

    # Evaluate
    evaluator = KFoldEvaluator(synthetic_weight=1.0)
    result = evaluator.evaluate(
        cache.embeddings,
        cache.labels,
        X_ensemble,
        y_ensemble,
        config_name=ensemble_name
    )

    # Save
    save_result(result, "ensembles")
    print_summary(result)

    return result


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase G Validation Runner")
    print("=" * 70)
    print(f"\n  K-Fold config: {KFOLD_CONFIG['n_splits']} splits x {KFOLD_CONFIG['n_repeats']} repeats")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Device: {DEVICE}")
    print(f"  LLM model: {LLM_MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max parallel API calls: {MAX_CONCURRENT_API_CALLS}")
    print(f"  Synthetic weight: {BASE_PARAMS['synthetic_weight']}")
    print("=" * 70)

    # Load data and cache embeddings
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    print(f"\nReady for experiments.")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique classes: {len(np.unique(labels))}")
    print(f"  Total configs: {len(ALL_CONFIGS)}")
