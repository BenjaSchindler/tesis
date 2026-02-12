#!/usr/bin/env python3
"""
Baseline Augmentation Methods for Fair Comparison

Provides:
1. Classical SMOTE on embeddings (imbalanced-learn)
2. Random oversampling (simple duplication with optional noise)
3. No augmentation (control baseline)

All baselines support fixed-N per class for fair comparison with SMOTE-LLM.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Run: pip install imbalanced-learn")


class SMOTEBaseline:
    """
    Classical SMOTE on embedding space.

    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples by interpolating between existing samples in feature space.
    When applied to text embeddings, this creates synthetic embeddings
    that are linear combinations of real embeddings.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        """
        Initialize SMOTE baseline.

        Args:
            k_neighbors: Number of nearest neighbors for SMOTE interpolation
            random_state: Random seed for reproducibility
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imbalanced-learn required. Run: pip install imbalanced-learn")

        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_n_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples using SMOTE on embeddings.

        Args:
            X: Original embeddings (N, D)
            y: Original labels (N,)
            target_n_per_class: If set, generate exactly this many per class

        Returns:
            (X_synthetic, y_synthetic) - ONLY the new synthetic samples
        """
        counts = Counter(y)

        # Determine k_neighbors (must be < min class size)
        min_class_size = min(counts.values())
        k = min(self.k_neighbors, min_class_size - 1)

        if k < 1:
            print(f"  Warning: min class size {min_class_size} too small for SMOTE, using k=1")
            k = 1

        # Create sampling strategy
        if target_n_per_class is not None:
            # Generate exactly target_n_per_class new samples per class
            sampling_strategy = {
                cls: count + target_n_per_class
                for cls, count in counts.items()
            }
        else:
            # Default: balance to majority class
            sampling_strategy = "auto"

        smote = SMOTE(
            k_neighbors=k,
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Extract ONLY the new synthetic samples
        n_original = len(X)
        X_synthetic = X_resampled[n_original:]
        y_synthetic = y_resampled[n_original:]

        return X_synthetic, y_synthetic

    def generate_per_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_n_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate exactly target_n_per_class for each class.

        This ensures fair comparison with SMOTE-LLM fixed-N experiments.
        """
        return self.generate(X, y, target_n_per_class=target_n_per_class)


class RandomOversamplingBaseline:
    """
    Random oversampling (duplication with optional noise).

    This is the simplest baseline: randomly select samples from each class
    and optionally add small Gaussian noise to create "new" samples.
    """

    def __init__(
        self,
        noise_level: float = 0.0,
        random_state: int = 42
    ):
        """
        Initialize random oversampling baseline.

        Args:
            noise_level: Standard deviation of Gaussian noise to add.
                         0 = pure duplication, 0.01 = small noise, 0.1 = large noise
            random_state: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.random_state = random_state

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_n_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples by random duplication.

        Args:
            X: Original embeddings (N, D)
            y: Original labels (N,)
            target_n_per_class: Number of synthetic samples per class

        Returns:
            (X_synthetic, y_synthetic)
        """
        np.random.seed(self.random_state)

        X_synthetic_list = []
        y_synthetic_list = []

        unique_classes = np.unique(y)

        for cls in unique_classes:
            class_mask = y == cls
            X_class = X[class_mask]

            if len(X_class) == 0:
                continue

            # Random sample with replacement
            indices = np.random.choice(
                len(X_class),
                target_n_per_class,
                replace=True
            )
            X_sampled = X_class[indices].copy()

            # Add optional noise
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, X_sampled.shape)
                X_sampled = X_sampled + noise

                # Renormalize if embeddings are normalized (common for sentence transformers)
                norms = np.linalg.norm(X_sampled, axis=1, keepdims=True)
                X_sampled = X_sampled / (norms + 1e-8)

            X_synthetic_list.append(X_sampled)
            y_synthetic_list.extend([cls] * target_n_per_class)

        X_synthetic = np.vstack(X_synthetic_list)
        y_synthetic = np.array(y_synthetic_list)

        return X_synthetic, y_synthetic

    def generate_per_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_n_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate exactly target_n_per_class for each class.

        This ensures fair comparison with SMOTE-LLM fixed-N experiments.
        """
        return self.generate(X, y, target_n_per_class=target_n_per_class)


class NoAugmentationBaseline:
    """
    No augmentation baseline (control).

    Returns empty arrays - used as control to measure baseline performance
    without any data augmentation.
    """

    def __init__(self):
        pass

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_n_per_class: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return empty synthetic data.

        Args:
            X: Original embeddings (N, D)
            y: Original labels (N,)
            target_n_per_class: Ignored

        Returns:
            Empty arrays (0, D) and (0,)
        """
        return np.array([]).reshape(0, X.shape[1]), np.array([])


def get_baseline(name: str, **kwargs):
    """
    Factory function to get baseline by name.

    Args:
        name: One of "smote", "random", "none"
        **kwargs: Additional arguments for the baseline

    Returns:
        Baseline instance
    """
    baselines = {
        "smote": SMOTEBaseline,
        "random": RandomOversamplingBaseline,
        "none": NoAugmentationBaseline,
    }

    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")

    return baselines[name](**kwargs)


def run_baseline_experiment(
    baseline_name: str,
    X: np.ndarray,
    y: np.ndarray,
    target_n_per_class: int = 100,
    **baseline_kwargs
) -> Dict:
    """
    Run a baseline experiment and return synthetic data.

    Args:
        baseline_name: One of "smote", "random", "none"
        X: Original embeddings
        y: Original labels
        target_n_per_class: Number of synthetic samples per class
        **baseline_kwargs: Additional arguments for the baseline

    Returns:
        Dictionary with X_synthetic, y_synthetic, and metadata
    """
    baseline = get_baseline(baseline_name, **baseline_kwargs)

    if baseline_name == "none":
        X_synth, y_synth = baseline.generate(X, y)
    else:
        X_synth, y_synth = baseline.generate(X, y, target_n_per_class=target_n_per_class)

    return {
        "X_synthetic": X_synth,
        "y_synthetic": y_synth,
        "method": baseline_name,
        "n_synthetic": len(X_synth),
        "n_per_class": target_n_per_class if baseline_name != "none" else 0,
        "params": baseline_kwargs
    }


if __name__ == "__main__":
    # Quick test
    print("Testing baselines...")

    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 768)
    y = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

    print(f"\nOriginal: {len(X)} samples, classes: {Counter(y)}")

    # Test SMOTE
    if IMBLEARN_AVAILABLE:
        smote = SMOTEBaseline(k_neighbors=3)
        X_smote, y_smote = smote.generate(X, y, target_n_per_class=10)
        print(f"SMOTE: {len(X_smote)} synthetic, classes: {Counter(y_smote)}")

    # Test Random
    random_os = RandomOversamplingBaseline(noise_level=0.01)
    X_rand, y_rand = random_os.generate(X, y, target_n_per_class=10)
    print(f"Random: {len(X_rand)} synthetic, classes: {Counter(y_rand)}")

    # Test None
    none_os = NoAugmentationBaseline()
    X_none, y_none = none_os.generate(X, y)
    print(f"None: {len(X_none)} synthetic")

    print("\nBaselines OK!")
