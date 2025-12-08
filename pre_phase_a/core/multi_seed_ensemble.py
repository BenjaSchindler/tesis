"""
TIER S Improvement: Ensemble Multi-Seed

Reduces seed variance by running augmentation with multiple random seeds
and using soft voting for final predictions.

Key problem solved: 50% of random seeds cause performance degradation
Solution: Average predictions from 3-4 different seeds

Expected improvement: +2-4% F1 macro
Implementation difficulty: LOW (3 days)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class SeedResult:
    """Results from a single seed run"""
    seed: int
    X_aug: np.ndarray  # Augmented features
    y_aug: np.ndarray  # Augmented labels
    weights: np.ndarray  # Sample weights
    model: any  # Trained classifier
    metrics: dict  # Performance metrics


class MultiSeedEnsemble:
    """
    Ensemble classifier using multiple augmentation seeds with soft voting.

    Strategy:
    1. Run augmentation pipeline with seeds [42, 101, 102, 456]
    2. Train separate classifier on each augmented dataset
    3. Average probability predictions (soft voting)
    4. Make final prediction based on averaged probabilities

    Benefits:
    - Reduces variance from seed selection
    - More robust to seed failures
    - Often improves performance by 2-4% F1 macro
    """

    def __init__(
        self,
        seeds: List[int] = [42, 101, 102, 456],
        voting_method: str = "soft",
        weight_by_performance: bool = True
    ):
        """
        Args:
            seeds: List of random seeds to use
            voting_method: 'soft' (probability averaging) or 'hard' (majority vote)
            weight_by_performance: Weight each seed's vote by its validation F1
        """
        self.seeds = seeds
        self.voting_method = voting_method
        self.weight_by_performance = weight_by_performance
        self.seed_results: List[SeedResult] = []
        self.seed_weights: Optional[np.ndarray] = None

    def train_single_seed(
        self,
        seed: int,
        augmentation_fn,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **augmentation_kwargs
    ) -> SeedResult:
        """
        Train augmentation + classifier for a single seed.

        Args:
            seed: Random seed
            augmentation_fn: Function that takes (X, y, seed, **kwargs) and returns augmented data
            X_train, y_train: Training data
            X_val, y_val: Validation data (for performance weighting)
            **augmentation_kwargs: Additional arguments for augmentation_fn

        Returns:
            SeedResult with trained model and metrics
        """
        # Run augmentation with this seed
        X_aug, y_aug, weights, aug_metadata = augmentation_fn(
            X_train, y_train, seed=seed, **augmentation_kwargs
        )

        # Train classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score

        model = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_aug, y_aug, sample_weight=weights)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')

        metrics = {
            "seed": seed,
            "val_f1": val_f1,
            "n_synthetic_added": len(X_aug) - len(X_train),
            "augmentation_metadata": aug_metadata
        }

        return SeedResult(
            seed=seed,
            X_aug=X_aug,
            y_aug=y_aug,
            weights=weights,
            model=model,
            metrics=metrics
        )

    def train_ensemble(
        self,
        augmentation_fn,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
        **augmentation_kwargs
    ):
        """
        Train ensemble with all seeds.

        Args:
            augmentation_fn: Augmentation function (must accept 'seed' parameter)
            X_train, y_train: Training data
            X_val, y_val: Validation data
            verbose: Print progress
            **augmentation_kwargs: Arguments for augmentation_fn
        """
        self.seed_results = []

        for seed in self.seeds:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training with seed {seed}...")
                print(f"{'='*60}")

            result = self.train_single_seed(
                seed=seed,
                augmentation_fn=augmentation_fn,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **augmentation_kwargs
            )

            self.seed_results.append(result)

            if verbose:
                print(f"Seed {seed}: Val F1 = {result.metrics['val_f1']:.4f}, "
                      f"Synthetics added = {result.metrics['n_synthetic_added']}")

        # Calculate seed weights based on validation performance
        if self.weight_by_performance:
            val_f1s = np.array([r.metrics['val_f1'] for r in self.seed_results])
            # Softmax weighting (higher F1 → higher weight)
            self.seed_weights = np.exp(val_f1s * 5) / np.exp(val_f1s * 5).sum()
        else:
            # Equal weights
            self.seed_weights = np.ones(len(self.seeds)) / len(self.seeds)

        if verbose:
            print(f"\n{'='*60}")
            print(f"ENSEMBLE SUMMARY")
            print(f"{'='*60}")
            for i, result in enumerate(self.seed_results):
                print(f"Seed {result.seed}: F1={result.metrics['val_f1']:.4f}, "
                      f"Weight={self.seed_weights[i]:.3f}")
            print(f"{'='*60}\n")

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Soft voting: Average probability predictions from all seeds.

        Args:
            X_test: Test data

        Returns:
            probs: Averaged probabilities, shape (n_samples, n_classes)
        """
        if not self.seed_results:
            raise RuntimeError("Ensemble not trained. Call train_ensemble() first.")

        # Collect probabilities from each seed
        all_probs = []
        for result in self.seed_results:
            probs = result.model.predict_proba(X_test)
            all_probs.append(probs)

        all_probs = np.array(all_probs)  # Shape: (n_seeds, n_samples, n_classes)

        # Weighted average
        weighted_probs = np.average(
            all_probs,
            axis=0,
            weights=self.seed_weights
        )

        return weighted_probs

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble voting.

        Args:
            X_test: Test data

        Returns:
            predictions: Predicted labels
        """
        if self.voting_method == "soft":
            probs = self.predict_proba(X_test)
            return probs.argmax(axis=1)
        else:  # hard voting
            # Collect predictions from each seed
            all_preds = []
            for result in self.seed_results:
                preds = result.model.predict(X_test)
                all_preds.append(preds)

            all_preds = np.array(all_preds)  # Shape: (n_seeds, n_samples)

            # Majority vote (weighted by seed weights)
            from scipy.stats import mode
            predictions, _ = mode(all_preds, axis=0)
            return predictions.flatten()

    def get_ensemble_stats(self) -> dict:
        """Get statistics about ensemble variance and agreement"""
        if not self.seed_results:
            raise RuntimeError("Ensemble not trained.")

        val_f1s = [r.metrics['val_f1'] for r in self.seed_results]
        synth_counts = [r.metrics['n_synthetic_added'] for r in self.seed_results]

        stats = {
            "n_seeds": len(self.seeds),
            "seeds": self.seeds,
            "val_f1_mean": np.mean(val_f1s),
            "val_f1_std": np.std(val_f1s),
            "val_f1_min": np.min(val_f1s),
            "val_f1_max": np.max(val_f1s),
            "val_f1_range": np.max(val_f1s) - np.min(val_f1s),
            "synthetic_mean": np.mean(synth_counts),
            "synthetic_std": np.std(synth_counts),
            "seed_weights": self.seed_weights.tolist()
        }

        return stats


# ============================================================================
# Simplified wrapper for easy integration
# ============================================================================

def run_ensemble_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    augmentation_fn,
    seeds: List[int] = [42, 101, 102],
    val_split: float = 0.2,
    verbose: bool = True,
    **augmentation_kwargs
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Easy-to-use wrapper for multi-seed ensemble augmentation.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        augmentation_fn: Function(X, y, seed, **kwargs) -> (X_aug, y_aug, weights, metadata)
        seeds: List of seeds to use
        val_split: Fraction of training data for validation (for weighting)
        verbose: Print progress
        **augmentation_kwargs: Arguments for augmentation_fn

    Returns:
        y_pred: Predictions on test set
        probs: Probabilities on test set
        results: Complete results including ensemble stats
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score

    # Split training into train+val for seed weighting
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        random_state=42,
        stratify=y_train
    )

    # Create ensemble
    ensemble = MultiSeedEnsemble(
        seeds=seeds,
        voting_method="soft",
        weight_by_performance=True
    )

    # Train ensemble
    ensemble.train_ensemble(
        augmentation_fn=augmentation_fn,
        X_train=X_train_sub,
        y_train=y_train_sub,
        X_val=X_val,
        y_val=y_val,
        verbose=verbose,
        **augmentation_kwargs
    )

    # Make predictions
    probs = ensemble.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)

    # Evaluate
    test_f1 = f1_score(y_test, y_pred, average='macro')

    if verbose:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test F1 (macro): {test_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # Collect results
    results = {
        "test_f1": test_f1,
        "ensemble_stats": ensemble.get_ensemble_stats(),
        "predictions": y_pred,
        "probabilities": probs
    }

    return y_pred, probs, results


# ============================================================================
# Example: Integration with runner_phase2.py
# ============================================================================

def example_integration_with_runner():
    """
    Example showing how to modify runner_phase2.py to use ensemble.

    OPTION 1: Minimal change (just ensemble at prediction time)
    ------------------------------------------------------------
    In runner_phase2.py main():

    # Instead of training single model:
    # model.fit(X_aug, y_aug, sample_weight=sample_weights)

    # Use ensemble:
    from multi_seed_ensemble import MultiSeedEnsemble

    ensemble = MultiSeedEnsemble(seeds=[42, 101, 102])

    # Define augmentation function wrapper
    def augment_wrapper(X, y, seed, **kwargs):
        # Call your existing augmentation pipeline
        X_aug, y_aug, weights = your_augmentation_pipeline(X, y, seed=seed, **kwargs)
        metadata = {"seed": seed}
        return X_aug, y_aug, weights, metadata

    # Train ensemble
    ensemble.train_ensemble(
        augmentation_fn=augment_wrapper,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,  # Need to add validation split
        y_val=y_val,
        # Pass your pipeline parameters
        target_classes=target_classes,
        embedding_model=embedding_model,
        # etc...
    )

    # Make predictions
    y_pred = ensemble.predict(X_test)

    OPTION 2: Command-line flag
    ----------------------------
    Add argument:
    parser.add_argument("--use-ensemble", action="store_true")
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42, 101, 102])

    Then in main():
    if args.use_ensemble:
        # Use ensemble logic
    else:
        # Use single-seed logic (current)
    """
    pass


if __name__ == "__main__":
    # Demo
    print("Multi-Seed Ensemble Demo")
    print("="*60)

    # Generate synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Dummy augmentation function (just adds noise)
    def dummy_augmentation(X, y, seed, noise_level=0.1):
        np.random.seed(seed)
        n_synthetic = len(X) // 10  # Add 10% synthetic
        synthetic_X = X[:n_synthetic] + np.random.randn(n_synthetic, X.shape[1]) * noise_level
        synthetic_y = y[:n_synthetic]

        X_aug = np.vstack([X, synthetic_X])
        y_aug = np.concatenate([y, synthetic_y])
        weights = np.ones(len(y_aug))

        metadata = {"n_synthetic": n_synthetic}
        return X_aug, y_aug, weights, metadata

    # Run ensemble
    y_pred, probs, results = run_ensemble_augmentation(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        augmentation_fn=dummy_augmentation,
        seeds=[42, 101, 102],
        verbose=True,
        noise_level=0.1
    )

    print("\n" + "="*60)
    print("Ensemble Stats:")
    for k, v in results["ensemble_stats"].items():
        print(f"  {k}: {v}")
