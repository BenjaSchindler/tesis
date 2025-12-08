"""
TIER S Improvement: Adversarial Discriminator

Filters synthetic samples based on how easily they can be distinguished from real samples.
The idea: Reject synthetics that are TOO easy to detect (low quality) or keep ones that
are indistinguishable from real samples.

This is inspired by GAN discriminators, but applied to LLM-generated text augmentation.

Expected improvement: +1-2% F1 macro
Implementation difficulty: LOW (2 days)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Optional
import warnings


class AdversarialDiscriminator:
    """
    Binary classifier that learns to distinguish real vs synthetic samples.

    Usage in augmentation pipeline:
    1. Train discriminator on (real_embeddings, synthetic_embeddings)
    2. Score new synthetic candidates
    3. Filter based on difficulty-to-detect threshold

    Filtering strategy:
    - prob_synthetic > 0.9: TOO EASY to detect → REJECT (obviously fake)
    - prob_synthetic < threshold: HARD to detect → ACCEPT (indistinguishable)
    - 0.7 < prob_synthetic < 0.9: MEDIUM → Maybe accept with lower weight
    """

    def __init__(
        self,
        model_type: str = "logistic",
        difficulty_threshold: float = 0.7,
        medium_zone_weight: float = 0.5,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'logistic' or 'random_forest'
            difficulty_threshold: Synthetics with prob < threshold are accepted
            medium_zone_weight: Weight for samples in medium difficulty zone
            random_state: Random seed
        """
        self.model_type = model_type
        self.difficulty_threshold = difficulty_threshold
        self.medium_zone_weight = medium_zone_weight
        self.random_state = random_state

        if model_type == "logistic":
            self.discriminator = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'  # Handle imbalance
            )
        elif model_type == "random_forest":
            self.discriminator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.is_trained = False

    def train(
        self,
        real_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray
    ) -> dict:
        """
        Train discriminator to distinguish real from synthetic.

        Args:
            real_embeddings: Shape (n_real, embedding_dim)
            synthetic_embeddings: Shape (n_synthetic, embedding_dim)

        Returns:
            metrics: Training accuracy and other stats
        """
        # Combine datasets
        X = np.vstack([real_embeddings, synthetic_embeddings])
        y = np.array([0] * len(real_embeddings) + [1] * len(synthetic_embeddings))

        # Train discriminator
        self.discriminator.fit(X, y)
        self.is_trained = True

        # Evaluate discriminator accuracy (on training set - indicative only)
        y_pred = self.discriminator.predict(X)
        accuracy = np.mean(y_pred == y)

        # Get probabilities for analysis
        probs = self.discriminator.predict_proba(X)[:, 1]  # P(synthetic)

        # Statistics
        real_probs = probs[:len(real_embeddings)]
        synth_probs = probs[len(real_embeddings):]

        metrics = {
            "accuracy": accuracy,
            "real_mean_prob": real_probs.mean(),
            "real_std_prob": real_probs.std(),
            "synth_mean_prob": synth_probs.mean(),
            "synth_std_prob": synth_probs.std(),
            "n_real": len(real_embeddings),
            "n_synthetic": len(synthetic_embeddings)
        }

        return metrics

    def filter_synthetics(
        self,
        synthetic_candidates: np.ndarray,
        return_weights: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Filter synthetic candidates based on discriminator scores.

        Strategy:
        - prob_synth > 0.9: REJECT (too obviously fake)
        - prob_synth > threshold: REJECT or DOWNWEIGHT (mediocre quality)
        - prob_synth < threshold: ACCEPT (indistinguishable from real)

        Args:
            synthetic_candidates: Shape (n_candidates, embedding_dim)
            return_weights: If True, return sample weights instead of binary filter

        Returns:
            filtered_indices: Indices of accepted samples
            weights: Sample weights (1.0 for high quality, 0.5 for medium, 0.0 for rejected)
            stats: Filtering statistics
        """
        if not self.is_trained:
            raise RuntimeError("Discriminator not trained. Call train() first.")

        # Get probabilities
        probs = self.discriminator.predict_proba(synthetic_candidates)[:, 1]

        # Filter based on thresholds
        # Zone 1: prob < threshold → ACCEPT (hard to detect)
        # Zone 2: threshold <= prob < 0.9 → MEDIUM (downweight)
        # Zone 3: prob >= 0.9 → REJECT (too easy to detect)

        weights = np.zeros(len(probs))

        zone1_mask = probs < self.difficulty_threshold
        zone2_mask = (probs >= self.difficulty_threshold) & (probs < 0.9)
        zone3_mask = probs >= 0.9

        weights[zone1_mask] = 1.0  # Full weight
        weights[zone2_mask] = self.medium_zone_weight  # Downweight
        weights[zone3_mask] = 0.0  # Reject

        if return_weights:
            # Return all samples with weights
            filtered_indices = np.arange(len(probs))
        else:
            # Return only accepted samples (weight > 0)
            filtered_indices = np.where(weights > 0)[0]

        stats = {
            "n_candidates": len(probs),
            "n_accepted_full": int(zone1_mask.sum()),
            "n_accepted_medium": int(zone2_mask.sum()),
            "n_rejected": int(zone3_mask.sum()),
            "acceptance_rate": (zone1_mask.sum() + zone2_mask.sum()) / len(probs),
            "mean_prob": probs.mean(),
            "std_prob": probs.std(),
            "min_prob": probs.min(),
            "max_prob": probs.max()
        }

        return filtered_indices, weights, stats

    def get_discriminator_scores(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Get raw discriminator probabilities for embeddings.

        Args:
            embeddings: Shape (n_samples, embedding_dim)

        Returns:
            probs: P(synthetic | embedding) for each sample
        """
        if not self.is_trained:
            raise RuntimeError("Discriminator not trained. Call train() first.")

        probs = self.discriminator.predict_proba(embeddings)[:, 1]
        return probs


def integrate_discriminator_with_pipeline(
    real_embeddings: np.ndarray,
    synthetic_candidates_embeddings: np.ndarray,
    difficulty_threshold: float = 0.7,
    model_type: str = "logistic",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Easy-to-use wrapper for integrating discriminator into existing pipeline.

    Args:
        real_embeddings: Real samples from training set
        synthetic_candidates_embeddings: Generated synthetic candidates
        difficulty_threshold: Acceptance threshold (lower = stricter)
        model_type: 'logistic' or 'random_forest'
        verbose: Print filtering statistics

    Returns:
        filtered_indices: Indices of accepted synthetics
        weights: Sample weights
        all_stats: Combined training + filtering statistics
    """
    # Initialize discriminator
    disc = AdversarialDiscriminator(
        model_type=model_type,
        difficulty_threshold=difficulty_threshold,
        random_state=42
    )

    # Train discriminator
    train_metrics = disc.train(real_embeddings, synthetic_candidates_embeddings)

    # Filter synthetics
    filtered_indices, weights, filter_stats = disc.filter_synthetics(
        synthetic_candidates_embeddings,
        return_weights=False  # Binary filter
    )

    # Combine statistics
    all_stats = {
        **train_metrics,
        **filter_stats,
        "discriminator_type": model_type,
        "threshold": difficulty_threshold
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL DISCRIMINATOR RESULTS")
        print(f"{'='*60}")
        print(f"Training:")
        print(f"  Accuracy: {train_metrics['accuracy']:.3f}")
        print(f"  Real mean P(synth): {train_metrics['real_mean_prob']:.3f} (should be LOW)")
        print(f"  Synth mean P(synth): {train_metrics['synth_mean_prob']:.3f} (should be HIGH)")
        print(f"\nFiltering:")
        print(f"  Candidates: {filter_stats['n_candidates']}")
        print(f"  Accepted (full weight): {filter_stats['n_accepted_full']}")
        print(f"  Accepted (medium weight): {filter_stats['n_accepted_medium']}")
        print(f"  Rejected: {filter_stats['n_rejected']}")
        print(f"  Acceptance rate: {filter_stats['acceptance_rate']:.1%}")
        print(f"  Mean P(synth): {filter_stats['mean_prob']:.3f}")
        print(f"{'='*60}\n")

    return filtered_indices, weights, all_stats


# ============================================================================
# Example usage for integration with runner_phase2.py
# ============================================================================

def example_usage_in_runner():
    """
    Example showing how to integrate discriminator into runner_phase2.py

    In runner_phase2.py, after generating synthetic candidates:

    # After generating synthetics with LLM
    synthetic_candidates = [...]  # List of text samples
    synthetic_embeddings = embedding_model.encode(synthetic_candidates)

    # Get real embeddings for this class
    real_class_embeddings = X_train[y_train == target_class]

    # Apply adversarial discriminator
    from adversarial_discriminator import integrate_discriminator_with_pipeline

    filtered_indices, weights, disc_stats = integrate_discriminator_with_pipeline(
        real_embeddings=real_class_embeddings,
        synthetic_candidates_embeddings=synthetic_embeddings,
        difficulty_threshold=0.7,  # Tune this
        model_type="logistic",
        verbose=True
    )

    # Keep only filtered synthetics
    filtered_synthetics = [synthetic_candidates[i] for i in filtered_indices]
    filtered_embeddings = synthetic_embeddings[filtered_indices]

    # Continue with existing quality gate filtering on filtered_synthetics
    # ...
    """
    pass


if __name__ == "__main__":
    # Demo with synthetic data
    print("Adversarial Discriminator Demo")
    print("="*60)

    # Generate fake data
    np.random.seed(42)
    embedding_dim = 384

    # Real samples: Cluster 1 centered at origin
    real_embeddings = np.random.randn(200, embedding_dim) * 0.5

    # Synthetic candidates:
    # - Good synthetics (similar to real): 100 samples
    # - Bad synthetics (obviously different): 50 samples
    good_synthetics = np.random.randn(100, embedding_dim) * 0.5 + 0.1
    bad_synthetics = np.random.randn(50, embedding_dim) * 2.0 + 3.0
    synthetic_embeddings = np.vstack([good_synthetics, bad_synthetics])

    # Apply discriminator
    filtered_indices, weights, stats = integrate_discriminator_with_pipeline(
        real_embeddings=real_embeddings,
        synthetic_candidates_embeddings=synthetic_embeddings,
        difficulty_threshold=0.7,
        model_type="logistic",
        verbose=True
    )

    print(f"Result: Kept {len(filtered_indices)}/150 synthetics")
    print(f"Expected: Should reject most of the 50 'bad' synthetics")
    print(f"Actual rejection: {stats['n_rejected']} synthetics")
