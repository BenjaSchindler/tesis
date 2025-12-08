"""
Focal Loss and Tier-Based Class Weighting for SMOTE-LLM

This module implements focal loss training and tier-based class weights
to improve performance on LOW and MID tier classes.

Focal Loss: (1 - p_t)^gamma * CE_loss
- Down-weights easy examples (high confidence)
- Up-weights hard examples (low confidence)
- Helps model focus on difficult classes

Tier-Based Weights:
- LOW tier (F1 < 0.20): 2.0x boost
- MID tier (F1 0.20-0.45): 1.5x boost
- HIGH tier (F1 >= 0.45): 1.0x (no boost)

This addresses the MID tier degradation problem by ensuring
the model pays more attention to difficult classes.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Optional, Tuple, Union
import warnings


# Tier thresholds based on baseline F1
TIER_THRESHOLDS = {
    "very_low": 0.15,  # F1 < 0.15
    "low": 0.20,       # F1 < 0.20
    "mid": 0.45,       # F1 < 0.45
    "high": 1.0        # F1 >= 0.45
}

# Default boost multipliers per tier
DEFAULT_TIER_BOOSTS = {
    "very_low": 2.5,
    "low": 2.0,
    "mid": 1.5,
    "high": 1.0
}


def get_class_tier(baseline_f1: float) -> str:
    """
    Determine the tier of a class based on its baseline F1.

    Args:
        baseline_f1: The baseline F1 score for the class

    Returns:
        Tier name: 'very_low', 'low', 'mid', or 'high'
    """
    if baseline_f1 < TIER_THRESHOLDS["very_low"]:
        return "very_low"
    elif baseline_f1 < TIER_THRESHOLDS["low"]:
        return "low"
    elif baseline_f1 < TIER_THRESHOLDS["mid"]:
        return "mid"
    else:
        return "high"


def compute_focal_weights(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    gamma: float = 2.0
) -> np.ndarray:
    """
    Compute focal loss weights for each sample.

    Focal weight = (1 - p_t)^gamma
    where p_t is the predicted probability for the true class.

    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)

    Returns:
        Array of focal weights, normalized to mean=1
    """
    n_samples = len(y_true)

    # Get predicted probability for the true class
    p_t = y_pred_proba[np.arange(n_samples), y_true]

    # Focal weight: (1 - p_t)^gamma
    # High p_t (easy) -> low weight
    # Low p_t (hard) -> high weight
    focal_weights = np.power(1.0 - p_t, gamma)

    # Normalize to mean=1 to maintain training dynamics
    focal_weights = focal_weights / (focal_weights.mean() + 1e-8)

    return focal_weights


def compute_tier_class_weights(
    y_train: np.ndarray,
    baseline_f1_scores: Dict[str, float],
    label_encoder: LabelEncoder,
    tier_boosts: Optional[Dict[str, float]] = None,
    use_balanced_base: bool = True
) -> Dict[int, float]:
    """
    Compute class weights with tier-based boosting.

    Args:
        y_train: Training labels (integer encoded)
        baseline_f1_scores: Dict mapping class names to baseline F1 scores
        label_encoder: LabelEncoder used for encoding classes
        tier_boosts: Optional custom tier boost multipliers
        use_balanced_base: If True, use sklearn's balanced weights as base

    Returns:
        Dict mapping class indices to weights
    """
    if tier_boosts is None:
        tier_boosts = DEFAULT_TIER_BOOSTS

    n_classes = len(label_encoder.classes_)
    class_counts = np.bincount(y_train, minlength=n_classes)
    total_samples = len(y_train)

    # Base weights (balanced or uniform)
    if use_balanced_base:
        # sklearn's balanced formula: n_samples / (n_classes * n_samples_per_class)
        base_weights = total_samples / (n_classes * (class_counts + 1e-8))
    else:
        base_weights = np.ones(n_classes)

    # Apply tier-based boosts
    class_weights = {}
    for idx, class_name in enumerate(label_encoder.classes_):
        weight = base_weights[idx]

        # Get baseline F1 for this class
        f1 = baseline_f1_scores.get(class_name, 0.5)

        # Determine tier and apply boost
        tier = get_class_tier(f1)
        boost = tier_boosts.get(tier, 1.0)

        class_weights[idx] = weight * boost

    return class_weights


def compute_combined_sample_weights(
    y_train: np.ndarray,
    class_weights: Dict[int, float],
    synthetic_mask: Optional[np.ndarray] = None,
    synthetic_weight_multiplier: float = 0.5,
    focal_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Combine class weights, synthetic weights, and focal weights into sample weights.

    Args:
        y_train: Training labels (integer encoded)
        class_weights: Dict mapping class indices to weights
        synthetic_mask: Boolean mask indicating synthetic samples
        synthetic_weight_multiplier: Weight multiplier for synthetic samples
        focal_weights: Optional focal loss weights per sample

    Returns:
        Array of combined sample weights
    """
    n_samples = len(y_train)

    # Start with class weights
    sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])

    # Apply synthetic weight reduction
    if synthetic_mask is not None:
        sample_weights[synthetic_mask] *= synthetic_weight_multiplier

    # Apply focal weights
    if focal_weights is not None:
        sample_weights *= focal_weights

    # Normalize to maintain training dynamics
    sample_weights = sample_weights / (sample_weights.mean() + 1e-8)

    return sample_weights


def train_with_tier_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_f1_scores: Dict[str, float],
    label_encoder: LabelEncoder,
    tier_boosts: Optional[Dict[str, float]] = None,
    synthetic_mask: Optional[np.ndarray] = None,
    synthetic_weight_multiplier: float = 0.5,
    max_iter: int = 2000,
    solver: str = 'lbfgs'
) -> Pipeline:
    """
    Train a classifier with tier-based class weights.

    This is the simplest approach - just tier-based weights without focal loss.
    Good baseline for comparison.

    Args:
        X_train: Training features (embeddings)
        y_train: Training labels (integer encoded)
        baseline_f1_scores: Dict mapping class names to baseline F1 scores
        label_encoder: LabelEncoder used for encoding classes
        tier_boosts: Optional custom tier boost multipliers
        synthetic_mask: Boolean mask indicating synthetic samples
        synthetic_weight_multiplier: Weight multiplier for synthetic samples
        max_iter: Maximum iterations for LogisticRegression
        solver: Solver for LogisticRegression

    Returns:
        Trained Pipeline with scaler and classifier
    """
    # Compute class weights
    class_weights = compute_tier_class_weights(
        y_train, baseline_f1_scores, label_encoder, tier_boosts
    )

    # Compute sample weights
    sample_weights = compute_combined_sample_weights(
        y_train, class_weights, synthetic_mask, synthetic_weight_multiplier
    )

    # Create and train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=None,  # We handle weights via sample_weight
            random_state=42
        ))
    ])

    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    return model


def train_with_focal_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_f1_scores: Dict[str, float],
    label_encoder: LabelEncoder,
    tier_boosts: Optional[Dict[str, float]] = None,
    synthetic_mask: Optional[np.ndarray] = None,
    synthetic_weight_multiplier: float = 0.5,
    gamma: float = 2.0,
    n_iterations: int = 2,
    max_iter: int = 2000,
    solver: str = 'lbfgs'
) -> Pipeline:
    """
    Train a classifier with iterative focal loss reweighting.

    This is a two-step process:
    1. Train initial model with tier-based weights
    2. Reweight samples using focal loss and retrain

    The focal loss step helps the model focus on hard examples
    (samples it's currently getting wrong).

    Args:
        X_train: Training features (embeddings)
        y_train: Training labels (integer encoded)
        baseline_f1_scores: Dict mapping class names to baseline F1 scores
        label_encoder: LabelEncoder used for encoding classes
        tier_boosts: Optional custom tier boost multipliers
        synthetic_mask: Boolean mask indicating synthetic samples
        synthetic_weight_multiplier: Weight multiplier for synthetic samples
        gamma: Focal loss focusing parameter (default 2.0)
        n_iterations: Number of focal loss retraining iterations (default 2)
        max_iter: Maximum iterations for LogisticRegression
        solver: Solver for LogisticRegression

    Returns:
        Trained Pipeline with scaler and classifier
    """
    # Compute tier-based class weights
    class_weights = compute_tier_class_weights(
        y_train, baseline_f1_scores, label_encoder, tier_boosts
    )

    # Initial sample weights (without focal)
    sample_weights = compute_combined_sample_weights(
        y_train, class_weights, synthetic_mask, synthetic_weight_multiplier
    )

    # Create model with warm_start for iterative training
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=None,
            warm_start=True,  # Enable for iterative training
            random_state=42
        ))
    ])

    # Initial training
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # Iterative focal loss retraining
    for iteration in range(n_iterations):
        # Get predictions for focal weight computation
        y_pred_proba = model.predict_proba(X_train)

        # Compute focal weights
        focal_weights = compute_focal_weights(y_train, y_pred_proba, gamma)

        # Combine with base sample weights
        combined_weights = compute_combined_sample_weights(
            y_train, class_weights, synthetic_mask, synthetic_weight_multiplier,
            focal_weights=focal_weights
        )

        # Retrain with updated weights
        model.fit(X_train, y_train, clf__sample_weight=combined_weights)

    return model


def get_tier_summary(
    baseline_f1_scores: Dict[str, float],
    tier_boosts: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate a summary of tier assignments and boosts.

    Useful for logging and debugging.

    Args:
        baseline_f1_scores: Dict mapping class names to baseline F1 scores
        tier_boosts: Optional custom tier boost multipliers

    Returns:
        Formatted string summary
    """
    if tier_boosts is None:
        tier_boosts = DEFAULT_TIER_BOOSTS

    lines = ["Tier-Based Weight Summary:"]
    lines.append("-" * 50)
    lines.append(f"{'Class':<10} {'F1':<8} {'Tier':<10} {'Boost':<8}")
    lines.append("-" * 50)

    # Sort by F1 ascending
    sorted_classes = sorted(baseline_f1_scores.items(), key=lambda x: x[1])

    for class_name, f1 in sorted_classes:
        tier = get_class_tier(f1)
        boost = tier_boosts.get(tier, 1.0)
        lines.append(f"{class_name:<10} {f1:<8.3f} {tier:<10} {boost:<8.1f}x")

    lines.append("-" * 50)

    # Count per tier
    tier_counts = {}
    for class_name, f1 in baseline_f1_scores.items():
        tier = get_class_tier(f1)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    lines.append("Tier Distribution:")
    for tier in ["very_low", "low", "mid", "high"]:
        count = tier_counts.get(tier, 0)
        if count > 0:
            lines.append(f"  {tier}: {count} classes")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("=== Focal Loss Training Module Test ===\n")

    # Mock baseline F1 scores (similar to MBTI dataset)
    baseline_f1 = {
        "ESFJ": 0.226,  # VERY LOW
        "ESFP": 0.312,  # LOW
        "ISFJ": 0.204,  # LOW
        "ISFP": 0.293,  # LOW
        "ENFJ": 0.261,  # MID
        "ESTJ": 0.541,  # HIGH
        "ENTP": 0.527,  # HIGH
        "INFP": 0.650,  # HIGH
        "INTJ": 0.580,  # HIGH
    }

    print(get_tier_summary(baseline_f1))

    print("\n\nTier assignments:")
    for cls, f1 in baseline_f1.items():
        tier = get_class_tier(f1)
        print(f"  {cls}: F1={f1:.3f} -> {tier}")
