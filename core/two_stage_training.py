"""
Two-Stage Training for SMOTE-LLM

This module implements two-stage training to prevent contamination
of the model with low-quality synthetic samples.

Stage 1: Train a robust baseline on REAL data only
Stage 2: Fine-tune with HIGH-CONFIDENCE synthetics that agree with Stage 1

The key insight is that low-quality synthetics can corrupt the decision
boundaries learned from real data. By training on real data first,
we establish robust boundaries, then only add synthetics that are
consistent with those boundaries.

This addresses the MID tier degradation problem by filtering out
synthetics that would push the model toward incorrect decisions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple, Union
import warnings


def select_high_confidence_synthetics(
    synthetic_embeddings: np.ndarray,
    synthetic_labels: np.ndarray,
    synthetic_confidences: np.ndarray,
    baseline_model: Optional[Pipeline],
    confidence_threshold: float = 0.7,
    require_agreement: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select synthetic samples that pass confidence and agreement checks.

    Args:
        synthetic_embeddings: Embeddings of synthetic samples
        synthetic_labels: Labels of synthetic samples (integer encoded)
        synthetic_confidences: Generation confidence scores (from LLM/filter)
        baseline_model: Trained baseline model for agreement check
        confidence_threshold: Minimum confidence to include (default 0.7)
        require_agreement: If True, also require baseline prediction matches label

    Returns:
        Tuple of (selected_embeddings, selected_labels, selected_confidences, selection_mask)
    """
    n_samples = len(synthetic_embeddings)

    if n_samples == 0:
        return (np.array([]).reshape(0, synthetic_embeddings.shape[1] if len(synthetic_embeddings.shape) > 1 else 0),
                np.array([]),
                np.array([]),
                np.array([], dtype=bool))

    # Confidence filter
    confidence_mask = synthetic_confidences >= confidence_threshold

    # Agreement filter (optional)
    if require_agreement and baseline_model is not None:
        try:
            predictions = baseline_model.predict(synthetic_embeddings)
            agreement_mask = predictions == synthetic_labels
        except Exception as e:
            warnings.warn(f"Agreement check failed: {e}. Skipping agreement filter.")
            agreement_mask = np.ones(n_samples, dtype=bool)
    else:
        agreement_mask = np.ones(n_samples, dtype=bool)

    # Combined selection
    selection_mask = confidence_mask & agreement_mask

    return (
        synthetic_embeddings[selection_mask],
        synthetic_labels[selection_mask],
        synthetic_confidences[selection_mask],
        selection_mask
    )


def compute_stage2_weights(
    selected_confidences: np.ndarray,
    weight_multiplier: float = 0.5,
    use_confidence_scaling: bool = True
) -> np.ndarray:
    """
    Compute sample weights for Stage 2 training.

    Higher confidence synthetics get higher weights.

    Args:
        selected_confidences: Confidence scores for selected synthetics
        weight_multiplier: Base weight multiplier for synthetics (default 0.5)
        use_confidence_scaling: If True, scale weights by confidence

    Returns:
        Array of sample weights for selected synthetics
    """
    if len(selected_confidences) == 0:
        return np.array([])

    if use_confidence_scaling:
        # Scale weights by confidence: high confidence = higher weight
        # Normalized to mean = weight_multiplier
        weights = selected_confidences * weight_multiplier
        weights = weights / (weights.mean() + 1e-8) * weight_multiplier
    else:
        weights = np.full(len(selected_confidences), weight_multiplier)

    return weights


def two_stage_train(
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    synthetic_embeddings: np.ndarray,
    synthetic_labels: np.ndarray,
    synthetic_confidences: np.ndarray,
    class_weight: Optional[Union[str, Dict[int, float]]] = 'balanced',
    confidence_threshold: float = 0.7,
    stage2_weight_multiplier: float = 0.5,
    require_agreement: bool = True,
    use_confidence_scaling: bool = True,
    max_iter: int = 2000,
    solver: str = 'lbfgs',
    verbose: bool = False
) -> Tuple[Pipeline, Dict[str, any]]:
    """
    Two-stage training: Baseline -> Fine-tune with high-confidence synthetics.

    Stage 1: Train a robust classifier on REAL data only.
             This establishes clean decision boundaries.

    Stage 2: Fine-tune by adding selected high-confidence synthetics.
             Only synthetics that agree with Stage 1 predictions are included.

    Args:
        real_embeddings: Embeddings of real training samples
        real_labels: Labels of real samples (integer encoded)
        synthetic_embeddings: Embeddings of synthetic samples
        synthetic_labels: Labels of synthetic samples (integer encoded)
        synthetic_confidences: Generation confidence scores for synthetics
        class_weight: Class weight strategy ('balanced' or dict)
        confidence_threshold: Minimum confidence to include synthetics (default 0.7)
        stage2_weight_multiplier: Weight multiplier for synthetics in Stage 2
        require_agreement: If True, require baseline prediction matches synthetic label
        use_confidence_scaling: If True, scale synthetic weights by confidence
        max_iter: Maximum iterations for LogisticRegression
        solver: Solver for LogisticRegression
        verbose: If True, print training statistics

    Returns:
        Tuple of (trained_model, training_stats)
    """
    stats = {
        "n_real": len(real_labels),
        "n_synthetic_total": len(synthetic_labels),
        "n_synthetic_selected": 0,
        "selection_rate": 0.0,
        "stage1_trained": False,
        "stage2_trained": False,
    }

    # ========== STAGE 1: Train on REAL data only ==========
    if verbose:
        print(f"[Stage 1] Training baseline on {len(real_labels)} real samples...")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            warm_start=True,  # Enable for Stage 2 continuation
            random_state=42
        ))
    ])

    model.fit(real_embeddings, real_labels)
    stats["stage1_trained"] = True

    # ========== SYNTHETIC SELECTION ==========
    if len(synthetic_embeddings) == 0:
        if verbose:
            print("[Stage 2] No synthetics provided. Returning Stage 1 model.")
        return model, stats

    selected_embs, selected_labels, selected_confs, selection_mask = \
        select_high_confidence_synthetics(
            synthetic_embeddings,
            synthetic_labels,
            synthetic_confidences,
            baseline_model=model,
            confidence_threshold=confidence_threshold,
            require_agreement=require_agreement
        )

    stats["n_synthetic_selected"] = len(selected_labels)
    stats["selection_rate"] = len(selected_labels) / len(synthetic_labels) if len(synthetic_labels) > 0 else 0.0

    if verbose:
        print(f"[Selection] {len(selected_labels)}/{len(synthetic_labels)} synthetics passed "
              f"(threshold={confidence_threshold}, agreement={require_agreement})")

    # ========== STAGE 2: Fine-tune with selected synthetics ==========
    if len(selected_embs) == 0:
        if verbose:
            print("[Stage 2] No synthetics passed filters. Returning Stage 1 model.")
        return model, stats

    if verbose:
        print(f"[Stage 2] Fine-tuning with {len(selected_labels)} high-confidence synthetics...")

    # Combine real and selected synthetic data
    combined_embs = np.vstack([real_embeddings, selected_embs])
    combined_labels = np.concatenate([real_labels, selected_labels])

    # Compute sample weights
    real_weights = np.ones(len(real_labels))
    synth_weights = compute_stage2_weights(
        selected_confs,
        weight_multiplier=stage2_weight_multiplier,
        use_confidence_scaling=use_confidence_scaling
    )
    combined_weights = np.concatenate([real_weights, synth_weights])

    # Normalize weights
    combined_weights = combined_weights / combined_weights.mean()

    # Fine-tune model (warm_start continues from Stage 1)
    model.fit(combined_embs, combined_labels, clf__sample_weight=combined_weights)
    stats["stage2_trained"] = True

    if verbose:
        print(f"[Complete] Two-stage training finished.")
        print(f"  Real: {len(real_labels)}, Synthetic: {len(selected_labels)}, Total: {len(combined_labels)}")

    return model, stats


def two_stage_train_with_validation(
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    synthetic_embeddings: np.ndarray,
    synthetic_labels: np.ndarray,
    synthetic_confidences: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    class_weight: Optional[Union[str, Dict[int, float]]] = 'balanced',
    confidence_threshold: float = 0.7,
    stage2_weight_multiplier: float = 0.5,
    max_val_degradation: float = 0.02,
    max_iter: int = 2000,
    solver: str = 'lbfgs',
    verbose: bool = False
) -> Tuple[Pipeline, Dict[str, any]]:
    """
    Two-stage training with validation-based early stopping.

    Same as two_stage_train, but with an additional check:
    If Stage 2 degrades validation performance by more than max_val_degradation,
    return the Stage 1 model instead.

    Args:
        real_embeddings: Embeddings of real training samples
        real_labels: Labels of real samples (integer encoded)
        synthetic_embeddings: Embeddings of synthetic samples
        synthetic_labels: Labels of synthetic samples (integer encoded)
        synthetic_confidences: Generation confidence scores for synthetics
        val_embeddings: Validation set embeddings
        val_labels: Validation set labels
        class_weight: Class weight strategy ('balanced' or dict)
        confidence_threshold: Minimum confidence to include synthetics
        stage2_weight_multiplier: Weight multiplier for synthetics in Stage 2
        max_val_degradation: Max allowed validation F1 drop (default 0.02 = 2%)
        max_iter: Maximum iterations for LogisticRegression
        solver: Solver for LogisticRegression
        verbose: If True, print training statistics

    Returns:
        Tuple of (trained_model, training_stats)
    """
    from sklearn.metrics import f1_score

    stats = {
        "n_real": len(real_labels),
        "n_synthetic_total": len(synthetic_labels),
        "n_synthetic_selected": 0,
        "selection_rate": 0.0,
        "stage1_val_f1": 0.0,
        "stage2_val_f1": 0.0,
        "val_delta": 0.0,
        "used_stage": 1,
    }

    # ========== STAGE 1 ==========
    if verbose:
        print(f"[Stage 1] Training baseline on {len(real_labels)} real samples...")

    stage1_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    stage1_model.fit(real_embeddings, real_labels)

    # Evaluate Stage 1
    stage1_preds = stage1_model.predict(val_embeddings)
    stage1_val_f1 = f1_score(val_labels, stage1_preds, average='macro')
    stats["stage1_val_f1"] = stage1_val_f1

    if verbose:
        print(f"[Stage 1] Validation macro F1: {stage1_val_f1:.4f}")

    # ========== SYNTHETIC SELECTION ==========
    if len(synthetic_embeddings) == 0:
        if verbose:
            print("[Stage 2] No synthetics provided. Returning Stage 1 model.")
        return stage1_model, stats

    selected_embs, selected_labels, selected_confs, _ = \
        select_high_confidence_synthetics(
            synthetic_embeddings,
            synthetic_labels,
            synthetic_confidences,
            baseline_model=stage1_model,
            confidence_threshold=confidence_threshold,
            require_agreement=True
        )

    stats["n_synthetic_selected"] = len(selected_labels)
    stats["selection_rate"] = len(selected_labels) / len(synthetic_labels) if len(synthetic_labels) > 0 else 0.0

    if len(selected_embs) == 0:
        if verbose:
            print("[Stage 2] No synthetics passed filters. Returning Stage 1 model.")
        return stage1_model, stats

    # ========== STAGE 2 ==========
    if verbose:
        print(f"[Stage 2] Training with {len(selected_labels)} high-confidence synthetics...")

    # Combine data
    combined_embs = np.vstack([real_embeddings, selected_embs])
    combined_labels = np.concatenate([real_labels, selected_labels])

    # Weights
    real_weights = np.ones(len(real_labels))
    synth_weights = compute_stage2_weights(selected_confs, stage2_weight_multiplier)
    combined_weights = np.concatenate([real_weights, synth_weights])
    combined_weights = combined_weights / combined_weights.mean()

    # Train Stage 2
    stage2_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    stage2_model.fit(combined_embs, combined_labels, clf__sample_weight=combined_weights)

    # Evaluate Stage 2
    stage2_preds = stage2_model.predict(val_embeddings)
    stage2_val_f1 = f1_score(val_labels, stage2_preds, average='macro')
    stats["stage2_val_f1"] = stage2_val_f1

    val_delta = stage2_val_f1 - stage1_val_f1
    stats["val_delta"] = val_delta

    if verbose:
        print(f"[Stage 2] Validation macro F1: {stage2_val_f1:.4f} (delta: {val_delta:+.4f})")

    # ========== DECISION ==========
    if val_delta < -max_val_degradation:
        if verbose:
            print(f"[Decision] Stage 2 degraded by {-val_delta:.4f} > {max_val_degradation}. Using Stage 1 model.")
        stats["used_stage"] = 1
        return stage1_model, stats
    else:
        if verbose:
            print(f"[Decision] Using Stage 2 model (delta: {val_delta:+.4f})")
        stats["used_stage"] = 2
        return stage2_model, stats


def get_training_summary(stats: Dict[str, any]) -> str:
    """
    Generate a human-readable summary of two-stage training.

    Args:
        stats: Training statistics dict from two_stage_train

    Returns:
        Formatted string summary
    """
    lines = ["Two-Stage Training Summary:"]
    lines.append("-" * 50)
    lines.append(f"Real samples: {stats.get('n_real', 0)}")
    lines.append(f"Synthetic samples (total): {stats.get('n_synthetic_total', 0)}")
    lines.append(f"Synthetic samples (selected): {stats.get('n_synthetic_selected', 0)}")
    lines.append(f"Selection rate: {stats.get('selection_rate', 0):.1%}")

    if "stage1_val_f1" in stats:
        lines.append(f"\nValidation Results:")
        lines.append(f"  Stage 1 F1: {stats['stage1_val_f1']:.4f}")
        if "stage2_val_f1" in stats:
            lines.append(f"  Stage 2 F1: {stats['stage2_val_f1']:.4f}")
            lines.append(f"  Delta: {stats.get('val_delta', 0):+.4f}")
        lines.append(f"  Final model: Stage {stats.get('used_stage', 1)}")

    lines.append("-" * 50)
    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("=== Two-Stage Training Module Test ===\n")

    # Create mock data
    np.random.seed(42)
    n_real = 100
    n_synth = 20
    n_features = 10

    real_embs = np.random.randn(n_real, n_features)
    real_labels = np.random.randint(0, 3, n_real)

    synth_embs = np.random.randn(n_synth, n_features)
    synth_labels = np.random.randint(0, 3, n_synth)
    synth_confs = np.random.uniform(0.5, 1.0, n_synth)

    print(f"Real samples: {n_real}")
    print(f"Synthetic samples: {n_synth}")
    print(f"Confidence range: {synth_confs.min():.2f} - {synth_confs.max():.2f}")

    # Test training
    model, stats = two_stage_train(
        real_embs, real_labels,
        synth_embs, synth_labels, synth_confs,
        confidence_threshold=0.7,
        verbose=True
    )

    print("\n" + get_training_summary(stats))
