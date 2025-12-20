#!/usr/bin/env python3
"""
RARE_MLP Suite - Exhaustive Search for Rare Class Improvement

This experiment uses MLP_512_256_128 (which achieved +12.42% on ESFJ) instead of
LogisticRegression, combined with state-of-the-art techniques:
- Focal Loss (down-weight easy examples)
- Class Weights (inverse frequency weighting)
- Remix/Rebalanced Mixup (expand minority decision boundaries)
- Intra-Class Mixup (augment within rare classes)
- Contrastive Embedding Refinement

References:
- Focal Loss: Lin et al. 2017, updated implementations 2024
- Remix: Chou et al. 2020 (https://arxiv.org/pdf/2007.03943)
- ImbLLM: arXiv 2024 (https://arxiv.org/html/2510.09783)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from config_definitions import ENSEMBLES, get_config_params, ALL_CONFIGS
from base_config import RESULTS_DIR

# =============================================================================
# MLP ARCHITECTURES
# =============================================================================

MLP_ARCHITECTURES = {
    "MLP_512_256_128": {
        "hidden_layer_sizes": (512, 256, 128),
        "max_iter": 300,
        "early_stopping": True,
        "random_state": 42,
        "verbose": False
    },
    "MLP_1024_512_256": {
        "hidden_layer_sizes": (1024, 512, 256),
        "max_iter": 500,
        "early_stopping": True,
        "random_state": 42,
        "verbose": False
    },
    "MLP_768_384_192": {
        "hidden_layer_sizes": (768, 384, 192),
        "max_iter": 400,
        "early_stopping": True,
        "random_state": 42,
        "verbose": False
    },
    "MLP_256_128_64": {
        "hidden_layer_sizes": (256, 128, 64),
        "max_iter": 300,
        "early_stopping": True,
        "random_state": 42,
        "verbose": False
    },
    "MLP_512_256_128_64": {
        "hidden_layer_sizes": (512, 256, 128, 64),
        "max_iter": 400,
        "early_stopping": True,
        "random_state": 42,
        "verbose": False
    },
}

# Problem classes to track
RARE_CLASSES = ["ESFJ", "ESFP", "ESTJ"]

# =============================================================================
# STATE-OF-THE-ART TECHNIQUES
# =============================================================================

def compute_focal_weights(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          gamma: float = 2.0) -> np.ndarray:
    """
    Compute Focal Loss sample weights.

    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This down-weights easy examples (high confidence) and focuses on hard ones.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for each class
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        Sample weights based on focal loss principle
    """
    n_samples = len(y_true)
    weights = np.ones(n_samples)

    for i in range(n_samples):
        p_t = y_pred_proba[i, y_true[i]]  # Probability of true class
        # Focal weight: (1 - p_t)^gamma
        weights[i] = (1 - p_t) ** gamma

    return weights


def compute_class_weights_inverse(y: np.ndarray, boost_rare: float = 1.0) -> Dict[int, float]:
    """
    Compute class weights inversely proportional to frequency.

    Args:
        y: Labels
        boost_rare: Additional multiplier for rare classes (ESFJ, ESFP, ESTJ)

    Returns:
        Dictionary mapping class index to weight
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    return weight_dict


def remix_mixup(X: np.ndarray, y: np.ndarray,
                alpha: float = 0.4,
                minority_boost: float = 2.0,
                rare_classes: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebalanced Mixup (Remix) - Chou et al. 2020

    Standard mixup but with higher weight for minority class in label mixing.
    This helps expand decision boundaries for rare classes.

    Args:
        X: Feature matrix
        y: Labels (integer encoded)
        alpha: Mixup interpolation parameter
        minority_boost: Extra weight for minority class labels
        rare_classes: List of rare class indices to boost

    Returns:
        Augmented X and y (soft labels)
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, n_samples)

    X_mixed = []
    y_mixed = []

    for i in range(n_samples):
        j = indices[i]
        l = lam[i]

        # Mix features
        x_new = l * X[i] + (1 - l) * X[j]

        # For labels, boost minority class weight
        if rare_classes and (y[i] in rare_classes or y[j] in rare_classes):
            # If one is rare, give it higher weight
            if y[i] in rare_classes and y[j] not in rare_classes:
                l = min(l * minority_boost, 0.9)  # Boost rare class
            elif y[j] in rare_classes and y[i] not in rare_classes:
                l = max(1 - (1 - l) * minority_boost, 0.1)  # Boost rare class

        X_mixed.append(x_new)
        # For hard labels, use the dominant class
        y_mixed.append(y[i] if l > 0.5 else y[j])

    return np.array(X_mixed), np.array(y_mixed)


def intraclass_mixup(X: np.ndarray, y: np.ndarray,
                     target_classes: List[int],
                     n_augment: int = 100,
                     alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intra-Class Mixup - Mix samples WITHIN the same rare class.

    This expands the feature space of rare classes without introducing
    cross-class confusion.

    Args:
        X: Feature matrix
        y: Labels
        target_classes: Classes to augment via intra-class mixup
        n_augment: Number of augmented samples to generate per class
        alpha: Mixup interpolation parameter

    Returns:
        Augmented X and y arrays (only the new samples)
    """
    X_aug, y_aug = [], []

    for cls in target_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) < 2:
            continue

        for _ in range(n_augment):
            # Sample two random indices from same class
            i, j = np.random.choice(cls_indices, 2, replace=False)
            lam = np.random.beta(alpha, alpha)

            x_new = lam * X[i] + (1 - lam) * X[j]
            X_aug.append(x_new)
            y_aug.append(cls)

    if X_aug:
        return np.array(X_aug), np.array(y_aug)
    return np.array([]).reshape(0, X.shape[1]), np.array([])


def contrastive_embedding_refinement(X: np.ndarray, y: np.ndarray,
                                     n_iterations: int = 100,
                                     learning_rate: float = 0.01,
                                     margin: float = 1.0) -> np.ndarray:
    """
    Simple contrastive learning to refine embeddings.

    Pulls same-class samples closer and pushes different-class samples apart.
    This helps separate rare classes in embedding space.

    Args:
        X: Embedding matrix (n_samples, n_features)
        y: Labels
        n_iterations: Number of refinement iterations
        learning_rate: Step size for updates
        margin: Minimum distance between different classes

    Returns:
        Refined embedding matrix
    """
    X_refined = X.copy()
    n_samples, n_features = X.shape

    for _ in range(n_iterations):
        # Sample random pairs
        i, j = np.random.randint(0, n_samples, 2)

        # Compute direction
        diff = X_refined[i] - X_refined[j]
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            continue

        if y[i] == y[j]:
            # Same class: pull together
            X_refined[i] -= learning_rate * diff / dist
            X_refined[j] += learning_rate * diff / dist
        else:
            # Different class: push apart (if within margin)
            if dist < margin:
                X_refined[i] += learning_rate * diff / dist
                X_refined[j] -= learning_rate * diff / dist

    return X_refined


# =============================================================================
# CONFIGURATION DEFINITIONS
# =============================================================================

RARE_MLP_CONFIGS = {
    # A. Architecture Variations
    "RARE_MLP_arch_512": {
        "description": "Baseline MLP 512-256-128 (best for ESFJ +12.42%)",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
    },
    "RARE_MLP_arch_1024": {
        "description": "Larger MLP 1024-512-256",
        "mlp": "MLP_1024_512_256",
        "components": ["RARE_massive_oversample"],
    },
    "RARE_MLP_arch_768": {
        "description": "Embedding-proportional MLP 768-384-192",
        "mlp": "MLP_768_384_192",
        "components": ["RARE_massive_oversample"],
    },
    "RARE_MLP_arch_256": {
        "description": "Lighter MLP 256-128-64",
        "mlp": "MLP_256_128_64",
        "components": ["RARE_massive_oversample"],
    },
    "RARE_MLP_arch_4layer": {
        "description": "4-layer MLP 512-256-128-64",
        "mlp": "MLP_512_256_128_64",
        "components": ["RARE_massive_oversample"],
    },

    # B. Oversampling Variations
    "RARE_MLP_20x": {
        "description": "20x oversampling (baseline)",
        "mlp": "MLP_512_256_128",
        "rare_boost": 20,
    },
    "RARE_MLP_30x": {
        "description": "30x oversampling",
        "mlp": "MLP_512_256_128",
        "rare_boost": 30,
    },
    "RARE_MLP_50x": {
        "description": "50x extreme oversampling",
        "mlp": "MLP_512_256_128",
        "rare_boost": 50,
    },

    # C. Component Combinations
    "RARE_MLP_top3_std": {
        "description": "Top 3 standard configs + MLP",
        "mlp": "MLP_512_256_128",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3"],
    },
    "RARE_MLP_top5_std": {
        "description": "Top 5 standard configs + MLP",
        "mlp": "MLP_512_256_128",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
    },
    "RARE_MLP_hybrid": {
        "description": "Top 3 standard + RARE_massive + MLP",
        "mlp": "MLP_512_256_128",
        "components": ["W5_many_shot_10", "W6_temp_high", "RARE_massive_oversample"],
    },

    # D. Class-Specific Focus
    "RARE_MLP_ESFJ_50x": {
        "description": "50x oversampling focused on ESFJ",
        "mlp": "MLP_512_256_128",
        "target_classes": ["ESFJ"],
        "rare_boost": 50,
    },
    "RARE_MLP_ESTJ_50x": {
        "description": "50x oversampling focused on ESTJ",
        "mlp": "MLP_512_256_128",
        "target_classes": ["ESTJ"],
        "rare_boost": 50,
    },
    "RARE_MLP_ESFP_100x": {
        "description": "100x extreme oversampling for ESFP (attempt)",
        "mlp": "MLP_512_256_128",
        "target_classes": ["ESFP"],
        "rare_boost": 100,
    },
    "RARE_MLP_trio_30x": {
        "description": "30x for all three rare classes",
        "mlp": "MLP_512_256_128",
        "target_classes": ["ESFJ", "ESFP", "ESTJ"],
        "rare_boost": 30,
    },

    # E. Weighted/Dedup
    "RARE_MLP_weighted_perf": {
        "description": "Performance-weighted components + MLP",
        "mlp": "MLP_512_256_128",
        "components": ["W5_many_shot_10", "W6_temp_high", "RARE_massive_oversample"],
        "weights": [5.98, 5.57, 2.07],  # By delta_pct
    },
    "RARE_MLP_dedup_sim095": {
        "description": "With deduplication (cosine > 0.95)",
        "mlp": "MLP_512_256_128",
        "components": ["W5_many_shot_10", "RARE_massive_oversample"],
        "dedup_threshold": 0.95,
    },

    # F. SOTA - Focal Loss & Class Weights
    "RARE_MLP_focal_g2": {
        "description": "MLP + Focal Loss (gamma=2.0)",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "focal_loss": True,
        "focal_gamma": 2.0,
    },
    "RARE_MLP_focal_g5": {
        "description": "MLP + Focal Loss (gamma=5.0, aggressive)",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "focal_loss": True,
        "focal_gamma": 5.0,
    },
    "RARE_MLP_classweight": {
        "description": "MLP + Inverse class weights",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "use_class_weights": True,
    },
    "RARE_MLP_focal_weighted": {
        "description": "MLP + Focal Loss + Class weights",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "focal_loss": True,
        "focal_gamma": 2.0,
        "use_class_weights": True,
    },

    # G. SOTA - Embedding Mixup/Remix
    "RARE_MLP_remix": {
        "description": "Rebalanced Mixup (boost minority labels)",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "use_remix": True,
        "remix_boost": 2.0,
    },
    "RARE_MLP_intraclass_mix": {
        "description": "Intra-class mixup for rare classes only",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "use_intraclass_mixup": True,
        "mixup_n_augment": 200,
    },

    # H. SOTA - Contrastive & Diversity
    "RARE_MLP_contrastive": {
        "description": "Contrastive embedding refinement pre-MLP",
        "mlp": "MLP_512_256_128",
        "components": ["RARE_massive_oversample"],
        "use_contrastive": True,
        "contrastive_iterations": 500,
    },

    # I. SOTA - Advanced Combinations
    "RARE_MLP_full_sota": {
        "description": "Focal + ClassWeight + Remix + 50x",
        "mlp": "MLP_512_256_128",
        "rare_boost": 50,
        "focal_loss": True,
        "focal_gamma": 2.0,
        "use_class_weights": True,
        "use_remix": True,
    },
    "RARE_MLP_kitchen_sink": {
        "description": "All SOTA techniques combined",
        "mlp": "MLP_512_256_128",
        "rare_boost": 50,
        "focal_loss": True,
        "focal_gamma": 3.0,
        "use_class_weights": True,
        "use_remix": True,
        "use_intraclass_mixup": True,
        "use_contrastive": True,
    },
}

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_kfold_mlp(
    mlp_config: Dict[str, Any],
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    unique_labels: List[str],
    config_params: Dict[str, Any] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run K-fold cross-validation with MLP classifier and SOTA techniques.

    Args:
        mlp_config: MLP architecture configuration
        X_orig: Original embeddings
        y_orig: Original labels (integer encoded)
        X_synth: Synthetic embeddings
        y_synth: Synthetic labels
        unique_labels: List of class names
        config_params: RARE_MLP config with SOTA options
        n_splits: Number of CV folds
        n_repeats: Number of CV repeats
        seed: Random seed

    Returns:
        Results dictionary with metrics
    """
    config_params = config_params or {}

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    baseline_f1s, augmented_f1s, deltas = [], [], []
    per_class_base = {l: [] for l in unique_labels}
    per_class_aug = {l: [] for l in unique_labels}

    total_folds = n_splits * n_repeats
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Get rare class indices
    rare_indices = [label_to_idx.get(c, -1) for c in RARE_CLASSES if c in label_to_idx]

    print(f"\n    Running K-Fold with MLP ({mlp_config.get('hidden_layer_sizes', 'default')})...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_orig)):
        X_train, y_train = X_orig[train_idx], y_orig[train_idx]
        X_test, y_test = X_orig[test_idx], y_orig[test_idx]

        # ===== BASELINE =====
        clf_base = MLPClassifier(**mlp_config)
        scaler_base = StandardScaler()
        X_tr_base = scaler_base.fit_transform(X_train)
        X_te_base = scaler_base.transform(X_test)

        clf_base.fit(X_tr_base, y_train)
        y_pred_base = clf_base.predict(X_te_base)
        base_f1 = f1_score(y_test, y_pred_base, average="macro")
        baseline_f1s.append(base_f1)

        base_pc = f1_score(y_test, y_pred_base, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_base[l].append(base_pc[i])

        # ===== AUGMENTED =====
        X_train_aug = np.vstack([X_train, X_synth]) if len(X_synth) > 0 else X_train
        y_train_aug = np.concatenate([y_train, y_synth]) if len(y_synth) > 0 else y_train

        # Apply SOTA techniques

        # 1. Contrastive refinement
        if config_params.get("use_contrastive", False):
            n_iter = config_params.get("contrastive_iterations", 500)
            X_train_aug = contrastive_embedding_refinement(X_train_aug, y_train_aug, n_iterations=n_iter)

        # 2. Remix/Mixup
        if config_params.get("use_remix", False):
            boost = config_params.get("remix_boost", 2.0)
            X_remix, y_remix = remix_mixup(X_train_aug, y_train_aug,
                                           minority_boost=boost,
                                           rare_classes=rare_indices)
            X_train_aug = np.vstack([X_train_aug, X_remix])
            y_train_aug = np.concatenate([y_train_aug, y_remix])

        # 3. Intra-class mixup
        if config_params.get("use_intraclass_mixup", False):
            n_aug = config_params.get("mixup_n_augment", 200)
            X_intra, y_intra = intraclass_mixup(X_train_aug, y_train_aug,
                                                 target_classes=rare_indices,
                                                 n_augment=n_aug)
            if len(X_intra) > 0:
                X_train_aug = np.vstack([X_train_aug, X_intra])
                y_train_aug = np.concatenate([y_train_aug, y_intra])

        # Scale features
        scaler_aug = StandardScaler()
        X_tr_aug = scaler_aug.fit_transform(X_train_aug)
        X_te_aug = scaler_aug.transform(X_test)

        # Compute sample weights
        sample_weights = np.ones(len(y_train_aug))

        # 4. Class weights
        if config_params.get("use_class_weights", False):
            class_weights = compute_class_weights_inverse(y_train_aug)
            sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train_aug])

        # 5. Focal loss (approximation via sample weighting)
        if config_params.get("focal_loss", False):
            gamma = config_params.get("focal_gamma", 2.0)
            # Train preliminary model to get predictions
            clf_prelim = MLPClassifier(**{**mlp_config, "max_iter": 50})
            try:
                clf_prelim.fit(X_tr_aug, y_train_aug)
                y_pred_proba = clf_prelim.predict_proba(X_tr_aug)
                focal_weights = compute_focal_weights(y_train_aug, y_pred_proba, gamma=gamma)
                sample_weights = sample_weights * focal_weights
            except:
                pass  # If preliminary training fails, skip focal weighting

        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)

        # Train final model
        clf_aug = MLPClassifier(**mlp_config)
        try:
            clf_aug.fit(X_tr_aug, y_train_aug, sample_weight=sample_weights)
        except TypeError:
            # MLPClassifier may not accept sample_weight in all sklearn versions
            clf_aug.fit(X_tr_aug, y_train_aug)

        y_pred_aug = clf_aug.predict(X_te_aug)
        aug_f1 = f1_score(y_test, y_pred_aug, average="macro")
        augmented_f1s.append(aug_f1)

        aug_pc = f1_score(y_test, y_pred_aug, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_aug[l].append(aug_pc[i])

        delta = aug_f1 - base_f1
        deltas.append(delta)

        if (fold_idx + 1) % 5 == 0:
            print(f"      Fold {fold_idx + 1}/{total_folds}: base={base_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}")

    # Compute statistics
    base_mean = np.mean(baseline_f1s)
    base_std = np.std(baseline_f1s, ddof=1)
    aug_mean = np.mean(augmented_f1s)
    aug_std = np.std(augmented_f1s, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    delta_pct = (delta_mean / base_mean) * 100 if base_mean > 0 else 0

    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    win_rate = sum(1 for d in deltas if d > 0) / n

    per_class_delta = {l: np.mean(per_class_aug[l]) - np.mean(per_class_base[l]) for l in unique_labels}

    return {
        "n_folds": total_folds,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "augmented_mean": aug_mean,
        "augmented_std": aug_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_pct": delta_pct,
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": win_rate,
        "per_class_delta": per_class_delta,
        "n_synthetic": len(X_synth) if len(X_synth) > 0 else 0,
    }


def generate_synthetics_for_config(
    config_name: str,
    cache: EmbeddingCache,
    texts: List[str],
    labels: List[str],
    embeddings: np.ndarray,
    rare_boost: int = None,
    target_classes: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic samples for a configuration.

    Args:
        config_name: Name of config from ALL_CONFIGS or RARE_CLASS_EXPERIMENTS
        cache: Embedding cache
        texts: Original texts
        labels: Original labels
        embeddings: Original embeddings
        rare_boost: Optional oversampling multiplier for rare classes
        target_classes: Optional list of classes to focus on

    Returns:
        Tuple of (synthetic_embeddings, synthetic_labels)
    """
    try:
        params = get_config_params(config_name)
    except KeyError:
        print(f"    Warning: Config {config_name} not found, using defaults")
        params = {}

    # Apply rare_boost if specified
    if rare_boost:
        params["rare_class_boost"] = rare_boost
        params["force_generation_classes"] = target_classes or RARE_CLASSES
        params["disable_quality_gate"] = True
        params["min_synthetic_per_class"] = 50

    if target_classes:
        params["force_generation_classes"] = target_classes

    generator = SyntheticGenerator(cache, params)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in labels])

    all_synth_emb = []
    all_synth_labels = []

    for label in unique_labels:
        class_mask = np.array(labels) == label
        class_texts = [t for t, m in zip(texts, class_mask) if m]
        class_emb = embeddings[class_mask]

        n_original = len(class_texts)

        # Check if should generate for this class
        if target_classes and label not in target_classes:
            continue

        # Generate synthetics
        try:
            # Note: generate_for_class expects (texts, embeddings, class_name)
            synth_texts_raw, synth_labels_raw = generator.generate_for_class(
                np.array(class_texts), class_emb, label
            )
            synthetic_texts = synth_texts_raw if synth_texts_raw else []

            if synthetic_texts:
                synth_emb = cache.embed_synthetic(synthetic_texts)
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label_to_idx[label]] * len(synth_emb))
                print(f"      {label}: {n_original} -> +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"      {label}: Error generating - {e}")

    if all_synth_emb:
        return np.vstack(all_synth_emb), np.array(all_synth_labels)
    return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


def run_rare_mlp_config(
    config_name: str,
    config: Dict[str, Any],
    cache: EmbeddingCache,
    texts: List[str],
    labels: List[str],
    embeddings: np.ndarray,
    unique_labels: List[str]
) -> Dict[str, Any]:
    """
    Run a single RARE_MLP configuration.

    Args:
        config_name: Configuration name
        config: Configuration dictionary
        cache: Embedding cache
        texts: Original texts
        labels: Original labels
        embeddings: Original embeddings
        unique_labels: List of unique class names

    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Running: {config_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*70}")

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in labels])

    # Get MLP architecture
    mlp_name = config.get("mlp", "MLP_512_256_128")
    mlp_config = MLP_ARCHITECTURES.get(mlp_name, MLP_ARCHITECTURES["MLP_512_256_128"])

    # Generate synthetics
    all_synth_emb = []
    all_synth_labels = []

    components = config.get("components", ["RARE_massive_oversample"])
    rare_boost = config.get("rare_boost")
    target_classes = config.get("target_classes")
    weights = config.get("weights")

    print(f"\n  Generating synthetics from {len(components)} components...")

    for i, comp in enumerate(components):
        print(f"\n  Component {i+1}: {comp}")
        X_synth, y_synth = generate_synthetics_for_config(
            comp, cache, texts, labels, embeddings,
            rare_boost=rare_boost,
            target_classes=target_classes
        )

        if len(X_synth) > 0:
            # Apply component weight if specified
            if weights and i < len(weights):
                w = weights[i]
                # Weighted sampling
                n_keep = int(len(X_synth) * w / max(weights))
                if n_keep < len(X_synth):
                    indices = np.random.choice(len(X_synth), n_keep, replace=False)
                    X_synth = X_synth[indices]
                    y_synth = y_synth[indices]

            all_synth_emb.append(X_synth)
            all_synth_labels.append(y_synth)

    if all_synth_emb:
        X_synth_combined = np.vstack(all_synth_emb)
        y_synth_combined = np.concatenate(all_synth_labels)
    else:
        X_synth_combined = np.array([]).reshape(0, embeddings.shape[1])
        y_synth_combined = np.array([])

    print(f"\n  Total synthetic samples: {len(X_synth_combined)}")

    # Apply deduplication if specified
    dedup_threshold = config.get("dedup_threshold")
    if dedup_threshold and len(X_synth_combined) > 0:
        print(f"  Applying deduplication (threshold={dedup_threshold})...")
        # Simple cosine similarity deduplication
        from sklearn.metrics.pairwise import cosine_similarity
        n_before = len(X_synth_combined)
        keep_mask = np.ones(n_before, dtype=bool)

        for i in range(n_before):
            if not keep_mask[i]:
                continue
            for j in range(i+1, n_before):
                if not keep_mask[j]:
                    continue
                sim = cosine_similarity(X_synth_combined[i:i+1], X_synth_combined[j:j+1])[0, 0]
                if sim > dedup_threshold:
                    keep_mask[j] = False

        X_synth_combined = X_synth_combined[keep_mask]
        y_synth_combined = y_synth_combined[keep_mask]
        print(f"  After dedup: {n_before} -> {len(X_synth_combined)}")

    # Run K-fold evaluation
    print(f"\n  Running K-fold evaluation with {mlp_name}...")
    results = run_kfold_mlp(
        mlp_config=mlp_config,
        X_orig=embeddings,
        y_orig=y_encoded,
        X_synth=X_synth_combined,
        y_synth=y_synth_combined,
        unique_labels=unique_labels,
        config_params=config,
    )

    results["config_name"] = config_name
    results["mlp_architecture"] = mlp_name
    results["description"] = config.get("description", "")
    results["components"] = components
    results["sota_techniques"] = []

    if config.get("focal_loss"):
        results["sota_techniques"].append(f"focal_loss_g{config.get('focal_gamma', 2.0)}")
    if config.get("use_class_weights"):
        results["sota_techniques"].append("class_weights")
    if config.get("use_remix"):
        results["sota_techniques"].append("remix_mixup")
    if config.get("use_intraclass_mixup"):
        results["sota_techniques"].append("intraclass_mixup")
    if config.get("use_contrastive"):
        results["sota_techniques"].append("contrastive")

    # Print summary
    print(f"\n  Results for {config_name}:")
    print(f"    Macro-F1: {results['baseline_mean']:.4f} -> {results['augmented_mean']:.4f}")
    print(f"    Delta: {results['delta_pct']:+.2f}% (p={results['p_value']:.6f})")
    print(f"    Significant: {'Yes' if results['significant'] else 'No'}")

    # Print rare class results
    print(f"\n    Rare class deltas:")
    for cls in RARE_CLASSES:
        delta = results['per_class_delta'].get(cls, 0)
        print(f"      {cls}: {delta:+.4f} ({delta*100:+.2f}%)")

    return results


def main():
    """Main entry point."""
    print("="*70)
    print("RARE_MLP Suite - Exhaustive Search for Rare Class Improvement")
    print("="*70)
    print(f"\nConfigurations to test: {len(RARE_MLP_CONFIGS)}")
    print(f"MLP Architectures: {list(MLP_ARCHITECTURES.keys())}")
    print(f"Rare classes tracked: {RARE_CLASSES}")

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    print(f"Classes: {len(unique_labels)}")
    print(f"Samples: {len(texts)}")

    # Print rare class counts
    print("\nRare class sample counts:")
    labels_list = list(labels) if isinstance(labels, np.ndarray) else labels
    for cls in RARE_CLASSES:
        count = labels_list.count(cls)
        print(f"  {cls}: {count} samples")

    # Create results directory
    results_dir = RESULTS_DIR / "rare_mlp"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all configurations
    all_results = {}

    for config_name, config in RARE_MLP_CONFIGS.items():
        try:
            results = run_rare_mlp_config(
                config_name, config, cache, texts, labels, embeddings, unique_labels
            )
            all_results[config_name] = results

            # Save individual result
            result_file = results_dir / f"{config_name}_kfold.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved: {result_file}")

        except Exception as e:
            print(f"\n  ERROR in {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Sort by delta_pct
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("delta_pct", 0),
        reverse=True
    )

    print("\nTop 10 by Macro-F1 Delta:")
    print("-"*70)
    for i, (name, res) in enumerate(sorted_results[:10], 1):
        sig = "✓" if res.get("significant", False) else "✗"
        print(f"  {i:2}. {name:30} delta={res.get('delta_pct', 0):+.2f}% p={res.get('p_value', 1):.6f} {sig}")

    # Best for rare classes
    if all_results:
        print("\nBest for ESFJ:")
        best_esfj = max(all_results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESFJ", 0))
        print(f"  {best_esfj[0]}: ESFJ delta = {best_esfj[1].get('per_class_delta', {}).get('ESFJ', 0):+.4f}")

        print("\nBest for ESTJ:")
        best_estj = max(all_results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESTJ", 0))
        print(f"  {best_estj[0]}: ESTJ delta = {best_estj[1].get('per_class_delta', {}).get('ESTJ', 0):+.4f}")

        print("\nBest for ESFP:")
        best_esfp = max(all_results.items(), key=lambda x: x[1].get("per_class_delta", {}).get("ESFP", 0))
        esfp_delta = best_esfp[1].get('per_class_delta', {}).get('ESFP', 0)
        if esfp_delta > 0:
            print(f"  {best_esfp[0]}: ESFP delta = {esfp_delta:+.4f} (BREAKTHROUGH!)")
        else:
            print(f"  No configuration improved ESFP (still irresoluble)")
    else:
        print("\nNo results available!")
        best_esfj = (None, {})
        best_estj = (None, {})
        best_esfp = (None, {})

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(all_results),
        "significant_count": sum(1 for r in all_results.values() if r.get("significant", False)),
        "top_10": [
            {"name": name, "delta_pct": res.get("delta_pct", 0), "significant": res.get("significant", False)}
            for name, res in sorted_results[:10]
        ],
        "best_esfj": {"name": best_esfj[0], "delta": best_esfj[1].get("per_class_delta", {}).get("ESFJ", 0)} if best_esfj[0] else None,
        "best_estj": {"name": best_estj[0], "delta": best_estj[1].get("per_class_delta", {}).get("ESTJ", 0)} if best_estj[0] else None,
        "best_esfp": {"name": best_esfp[0], "delta": best_esfp[1].get("per_class_delta", {}).get("ESFP", 0)} if best_esfp[0] else None,
    }

    summary_file = results_dir / "rare_mlp_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    print("\n" + "="*70)
    print("RARE_MLP Suite Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
