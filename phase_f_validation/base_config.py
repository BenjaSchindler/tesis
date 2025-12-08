#!/usr/bin/env python3
"""
Base Configuration for Phase F Validation Experiments

Parameters from CMB3_skip (best single performer +0.57%)
All experiments use these as defaults, overriding only the parameter being tested.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
VALIDATION_DIR = PROJECT_ROOT / "phase_f_validation"
CACHE_DIR = VALIDATION_DIR / "cache"
RESULTS_DIR = VALIDATION_DIR / "results"
LATEX_DIR = VALIDATION_DIR / "latex_output"

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embeddings_mpnet.npy"
LABELS_CACHE_PATH = CACHE_DIR / "labels.npy"
TEXTS_CACHE_PATH = CACHE_DIR / "texts.pkl"

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
MAX_CONCURRENT_API_CALLS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150

# Base parameters from CMB3_skip (best single config)
BASE_PARAMS = {
    # Generation volume
    "max_clusters": 5,
    "prompts_per_cluster": 9,
    "samples_per_prompt": 5,

    # Anchor selection
    "auto_anchor_margin": 0.05,
    "anchor_selection_ratio": 0.8,
    "anchor_outlier_threshold": 1.5,

    # Filtering thresholds
    "similarity_threshold": 0.90,
    "min_classifier_confidence": 0.05,
    "contamination_threshold": 0.95,
    "filter_mode": "hybrid",

    # F1 budget scaling
    "use_f1_budget_scaling": True,
    "f1_budget_thresholds": [0.35, 0.20],
    "f1_budget_multipliers": [0.0, 0.5, 2.5],

    # Evaluation
    "synthetic_weight": 0.5,
    "synthetic_weight_mode": "flat",

    # Validation gating
    "use_val_gating": True,
    "val_size": 0.15,
    "val_tolerance": 0.02,

    # K-NN support for prompts (parameter being validated in exp03)
    "k_neighbors": 8,

    # Class ratio cap
    "cap_class_ratio": 0.15,
}

# K-Fold CV configuration
KFOLD_CONFIG = {
    "n_splits": 5,
    "n_repeats": 3,
    "random_state": 42,
}

# Experiment-specific parameter ranges
EXPERIMENT_PARAMS = {
    "clustering": {
        "K_MAX_VALUES": [1, 2, 3, 6, 12, 24],
    },
    "anchor_strategies": {
        "STRATEGIES": ["random", "nearest_neighbor", "medoid",
                       "quality_gated", "diverse", "ensemble"],
    },
    "k_neighbors": {
        "K_VALUES": [5, 10, 15, 25, 50, 75, 100],
    },
    "filter_cascade": {
        "CONFIGS": ["length_only", "length_similarity",
                    "three_partial", "full_cascade"],
    },
    "adaptive_thresholds": {
        "CONFIGS": [
            {"name": "fixed_permissive", "threshold": 0.60, "adaptive": False},
            {"name": "fixed_medium", "threshold": 0.70, "adaptive": False},
            {"name": "fixed_strict", "threshold": 0.90, "adaptive": False},
            {"name": "adaptive", "threshold": None, "adaptive": True},
        ],
    },
    "tier_impact": {
        "TIERS": {
            "LOW": {"min": 0.0, "max": 0.20},
            "MID": {"min": 0.20, "max": 0.45},
            "HIGH": {"min": 0.45, "max": 1.0},
        },
    },
}

# MBTI class info
MBTI_CLASSES = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]
