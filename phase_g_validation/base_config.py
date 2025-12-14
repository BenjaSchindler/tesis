#!/usr/bin/env python3
"""
Base Configuration for Phase G Validation Experiments

OPTIMAL parameters discovered in Phase F validation:
- synthetic_weight: 1.0 (was 0.5) - +1.41% improvement
- temperature: 0.3 (was 0.7) - +0.34% improvement
- max_clusters: 12 (was 5) - better class coverage
- anchor_strategy: medoid (p=0.005 significant)
- k_neighbors: 15 (was 8) - better prompt context

All experiments use these as defaults, with config-specific overrides
for parameters that define each config's unique purpose.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
VALIDATION_DIR = PROJECT_ROOT / "phase_g_validation"
CACHE_DIR = VALIDATION_DIR / "cache"
RESULTS_DIR = VALIDATION_DIR / "results"
LATEX_DIR = VALIDATION_DIR / "latex_output"

# Phase G reference (for reading original configs)
PHASE_G_DIR = PROJECT_ROOT / "phase_g"
PHASE_G_CONFIGS = PHASE_G_DIR / "configs"

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embeddings_mpnet.npy"
LABELS_CACHE_PATH = CACHE_DIR / "labels.npy"
TEXTS_CACHE_PATH = CACHE_DIR / "texts.pkl"

# LLM configuration - UPDATED with optimal temperature
LLM_MODEL = "gpt-4o-mini"
MAX_CONCURRENT_API_CALLS = 25  # High tier - increased from 10
TEMPERATURE = 0.3  # OPTIMAL (was 0.7)
MAX_TOKENS = 150

# Hardware optimization
DEVICE = "cuda"  # RTX 3090
EMBEDDING_BATCH_SIZE = 256  # Optimized for 24GB VRAM

# OPTIMAL base parameters from Phase F validation
BASE_PARAMS = {
    # Generation volume - UPDATED
    "max_clusters": 12,  # OPTIMAL (was 5)
    "prompts_per_cluster": 9,
    "samples_per_prompt": 5,

    # Anchor selection - UPDATED
    "anchor_strategy": "medoid",  # OPTIMAL (was random), p=0.005
    "auto_anchor_margin": 0.05,
    "anchor_selection_ratio": 0.8,
    "anchor_outlier_threshold": 1.5,

    # Prompt context - UPDATED
    "k_neighbors": 15,  # OPTIMAL (was 8)

    # Filtering thresholds
    "similarity_threshold": 0.90,
    "min_classifier_confidence": 0.05,
    "contamination_threshold": 0.95,
    "filter_mode": "hybrid",

    # F1 budget scaling
    "use_f1_budget_scaling": True,
    "f1_budget_thresholds": [0.35, 0.20],
    "f1_budget_multipliers": [0.0, 0.5, 2.5],

    # Evaluation - UPDATED (MAJOR CHANGE)
    "synthetic_weight": 1.0,  # OPTIMAL (was 0.5) - +1.41% improvement
    "synthetic_weight_mode": "flat",

    # Validation gating
    "use_val_gating": True,
    "val_size": 0.15,
    "val_tolerance": 0.02,

    # Class ratio cap
    "cap_class_ratio": 0.15,

    # LLM settings
    "temperature": 0.3,  # OPTIMAL (was 0.7)
    "llm_model": "gpt-4o-mini",
}

# K-Fold CV configuration
KFOLD_CONFIG = {
    "n_splits": 5,
    "n_repeats": 3,
    "random_state": 42,
}

# MBTI class info
MBTI_CLASSES = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

# Problem classes (from Phase G analysis)
PROBLEM_CLASSES = ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"]

# Parallel execution settings
PARALLEL_CONFIG = {
    "max_concurrent_configs": 4,  # Run 4 configs in parallel
    "max_api_calls_per_config": 25,  # High OpenAI tier
    "embedding_batch_size": 256,  # RTX 3090 optimized
}
