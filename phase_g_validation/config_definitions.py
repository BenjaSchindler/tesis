#!/usr/bin/env python3
"""
Configuration Definitions for Phase G Validation

Each config inherits OPTIMAL base parameters from Phase F and overrides
specific parameters that define its unique experimental purpose.

CRUCIAL parameters are marked - these MUST be preserved because they
define what the experiment is testing.
"""

from typing import Dict, Any, List
import numpy as np
from base_config import BASE_PARAMS

# ==============================================================================
# PHASE F ENSEMBLE COMPONENTS (4 configs)
# ==============================================================================

PHASE_F_COMPONENTS = {
    "CMB3_skip": {
        "description": "F1-budget scaling with relaxed confidence (Phase F best single)",
        "wave": "component",
        "crucial_params": ["use_f1_budget_scaling", "f1_budget_thresholds", "f1_budget_multipliers"],
        "overrides": {
            "auto_anchor_margin": 0.05,
            "max_clusters": 5,  # Original value - part of design
            "prompts_per_cluster": 9,
            "samples_per_prompt": 5,
            "min_classifier_confidence": 0.05,
            "filter_mode": "hybrid",
            "use_f1_budget_scaling": True,  # CRUCIAL
            "f1_budget_thresholds": [0.35, 0.20],  # CRUCIAL
            "f1_budget_multipliers": [0.0, 0.5, 2.5],  # CRUCIAL
        }
    },
    "CF1_conf_band": {
        "description": "Confidence band 0.3-0.7 filtering",
        "wave": "component",
        "crucial_params": ["min_classifier_confidence", "max_classifier_confidence"],
        "overrides": {
            "auto_anchor_margin": 0.02,
            "max_clusters": 5,
            "prompts_per_cluster": 9,
            "samples_per_prompt": 5,
            "min_classifier_confidence": 0.3,  # CRUCIAL - band lower
            "max_classifier_confidence": 0.7,  # CRUCIAL - band upper
            "filter_mode": "hybrid",
        }
    },
    "V4_ultra": {
        "description": "High volume generation (8x12x7=672 candidates)",
        "wave": "component",
        "crucial_params": ["max_clusters", "prompts_per_cluster", "samples_per_prompt"],
        "overrides": {
            "auto_anchor_margin": 0.02,
            "max_clusters": 8,  # CRUCIAL - volume
            "prompts_per_cluster": 12,  # CRUCIAL - volume
            "samples_per_prompt": 7,  # CRUCIAL - volume
        }
    },
    "G5_K25_medium": {
        "description": "High K=25 samples per prompt with reasoning",
        "wave": "component",
        "crucial_params": ["samples_per_prompt", "max_completion_tokens"],
        "overrides": {
            "auto_anchor_margin": 0.02,
            "max_clusters": 5,
            "prompts_per_cluster": 9,
            "samples_per_prompt": 25,  # CRUCIAL - high sample depth
            "max_completion_tokens": 2048,  # CRUCIAL - extended output
        }
    },
}

# ==============================================================================
# WAVE 1: GATE EXPERIMENTS (3 configs)
# ==============================================================================

WAVE1_GATES = {
    "W1_low_gate": {
        "description": "Very low gate threshold (0.05) - tests permissive gating",
        "wave": "wave1",
        "crucial_params": ["anchor_quality_threshold", "purity_gate_threshold", "min_classifier_confidence"],
        "overrides": {
            "anchor_quality_threshold": 0.05,  # CRUCIAL - very low
            "purity_gate_threshold": 0.005,  # CRUCIAL - very low
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
            "min_classifier_confidence": 0.01,  # CRUCIAL - very permissive
        }
    },
    "W1_no_gate": {
        "description": "Disable ALL quality gates - baseline no-filter approach",
        "wave": "wave1",
        "crucial_params": ["disable_quality_gate"],
        "overrides": {
            "disable_quality_gate": True,  # CRUCIAL - no gates
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
            "min_classifier_confidence": 0.01,
            "filter_mode": "hybrid",
        }
    },
    "W1_force_problem": {
        "description": "Force generation for problem classes (ENFJ,ESFJ,ESFP,ESTJ,ISTJ)",
        "wave": "wave1",
        "crucial_params": ["force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            "max_clusters": 8,
            "prompts_per_cluster": 15,
            "samples_per_prompt": 7,
            "min_classifier_confidence": 0.01,
        }
    },
}

# ==============================================================================
# WAVE 2: VOLUME EXPERIMENTS (2 configs)
# ==============================================================================

WAVE2_VOLUME = {
    "W2_ultra_vol": {
        "description": "Ultra high volume (10x15x10=1500 candidates per class)",
        "wave": "wave2",
        "crucial_params": ["max_clusters", "prompts_per_cluster", "samples_per_prompt", "disable_quality_gate"],
        "overrides": {
            "disable_quality_gate": True,  # CRUCIAL - no filtering
            "max_clusters": 10,  # CRUCIAL - volume
            "prompts_per_cluster": 15,  # CRUCIAL - volume
            "samples_per_prompt": 10,  # CRUCIAL - volume
            "min_classifier_confidence": 0.01,
        }
    },
    "W2_mega_vol": {
        "description": "Mega volume (12x20x12=2880 candidates per class)",
        "wave": "wave2",
        "crucial_params": ["max_clusters", "prompts_per_cluster", "samples_per_prompt", "cap_class_ratio"],
        "overrides": {
            "disable_quality_gate": True,
            "max_clusters": 12,  # CRUCIAL - volume
            "prompts_per_cluster": 20,  # CRUCIAL - volume
            "samples_per_prompt": 12,  # CRUCIAL - volume
            "min_classifier_confidence": 0.01,
            "cap_class_ratio": 0.30,  # CRUCIAL - higher cap for mega volume
        }
    },
}

# ==============================================================================
# WAVE 3: FILTER EXPERIMENTS (2 configs)
# ==============================================================================

WAVE3_FILTERS = {
    "W3_permissive_filter": {
        "description": "Very permissive filters (high similarity threshold)",
        "wave": "wave3",
        "crucial_params": ["similarity_threshold", "dedup_embed_sim", "filter_mode"],
        "overrides": {
            "disable_quality_gate": True,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
            "min_classifier_confidence": 0.01,
            "similarity_threshold": 0.98,  # CRUCIAL - very permissive
            "dedup_embed_sim": 0.99,  # CRUCIAL - minimal dedup
            "filter_mode": "classifier",  # CRUCIAL - classifier only
        }
    },
    "W3_no_dedup": {
        "description": "Disable deduplication entirely",
        "wave": "wave3",
        "crucial_params": ["dedup_embed_sim", "duplicate_threshold"],
        "overrides": {
            "disable_quality_gate": True,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
            "min_classifier_confidence": 0.01,
            "dedup_embed_sim": 1.0,  # CRUCIAL - no dedup
            "duplicate_threshold": 1.0,  # CRUCIAL - no dedup
        }
    },
}

# ==============================================================================
# WAVE 4: TARGETING EXPERIMENTS (1 config)
# ==============================================================================

WAVE4_TARGETING = {
    "W4_target_only": {
        "description": "Target ONLY problematic classes with F1 budget scaling",
        "wave": "wave4",
        "crucial_params": ["force_generation_classes", "use_f1_budget_scaling", "f1_budget_thresholds", "f1_budget_multipliers"],
        "overrides": {
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            "use_f1_budget_scaling": True,  # CRUCIAL
            "f1_budget_thresholds": [0.20, 0.10],  # CRUCIAL - more aggressive
            "f1_budget_multipliers": [0.0, 0.5, 3.0],  # CRUCIAL - higher multiplier
            "max_clusters": 10,
            "prompts_per_cluster": 15,
            "samples_per_prompt": 10,
        }
    },
}

# ==============================================================================
# WAVE 5: PROMPTING EXPERIMENTS (3 configs)
# ==============================================================================

WAVE5_PROMPTING = {
    "W5_zero_shot": {
        "description": "Zero-shot prompting (no examples in prompt)",
        "wave": "wave5",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 0,  # CRUCIAL - zero shot
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_few_shot_3": {
        "description": "3-shot prompting",
        "wave": "wave5",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 3,  # CRUCIAL - 3 examples
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_many_shot_10": {
        "description": "10-shot prompting (many examples)",
        "wave": "wave5",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 10,  # CRUCIAL - 10 examples
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
}

# ==============================================================================
# WAVE 5 EXTENDED: MANY-SHOT EXPERIMENTS
# Testing n_shot from 20 to 200 with fine granularity around optimal (100)
# ==============================================================================

WAVE5_EXTENDED_NSHOT = {
    "W5_shot_20": {
        "description": "20-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 20,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_30": {
        "description": "30-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 30,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_40": {
        "description": "40-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 40,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_50": {
        "description": "50-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 50,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_60": {
        "description": "60-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_70": {
        "description": "70-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 70,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_80": {
        "description": "80-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 80,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_90": {
        "description": "90-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 90,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_100": {
        "description": "100-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 100,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_110": {
        "description": "110-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 110,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_120": {
        "description": "120-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 120,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_130": {
        "description": "130-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 130,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_140": {
        "description": "140-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 140,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_150": {
        "description": "150-shot prompting",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 150,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5_shot_200": {
        "description": "200-shot prompting - maximum",
        "wave": "wave5_ext",
        "crucial_params": ["n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 200,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
}

# ==============================================================================
# WAVE 5b: TEMPERATURE WITH OPTIMAL N_SHOT=60 (5 configs)
# Tests temperature interaction with optimal prompting
# ==============================================================================

WAVE5B_TEMP_NSHOT60 = {
    "W5b_temp03_n60": {
        "description": "Temperature 0.3 with optimal n_shot=60",
        "wave": "wave5b_temp",
        "crucial_params": ["temperature", "n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "temperature": 0.3,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5b_temp06_n60": {
        "description": "Temperature 0.6 with optimal n_shot=60",
        "wave": "wave5b_temp",
        "crucial_params": ["temperature", "n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "temperature": 0.6,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5b_temp09_n60": {
        "description": "Temperature 0.9 with optimal n_shot=60",
        "wave": "wave5b_temp",
        "crucial_params": ["temperature", "n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "temperature": 0.9,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5b_temp12_n60": {
        "description": "Temperature 1.2 with optimal n_shot=60",
        "wave": "wave5b_temp",
        "crucial_params": ["temperature", "n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "temperature": 1.2,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W5b_temp15_n60": {
        "description": "Temperature 1.5 with optimal n_shot=60",
        "wave": "wave5b_temp",
        "crucial_params": ["temperature", "n_shot"],
        "overrides": {
            "disable_quality_gate": True,
            "n_shot": 60,
            "temperature": 1.5,
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
}

# ==============================================================================
# WAVE 6: TEMPERATURE EXPERIMENTS (3 configs)
# Note: W6_temp_low uses 0.3, which matches the optimal!
# ==============================================================================

WAVE6_TEMPERATURE = {
    "W6_temp_low": {
        "description": "Low temperature (0.3) - MATCHES OPTIMAL!",
        "wave": "wave6",
        "crucial_params": ["temperature"],
        "overrides": {
            "disable_quality_gate": True,
            "temperature": 0.3,  # CRUCIAL - matches Phase F optimal!
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W6_temp_high": {
        "description": "High temperature (0.9) for diversity",
        "wave": "wave6",
        "crucial_params": ["temperature"],
        "overrides": {
            "disable_quality_gate": True,
            "temperature": 0.9,  # CRUCIAL - high temp test
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W6_temp_extreme": {
        "description": "Extreme temperature (1.0) for maximum diversity",
        "wave": "wave6",
        "crucial_params": ["temperature"],
        "overrides": {
            "disable_quality_gate": True,
            "temperature": 1.0,  # CRUCIAL - extreme temp test
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
}

# ==============================================================================
# WAVE 7: YOLO EXPERIMENTS (2 configs)
# ==============================================================================

WAVE7_YOLO = {
    "W7_yolo": {
        "description": "YOLO mode - ALL filters disabled, maximum volume",
        "wave": "wave7",
        "crucial_params": ["disable_quality_gate", "min_classifier_confidence", "similarity_threshold", "dedup_embed_sim", "cap_class_ratio"],
        "overrides": {
            "disable_quality_gate": True,  # CRUCIAL - no gates
            "max_clusters": 10,
            "prompts_per_cluster": 15,
            "samples_per_prompt": 10,
            "min_classifier_confidence": 0.0,  # CRUCIAL - accept everything
            "similarity_threshold": 0.99,  # CRUCIAL - no similarity filter
            "dedup_embed_sim": 1.0,  # CRUCIAL - no dedup
            "filter_mode": "classifier",
            "cap_class_ratio": 0.50,  # CRUCIAL - high cap
        }
    },
    "W7_yolo_force": {
        "description": "YOLO mode + force problem classes",
        "wave": "wave7",
        "crucial_params": ["disable_quality_gate", "force_generation_classes", "min_classifier_confidence"],
        "overrides": {
            "disable_quality_gate": True,  # CRUCIAL
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            "max_clusters": 10,
            "prompts_per_cluster": 15,
            "samples_per_prompt": 10,
            "min_classifier_confidence": 0.0,  # CRUCIAL
            "similarity_threshold": 0.99,
            "dedup_embed_sim": 1.0,
            "cap_class_ratio": 0.50,
        }
    },
}

# ==============================================================================
# WAVE 8: MODEL EXPERIMENTS (2 configs)
# Note: These use gpt-5-mini instead of gpt-4o-mini
# ==============================================================================

WAVE8_MODELS = {
    "W8_gpt5_reasoning": {
        "description": "GPT-5-mini with medium reasoning effort",
        "wave": "wave8",
        "crucial_params": ["llm_model", "reasoning_effort"],
        "overrides": {
            "disable_quality_gate": True,
            "llm_model": "gpt-5-mini",  # CRUCIAL - different model
            "reasoning_effort": "medium",  # CRUCIAL - reasoning mode
            "max_clusters": 8,
            "prompts_per_cluster": 10,
            "samples_per_prompt": 5,
        }
    },
    "W8_gpt5_high": {
        "description": "GPT-5-mini with high reasoning effort",
        "wave": "wave8",
        "crucial_params": ["llm_model", "reasoning_effort"],
        "overrides": {
            "disable_quality_gate": True,
            "llm_model": "gpt-5-mini",  # CRUCIAL - different model
            "reasoning_effort": "high",  # CRUCIAL - high reasoning
            "max_clusters": 6,
            "prompts_per_cluster": 8,
            "samples_per_prompt": 4,
        }
    },
}

# ==============================================================================
# WAVE 9: COMBINATION EXPERIMENTS (2 configs)
# ==============================================================================

WAVE9_COMBINATIONS = {
    "W9_contrastive": {
        "description": "Contrastive prompting to differentiate from confuser classes",
        "wave": "wave9",
        "crucial_params": ["use_contrastive_prompting", "contrastive_top_k"],
        "overrides": {
            "disable_quality_gate": True,
            "use_contrastive_prompting": True,  # CRUCIAL - contrastive mode
            "contrastive_top_k": 3,  # CRUCIAL - top 3 confusers
            "max_clusters": 8,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 7,
        }
    },
    "W9_best_combo": {
        "description": "Best combination from previous phases",
        "wave": "wave9",
        "crucial_params": ["force_generation_classes", "use_contrastive_prompting", "use_f1_budget_scaling"],
        "overrides": {
            "disable_quality_gate": True,
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            "use_contrastive_prompting": True,  # CRUCIAL
            "contrastive_top_k": 2,
            "max_clusters": 10,
            "prompts_per_cluster": 15,
            "samples_per_prompt": 10,
            "use_f1_budget_scaling": True,  # CRUCIAL
            "f1_budget_thresholds": [0.25, 0.15],
            "f1_budget_multipliers": [0.0, 0.5, 3.0],
        }
    },
}

# ==============================================================================
# PHASE F DERIVED: Focused on Problem Classes (3 configs)
# Based on Phase F validation findings targeting LOW tier classes
# ==============================================================================

PHASE_F_DERIVED = {
    "PF_tier_boost": {
        "description": "Tier-based weighting: LOW=2.0, MID=0.8, HIGH=0.3 (from exp07a)",
        "wave": "pf_derived",
        "crucial_params": ["tier_weights", "synthetic_weight_mode"],
        "overrides": {
            # Use tier-based weighting instead of flat weight
            "synthetic_weight_mode": "tier",  # CRUCIAL - tier-based
            "tier_weights": {  # CRUCIAL - boost LOW tier
                "LOW": 2.0,  # Problem classes get 2x weight
                "MID": 0.8,
                "HIGH": 0.3,
            },
            # Standard generation params
            "max_clusters": 12,
            "prompts_per_cluster": 9,
            "samples_per_prompt": 5,
            "temperature": 0.3,  # Optimal from Phase F
        }
    },
    "PF_high_budget_problem": {
        "description": "High budget (25%) + force problem classes only",
        "wave": "pf_derived",
        "crucial_params": ["budget", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            "budget": 0.25,  # CRUCIAL - 25% (vs 12% optimal for all classes)
            "max_clusters": 12,
            "prompts_per_cluster": 12,
            "samples_per_prompt": 8,
            "temperature": 0.3,
            # More aggressive F1 scaling for problem classes
            "use_f1_budget_scaling": True,
            "f1_budget_thresholds": [0.15, 0.05],  # Lower thresholds
            "f1_budget_multipliers": [0.0, 1.0, 4.0],  # Higher multiplier for worst
        }
    },
    "PF_optimal_focused": {
        "description": "Full Phase F optimal + force problem classes + contrastive",
        "wave": "pf_derived",
        "crucial_params": ["force_generation_classes", "use_contrastive_prompting", "filter_method"],
        "overrides": {
            # Phase F optimal params
            "max_clusters": 12,
            "anchor_strategy": "medoid",
            "k_neighbors": 15,
            "filter_method": "full_cascade",  # CRUCIAL - best filter
            "selection_method": "adaptive_ranking",
            "temperature": 0.3,
            "budget": 0.15,  # Slightly higher
            # Problem class focus
            "force_generation_classes": ["ENFJ", "ESFJ", "ESFP", "ESTJ", "ISTJ"],  # CRUCIAL
            # Add contrastive to help differentiate from confuser classes
            "use_contrastive_prompting": True,  # CRUCIAL
            "contrastive_top_k": 2,
        }
    },
}

# ==============================================================================
# RARE CLASS EXPERIMENTS (5 configs) - Target ESFJ, ESFP, ESTJ (<50 samples)
# ==============================================================================

RARE_CLASS_EXPERIMENTS = {
    "RARE_massive_oversample": {
        "description": "Generate 5x more synthetics for rare classes (min 100 per class)",
        "wave": "rare_class",
        "crucial_params": ["min_synthetic_per_class", "rare_class_boost", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],  # CRUCIAL - only rare classes
            "min_synthetic_per_class": 100,  # CRUCIAL - force minimum 100
            "rare_class_boost": 5.0,  # CRUCIAL - 5x multiplier for rare classes
            "samples_per_prompt": 15,  # More samples per prompt
            "prompts_per_cluster": 15,  # More prompts
            "max_clusters": 3,  # Fewer clusters = more samples per cluster
            "disable_quality_gate": True,  # Don't filter - we need volume
            "similarity_threshold": 0.99,  # Very permissive
        }
    },
    "RARE_yolo_extreme": {
        "description": "No filtering at all - maximum generation for rare classes",
        "wave": "rare_class",
        "crucial_params": ["disable_quality_gate", "disable_all_filters", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],  # CRUCIAL
            "disable_quality_gate": True,  # CRUCIAL
            "disable_all_filters": True,  # CRUCIAL - no similarity, no confidence filter
            "samples_per_prompt": 20,  # Maximum samples
            "prompts_per_cluster": 10,
            "max_clusters": 2,
            "min_classifier_confidence": 0.0,  # Accept everything
            "similarity_threshold": 1.0,  # No similarity filtering
            "contamination_threshold": 1.0,  # No contamination filtering
        }
    },
    "RARE_few_shot_expert": {
        "description": "Use ALL samples from rare class in each prompt (max context)",
        "wave": "rare_class",
        "crucial_params": ["n_shot", "use_all_examples", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],  # CRUCIAL
            "n_shot": 40,  # CRUCIAL - use up to 40 examples (nearly all samples)
            "use_all_examples": True,  # CRUCIAL - don't sample, use all
            "samples_per_prompt": 10,
            "prompts_per_cluster": 20,  # More prompts since we have fixed examples
            "max_clusters": 1,  # Single cluster = all examples available
            "temperature": 0.5,  # Moderate diversity
        }
    },
    "RARE_high_temperature": {
        "description": "High temperature for maximum diversity in rare class generation",
        "wave": "rare_class",
        "crucial_params": ["temperature", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],  # CRUCIAL
            "temperature": 0.9,  # CRUCIAL - high diversity
            "samples_per_prompt": 10,
            "prompts_per_cluster": 12,
            "max_clusters": 3,
            "similarity_threshold": 0.85,  # Allow more diverse samples
        }
    },
    "RARE_contrastive_transfer": {
        "description": "Use similar larger classes to guide rare class generation",
        "wave": "rare_class",
        "crucial_params": ["use_contrastive_prompting", "contrast_source_classes", "force_generation_classes"],
        "overrides": {
            "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],  # CRUCIAL
            "use_contrastive_prompting": True,  # CRUCIAL
            # ESFJ similar to ISFJ (introverted version), ENFJ (feeling version)
            # ESFP similar to ISFP, ENFP
            # ESTJ similar to ISTJ, ENTJ
            "contrast_source_classes": {  # CRUCIAL
                "ESFJ": ["ISFJ", "ENFJ"],  # Use these as reference
                "ESFP": ["ISFP", "ENFP"],
                "ESTJ": ["ISTJ", "ENTJ"],
            },
            "contrastive_prompt_style": "transfer",  # Generate "like X but more extroverted"
            "samples_per_prompt": 8,
            "prompts_per_cluster": 10,
            "max_clusters": 5,
        }
    },
}

# ==============================================================================
# ALL CONFIGS COMBINED
# ==============================================================================

ALL_CONFIGS: Dict[str, Dict[str, Any]] = {
    **PHASE_F_COMPONENTS,
    **WAVE1_GATES,
    **WAVE2_VOLUME,
    **WAVE3_FILTERS,
    **WAVE4_TARGETING,
    **WAVE5_PROMPTING,
    **WAVE5_EXTENDED_NSHOT,  # Extended many-shot experiments
    **WAVE5B_TEMP_NSHOT60,   # Temperature with optimal n_shot=60
    **WAVE6_TEMPERATURE,
    **WAVE7_YOLO,
    **WAVE8_MODELS,
    **WAVE9_COMBINATIONS,
    **PHASE_F_DERIVED,
    **RARE_CLASS_EXPERIMENTS,
}

# ==============================================================================
# ENSEMBLE DEFINITIONS
# ==============================================================================

ENSEMBLES = {
    "ENS_Top3_G5": {
        "description": "Top 4 Phase F components ensemble",
        "components": ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium"],
    },
    "ENS_SUPER_G5_F7_v2": {
        "description": "Extended ensemble with Phase G winners",
        "components": ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium",
                      "W1_force_problem", "W3_no_dedup"],
    },
    "ENS_TopG5_Extended": {
        "description": "Top ensemble extended with contrastive",
        "components": ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium",
                      "W9_contrastive", "W1_low_gate"],
    },
    "ENS_WaveChampions": {
        "description": "Best config from each wave",
        "components": ["W1_force_problem", "W2_ultra_vol", "W3_no_dedup",
                      "W4_target_only", "W9_contrastive"],
    },
    "ENS_ProblemClass_Focus": {
        "description": "All configs focused on problem classes (NEW)",
        "components": ["PF_tier_boost", "PF_high_budget_problem", "PF_optimal_focused",
                      "W1_force_problem", "W4_target_only"],
    },

    # =========================================================================
    # PHASE G EXTENDED ENSEMBLES (70 configurations)
    # =========================================================================

    # Category 1: Weighted Top-K Ensembles (12 tests)
    # -------------------------------------------------------------------------
    "WGT_Top3_equal": {
        "description": "Top 3 configs, equal weights",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3"],
        "weights": None,  # Equal weights (default)
    },
    "WGT_Top3_perf": {
        "description": "Top 3 configs, performance-weighted",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3"],
        "weights": [5.98, 5.57, 5.34],  # By delta_pct
    },
    "WGT_Top3_exp": {
        "description": "Top 3 configs, exponential weighting",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3"],
        "weights": [np.exp(5.98/100), np.exp(5.57/100), np.exp(5.34/100)],  # e^(delta/100)
    },
    "WGT_Top5_equal": {
        "description": "Top 5 configs, equal weights",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "weights": None,
    },
    "WGT_Top5_perf": {
        "description": "Top 5 configs, performance-weighted",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "weights": [5.98, 5.57, 5.34, 5.22, 5.05],
    },
    "WGT_Top5_rank": {
        "description": "Top 5 configs, rank-weighted (1/rank)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "weights": [1/1, 1/2, 1/3, 1/4, 1/5],
    },
    "WGT_Top7_equal": {
        "description": "Top 7 configs, equal weights",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip"],
        "weights": None,
    },
    "WGT_Top7_perf": {
        "description": "Top 7 configs, performance-weighted",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip"],
        "weights": [5.98, 5.57, 5.34, 5.22, 5.05, 4.35, 4.32],
    },
    "WGT_Top10_equal": {
        "description": "Top 10 configs, equal weights (diversity test)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip", "W6_temp_low",
                      "W3_no_dedup", "W2_ultra_vol"],
        "weights": None,
    },
    "WGT_Top10_perf": {
        "description": "Top 10 configs, performance-weighted",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip", "W6_temp_low",
                      "W3_no_dedup", "W2_ultra_vol"],
        "weights": [5.98, 5.57, 5.34, 5.22, 5.05, 4.35, 4.32, 3.89, 3.88, 3.55],
    },
    "WGT_Prompting_only": {
        "description": "Prompting-focused ensemble",
        "components": ["W5_many_shot_10", "W5_few_shot_3"],
        "weights": [5.98, 5.34],
    },
    "WGT_Temperature_only": {
        "description": "Temperature-focused ensemble",
        "components": ["W6_temp_high", "W6_temp_low"],
        "weights": [5.57, 3.89],
    },

    # Category 2: Diversity-Maximizing Ensembles (10 tests)
    # -------------------------------------------------------------------------
    # NOTE: Actual component selection will be done dynamically by diversity_selector.py
    # These definitions specify the strategy, not the exact components
    "DIV_k3_maxdist": {
        "description": "3 configs with maximum diversity (selected by diversity_selector)",
        "strategy": "diversity",
        "k": 3,
        "metric": "euclidean",
        "components": [],  # Filled dynamically
    },
    "DIV_k5_maxdist": {
        "description": "5 configs with maximum diversity",
        "strategy": "diversity",
        "k": 5,
        "metric": "euclidean",
        "components": [],
    },
    "DIV_k7_maxdist": {
        "description": "7 configs with maximum diversity",
        "strategy": "diversity",
        "k": 7,
        "metric": "euclidean",
        "components": [],
    },
    "DIV_k5_class_balanced": {
        "description": "5 configs ensuring all classes covered",
        "strategy": "class_balanced",
        "k": 5,
        "components": [],
    },
    "DIV_prompting_temp_vol": {
        "description": "One best from each strategy type",
        "strategy": "strategy_diverse",
        "strategy_types": {
            "prompting": ["W5_many_shot_10", "W5_few_shot_3", "W5_zero_shot"],
            "temperature": ["W6_temp_high", "W6_temp_low", "W6_temp_extreme"],
            "volume": ["V4_ultra", "W2_ultra_vol", "W2_mega_vol"],
            "filtering": ["W3_permissive_filter", "W7_yolo"],
            "budget": ["CMB3_skip"],
        },
        "components": [],
    },
    "DIV_wave_diverse": {
        "description": "One from each wave, maximized diversity",
        "strategy": "diversity",
        "k": 9,  # 9 waves
        "exclude_wave8": True,  # Wave 8 failed
        "components": [],
    },
    "DIV_ENTJ_ISFJ_focus": {
        "description": "Configs that excel at different top classes",
        "strategy": "class_specific",
        "target_classes": ["ENTJ", "ISFJ", "ISTJ"],
        "components": [],
    },
    "DIV_rare_vs_common": {
        "description": "Configs good at rare + configs good at common",
        "strategy": "rare_vs_common",
        "components": [],
    },
    "DIV_orthogonal": {
        "description": "Configs with orthogonal per-class improvements",
        "strategy": "diversity",
        "k": 5,
        "metric": "cosine",  # Cosine distance for orthogonality
        "components": [],
    },
    "DIV_complementary_pairs": {
        "description": "Pairs of configs with complementary strengths",
        "strategy": "complementary",
        "n_pairs": 3,
        "components": [],
    },

    # Category 3: Hybrid Strategy Ensembles (15 tests)
    # -------------------------------------------------------------------------
    "HYB_prompt_temp": {
        "description": "Best prompting + best temperature",
        "components": ["W5_many_shot_10", "W6_temp_high"],
    },
    "HYB_prompt_vol": {
        "description": "Best prompting + best volume",
        "components": ["W5_many_shot_10", "V4_ultra"],
    },
    "HYB_prompt_filter": {
        "description": "Best prompting + best filtering",
        "components": ["W5_many_shot_10", "W3_permissive_filter"],
    },
    "HYB_temp_vol": {
        "description": "Best temperature + best volume",
        "components": ["W6_temp_high", "V4_ultra"],
    },
    "HYB_temp_filter": {
        "description": "Best temperature + best filtering",
        "components": ["W6_temp_high", "W3_permissive_filter"],
    },
    "HYB_vol_filter": {
        "description": "Best volume + best filtering",
        "components": ["V4_ultra", "W3_permissive_filter"],
    },
    "HYB_all_strategies": {
        "description": "One best from each strategy (5 configs)",
        "components": ["W5_many_shot_10", "W6_temp_high", "V4_ultra", "W3_permissive_filter", "CMB3_skip"],
    },
    "HYB_manyshot_hightemp": {
        "description": "Top 2: W5_many_shot_10 + W6_temp_high",
        "components": ["W5_many_shot_10", "W6_temp_high"],
    },
    "HYB_manyshot_yolo": {
        "description": "Many-shot + YOLO",
        "components": ["W5_many_shot_10", "W7_yolo"],
    },
    "HYB_manyshot_ultra": {
        "description": "Many-shot + ultra volume",
        "components": ["W5_many_shot_10", "V4_ultra"],
    },
    "HYB_triple_prompting": {
        "description": "All prompting strategies",
        "components": ["W5_many_shot_10", "W5_few_shot_3", "W5_zero_shot"],
    },
    "HYB_triple_temp": {
        "description": "All temperature strategies",
        "components": ["W6_temp_high", "W6_temp_low", "W6_temp_extreme"],
    },
    "HYB_conservative": {
        "description": "Best configs with quality gates enabled",
        "components": ["V4_ultra", "CMB3_skip", "CF1_conf_band"],
    },
    "HYB_aggressive": {
        "description": "Best configs with gates disabled (YOLO style)",
        "components": ["W7_yolo", "W5_many_shot_10", "W6_temp_high"],
    },
    "HYB_balanced": {
        "description": "Mix of conservative and aggressive",
        "components": ["W5_many_shot_10", "V4_ultra", "W7_yolo"],
    },

    # Category 4: Deduplication-Based Ensembles (8 tests)
    # -------------------------------------------------------------------------
    "DEDUP_Top5_sim095": {
        "description": "Top 5, cosine_sim > 0.95 removed",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.95, "method": "cosine"},
    },
    "DEDUP_Top5_sim098": {
        "description": "Top 5, cosine_sim > 0.98 removed (more permissive)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.98, "method": "cosine"},
    },
    "DEDUP_Top5_classwise": {
        "description": "Top 5, deduplicate per class",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "dedup_method": "classwise",
        "dedup_params": {"threshold": 0.95, "method": "cosine"},
    },
    "DEDUP_Top5_cluster": {
        "description": "Top 5, cluster and keep centroids",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "dedup_method": "clustering",
        "dedup_params": {"n_clusters": None, "keep_strategy": "centroid"},
    },
    "DEDUP_WaveChampions_sim095": {
        "description": "ENS_WaveChampions with dedup",
        "components": ["W1_force_problem", "W2_ultra_vol", "W3_no_dedup", "W4_target_only", "W9_contrastive"],
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.95, "method": "cosine"},
    },
    "DEDUP_diverse_k7": {
        "description": "7 diverse configs with dedup (DIV_k7_maxdist + dedup)",
        "strategy": "diversity",
        "k": 7,
        "components": [],  # Filled by diversity_selector
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.95, "method": "cosine"},
    },
    "DEDUP_hybrid_all": {
        "description": "HYB_all_strategies with dedup",
        "components": ["W5_many_shot_10", "W6_temp_high", "V4_ultra", "W3_permissive_filter", "CMB3_skip"],
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.95, "method": "cosine"},
    },
    "DEDUP_aggressive": {
        "description": "Top 10 configs, aggressive dedup (sim > 0.90)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip", "W6_temp_low",
                      "W3_no_dedup", "W2_ultra_vol"],
        "dedup_method": "similarity",
        "dedup_params": {"threshold": 0.90, "method": "cosine"},
    },

    # Category 5: Class-Targeted Ensembles (12 tests)
    # -------------------------------------------------------------------------
    # A. Rare Class Focus (with MLP_512_256_128)
    "RARE_MLP_massive": {
        "description": "RARE_massive_oversample (best for ESFJ) + MLP classifier",
        "components": ["RARE_massive_oversample"],
        "classifier": "MLP_512_256_128",
    },
    "RARE_MLP_all": {
        "description": "All 5 RARE configs + MLP classifier",
        "components": ["RARE_massive_oversample", "RARE_high_temperature", "RARE_yolo_extreme",
                      "RARE_contrastive_transfer", "RARE_few_shot_expert"],
        "classifier": "MLP_512_256_128",
    },
    "RARE_MLP_top5_standard": {
        "description": "Top 5 standard configs + MLP for rare classes",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "classifier": "MLP_512_256_128",
    },
    "RARE_hybrid_MLP": {
        "description": "Top 3 standard + 2 rare-focused + MLP",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3",
                      "RARE_massive_oversample", "RARE_high_temperature"],
        "classifier": "MLP_512_256_128",
    },

    # B. Top Class Optimization
    "TOP_ENTJ_focus": {
        "description": "Configs where ENTJ improves most",
        "components": ["W5_many_shot_10", "W7_yolo", "W3_permissive_filter", "W1_no_gate"],
    },
    "TOP_ISTJ_focus": {
        "description": "Configs where ISTJ improves most",
        "components": ["W7_yolo", "W1_low_gate", "W6_temp_low", "W2_ultra_vol"],
    },
    "TOP_ISFJ_focus": {
        "description": "Configs where ISFJ improves most",
        "components": ["W6_temp_high", "V4_ultra", "W5_few_shot_3", "W3_permissive_filter"],
    },
    "TOP_all_common": {
        "description": "Ensemble optimized for all common classes",
        "components": ["W5_many_shot_10", "W6_temp_high", "V4_ultra", "W7_yolo"],
    },

    # C. Balanced Class Coverage
    "BAL_16class_coverage": {
        "description": "One config per class (best for that class)",
        "strategy": "per_class_best",
        "components": [],  # Filled dynamically (16 configs, one per class)
    },
    "BAL_weighted_need": {
        "description": "Weight by inverse baseline F1 (help struggling classes)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "weighting_strategy": "inverse_baseline",
    },
    "BAL_equal_improvement": {
        "description": "Configs with most uniform per-class improvements",
        "strategy": "min_variance",
        "components": [],  # Filled dynamically
    },
    "BAL_rare_plus_common": {
        "description": "50% rare class configs + 50% common class configs",
        "components": ["RARE_massive_oversample", "RARE_high_temperature",
                      "W5_many_shot_10", "W6_temp_high"],
    },

    # Category 6: Advanced Combination Strategies (8 tests)
    # -------------------------------------------------------------------------
    # 6A. Stacking Ensembles
    "STACK_Top5_LogReg": {
        "description": "Top 5 configs, LogReg meta-classifier (Level-1)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "ensemble_method": "stacking",
        "meta_classifier": "LogisticRegression",
    },
    "STACK_Top5_MLP": {
        "description": "Top 5 configs, MLP meta-classifier",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "ensemble_method": "stacking",
        "meta_classifier": "MLP_256_128",
    },
    "STACK_diverse_k7_MLP": {
        "description": "7 diverse configs, MLP meta-classifier",
        "strategy": "diversity",
        "k": 7,
        "components": [],  # Filled by diversity_selector
        "ensemble_method": "stacking",
        "meta_classifier": "MLP_256_128",
    },

    # 6B. Voting Ensembles
    "VOTE_Top5_hard": {
        "description": "Top 5, hard voting (majority)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "ensemble_method": "voting",
        "voting_type": "hard",
    },
    "VOTE_Top5_soft": {
        "description": "Top 5, soft voting (probability averaging)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
        "ensemble_method": "voting",
        "voting_type": "soft",
    },
    "VOTE_Top7_weighted": {
        "description": "Top 7, weighted by delta_pct",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra",
                      "W7_yolo", "W3_permissive_filter", "CMB3_skip"],
        "ensemble_method": "voting",
        "voting_type": "weighted",
        "weights": [5.98, 5.57, 5.34, 5.22, 5.05, 4.35, 4.32],
    },

    # 6C. Selective Ensemble
    "SELECT_per_class": {
        "description": "Best config for each class separately",
        "ensemble_method": "selective",
        "selection_strategy": "per_class",
        "components": [],  # Filled dynamically (16 configs, one per class)
    },
    "SELECT_adaptive": {
        "description": "Choose config based on sample characteristics",
        "ensemble_method": "selective",
        "selection_strategy": "adaptive",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra", "W7_yolo"],
    },

    # Category 7: Experimental & Novel Strategies (5 tests)
    # -------------------------------------------------------------------------
    "NOVEL_boosted_rare": {
        "description": "Oversample rare classes 5x in ensemble",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3"],
        "rare_boost": 5.0,  # Multiply rare class samples by 5
    },
    "NOVEL_temperature_ladder": {
        "description": "Combine low/med/high temps with equal weight",
        "components": ["W6_temp_low", "W6_temp_extreme", "W6_temp_high"],
        "weights": None,  # Equal weights
    },
    "NOVEL_nshot_ladder": {
        "description": "Combine 0/3/5/10-shot configs",
        "components": ["W5_zero_shot", "W5_few_shot_3", "W5_many_shot_10"],
        "weights": None,
    },
    "NOVEL_confidence_filtered": {
        "description": "Only include high-confidence synthetics (>0.80)",
        "components": ["W5_many_shot_10", "W6_temp_high", "W5_few_shot_3", "V4_ultra"],
        "confidence_threshold": 0.80,
    },
    "NOVEL_quality_tiers": {
        "description": "Separate ensembles for high/med/low quality, then combine",
        "ensemble_method": "tiered",
        "tier_high": ["W5_many_shot_10", "W6_temp_high"],
        "tier_med": ["W5_few_shot_3", "V4_ultra"],
        "tier_low": ["W7_yolo", "W2_ultra_vol"],
    },
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_config_params(config_name: str) -> Dict[str, Any]:
    """
    Get merged parameters for a config.

    Merges BASE_PARAMS with config-specific overrides.
    Crucial parameters from the config override base params.

    Args:
        config_name: Name of the config (e.g., "W1_low_gate")

    Returns:
        Complete parameter dictionary
    """
    if config_name not in ALL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ALL_CONFIGS.keys())}")

    config = ALL_CONFIGS[config_name]

    # Start with base params
    params = BASE_PARAMS.copy()

    # Apply config-specific overrides
    params.update(config["overrides"])

    return params


def get_configs_by_wave(wave: str) -> Dict[str, Dict[str, Any]]:
    """Get all configs belonging to a specific wave."""
    return {
        name: config
        for name, config in ALL_CONFIGS.items()
        if config.get("wave") == wave
    }


def list_all_configs() -> List[str]:
    """List all available config names."""
    return list(ALL_CONFIGS.keys())


def get_ensemble_components(ensemble_name: str) -> List[str]:
    """Get component config names for an ensemble."""
    if ensemble_name not in ENSEMBLES:
        raise ValueError(f"Unknown ensemble: {ensemble_name}")
    return ENSEMBLES[ensemble_name]["components"]


# ==============================================================================
# CONFIG SUMMARY FOR REFERENCE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase G Validation - Configuration Summary")
    print("=" * 70)
    print(f"\nTotal configs: {len(ALL_CONFIGS)}")
    print(f"Total ensembles: {len(ENSEMBLES)}")

    print("\n" + "-" * 70)
    print("CONFIGS BY WAVE:")
    print("-" * 70)

    waves = ["component", "wave1", "wave2", "wave3", "wave4",
             "wave5", "wave6", "wave7", "wave8", "wave9", "pf_derived", "rare_class"]

    for wave in waves:
        wave_configs = get_configs_by_wave(wave)
        print(f"\n{wave.upper()} ({len(wave_configs)} configs):")
        for name, config in wave_configs.items():
            print(f"  - {name}: {config['description']}")
            print(f"    Crucial: {config['crucial_params']}")

    print("\n" + "-" * 70)
    print("ENSEMBLES:")
    print("-" * 70)

    for name, ensemble in ENSEMBLES.items():
        print(f"\n{name}: {ensemble['description']}")
        print(f"  Components: {ensemble['components']}")
