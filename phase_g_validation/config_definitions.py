#!/usr/bin/env python3
"""
Configuration Definitions for Phase G Validation

Each config inherits OPTIMAL base parameters from Phase F and overrides
specific parameters that define its unique experimental purpose.

CRUCIAL parameters are marked - these MUST be preserved because they
define what the experiment is testing.
"""

from typing import Dict, Any, List
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
