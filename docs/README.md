# Documentation - SMOTE-LLM Project

This directory contains comprehensive technical documentation for the SMOTE-LLM synthetic data augmentation system.

## Quick Navigation

### Executive Summary
- [00_RESUMEN_EJECUTIVO.md](00_RESUMEN_EJECUTIVO.md) - Complete executive summary of results (Spanish)

### Technical Foundation
- [01_TECHNICAL_OVERVIEW.md](01_TECHNICAL_OVERVIEW.md) - Complete technical overview of the system
- [01_pipeline_completo.md](01_pipeline_completo.md) - End-to-end pipeline description
- [02_PROBLEM_STATEMENT.md](02_PROBLEM_STATEMENT.md) - Core problems: proportional contamination theory

### Configuration & Parameters
- [02_parametros_justificados.md](02_parametros_justificados.md) - Parameter justification
- [04_PARAMETER_DEEP_DIVE.md](04_PARAMETER_DEEP_DIVE.md) - Exhaustive parameter analysis (45 pages)
- [05_best_config.md](05_best_config.md) - Optimal configuration found

### Improvements & Methodology
- [03_mejoras_implementadas.md](03_mejoras_implementadas.md) - Implemented improvements (TIER S, Phase 1-2-3)
- [04_estrategias_batches.md](04_estrategias_batches.md) - Batch experiment strategies

### Problems & Solutions
- [01_cross_contamination.md](01_cross_contamination.md) - Cross-class contamination problem and mitigation
- [02_seed_variance.md](02_seed_variance.md) - Seed variance problem (54pp → 3.75pp reduction)
- [03_high_f1_protection.md](03_high_f1_protection.md) - Protecting high-performing classes
- [04_mid_tier_vulnerability.md](04_mid_tier_vulnerability.md) - Mid-tier F1 vulnerability issue

### Results & Analysis
- [01_macro_f1_evolution.md](01_macro_f1_evolution.md) - Evolution of macro F1 scores
- [02_per_class_analysis.md](02_per_class_analysis.md) - Per-class performance analysis (16 MBTI classes)
- [03_tier_analysis.md](03_tier_analysis.md) - Analysis by F1 tier (LOW/MID/HIGH)
- [04_statistical_validation.md](04_statistical_validation.md) - Statistical significance testing

---

## Documentation by Topic

### Understanding the Problem
Start here if you're new to the project:
1. [02_PROBLEM_STATEMENT.md](02_PROBLEM_STATEMENT.md) - What problem are we solving?
2. [01_cross_contamination.md](01_cross_contamination.md) - Cross-contamination details
3. [02_seed_variance.md](02_seed_variance.md) - Why seed variance matters

### Understanding the Solution
1. [01_TECHNICAL_OVERVIEW.md](01_TECHNICAL_OVERVIEW.md) - System architecture
2. [01_pipeline_completo.md](01_pipeline_completo.md) - Pipeline flow
3. [03_mejoras_implementadas.md](03_mejoras_implementadas.md) - What improvements were made

### Configuration & Tuning
1. [02_parametros_justificados.md](02_parametros_justificados.md) - Why each parameter
2. [04_PARAMETER_DEEP_DIVE.md](04_PARAMETER_DEEP_DIVE.md) - Deep dive into parameter selection
3. [05_best_config.md](05_best_config.md) - Final optimal configuration

### Results & Performance
1. [00_RESUMEN_EJECUTIVO.md](00_RESUMEN_EJECUTIVO.md) - Executive summary
2. [01_macro_f1_evolution.md](01_macro_f1_evolution.md) - Overall performance evolution
3. [02_per_class_analysis.md](02_per_class_analysis.md) - Class-by-class breakdown
4. [03_tier_analysis.md](03_tier_analysis.md) - Performance by difficulty tier

---

## Key Findings

### Phase A Results
- **Macro F1:** +1.00% ± 0.07%
- **Seed Variance:** 93% reduction (54pp → 3.75pp)
- **HIGH F1 Protection:** 100% (9/9 classes protected)
- **LOW F1 Improvement:** +12.17% average

### Core Problems Identified
1. **Proportional Contamination:** Synthetics contaminate other classes
2. **Seed Variance:** Results varied wildly across random seeds
3. **High-F1 Degradation:** Strong classes got worse with augmentation
4. **Mid-Tier Vulnerability:** Classes with F1 20-45% showed degradation

### Solutions Implemented
1. **F1-Budget Scaling:** Allocate synthetics based on baseline F1
2. **Ensemble Selection:** Mathematical guarantee of non-degradation
3. **Validation Gating:** Early stopping based on validation performance
4. **Anchor Selection:** Quality-based anchor filtering
5. **Adaptive Filters:** Dynamic threshold adjustment

---

## Recommended Reading Order

### Quick Overview (30 minutes)
1. [00_RESUMEN_EJECUTIVO.md](00_RESUMEN_EJECUTIVO.md)
2. [05_best_config.md](05_best_config.md)
3. [03_tier_analysis.md](03_tier_analysis.md)

### Technical Deep Dive (3-4 hours)
1. [01_TECHNICAL_OVERVIEW.md](01_TECHNICAL_OVERVIEW.md)
2. [02_PROBLEM_STATEMENT.md](02_PROBLEM_STATEMENT.md)
3. [01_pipeline_completo.md](01_pipeline_completo.md)
4. [03_mejoras_implementadas.md](03_mejoras_implementadas.md)
5. [02_parametros_justificados.md](02_parametros_justificados.md)

### Complete Understanding (1-2 days)
Read all documents in numerical order, starting with:
- 00_RESUMEN_EJECUTIVO.md
- 01_*.md files
- 02_*.md files
- etc.

---

## Related Resources

- [Main README](../README.md) - Project overview and quick start
- [CLAUDE.md](../CLAUDE.md) - GCP deployment guide
- [phase_a/README.md](../phase_a/README.md) - Phase A configuration
- [phase_b/README.md](../phase_b/README.md) - Phase B adaptive weighting
- [scripts/](../scripts/) - Analysis and plotting scripts

---

## Document Statistics

- **Total Documents:** 21 markdown files
- **Total Pages (est.):** ~200+ pages
- **Key Topics:** Pipeline, Parameters, Problems, Results, Solutions
- **Languages:** English (technical docs), Spanish (executive summary)

---

## Contributing

If you find errors or want to add documentation:
1. Follow existing naming convention: `##_topic_name.md`
2. Use clear section headers and markdown formatting
3. Include code examples where relevant
4. Cross-reference related documents

---

Last updated: 2025-11-15
