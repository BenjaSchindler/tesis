# Pre-Phase A: Original Clean Baseline

This folder contains the **original, unmodified** Phase A code from `Tesis_Limpio/Laptop Runs/`.

## Purpose

This is the TRUE clean Phase A baseline, preserved for reference and comparison. It has:
- **0 Phase C/D references** (no focal_loss, two_stage, contrastive imports)
- **Original anchor selection** (no "hard anchors" strategy)
- **Original quality gate** (no seeded RNG)

## Differences from `phase_a/core/`

| File | pre_phase_a | phase_a/core | Change |
|------|-------------|--------------|--------|
| `runner_phase2.py` | 2,750 lines | 3,325 lines | +575 lines (Phase C/D imports) |
| `ensemble_anchor_selector.py` | 468 lines | 548 lines | +80 lines (hard anchors) |
| `enhanced_quality_gate.py` | 404 lines | 417 lines | +13 lines (seeded RNG) |
| Other files | - | - | IDENTICAL |

## Key Differences

### 1. `ensemble_anchor_selector.py`
- **OLD**: 3 anchor strategies (medoid, quality, diverse)
- **NEW**: 4 anchor strategies (+ hard anchors near decision boundary)
- **Impact**: Slightly different anchor selection when `--use-ensemble-selection` is enabled

### 2. `enhanced_quality_gate.py`
- **OLD**: Uses numpy global RNG (non-deterministic)
- **NEW**: Supports seeded RNG for deterministic probabilistic decisions
- **Impact**: Better reproducibility (improvement)

### 3. `runner_phase2.py`
- **OLD**: No Phase C/D code
- **NEW**: Phase C/D imports present but **disabled by default**
  - `use_contrastive_prompting=False` (opt-in)
  - `use_focal_loss=False` (opt-in)
  - `use_two_stage_training=False` (opt-in)

## When to Use This

Use `pre_phase_a/` when you need:
1. A reference for what the original Phase A code looked like
2. To compare results with the TRUE original baseline
3. To debug whether Phase C/D additions caused regressions

## Source

Copied from: `/home/benja/Desktop/Tesis/Tesis_Limpio/Laptop Runs/`
Date: 2025-11-26
