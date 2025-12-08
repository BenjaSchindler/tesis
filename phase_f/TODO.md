# Phase F - TODO List

**Updated:** 2025-12-04
**Standard:** V2_high_vol + K-Fold (k=5, r=3)

---

## COMPLETED

- [x] **Stratified K-Fold CV (k=5, r=3)** - Implemented in `kfold_evaluator.py`
  - Reduces variance ~85-95%
  - V2_high_vol: +1.88%, p=0.0016, 100% win rate
  - V3_low_vol: +1.42%, p=0.0269, 73% win rate

- [x] **Fix --max-completion-tokens for G5 experiments**

- [x] **Document K-Fold results** - See `Variance_tests/KFOLD_VARIANCE_ANALYSIS.md`

---

## NEW STANDARD (Always Apply)

- [ ] **Always use K-Fold (k=5, r=3) for evaluation**
- [ ] **Use V2_high_vol as baseline for all new experiments**

---

## PRIORITY HIGH - Next Experiments

### Volume Experiments (More Synthetic Candidates)
- [ ] **V4_ultra** (8×12×7 = 672 candidates) - 2.5x more than V2
- [ ] **V5_extreme** (10×15×10 = 1500 candidates) - 5.5x more than V2
- [ ] **V2_f1scaled** (V2 + f1-budget-multipliers 2.5x for weak classes)

### Model Comparison
- [ ] **K-Fold on G5_K15_none** - Fair gpt-5-mini vs gpt-4o-mini comparison
- [ ] **K-Fold on CMB3_skip** - Compare 61 synth vs V2's 46 synth

### Quality Improvements
- [ ] **Threshold tuning per class** - +1-3% F1 expected
- [ ] **Add presence_penalty=0.3 to LLM calls** - Better diversity

---

## PRIORITY MEDIUM

- [ ] **Test EasyEnsembleClassifier/BalancedRandomForest** - +3-7% F1
- [ ] **Increase classifier_confidence threshold** (0.10 → 0.30)
- [ ] **Quality > Quantity** - Prioritize high-confidence synthetics

---

## FINDINGS

| Finding | Evidence |
|---------|----------|
| K-Fold essential | Single-seed variance: 7.7pp, K-Fold std: 0.38pp |
| V2_high_vol best config | +1.88% (p=0.0016), 100% win rate |
| gpt-5-mini underperforms | 12/18 G5 results, avg -1.11% |
| More volume helps | V2 (270 cand) > V3 (180 cand) |

---

## G5 (gpt-5-mini) Results Summary

| Config | s42 | s100 | s123 | Avg |
|--------|-----|------|------|-----|
| G5_K5_none | -0.25% | -1.71% | -2.16% | -1.37% |
| G5_K15_none | **+2.81%** | -1.79% | -2.89% | -0.62% |
| G5_K25_none | -2.74% | +0.27% | -0.34% | -0.94% |
| G5_K15_low | -1.34% | -0.14% | -3.04% | -1.51% |

**Conclusion:** gpt-4o-mini > gpt-5-mini for this task (need K-Fold to confirm)

---

## Reference

| Metric | V2_high_vol (K-Fold) | V3_low_vol (K-Fold) |
|--------|---------------------|---------------------|
| Delta F1 | +1.88% | +1.42% |
| p-value | 0.0016 | 0.0269 |
| Win Rate | 100% (15/15) | 73% (11/15) |
| 95% CI | [+0.17%, +0.60%] | [+0.04%, +0.54%] |
| Synthetics | 46 | 44 |
