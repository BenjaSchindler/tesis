# Statistical Validation

**Status:** Partial (limited seeds)
**Pending:** Full multi-seed validation (n=10)

---

## 🎯 Current Evidence

### Batch 4 (10 seeds, no F1-budget)

```
Mean: +0.66%
Std: 0.45%
95% CI: [0.37%, 0.95%]

t-test: p < 0.01 ✓
Conclusion: Significantly positive
```

### Batch 1E (2 seeds, F1-budget)

```
Seed 42:  -0.88%
Seed 100: -0.92%
Difference: 0.04pp

Conclusion: Very low variance
```

### Fase A (1 seed)

```
Seed 42: +1.00%

Pending: Multi-seed validation
```

---

## 📊 Expected Full Validation

### Plan: 10-Seed Test

```python
seeds = [42, 100, 101, 200, 300, 400, 456, 500, 789, 2024]

Expected results:
  Mean: +1.20% to +1.40%
  Std: < 0.15%
  95% CI: Narrow (< 0.3pp)

Statistical tests:
  t-test: p < 0.001 (highly significant)
  Cohen's d: > 0.8 (large effect)
  Power: > 0.99
```

---

## 🎯 Significance Tests Planned

1. **One-sample t-test:** μ > 0
2. **Paired t-test:** Augmented vs Baseline
3. **F-test:** Variance reduction
4. **Effect size:** Cohen's d
5. **Power analysis:** Detection probability

---

## 📚 References

- [Macro F1 Evolution](01_macro_f1_evolution.md)
- [Best Config](05_best_config.md)

---

**Status:** Pending post-Fase B
