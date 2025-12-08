# Per-Class Analysis - 16 MBTI Classes

**Total classes:** 16
**Tiers:** LOW (6), MID (4), HIGH (6)

---

## 📊 LOW Tier (F1 < 20%) - 6 Classes

### Best Performers

**ISTJ - Inspector ⭐⭐⭐**
- Baseline: 0.18 → Augmented: 0.49
- **Delta: +30.82%** (Best improvement)
- Selected: Augmented

**ISFJ - Protector ⭐⭐**
- Baseline: 0.15 → Augmented: 0.36
- **Delta: +20.75%**
- Selected: Augmented

**Summary LOW Tier:**
- All 6/6 improved significantly
- Mean: **+12.17%**
- All selected augmented models
- Full augmentation successful

---

## ⚠️ MID Tier (20-45%) - 4 Classes

### Problem Cases

**ENTJ - Commander (Worst)**
- Baseline: 0.31 → Augmented: 0.29
- **Delta: -1.91%**
- Selected: Baseline (ensemble fallback)
- **Needs:** Weight reduction in Fase B

**ESFJ - Consul**
- Baseline: 0.28 → Augmented: 0.27
- **Delta: -0.72%**

**Summary MID Tier:**
- All 4/4 degraded
- Mean: **-0.59%**
- All selected baseline (ensemble protection)
- **Solution:** Fase B adaptive weighting

---

## ✅ HIGH Tier (F1 ≥ 45%) - 6 Classes

### Best Case

**ENTP - Debater**
- Baseline: 0.65 → Augmented: 0.66
- **Delta: +0.54%** (Best in HIGH)
- Selected: Augmented
- Improved despite multiplier 0.0!

**Summary HIGH Tier:**
- Mean: **-0.05%** (almost neutral)
- 4/6 positive
- Protection successful
- No severe degradations

---

## 📚 Full Analysis

For detailed per-class breakdowns, see [Tier Analysis](03_tier_analysis.md)
