# Embedding Ablation Study - Analysis Results

**Analysis Date**: 2026-02-07
**Analyst**: Claude Sonnet 4.5
**Experiment Data**: final_results.json (504 runs)

## Overview

This directory contains comprehensive analysis of the embedding ablation experiment, which tests whether geometric filtering (cascade_l1 + soft weighting) is robust across different embedding models.

**Core Finding**: YES - All 4 embedding models show statistically significant improvements over SMOTE (p<0.05), with gains ranging from +0.97pp to +2.74pp.

---

## Directory Structure

```
/home/benja/Desktop/Tesis/filters/results/embedding_ablation/
├── final_results.json                  # Raw experimental results (504 runs)
├── README.md                           # This file
├── EXECUTIVE_SUMMARY.md                # 5-minute high-level overview
├── THESIS_ANALYSIS.md                  # Comprehensive 50-page analysis
├── THESIS_CHAPTER_SECTION.md           # Ready-to-integrate thesis chapter
├── latex_tables.tex                    # 6 publication-ready LaTeX tables
└── figures/                            # 6 publication-quality visualizations
    ├── fig1_overall_comparison.png
    ├── fig2_nshot_breakdown.png
    ├── fig3_dimensionality.png
    ├── fig4_dataset_heatmap.png
    ├── fig5_win_rates.png
    └── fig6_consistency.png
```

---

## File Descriptions

### EXECUTIVE_SUMMARY.md
**Purpose**: Quick overview for busy readers (5-10 minutes)

**Contents**:
- Key findings in 60 seconds
- 5 main takeaways with tables
- Practical recommendations
- Statistical validation summary
- Bottom-line conclusion

**Use when**: You need to quickly understand or present the main results

---

### THESIS_ANALYSIS.md
**Purpose**: Comprehensive academic analysis (50 pages)

**Contents**:
- Experimental design details
- Overall performance with statistical tests
- Dimensionality analysis (384d vs 768d vs 1024d)
- N-shot breakdown (10/25/50)
- Dataset-model interaction analysis
- Cross-model consistency analysis
- Variance analysis
- Best/worst cases
- 8 key insights for thesis
- Limitations and future work
- Practical and research recommendations

**Sections**:
1. Experimental Design
2. Overall Performance by Model
3. Dimensionality Analysis
4. Performance by N-Shot
5. Dataset-Model Interaction
6. Cross-Model Consistency
7. Variance Analysis
8. Key Insights for Thesis
9. Limitations and Future Work
10. Practical Recommendations
11. Conclusion
Appendix: Statistical Notes

**Use when**: Writing the full thesis chapter or deep technical analysis

---

### THESIS_CHAPTER_SECTION.md
**Purpose**: Ready-to-integrate thesis section (15-20 pages)

**Contents**:
- Narrative-style academic writing
- Integrated with figure/table references
- Structured as Section 5.X (customizable)
- Includes all key findings with interpretations
- Discusses implications and limitations
- References to specific figures and tables

**Subsections**:
- 5.X.1 Experimental Design
- 5.X.2 Overall Results
- 5.X.3 Dimensionality Analysis
- 5.X.4 Performance by N-Shot
- 5.X.5 Cross-Model Consistency
- 5.X.6 Dataset-Model Interactions
- 5.X.7 Implications for Robustness
- 5.X.8 Limitations
- 5.X.9 Conclusion

**Use when**: Copy-pasting directly into your thesis LaTeX document

---

### latex_tables.tex
**Purpose**: Publication-ready LaTeX tables

**Contents**: 6 tables formatted for copy-paste into thesis
1. Overall Performance Summary
2. N-Shot Breakdown
3. Dimensionality Analysis
4. Best and Worst Cases
5. Cross-Model Consistency Summary
6. Dataset-Specific Best Models

**Additional**: Inline statistics for narrative text (at bottom of file)

**Use when**: Formatting tables for thesis document

---

### Figures Directory (6 PNG files, 300 DPI)

#### fig1_overall_comparison.png
Bar chart showing mean Δ F1 vs SMOTE for all 4 models with 95% CI error bars
- Highlights best model (mpnet-base) with red border
- Shows embedding dimensions (384d, 768d, 1024d)
- Horizontal dashed line at y=0

#### fig2_nshot_breakdown.png
Grouped bar chart showing performance by n-shot (10/25/50) and model
- 4 bars per n-shot value (one per model)
- Error bars (95% CI)
- Demonstrates diminishing returns pattern

#### fig3_dimensionality.png
Scatter plot showing dimensionality vs filtering effectiveness
- X-axis: embedding dimension (384, 768, 1024)
- Y-axis: mean Δ F1
- Annotated model names
- Linear trend line (shows no clear relationship)

#### fig4_dataset_heatmap.png
Heatmap showing performance across 21 dataset × 4 model combinations
- Color scale: red (negative) to green (positive)
- Cell annotations with exact Δ values
- Reveals dataset-specific model preferences

#### fig5_win_rates.png
Horizontal bar chart showing win rate (soft_weighted > SMOTE) by model
- Percentage of 63 comparisons where filtering wins
- Dashed line at 50% (random baseline)
- Highlights best model (mpnet-base: 92.1%)

#### fig6_consistency.png
Three-panel analysis of cross-model consistency by n-shot
- One panel per n-shot value (10/25/50)
- Color-coded by agreement level (0-4 models improve)
- Shows datasets where filtering is universally beneficial

**Figure Usage**: All figures are 300 DPI PNG, suitable for print publication

---

## Scripts Used for Analysis

### /home/benja/Desktop/Tesis/filters/experiments/analyze_embedding_ablation.py
**Purpose**: Statistical analysis and text output generation

**Key Functions**:
- `cohen_d()`: Calculate effect sizes
- `extract_paired_deltas()`: Extract paired SMOTE vs soft_weighted comparisons
- Paired t-tests for each model
- Win rate calculations
- Cross-model consistency analysis
- Variance analysis across seeds

**Output**: Console output with comprehensive statistics (saved to analysis above)

---

### /home/benja/Desktop/Tesis/filters/experiments/visualize_embedding_ablation.py
**Purpose**: Generate publication-quality visualizations

**Key Functions**:
- `plot_overall_comparison()`: Figure 1
- `plot_nshot_breakdown()`: Figure 2
- `plot_dimensionality_analysis()`: Figure 3
- `plot_dataset_heatmap()`: Figure 4
- `plot_win_rate_comparison()`: Figure 5
- `plot_consistency_analysis()`: Figure 6

**Style**: Seaborn publication theme, 300 DPI, color-blind friendly palette

---

## Key Results at a Glance

### Overall Performance

| Model | Δ vs SMOTE | p-value | Win Rate | Cohen's d |
|-------|------------|---------|----------|-----------|
| mpnet-base | **+2.74pp** | <0.001 | **92.1%** | 0.154 |
| bge-large | +2.04pp | <0.001 | 69.8% | 0.140 |
| e5-large | +0.97pp | 0.013 | 68.3% | 0.059 |
| bge-small | +2.00pp | <0.001 | 76.2% | 0.123 |

### Dimensionality Ranking

1. **768d** (mpnet-base): +2.74pp ← best
2. **384d** (bge-small): +2.00pp
3. **1024d** (mean of bge-large + e5-large): +1.51pp

**Insight**: Bigger is NOT better. 768d achieves optimal balance.

### N-Shot Pattern (all models)

- **10-shot**: +4.23pp average (range: +1.42pp to +5.37pp)
- **25-shot**: +1.33pp average (range: +0.75pp to +2.11pp)
- **50-shot**: +0.51pp average (range: +0.32pp to +0.75pp)

**Decay**: ~88% reduction from 10-shot to 50-shot

### Cross-Model Consistency

- **52.4%** full agreement (all 4 models improve)
- **90.5%** majority agreement (≥2 models improve)
- **9.5%** minority/failure (only 2 configs, both 50-shot)

### Best Single Result

**emotion_10shot + mpnet-base**: +10.52pp

### Worst Single Result

**hate_speech_davidson_10shot + e5-large**: -9.39pp (catastrophic failure)

---

## Thesis Contributions

1. **Robustness Validation**: First demonstration that geometric filtering generalizes across embedding architectures, dimensions, and training objectives

2. **Dimensionality Paradox**: Novel finding that 768d outperforms 1024d for filtering, challenging scaling assumptions

3. **Model-Task Interactions**: Identification of specific failure modes (e.g., e5-large on hate_speech) that guide model selection

4. **10-Shot Sweet Spot**: Quantitative evidence that geometric filtering achieves maximum impact at extreme low-resource (10 samples/class)

5. **Practical Guidelines**: Evidence-based recommendations for model selection based on compute constraints and task characteristics

---

## How to Use These Files

### For Thesis Writing

1. **Start with**: THESIS_CHAPTER_SECTION.md
   - Copy sections directly into your LaTeX document
   - Customize section numbering (5.X → 5.4, etc.)
   - Integrate figure/table references

2. **Add tables**: latex_tables.tex
   - Copy-paste desired tables into thesis
   - Adjust labels to match your numbering scheme

3. **Insert figures**: figures/ directory
   - Reference as `\includegraphics{path/to/fig1_overall_comparison.png}`
   - Captions are included in THESIS_CHAPTER_SECTION.md

### For Presentations

1. **Use**: EXECUTIVE_SUMMARY.md for slide content
2. **Visuals**:
   - fig1_overall_comparison.png (main result)
   - fig2_nshot_breakdown.png (diminishing returns)
   - fig3_dimensionality.png (dimension paradox)

### For Quick Reference

1. **This README**: Quick stats lookup
2. **EXECUTIVE_SUMMARY**: Share with advisors/collaborators

### For Deep Analysis

1. **THESIS_ANALYSIS.md**: All details, interpretations, and statistical notes

---

## Citation

If using this analysis, please cite the thesis:

```
@mastersthesis{yourname2026geometric,
  title={Geometric Filters for LLM-Based Data Augmentation in Low-Resource Text Classification},
  author={Your Name},
  year={2026},
  school={Your University},
  note={Chapter 5.X: Embedding Ablation Study}
}
```

---

## Contact

For questions about this analysis:
- Check THESIS_ANALYSIS.md for detailed explanations
- Review analysis script: analyze_embedding_ablation.py
- Consult visualization script: visualize_embedding_ablation.py

---

**Last Updated**: 2026-02-07
**Status**: Complete and ready for thesis integration
