# State of the Art: Seed Variance & Macro F1 Improvement

## Sources

### Seed Variance Reduction

1. **Cross-validation - scikit-learn**
   https://scikit-learn.org/stable/modules/cross_validation.html
   - Stratified K-Fold ensures class distribution in each fold
   - Reduces estimator variance mathematically

2. **Stratified K-Fold Cross Validation - GeeksforGeeks**
   https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/
   - Maintains class proportions across folds
   - Minimizes variability in model performance

3. **Embrace Randomness in ML - MachineLearningMastery**
   https://machinelearningmastery.com/randomness-in-machine-learning/
   - Ensemble of models with different seeds
   - Average predictions reduces variance

4. **Properly Setting Random Seed - ODSC Medium**
   https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
   - Repeat k-fold CV with different seeds
   - Average final results for robust estimate

5. **Stabilizing ML for Reproducible Results - ScienceDirect 2025**
   https://www.sciencedirect.com/science/article/pii/S0169260725003165
   - Novel repeated trials validation with random seed variation
   - Reduces variability in feature rankings

6. **PyTorch Reproducibility Docs**
   https://docs.pytorch.org/docs/stable/notes/randomness.html
   - Disable cuDNN benchmarking for determinism
   - Set seeds across all random sources

---

### Macro F1 Improvement (Imbalanced Data)

7. **F1 Score Guide for Imbalanced Classes - Number Analytics**
   https://www.numberanalytics.com/blog/f1-score-imbalanced-classes-guide
   - Use macro F1 when all classes equally important
   - Resampling + class_weight combination

8. **Tour of Evaluation Metrics - MachineLearningMastery**
   https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
   - Precision-Recall AUC for skewed data
   - Balanced accuracy with adjustment

9. **Computing F1-Score - Sebastian Raschka**
   https://sebastianraschka.com/faq/docs/computing-the-f1-score.html
   - Stratify folds to maintain class proportions
   - Avoid folds without minority class samples

10. **Stratified Cross-Validation - Number Analytics**
    https://www.numberanalytics.com/blog/practical-implementation-stratified-cross-validation
    - Critical for skewed datasets
    - More reliable performance estimates

---

### LLM Data Augmentation

11. **Data Augmentation using LLMs - ACL 2024**
    https://aclanthology.org/2024.findings-acl.97.pdf
    - LLM-driven augmentation dominant trend 2024
    - Quality filtering with validator models

12. **LLM Synthetic Data Quality - ACL 2024 Findings**
    https://aclanthology.org/2024.findings-acl.658.pdf
    - Adaptive Reliability Validation (ARV)
    - Discard/down-weight unrealistic samples

13. **Data Generation for Text Classification - arXiv 2024**
    https://arxiv.org/html/2407.12813v1
    - Nucleus sampling for diversity
    - Multiple candidates per prompt

14. **Improving Model Performance through Minimal Data - ACL DASH 2024**
    https://aclanthology.org/2024.dash-1.4.pdf
    - Error-case augmentation
    - Focus on misclassified samples

15. **Text Data Augmentation Survey - ACM Computing Surveys**
    https://dl.acm.org/doi/10.1145/3544558
    - Token-level augmentation most effective
    - Task complexity affects augmentation choice

16. **AugGPT: ChatGPT for Text Augmentation - arXiv**
    https://arxiv.org/abs/2302.13007
    - LLM paraphrasing for diversity
    - Quality vs quantity tradeoff

17. **Balancing Cost and Effectiveness - arXiv 2024**
    https://arxiv.org/html/2409.19759
    - Optimal strategy depends on budget/seed ratio
    - Low budget: new answers; High budget: new questions

---

## Key Strategies Summary

### Variance Reduction
| Strategy | Expected Impact | Complexity |
|----------|-----------------|------------|
| Stratified K-Fold (k=5) | -60% variance | Low |
| Repeated K-Fold (5×5) | -80% variance | Medium |
| Model Ensemble (5 seeds) | -50% variance | Low |
| Report mean±std, 95% CI | N/A (reporting) | Low |

### Macro F1 Improvement
| Strategy | Expected Impact | Complexity |
|----------|-----------------|------------|
| class_weight='balanced' | +2-5% F1 | Very Low |
| Threshold tuning per class | +1-3% F1 | Low |
| EasyEnsemble/BalancedRF | +3-7% F1 | Medium |
| SMOTE/ADASYN resampling | +2-4% F1 | Low |

### Augmentation Quality
| Strategy | Expected Impact | Complexity |
|----------|-----------------|------------|
| ARV (secondary validator) | +quality, -30% quantity | Medium |
| Error-case augmentation | +efficiency | Medium |
| Diversity sampling (nucleus) | +variety | Low |
| Similarity deduplication | -redundancy | Low |

---

*Generated: 2024-12-04*
*Phase F Experiments*
