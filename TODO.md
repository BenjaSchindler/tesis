# SMOTE-LLM Project TODO List

**Last Updated:** 2025-11-24

## 📍 Current Status

**Phase C v2.1 Multi-Seed Validation (5 seeds) - IN PROGRESS**

Running on GCP GPU VMs (doctor911 account):
- **Seeds:** 42, 100, 123, 456, 789
- **VMs:** 4x n1-standard-4 + Tesla T4 GPU
- **Status:** ~1h+ running, currently computing embeddings
- **Expected completion:** 4-5 hours per seed

---

## ✅ Completed Tasks

- [x] Add timeout to OpenAI API calls in runner_phase2.py (30s timeout)
- [x] Add unbuffered output flag to Python execution (`-u`)
- [x] Launch 5-seed experiments on GCP GPU VMs (42, 100, 123, 456, 789)

---

## 🔄 In Progress

- [ ] **Monitor experiments** - wait for completion (1h+ running, currently in embeddings)
  - Status: Processes active, GPU at ~100%, logs updating slowly
  - Expected: 4-5h total per seed

---

## 🎯 Immediate Tasks (Phase C v2.1 Completion)

Priority: **HIGH** | Timeline: **This week**

- [ ] Download results from all 4 VMs when complete
  - Files: `phaseC_v2.1_seed{SEED}_metrics.json`
  - Files: `phaseC_v2.1_seed{SEED}_synthetic.csv`
  - Files: `phaseC_v2.1_seed{SEED}_augmented.csv`

- [ ] Calculate Phase C v2.1 multi-seed statistics
  - Compute: mean, std, 95% confidence interval
  - Metrics: overall F1, per-tier F1 (LOW, MID, HIGH)

- [ ] Verify robustness requirements
  - Target: mean ≥ +0.377% overall improvement
  - Target: mean ≥ +1.72% MID-tier improvement
  - Low variance across seeds (std < 0.5%)

- [ ] Stop/delete VMs to avoid charges
  - Clean up: `gcloud compute instances delete vm-batch{1-4}-gpu`

---

## 🧪 Short-Term Experiments (1-2 weeks)

Priority: **HIGH** | Timeline: **Next 2 weeks**

### SMOTE Variants Comparison

- [ ] **Baseline:** Standard SMOTE
- [ ] **Borderline-SMOTE:** Focus on borderline samples
- [ ] **ADASYN:** Adaptive synthetic sampling
- [ ] **SVM-SMOTE:** Use SVM to identify difficult regions
- [ ] **Compare:** LLM-SMOTE vs all traditional methods
  - Metric: F1 improvement gap
  - Metric: Per-class performance
  - Metric: Cost-benefit (time + API cost)

### LOW-Tier Class Improvements

Focus: ESFJ (0.17%), ESFP (0.34%), ESTJ (0.45%)

- [ ] Design specialized prompts for LOW-tier classes
  - More descriptive class characteristics
  - Include contrastive examples
  - Emphasize rare patterns

- [ ] Increase generation budget for LOW classes
  - Current: 100x multiplier
  - Test: 150x, 200x, 300x

- [ ] Test higher temperature for LOW classes
  - Current: 1.0
  - Test: 1.2, 1.5, 1.8
  - Goal: Increase diversity

- [ ] Implement diversity penalty
  - Prevent modal collapse
  - Cosine similarity threshold
  - Reject too-similar samples

---

## 🚀 Medium-Term Research (1 month)

Priority: **MEDIUM** | Timeline: **Next month**

### LLM Comparison Study

- [ ] **GPT-4o** vs **GPT-4o-mini** (current)
  - Compare: Quality, cost, speed
  - Dataset: MBTI (for direct comparison)

- [ ] **Claude-3.5-Sonnet**
  - Known for strong reasoning
  - Test on MBTI

- [ ] **Llama-3** (70B or 8B)
  - Open source alternative
  - Cost analysis: self-hosting vs API

- [ ] **Cost-Benefit Analysis**
  - Metric: F1 improvement per dollar
  - Metric: Time to complete
  - Create comparison table for paper

### Advanced Prompting Strategies

- [ ] **Chain-of-Thought (CoT) Prompting**
  - Prompt: "Think step by step about characteristics..."
  - Expected: Better quality samples
  - Downside: Higher token cost

- [ ] **Few-shot vs Zero-shot**
  - Zero-shot: Current approach (class description only)
  - Few-shot: Include 2-3 real examples
  - Measure: Quality improvement vs cost

- [ ] **Iterative Refinement for MID-tier**
  - Generate → Evaluate quality → Regenerate low-quality
  - Multi-stage: Coarse description → Fine details
  - Target: MID-tier classes (0.61% - 1.87%)

- [ ] **Ensemble Multiple LLMs**
  - Generate with GPT-4o, Claude, Llama-3
  - Vote/average for synthetic samples
  - Hypothesis: Diversity improves robustness

### Multi-Dataset Validation

**🔑 CRITICAL FOR PAPER: Demonstrate generalization beyond MBTI**

#### Text Classification Benchmarks

- [ ] **AG News** (4 classes, news articles)
  - Well-established benchmark
  - Moderate imbalance
  - Expected: Easy baseline

- [ ] **IMDB** (2 classes, sentiment)
  - Binary classification
  - Different structure than MBTI
  - Test: Does LLM-SMOTE work on sentiment?

- [ ] **Yelp Reviews** (5 classes, stars + sentiment)
  - Multi-class
  - Natural imbalance
  - Domain: User reviews

#### Multiclass Datasets

- [ ] **20 Newsgroups** (20 classes)
  - High class count
  - High imbalance
  - Similar to MBTI structure

- [ ] **DBpedia** (14 classes, ontology)
  - Structured knowledge
  - Test: LLM understanding of ontology

- [ ] **Yahoo Answers** (10 classes, Q&A)
  - Conversational text
  - Different style

#### Domain-Specific Datasets

- [ ] **Medical Text Classification**
  - Diagnosis codes (ICD-10)
  - Clinical notes
  - High stakes: Accuracy critical

- [ ] **Legal Documents**
  - Case classification
  - Contract types
  - Test: Domain-specific language

- [ ] **Scientific Papers**
  - Topic classification
  - Abstract-based
  - Technical vocabulary

#### Controlled Imbalance Experiments

- [ ] Create artificial imbalance ratios
  - 1:10 (mild)
  - 1:100 (moderate)
  - 1:1000 (extreme)

- [ ] Measure degradation of performance
  - Baseline vs SMOTE vs LLM-SMOTE
  - At what ratio does LLM-SMOTE excel?

- [ ] Cross-dataset pattern analysis
  - Do same hyperparameters work?
  - What predicts success?

---

## 📊 Research & Analysis (Paper Material)

Priority: **MEDIUM-HIGH** | Timeline: **Ongoing**

### Ablation Studies

**Goal:** Understand contribution of each component

- [ ] **Ensemble Anchor Selection**
  - Compare: Single anchor vs ensemble
  - Metric: Quality of selected anchors
  - Expected: +X% from ensemble

- [ ] **Validation Gating**
  - Compare: With vs without val-gating
  - Metric: F1 improvement
  - Expected: Prevents overfitting

- [ ] **F1-Budget Scaling**
  - Compare: Fixed budget vs adaptive
  - Metric: Per-tier improvements
  - Expected: Allocates resources optimally

- [ ] **Adaptive Weighting**
  - Compare: Fixed weight (0.5) vs adaptive
  - Metric: Overall F1
  - Expected: Balances synthetic vs real

- [ ] **Contamination Filter**
  - Compare: With vs without
  - Metric: Precision of synthetic samples
  - Expected: Removes low-quality generations

### Class-Wise Analysis

- [ ] **Which classes benefit most?**
  - Hypothesis: LOW-tier benefits most
  - Analyze: F1 delta per class
  - Plot: Class size vs improvement

- [ ] **LLM vs Traditional SMOTE per class**
  - For each class: LLM-SMOTE F1 - SMOTE F1
  - Identify: Where LLM excels
  - Identify: Where LLM struggles

- [ ] **Error Analysis**
  - Analyze: Misclassified samples
  - Question: Are synthetic samples confusing?
  - Question: Which class pairs confuse most?

- [ ] **Correlation Studies**
  - Class size ↔ F1 improvement
  - Class similarity ↔ Quality of synthetics
  - Budget multiplier ↔ F1 gain

### Cross-Dataset Patterns

- [ ] **Does +0.377% maintain across datasets?**
  - Hypothesis: Consistent improvement
  - Test: Multiple datasets
  - Report: Mean improvement across all

- [ ] **Do hyperparameters transfer?**
  - Use MBTI hyperparameters on AG News
  - Measure: Performance vs tuned
  - Result: Generalization score

- [ ] **What characteristics predict success?**
  - Feature: Dataset size
  - Feature: Number of classes
  - Feature: Imbalance ratio
  - Feature: Text length distribution
  - Model: Predict F1 improvement

---

## ⚡ Optimization & Infrastructure

Priority: **LOW-MEDIUM** | Timeline: **As needed**

### GPU Performance

- [ ] **Investigate embedding delay**
  - Issue: 1h+ to compute embeddings (expected: 10-15 min)
  - Check: CPU/GPU bottleneck
  - Check: I/O bottleneck
  - Check: Batch size too small

- [ ] **Test larger batch sizes**
  - Current: 64
  - Test: 128, 256, 512
  - Metric: Time to completion
  - Constraint: GPU memory (15GB)

- [ ] **Mixed precision training (FP16)**
  - Use: Half precision for embeddings
  - Expected: 2x throughput
  - Risk: Numerical stability

### Infrastructure Improvements

- [ ] **Add auto-shutdown to launch_simple_gpu.sh**
  - Current: No auto-shutdown (manual cleanup)
  - Add: 2-minute delay after completion
  - Safety: Prevent forgotten VMs

- [ ] **Document actual vs estimated times**
  - Record: Real completion times
  - Compare: vs estimates
  - Update: CLAUDE.md with accurate times

- [ ] **Create monitoring dashboard**
  - Real-time: VM status
  - Real-time: Experiment progress
  - Alerts: Stuck processes

---

## 💡 Future Ideas (Long-term)

Priority: **LOW** | Timeline: **3+ months**

### Advanced Techniques

- [ ] **Active Learning Integration**
  - Idea: LLM generates samples for uncertain regions
  - Use: Model uncertainty to guide generation
  - Expected: More efficient use of budget

- [ ] **Multi-Modal SMOTE**
  - Idea: Combine text + metadata
  - Example: Text + user features
  - Expected: Better quality synthetics

- [ ] **Hierarchical Generation**
  - Idea: Generate coarse → refine → finalize
  - Stage 1: Class membership
  - Stage 2: Sub-characteristics
  - Stage 3: Final text

- [ ] **Curriculum Learning**
  - Idea: Start with easy classes → hard classes
  - Hypothesis: Improves stability
  - Expected: Better overall results

### Deployment & Productionization

- [ ] **API Service**
  - REST API: Upload dataset → get augmented dataset
  - Authentication: API keys
  - Pricing: Per-sample generation

- [ ] **Python Package**
  - Name: `llm-smote`
  - Install: `pip install llm-smote`
  - Usage: Simple sklearn-like API

- [ ] **Benchmarking Suite**
  - Automated: Run on multiple datasets
  - Output: Comparison tables
  - Purpose: Marketing & validation

---

## 📝 Notes & Reminders

### Current Issues

- **Embedding performance:** Taking longer than expected on GPU (~1h vs 10-15min). Needs investigation.
- **Output buffering:** Logs were stuck initially, resolved with `-u` flag but still some delay.
- **No auto-shutdown:** Current runs using `launch_simple_gpu.sh` don't auto-shutdown. Remember to manually stop VMs!

### Key Decisions Made

- **OpenAI timeout:** Set to 30 seconds to prevent infinite hangs
- **Unbuffered output:** Using `-u` flag for real-time logs
- **GPU over CPU:** ~3x faster for embeddings (when working properly)
- **5 seeds:** Good balance between statistical robustness and cost

### Questions to Answer

- Why do LOW-tier classes still underperform? (ESFJ, ESFP, ESTJ)
- What's the optimal synthetic-to-real ratio per class?
- Can we predict which datasets will benefit most from LLM-SMOTE?
- Is the improvement consistent across different LLM providers?

---

## 📚 Paper Sections to Write

### Introduction
- [ ] Problem: Class imbalance in text classification
- [ ] Gap: Traditional SMOTE doesn't understand semantic relationships
- [ ] Solution: LLM-SMOTE with validation gating

### Related Work
- [ ] Traditional oversampling (SMOTE, ADASYN, Borderline-SMOTE)
- [ ] Deep learning approaches (VAEs, GANs)
- [ ] LLM for data augmentation (prior work)
- [ ] Our contribution: Multi-component system with quality gates

### Methodology
- [ ] System architecture (6 phases)
- [ ] Ensemble anchor selection
- [ ] Validation-based gating
- [ ] F1-budget scaling
- [ ] Contamination-aware filtering

### Experiments
- [ ] Datasets (MBTI + others)
- [ ] Baselines (Standard, SMOTE, Borderline-SMOTE, ADASYN)
- [ ] Metrics (macro F1, per-tier F1)
- [ ] Statistical significance (5-seed validation, 95% CI)

### Results
- [ ] Main result: +0.377% overall, +1.72% MID-tier
- [ ] Ablation studies
- [ ] Class-wise analysis
- [ ] Cross-dataset validation
- [ ] Cost-benefit analysis

### Discussion
- [ ] Why does it work? (Semantic understanding)
- [ ] Where does it fail? (LOW-tier classes)
- [ ] Limitations (API cost, latency)
- [ ] Future work (hierarchical generation, active learning)

### Conclusion
- [ ] Summary of contributions
- [ ] Practical impact
- [ ] Open questions

---

## 🔗 Related Files

- **Main Runner:** `/core/runner_phase2.py`
- **Launch Scripts:** `/phase_c/launch_simple_gpu.sh`, `/phase_c/launch_5seeds_gpu_doctor911.sh`
- **Documentation:** `/docs/` (deployment guides, status reports)
- **Results:** Will be in `/phase_c/results/` after collection

---

## 🎓 Research Timeline

**Week 1-2:** Complete Phase C v2.1 multi-seed validation
**Week 3-4:** SMOTE variants comparison + LOW-tier improvements
**Month 2:** LLM comparison + Multi-dataset validation (3-5 datasets)
**Month 3:** Ablation studies + Cross-dataset analysis + Paper writing
**Month 4:** Review, experiments for paper revisions, final polishing

---

**Target Conference/Journal:** TBD (ACL, EMNLP, NAACL, or JMLR)
**Submission Deadline:** TBD
**Expected Completion:** ~4 months from now
