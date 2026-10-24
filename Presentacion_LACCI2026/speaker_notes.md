# LACCI 2026 — Speaker script (15 min)

Target: **~13.7 min of talk**, leaving a buffer for slide changes and the chair's timing.
Numbers match `Escrito_Tesis_IEEE/paper.tex`. Anchor phrases are in **bold**: say them verbatim.

The script below is ~1,100 words, about 8–9 min if read straight through. The per-slide times
assume you pause and point at the charts (slides 9–13). Rehearse once with a timer: if you
finish under 12 min, slow down on slides 9 and 12; if the chair says the 15 min include
questions, cut slides 8 and 14 to a single sentence each.

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | The problem | 0:45 | 1:05 |
| 3 | Two ways to create data | 0:50 | 1:55 |
| 4 | The idea | 0:50 | 2:45 |
| 5 | Pipeline | 0:50 | 3:35 |
| 6 | Geometric filters | 1:00 | 4:35 |
| 7 | Soft weighting | 0:50 | 5:25 |
| 8 | Experimental setup | 0:50 | 6:15 |
| 9 | Main result | 1:20 | 7:35 |
| 10 | t-SNE | 0:45 | 8:20 |
| 11 | n-shot + classifiers | 1:00 | 9:20 |
| 12 | Simple beats complex | 1:20 | 10:40 |
| 13 | LLMs + NER | 1:10 | 11:50 |
| 14 | When / limits | 0:50 | 12:40 |
| 15 | Take-home | 1:00 | 13:40 |

If you are running late, shorten slides 8 and 14: their details are on the slides.

---

### 1. Title (0:20)
Good morning. I'm Benjamín Schindler from Universidad Adolfo Ibáñez in Chile; this is joint work with Gonzalo Ruz.
We study how to make LLM-generated training data actually useful when you only have a handful of labeled examples.

### 2. The problem (0:45)
Many real tasks, such as hate speech, news or spam, start with only 10 to 50 labeled examples per class, because labeling is slow and expensive.
With so few points, many different decision boundaries fit the data equally well, and the model has no way to know which one is right.
The natural fix is to create more examples from the ones we have: reliable new examples pin the boundary down.
But if some generated examples carry the wrong content for their label, they pull the boundary the wrong way, and real examples end up misclassified.
**We need more training data, but only data we can trust.**

### 3. Two ways to create data (0:50)
There are two ways to create that data. SMOTE interpolates between neighbors in embedding space: it gives geometric control, but it produces vectors, not sentences, and it only fills space you already know.
LLMs give you fluent, diverse text, but they know nothing about where those samples land in the space the classifier actually uses.
**Can we get LLM-quality text and SMOTE-like geometric control at the same time?**

### 4. The idea (0:50)
The key observation: LLM samples have heterogeneous quality. Some land inside their class, others in peripheral or cross-class zones, and adding all of them can hurt.
So we filter after generation: generate a three-times surplus, embed the candidates with the same encoder, score each one by its distance to real examples of its class, keep the best and weight them by their score.
Our question: can this make LLM augmentation reliably better than SMOTE in few-shot settings?

### 5. Pipeline (0:50)
Here is the full pipeline. Real examples go to the LLM as in-context demonstrations; there's no fine-tuning. Candidates are embedded with all-mpnet-base-v2, filtered, weighted, and combined with the real samples, which always get weight one.
The filter only sees embeddings, so **any LLM can be plugged in**. Our two contributions are the geometric filter and the soft weighting.

### 6. Geometric filters (1:00)
We compared filters of increasing complexity: no filter as a control, Local Outlier Factor, a four-level cascade, and a combined filter that requires passing both LOF and a cosine threshold.
Cascade level 1 is the simplest: the Euclidean distance to the nearest real example of the class. We rank candidates and keep the top N, so there's no threshold to tune.
Spoiler: **the simplest filter wins**, and I'll explain why in a few slides.

### 7. Soft weighting (0:50)
Instead of a hard accept-or-reject, we turn scores into training weights: min-max normalize, then apply a temperature. With T equal to 0.5 the weight is the squared score, so the most class-consistent candidates count much more.
The weights go straight into the classifier as sample weights.

### 8. Experimental setup (0:50)
We evaluated on 13 datasets across 9 domains, from binary up to 150 classes, plus three NER corpora, at 10, 25 and 50 examples per class.
We compared 10 augmentation methods, 5 classifiers and 5 LLMs, more than 6,700 configurations in total, with paired t-tests, Bonferroni correction, Cohen's d and win rates.
The main comparison has 1,890 runs.

### 9. Main result (1:20)
This is the headline: soft-weighted filtering improves macro F1 by **2.61 points over SMOTE**, with p below 0.0001, a large effect size of 0.95, and it beats SMOTE in **89% of configurations**. Binary filtering alone already gives 2.39.
Just as interesting is the bottom of the chart: **T5 paraphrasing, BERT contextual augmentation and embedding mixup do not beat SMOTE**; their confidence intervals straddle the SMOTE line. Using a pretrained model is not enough; what matters is selecting the right samples.

### 10. t-SNE (0:45)
This is what the filter does on the emotion dataset at 10-shot. On the left, all LLM candidates: many fall between classes or in the periphery. On the right, the samples that survive cluster around the real examples of their class.
**Cross-class candidates are discarded, and class regions get denser.**

### 11. n-shot and classifiers (1:00)
The gains are largest where data is scarcest: almost **5 points at 10-shot**, going from 67 to 72% macro F1, then 2.2 at 25-shot and 0.8 at 50-shot. All are significant, but returns clearly diminish as real data grows.
By classifier, linear models benefit most: linear SVC and Ridge gain about 3 points with effect sizes above 1. Filtered samples densify class regions near hyperplane boundaries; random forests, with axis-parallel splits, benefit less.

### 12. Simple beats complex (1:20)
Now the counter-intuitive finding: **distance alone beats every multi-criteria filter.** Three reasons.
First, for L2-normalized embeddings, squared Euclidean distance equals two times one minus the cosine similarity, so adding cosine gives exactly the same ranking: no new information.
Second, LOF needs 20 neighbors, but at 10-shot a class only has 9 other real examples, so the density estimates break down.
Third, intersecting criteria over-filters: you keep only near-duplicates and lose diversity. The combined filter accepted zero samples in 6 of 9 NER configurations.
And the curriculum curve below confirms it: adding candidates best-first, performance peaks at the top half and then declines. **Generate a surplus, filter lightly.**

### 13. LLMs and NER (1:10)
Is this tied to one LLM? No. With five generators from four providers, including two open models, all improvements are significant, from 1.5 to 2.3 points, and the Friedman test finds no significant LLM effect.
We also applied the same filter, unchanged, to named entity recognition, treating the dominant entity type of each sentence as its class. We get **plus 9.26 points of entity F1 with a 100% win rate.** To be transparent, most of that comes from the LLM data itself, and filtering adds a smaller, non-significant refinement on top.

### 14. When / limits (0:50)
The method helps most on hard classes: classes starting below 30% F1 gain over 10 points; for example "anger" goes from 28 to 45%. The sweet spot is a handful of classes and 10 to 25 examples per class.
The chart shows the class-count effect: gains are positive from 2 to 20 classes, peak at 6, and with 77 or 150 classes they disappear, because class regions get too packed. Gains are small at 50-shot, we only tested English, and we use the same encoder for filtering and classification, which we mitigate with an ablation over four embedding models.

### 15. Take-home (1:00)
Three messages. One: filtering LLM samples by embedding distance works, with plus 2.6 points over SMOTE across 13 datasets. Two: **simple beats complex**, so don't over-filter. Three: it is decoupled from the LLM and from the task.
And the recipe fits on one slide: prompt with your few examples, generate a three-times surplus, keep the closest half, soft-weight them and train a linear classifier.
Thank you, I'm happy to take questions.

---

## Likely questions (backup slides at the end)

- **"Is the gain from filtering or from weighting?"** → Backup 1: binary filtering vs. no augmentation +1.75 pp; soft over binary only +0.22 (p = 0.053). The selection does the heavy lifting.
- **"Isn't using the same encoder circular?"** → 90.5% agreement across 4 embedding models; geometric filtering beats teacher-model filtering (+2.78 vs +2.43 pp, p = 0.0001).
- **"Caching makes seeds identical?"** → Backup 1: on stochastic classifiers only, the effect is still +1.79 pp, p = 0.0015.
- **"NER details?"** → Backup 2, full table.
- **"Cost?"** → In-context generation with a fast model; 3 candidates per desired sample; no fine-tuning.
- **"Why not fine-tune the LLM?"** → Out of scope: we wanted a filter that works with any off-the-shelf model; fine-tuning is complementary.
