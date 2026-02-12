#!/usr/bin/env python3
"""
Robust MBTI Experiment: SMOTE vs LLM+SMOTE Hybrid with Soft Weighting

Applies best techniques from thesis research:
- Cascade L1 (distance-only) geometric filter
- Soft weighting (minmax, temp=0.5)
- Multiple classifiers (Ridge, SVC, LogReg)
- Multiple seeds with paired statistics

Strategy:
- Classes with < THRESHOLD train samples → LLM augmentation (filtered, soft-weighted)
- Then SMOTE globally to balance all 16 classes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
CACHE_DIR = PROJECT_ROOT / "cache" / "mbti_llm"
RESULTS_DIR = PROJECT_ROOT / "results" / "mbti_robust"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [42, 123, 456]
TEST_SIZE = 0.2
SIZE_THRESHOLD = 150  # classes with < this in train split get LLM augmentation
OVERSAMPLE_FACTOR = 2.5  # generate 2.5x candidates, keep top-N

# Best soft weighting config from thesis experiments
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

# LLM generation
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"
N_SHOT = 15  # examples per prompt (individual posts, not full user strings)
BATCH_SIZE = 25
MAX_BATCHES_PER_CLASS = 40  # safety limit

# Classifiers (RF excluded — doesn't benefit from sample weights)
CLASSIFIERS = {
    "ridge": lambda seed: RidgeClassifier(alpha=1.0),
    "svc_linear": lambda seed: SVC(kernel="linear", random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
}

# Methods to compare
METHODS = ["baseline", "smote_only", "llm_binary_smote", "llm_soft_smote"]

# MBTI type descriptions for prompt enrichment
MBTI_DESCRIPTIONS = {
    "ESTJ": "practical, organized, direct — focused on rules, schedules, and concrete facts",
    "ESFJ": "warm, social, supportive — focused on people, harmony, and social norms",
    "ESFP": "energetic, spontaneous, fun-loving — about experiences, activities, and living in the moment",
    "ESTP": "action-oriented, pragmatic, blunt — about challenges, risk-taking, and problem-solving",
    "ISFJ": "careful, traditional, supportive — focused on duties, details, and people's needs",
    "ENFJ": "charismatic, empathetic, inspiring — about personal growth and helping others",
    "ISTJ": "methodical, reliable, factual — focused on procedures, responsibility, and order",
    "ENTJ": "commanding, strategic, decisive — about goals, efficiency, and leadership",
    "ISFP": "gentle, artistic, authentic — about personal values and sensory experiences",
    "ISTP": "analytical, independent, concise — about mechanics, tools, and problem-solving",
    "ENFP": "enthusiastic, creative, scattered — exploring ideas, possibilities, and connections",
    "ENTP": "witty, argumentative, inventive — challenging ideas and debating concepts",
    "INTJ": "strategic, independent, analytical — about systems, planning, and long-term vision",
    "INTP": "theoretical, precise, curious — exploring abstract concepts, logic, and frameworks",
    "INFJ": "insightful, idealistic, complex — about meaning, patterns, and human nature",
    "INFP": "introspective, idealistic, creative — about values, authenticity, and inner experiences",
}

ALL_MBTI_TYPES = set(MBTI_DESCRIPTIONS.keys())


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mbti_data():
    """Load MBTI dataset from CSV."""
    texts, labels = [], []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['posts'])
            labels.append(row['type'])
    return texts, np.array(labels)


def extract_individual_posts(full_posts, max_posts=5):
    """Extract clean individual posts from ||| separated string."""
    posts = [p.strip().strip("'\"") for p in full_posts.split("|||")]
    clean = []
    for p in posts:
        if len(p) < 20:
            continue
        if p.startswith("http"):
            continue
        upper = p.upper()
        if any(t in upper for t in ALL_MBTI_TYPES):
            continue
        # Filter cognitive function jargon
        for fn in ["Fi", "Fe", "Ti", "Te", "Si", "Se", "Ni", "Ne"]:
            if f" {fn} " in f" {p} " or f" {fn}>" in p:
                break
        else:
            clean.append(p[:300])
            continue
    return clean[:max_posts]


# ============================================================================
# PROMPT CREATION
# ============================================================================

def create_mbti_prompt(class_name, class_texts, n_generate):
    """Create MBTI generation prompt with negative prompting + style examples."""
    # Extract individual posts for cleaner examples
    example_posts = []
    for text in class_texts:
        posts = extract_individual_posts(text, max_posts=3)
        example_posts.extend(posts)
    example_posts = example_posts[:N_SHOT]

    if not example_posts:
        # Fallback: use raw texts truncated
        example_posts = [t[:200] for t in class_texts[:N_SHOT]]

    examples_text = "\n".join(f"- {ex}" for ex in example_posts)
    other_types = sorted(ALL_MBTI_TYPES - {class_name})
    description = MBTI_DESCRIPTIONS.get(class_name, "")

    return f"""Generate {n_generate} realistic online forum posts that reflect the communication style of someone with {class_name} personality type.

{class_name} traits: {description}

These are real examples of how {class_name} people write online:
{examples_text}

REQUIREMENTS:
- Match the casual, personal tone of the examples
- Reflect {class_name} thinking patterns and communication style naturally
- Cover diverse topics: relationships, work, hobbies, opinions, daily life
- Each post: 1-4 sentences, conversational tone
- Create genuinely different posts, not paraphrases of examples

DO NOT:
- Mention any personality type names ({', '.join(other_types[:8])}, etc.)
- Use personality jargon (cognitive functions, MBTI terminology)
- Write generic text that could belong to any type
- Include URLs, numbering prefixes, or metadata

Output {n_generate} posts, one per line, numbered 1-{n_generate}."""


# ============================================================================
# LLM GENERATION WITH CACHE
# ============================================================================

def get_cache_key(class_name, n_generate, batch_idx, seed):
    """MD5 cache key for LLM generation."""
    raw = f"mbti_{class_name}_{n_generate}_batch{batch_idx}_seed{seed}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def generate_llm_batch_cached(provider, class_name, class_texts, n_generate,
                                batch_idx, seed, embed_model):
    """Generate a batch of LLM samples, with file-based cache."""
    cache_key = get_cache_key(class_name, n_generate, batch_idx, seed)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("texts"):
            embeddings = embed_model.encode(cached["texts"], show_progress_bar=False)
            return embeddings, cached["texts"]

    prompt = create_mbti_prompt(class_name, class_texts, n_generate)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(5):
        try:
            response, _ = provider.generate(messages, temperature=0.9, max_tokens=4000)
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** attempt * 5
                print(f"        Rate limit, retry in {wait}s...")
                time.sleep(wait)
                continue
            print(f"        LLM error: {e}")
            return np.array([]).reshape(0, 768), []
    else:
        return np.array([]).reshape(0, 768), []

    # Parse response
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 15:
            generated.append(clean)

    if not generated:
        return np.array([]).reshape(0, 768), []

    # Cache
    with open(cache_file, "w") as f:
        json.dump({
            "class_name": class_name, "batch_idx": batch_idx,
            "seed": seed, "texts": generated,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    embeddings = embed_model.encode(generated, show_progress_bar=False)
    return embeddings, generated


def generate_pool_for_class(provider, class_name, class_texts, target_n,
                             seed, embed_model):
    """Generate a pool of LLM candidates for one class (multiple batches)."""
    pool_embeddings = []
    pool_texts = []
    needed = int(target_n * OVERSAMPLE_FACTOR)

    batch_idx = 0
    while len(pool_texts) < needed and batch_idx < MAX_BATCHES_PER_CLASS:
        batch_n = min(BATCH_SIZE, needed - len(pool_texts))
        emb, texts = generate_llm_batch_cached(
            provider, class_name, class_texts, batch_n, batch_idx, seed, embed_model
        )
        if len(emb) > 0:
            pool_embeddings.append(emb)
            pool_texts.extend(texts)
        batch_idx += 1

    if not pool_embeddings:
        return np.array([]).reshape(0, 768), []

    return np.vstack(pool_embeddings), pool_texts


# ============================================================================
# FILTERING + SCORING
# ============================================================================

def normalize_scores(raw_scores, method="minmax", temperature=1.0, min_weight=0.0):
    """Convert raw cascade scores to sample weights."""
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)
    weight_range = 1.0 - min_weight
    if method == "minmax":
        s_min, s_max = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    else:
        normalized = raw_scores
    if temperature != 1.0:
        normalized = np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / temperature)
    return min_weight + weight_range * normalized


def filter_and_score_class(pool_emb, target_n, real_embeddings, real_labels,
                            class_name, cascade):
    """Apply cascade filter, return top-N embeddings + soft weights."""
    if len(pool_emb) == 0:
        return np.array([]).reshape(0, 768), np.array([]), np.array([])

    labels_arr = np.array(real_labels) if not isinstance(real_labels, np.ndarray) else real_labels
    cls_mask = labels_arr == class_name
    cls_emb = real_embeddings[cls_mask]
    anchor = cls_emb.mean(axis=0) if len(cls_emb) > 0 else real_embeddings.mean(axis=0)

    scores, _ = cascade.compute_quality_scores(
        pool_emb, anchor, real_embeddings, labels_arr, class_name
    )

    # Select top-N
    actual_n = min(target_n, len(pool_emb))
    top_idx = np.argsort(scores)[-actual_n:]

    top_emb = pool_emb[top_idx]
    top_scores = scores[top_idx]
    top_weights = normalize_scores(top_scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)

    return top_emb, top_scores, top_weights


# ============================================================================
# SMOTE
# ============================================================================

def apply_smote_global(embeddings, labels, seed):
    """Apply SMOTE globally to balance all classes to majority class size."""
    counts = Counter(labels)
    max_count = max(counts.values())

    sampling_strategy = {}
    for cls, cnt in counts.items():
        if cnt < max_count:
            sampling_strategy[cls] = max_count

    if not sampling_strategy:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])

    min_class_size = min(counts.values())
    k_neighbors = min(5, min_class_size - 1)
    if k_neighbors < 1:
        print(f"    WARNING: Smallest class has {min_class_size} samples, skipping SMOTE")
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])

    try:
        smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=seed
        )
        X_res, y_res = smote.fit_resample(embeddings, labels)
        return X_res[len(embeddings):], y_res[len(embeddings):]
    except Exception as e:
        print(f"    SMOTE error: {e}")
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(train_emb, train_labels, test_emb, test_labels, clf,
             sample_weight=None):
    """Train and evaluate classifier."""
    if sample_weight is not None:
        clf.fit(train_emb, train_labels, sample_weight=sample_weight)
    else:
        clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)
    macro_f1 = f1_score(test_labels, preds, average='macro')
    per_class = {}
    all_classes = sorted(set(test_labels))
    per_class_f1 = f1_score(test_labels, preds, average=None, labels=all_classes)
    for cls, f in zip(all_classes, per_class_f1):
        per_class[cls] = float(f)
    return float(macro_f1), per_class


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class MBTIResult:
    method: str
    classifier: str
    seed: int
    macro_f1: float
    per_class_f1: dict
    n_train: int
    n_llm_samples: int
    n_smote_samples: int
    small_classes: list
    small_class_avg_f1: float
    large_class_avg_f1: float
    acceptance_rate: float
    weight_mean: float
    weight_std: float


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    start_time = time.time()
    print("=" * 80)
    print("MBTI ROBUST EXPERIMENT: SMOTE vs LLM+SMOTE (Soft Weighted)")
    print(f"Threshold: {SIZE_THRESHOLD} | Seeds: {SEEDS} | Classifiers: {list(CLASSIFIERS.keys())}")
    print("=" * 80)

    # Load data
    print("\nLoading MBTI data...")
    texts, labels = load_mbti_data()
    print(f"  Total: {len(texts)} samples, {len(set(labels))} classes")

    counts = Counter(labels)
    for t, c in sorted(counts.items(), key=lambda x: x[1]):
        print(f"    {t:6s}: {c:5d}")

    # Initialize models
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)
    cascade = FilterCascade(**FILTER_CONFIG)

    all_results = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*80}")

        # Stratified train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=seed, stratify=labels
        )
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        train_counts = Counter(train_labels)
        small_classes = sorted([c for c, n in train_counts.items() if n < SIZE_THRESHOLD])
        large_classes = sorted([c for c, n in train_counts.items() if n >= SIZE_THRESHOLD])

        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        print(f"  Small classes (< {SIZE_THRESHOLD}): {small_classes}")
        for c in small_classes:
            print(f"    {c}: {train_counts[c]} samples")

        # Compute embeddings
        print("\n  Computing embeddings...")
        train_emb = embed_model.encode(train_texts, show_progress_bar=True, batch_size=256)
        test_emb = embed_model.encode(test_texts, show_progress_bar=True, batch_size=256)

        # ---- LLM generation for small classes ----
        print("\n  Generating LLM samples for small classes...")
        llm_class_data = {}  # class -> {emb, weights_binary, weights_soft}
        total_llm_samples = 0
        acceptance_rates = []

        for cls in small_classes:
            cls_count = train_counts[cls]
            # Target: bring up to threshold
            target_n = SIZE_THRESHOLD - cls_count
            if target_n <= 0:
                continue

            cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]

            print(f"    {cls}: {cls_count} → {SIZE_THRESHOLD} (need {target_n} LLM samples)")

            pool_emb, pool_texts = generate_pool_for_class(
                provider, cls, cls_texts, target_n, seed, embed_model
            )

            if len(pool_emb) == 0:
                print(f"      WARNING: No LLM samples generated for {cls}")
                continue

            top_emb, top_scores, top_weights = filter_and_score_class(
                pool_emb, target_n, train_emb, train_labels, cls, cascade
            )

            acc_rate = len(top_emb) / len(pool_emb) if len(pool_emb) > 0 else 0
            acceptance_rates.append(acc_rate)
            total_llm_samples += len(top_emb)

            llm_class_data[cls] = {
                "emb": top_emb,
                "weights_soft": top_weights,
                "weights_binary": np.ones(len(top_emb)),
            }

            print(f"      Generated {len(pool_emb)} candidates → kept {len(top_emb)} "
                  f"(accept: {acc_rate*100:.1f}%, "
                  f"weight range: [{top_weights.min():.3f}, {top_weights.max():.3f}])")

        avg_acceptance = np.mean(acceptance_rates) if acceptance_rates else 0

        # ---- Run all methods × classifiers ----
        for clf_name, clf_factory in CLASSIFIERS.items():
            for method in METHODS:
                print(f"\n  [{clf_name}/{method}]", end=" ")

                if method == "baseline":
                    clf = clf_factory(seed)
                    macro_f1, per_class_f1 = evaluate(
                        train_emb, train_labels, test_emb, test_labels, clf
                    )
                    n_smote = 0
                    w_mean, w_std = 0, 0

                elif method == "smote_only":
                    smote_emb, smote_labels = apply_smote_global(
                        train_emb, train_labels, seed
                    )
                    if len(smote_emb) > 0:
                        aug_emb = np.vstack([train_emb, smote_emb])
                        aug_labels = np.concatenate([train_labels, smote_labels])
                    else:
                        aug_emb = train_emb
                        aug_labels = train_labels

                    clf = clf_factory(seed)
                    macro_f1, per_class_f1 = evaluate(
                        aug_emb, aug_labels, test_emb, test_labels, clf
                    )
                    n_smote = len(smote_emb)
                    w_mean, w_std = 1.0, 0.0

                elif method in ("llm_binary_smote", "llm_soft_smote"):
                    # Step 1: merge LLM samples into training data
                    aug_emb_parts = [train_emb]
                    aug_label_parts = [train_labels]
                    weight_parts = [np.ones(len(train_emb))]

                    use_soft = (method == "llm_soft_smote")

                    for cls, data in llm_class_data.items():
                        aug_emb_parts.append(data["emb"])
                        aug_label_parts.append(np.array([cls] * len(data["emb"])))
                        w = data["weights_soft"] if use_soft else data["weights_binary"]
                        weight_parts.append(w)

                    augmented_emb = np.vstack(aug_emb_parts)
                    augmented_labels = np.concatenate(aug_label_parts)
                    all_weights = np.concatenate(weight_parts)

                    # Step 2: SMOTE on (real + LLM-augmented)
                    smote_emb, smote_labels = apply_smote_global(
                        augmented_emb, augmented_labels, seed
                    )

                    if len(smote_emb) > 0:
                        final_emb = np.vstack([augmented_emb, smote_emb])
                        final_labels = np.concatenate([augmented_labels, smote_labels])
                        # SMOTE samples get weight 1.0
                        final_weights = np.concatenate([all_weights, np.ones(len(smote_emb))])
                    else:
                        final_emb = augmented_emb
                        final_labels = augmented_labels
                        final_weights = all_weights

                    clf = clf_factory(seed)
                    # Only pass sample_weight for soft — binary has all 1s anyway
                    if use_soft:
                        macro_f1, per_class_f1 = evaluate(
                            final_emb, final_labels, test_emb, test_labels, clf,
                            sample_weight=final_weights
                        )
                    else:
                        macro_f1, per_class_f1 = evaluate(
                            final_emb, final_labels, test_emb, test_labels, clf
                        )

                    n_smote = len(smote_emb)
                    syn_weights = np.concatenate([
                        data["weights_soft" if use_soft else "weights_binary"]
                        for data in llm_class_data.values()
                    ]) if llm_class_data else np.array([1.0])
                    w_mean = float(syn_weights.mean())
                    w_std = float(syn_weights.std())

                # Compute small/large class F1
                small_f1s = [per_class_f1.get(c, 0) for c in small_classes]
                large_f1s = [per_class_f1.get(c, 0) for c in large_classes]
                small_avg = float(np.mean(small_f1s)) if small_f1s else 0
                large_avg = float(np.mean(large_f1s)) if large_f1s else 0

                result = MBTIResult(
                    method=method,
                    classifier=clf_name,
                    seed=seed,
                    macro_f1=macro_f1,
                    per_class_f1=per_class_f1,
                    n_train=len(train_emb),
                    n_llm_samples=total_llm_samples if "llm" in method else 0,
                    n_smote_samples=n_smote if method != "baseline" else 0,
                    small_classes=small_classes,
                    small_class_avg_f1=small_avg,
                    large_class_avg_f1=large_avg,
                    acceptance_rate=avg_acceptance if "llm" in method else 0,
                    weight_mean=w_mean,
                    weight_std=w_std,
                )
                all_results.append(result)

                print(f"F1={macro_f1:.4f} (small={small_avg:.4f}, large={large_avg:.4f})")

        # Save incrementally
        save_results(all_results)

    # Final analysis
    generate_report(all_results)

    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print("Done!")


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def save_results(results):
    """Save results to JSON."""
    output = {
        "experiment": "mbti_robust",
        "config": {
            "threshold": SIZE_THRESHOLD,
            "seeds": SEEDS,
            "classifiers": list(CLASSIFIERS.keys()),
            "methods": METHODS,
            "filter": FILTER_CONFIG,
            "normalization": NORMALIZATION,
            "temperature": TEMPERATURE,
            "oversample_factor": OVERSAMPLE_FACTOR,
        },
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


def compute_paired_stats(method_f1s, baseline_f1s, n_comparisons=3):
    """Paired t-test, CI, Cohen's d, Bonferroni."""
    m = np.array(method_f1s)
    b = np.array(baseline_f1s)
    deltas = m - b
    n = len(deltas)
    mean_d = np.mean(deltas)
    std_d = np.std(deltas, ddof=1) if n > 1 else 0
    se = std_d / np.sqrt(n) if n > 1 else 0

    if n > 1 and se > 0:
        t_stat, p_raw = stats.ttest_rel(m, b)
        ci = stats.t.interval(0.95, n-1, loc=mean_d, scale=se)
    else:
        t_stat, p_raw = 0, 1
        ci = (mean_d, mean_d)

    p_bonf = min(p_raw * n_comparisons, 1.0)
    d = mean_d / std_d if std_d > 0 else 0  # Cohen's d
    win_rate = np.mean(deltas > 0) * 100

    return {
        "delta_pp": mean_d * 100,
        "std_pp": std_d * 100,
        "ci_low": ci[0] * 100,
        "ci_high": ci[1] * 100,
        "p_raw": p_raw,
        "p_bonferroni": p_bonf,
        "cohens_d": d,
        "win_rate": win_rate,
        "n": n,
    }


def generate_report(results):
    """Generate summary tables."""
    print("\n" + "=" * 90)
    print("FINAL REPORT")
    print("=" * 90)

    # ---- Table 1: Overall method comparison (vs smote_only) ----
    print("\n--- TABLE 1: Overall Method Comparison (vs SMOTE) ---")
    print(f"{'Method':<22s} {'Mean F1':>8s} {'Δ SMOTE':>10s} {'95% CI':>20s} "
          f"{'p(Bonf)':>10s} {'d':>6s} {'Win%':>6s}")
    print("-" * 85)

    for method in METHODS:
        if method == "smote_only":
            continue

        method_f1s = []
        smote_f1s = []

        # Pair by (classifier, seed)
        for clf_name in CLASSIFIERS:
            for seed in SEEDS:
                mf = [r.macro_f1 for r in results
                      if r.method == method and r.classifier == clf_name and r.seed == seed]
                sf = [r.macro_f1 for r in results
                      if r.method == "smote_only" and r.classifier == clf_name and r.seed == seed]
                if mf and sf:
                    method_f1s.append(mf[0])
                    smote_f1s.append(sf[0])

        if not method_f1s:
            continue

        s = compute_paired_stats(method_f1s, smote_f1s, n_comparisons=3)
        mean_f1 = np.mean(method_f1s) * 100
        sig = "*" if s["p_bonferroni"] < 0.05 else ""
        print(f"  {method:<20s} {mean_f1:>7.2f}% {s['delta_pp']:>+9.2f}pp "
              f"[{s['ci_low']:>+7.2f}, {s['ci_high']:>+7.2f}] "
              f"{s['p_bonferroni']:>9.4f}{sig} {s['cohens_d']:>5.2f} {s['win_rate']:>5.1f}%")

    # Print SMOTE reference
    smote_f1s_all = [r.macro_f1 for r in results if r.method == "smote_only"]
    if smote_f1s_all:
        print(f"  {'smote_only (ref)':<20s} {np.mean(smote_f1s_all)*100:>7.2f}%    (reference)")

    # ---- Table 2: By classifier ----
    print("\n--- TABLE 2: LLM Soft+SMOTE vs SMOTE by Classifier ---")
    print(f"{'Classifier':<22s} {'Δ SMOTE':>10s} {'95% CI':>20s} {'p':>10s} {'Win%':>6s}")
    print("-" * 72)

    for clf_name in CLASSIFIERS:
        method_f1s = [r.macro_f1 for r in results
                      if r.method == "llm_soft_smote" and r.classifier == clf_name]
        smote_f1s = [r.macro_f1 for r in results
                     if r.method == "smote_only" and r.classifier == clf_name]
        if method_f1s and smote_f1s:
            s = compute_paired_stats(method_f1s, smote_f1s, n_comparisons=1)
            print(f"  {clf_name:<20s} {s['delta_pp']:>+9.2f}pp "
                  f"[{s['ci_low']:>+7.2f}, {s['ci_high']:>+7.2f}] "
                  f"{s['p_raw']:>9.4f} {s['win_rate']:>5.1f}%")

    # ---- Table 3: Small vs Large class impact ----
    print("\n--- TABLE 3: Impact on Small vs Large Classes ---")
    print(f"{'Method':<22s} {'Small F1':>10s} {'Large F1':>10s} {'Small Δ':>10s} {'Large Δ':>10s}")
    print("-" * 65)

    for method in METHODS:
        small_avg = np.mean([r.small_class_avg_f1 for r in results if r.method == method])
        large_avg = np.mean([r.large_class_avg_f1 for r in results if r.method == method])
        smote_small = np.mean([r.small_class_avg_f1 for r in results if r.method == "smote_only"])
        smote_large = np.mean([r.large_class_avg_f1 for r in results if r.method == "smote_only"])
        delta_small = (small_avg - smote_small) * 100
        delta_large = (large_avg - smote_large) * 100
        print(f"  {method:<20s} {small_avg*100:>9.2f}% {large_avg*100:>9.2f}% "
              f"{delta_small:>+9.2f}pp {delta_large:>+9.2f}pp")

    # ---- Table 4: Per-class breakdown (averaged across seeds & classifiers) ----
    print("\n--- TABLE 4: Per-Class F1 Breakdown (avg across seeds & classifiers) ---")

    all_classes = sorted(set(c for r in results for c in r.per_class_f1.keys()))
    train_counts_approx = Counter(results[0].per_class_f1.keys())  # just for ordering

    print(f"{'Type':<8s} {'Baseline':>10s} {'SMOTE':>10s} {'LLM+SMOTE':>10s} {'Soft+SMOTE':>10s} {'Δ Soft':>10s}")
    print("-" * 62)

    for cls in all_classes:
        vals = {}
        for method in METHODS:
            f1s = [r.per_class_f1.get(cls, 0) for r in results if r.method == method]
            vals[method] = np.mean(f1s) * 100 if f1s else 0

        delta = vals.get("llm_soft_smote", 0) - vals.get("smote_only", 0)
        is_small = cls in (results[0].small_classes if results else [])
        marker = " *" if is_small else ""

        print(f"  {cls:<6s} {vals.get('baseline', 0):>9.2f}% {vals.get('smote_only', 0):>9.2f}% "
              f"{vals.get('llm_binary_smote', 0):>9.2f}% {vals.get('llm_soft_smote', 0):>9.2f}% "
              f"{delta:>+9.2f}pp{marker}")

    print("\n  * = LLM-augmented class")

    # Save report
    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write("MBTI Robust Experiment Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Threshold: {SIZE_THRESHOLD}\n")
        f.write(f"Total results: {len(results)}\n")


if __name__ == "__main__":
    main()
