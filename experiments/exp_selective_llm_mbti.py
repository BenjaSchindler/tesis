#!/usr/bin/env python3
"""
Selective LLM Augmentation for MBTI Classification

Uses LLM augmentation ONLY on small classes (below a threshold) to bring them
up to a target size, then applies SMOTE globally to balance all 16 classes.

Hypothesis: Targeted LLM augmentation on minority classes + global SMOTE
outperforms SMOTE-only by providing semantic diversity where it's needed most.

Research findings applied:
- cascade_l1 filter (best: +1.45pp vs SMOTE)
- Negative prompting (best: +5.59pp)
- LLM wins 88-90% in multi-class scenarios
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "mbti_1.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "selective_llm_mbti"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Thresholds: classes with fewer train samples than this get LLM augmentation
SIZE_THRESHOLDS = [100, 200]

# Target sizes: what to bring small classes up to after LLM augmentation
TARGET_SIZES = [200, 300]

# Seeds for reproducibility
SEEDS = [42, 123, 456]

# Train/test split
TEST_SIZE = 0.2

# LLM config (best from prior experiments)
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"

# Filter config (cascade level 1 = distance only, best performer)
LLM_FILTER = {"filter_level": 1, "k_neighbors": 10}

# Generation config
N_SHOT = 25
BATCH_SIZE = 25
MAX_LLM_CALLS_PER_CLASS = 30
OVERSAMPLE_FACTOR = 2.0

# MBTI type descriptions for prompt enrichment
MBTI_DESCRIPTIONS = {
    "ESTJ": "practical, organized, direct - focused on rules and concrete facts",
    "ESFJ": "warm, social, supportive - focused on people and harmony",
    "ESFP": "energetic, spontaneous, fun-loving - about experiences and activities",
    "ESTP": "action-oriented, pragmatic, blunt - about challenges and problem-solving",
    "ISFJ": "careful, traditional, supportive - focused on duties and people's needs",
    "ENFJ": "charismatic, empathetic, inspiring - about personal growth and relationships",
    "ISTJ": "methodical, reliable, factual - focused on details and procedures",
    "ENTJ": "commanding, strategic, decisive - about goals and efficiency",
    "ISFP": "gentle, artistic, authentic - about values and sensory experiences",
    "ISTP": "analytical, independent, concise - about mechanics and problem-solving",
    "ENFP": "enthusiastic, creative, scattered - exploring ideas and possibilities",
    "ENTP": "witty, argumentative, inventive - challenging ideas and exploring concepts",
    "INTJ": "strategic, independent, analytical - about systems and long-term vision",
    "INTP": "theoretical, precise, curious - exploring abstract concepts and logic",
    "INFJ": "insightful, idealistic, complex - about meaning and human nature",
    "INFP": "introspective, idealistic, creative - about values and inner experiences",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SelectiveResult:
    """Result for one experimental condition."""
    method: str
    size_threshold: Optional[int]
    target_size: Optional[int]
    seed: int
    macro_f1: float
    weighted_f1: float
    baseline_f1: float
    delta_vs_baseline: float
    per_class_f1: Dict[str, float]
    small_class_avg_f1: float
    large_class_avg_f1: float
    classes_augmented: List[str]
    total_llm_calls: int
    total_llm_samples: int
    total_smote_samples: int
    acceptance_rates: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mbti_data(data_path: Path) -> Tuple[List[str], np.ndarray]:
    """Load MBTI dataset from CSV. Each row = 1 sample."""
    texts = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['posts'])
            labels.append(row['type'])
    return texts, np.array(labels)


def identify_small_classes(
    labels: np.ndarray, threshold: int
) -> Tuple[List[str], List[str]]:
    """Identify classes below the threshold."""
    counts = Counter(labels)
    small = [cls for cls, cnt in counts.items() if cnt < threshold]
    large = [cls for cls, cnt in counts.items() if cnt >= threshold]
    return sorted(small), sorted(large)


def print_class_distribution(labels: np.ndarray, title: str = "Class Distribution"):
    """Print class distribution table."""
    counts = Counter(labels)
    print(f"\n  {title}:")
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1]):
        bar = "#" * min(cnt // 20, 50)
        print(f"    {cls:6s}: {cnt:5d} {bar}")
    print(f"    Total: {len(labels)}")


# ============================================================================
# PROMPT CREATION (Negative Prompting for MBTI)
# ============================================================================

MBTI_TYPES = {"ESTJ", "ESFJ", "ESFP", "ESTP", "ISFJ", "ENFJ", "ISTJ", "ENTJ",
              "ISFP", "ISTP", "ENFP", "ENTP", "INTJ", "INTP", "INFJ", "INFP"}


def extract_individual_posts(full_posts: str, max_posts: int = 5) -> List[str]:
    """Extract individual posts from a ||| separated string.

    Filters out:
    - Short posts (<20 chars)
    - Posts that are just URLs
    - Posts that explicitly mention MBTI types (to avoid label leakage)
    """
    posts = [p.strip().strip("'\"") for p in full_posts.split("|||")]
    clean = []
    for p in posts:
        if len(p) < 20:
            continue
        if p.startswith("http"):
            continue
        # Skip posts that mention MBTI types or cognitive function jargon (label leakage)
        upper = p.upper()
        if any(t in upper for t in MBTI_TYPES):
            continue
        # Filter cognitive function jargon (Fi, Fe, Ti, Te, Si, Se, Ni, Ne)
        if any(f" {fn} " in f" {p} " or f" {fn}>" in p or f">{fn}" in p
               for fn in ["Fi", "Fe", "Ti", "Te", "Si", "Se", "Ni", "Ne"]):
            continue
        clean.append(p[:300])
    return clean[:max_posts]


def create_mbti_negative_prompt(
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    all_classes: List[str]
) -> str:
    """Create MBTI-specific prompt using negative prompting."""
    # Extract individual posts from samples for cleaner examples
    example_posts = []
    for text in class_texts[:N_SHOT]:
        posts = extract_individual_posts(text, max_posts=3)
        example_posts.extend(posts)
    example_posts = example_posts[:20]  # Cap total examples

    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(example_posts))
    other_classes = [c for c in all_classes if c != class_name]
    description = MBTI_DESCRIPTIONS.get(class_name, "")

    return f"""Generate {n_generate} NEW online forum/discussion posts written by someone with {class_name} personality type.

{class_name} personality traits: {description}

These posts come from personality discussion forums where people share opinions, personal experiences, and reactions to topics. The writing style is casual, personal, and conversational.

REAL EXAMPLES from {class_name} users:
{examples_text}

REQUIREMENTS:
- Generate text that reflects {class_name} communication patterns and thinking style
- Keep the casual forum/discussion tone
- Cover diverse topics (relationships, work, hobbies, opinions, personal experiences)
- Each post should be 1-4 sentences
- Create diverse variations, not copies of examples

DO NOT generate posts that sound like these other personality types: {', '.join(other_classes)}
DO NOT:
- Generate generic or vague text that could belong to any personality type
- Copy examples exactly or paraphrase them too closely
- Mention any MBTI type names (no "ESTJ", "INFP", etc.), cognitive functions, or personality jargon
- Generate URLs, links, or references to external content
- Include numbering prefixes, labels, or metadata

Output ONLY {n_generate} generated posts, one per line, numbered 1-{n_generate}."""


# ============================================================================
# LLM GENERATION + FILTERING
# ============================================================================

def generate_llm_batch(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    embed_model: SentenceTransformer,
    all_classes: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Generate a batch of LLM samples for one MBTI class."""
    prompt = create_mbti_negative_prompt(class_name, class_texts, n_generate, all_classes)
    messages = [{"role": "user", "content": prompt}]

    try:
        response, _ = provider.generate(messages, temperature=1.0, max_tokens=4000)
    except Exception as e:
        print(f"      LLM error: {e}")
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

    embeddings = embed_model.encode(generated, show_progress_bar=False)
    return embeddings, generated


def generate_llm_for_class(
    provider,
    filter_obj: FilterCascade,
    class_name: str,
    class_texts: List[str],
    target_n: int,
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    embed_model: SentenceTransformer,
    all_classes: List[str]
) -> Dict:
    """Generate target_n filtered LLM samples for one class using pool-based approach."""
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    # Generate until we have enough candidates
    while llm_calls < MAX_LLM_CALLS_PER_CLASS:
        batch_emb, batch_texts = generate_llm_batch(
            provider, class_name, class_texts, BATCH_SIZE,
            embed_model, all_classes
        )
        llm_calls += 1

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        if not pool_embeddings:
            continue

        pool_arr = np.vstack(pool_embeddings)

        # Check if we have enough for filtering (oversample factor)
        if len(pool_arr) >= target_n * OVERSAMPLE_FACTOR:
            break

    if not pool_embeddings:
        return {
            "embeddings": np.array([]).reshape(0, 768),
            "texts": [],
            "llm_calls": llm_calls,
            "acceptance_rate": 0.0
        }

    pool_arr = np.vstack(pool_embeddings)

    # Apply cascade filter
    class_mask = real_labels == class_name
    if not class_mask.any():
        return {
            "embeddings": pool_arr[:target_n],
            "texts": pool_texts[:target_n],
            "llm_calls": llm_calls,
            "acceptance_rate": 1.0
        }

    filtered_emb, avg_quality, details = filter_obj.filter_samples(
        candidates=pool_arr,
        real_embeddings=real_embeddings,
        real_labels=real_labels,
        target_class=class_name,
        target_count=target_n
    )

    acceptance_rate = len(filtered_emb) / len(pool_arr) if len(pool_arr) > 0 else 0

    return {
        "embeddings": filtered_emb[:target_n],
        "texts": [],  # We don't need texts after filtering (only embeddings matter)
        "llm_calls": llm_calls,
        "acceptance_rate": acceptance_rate
    }


# ============================================================================
# SMOTE
# ============================================================================

def apply_smote_global(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE globally to balance all classes to majority class size."""
    counts = Counter(labels)
    max_count = max(counts.values())

    # Build sampling strategy: bring all classes up to majority size
    sampling_strategy = {}
    for cls, cnt in counts.items():
        if cnt < max_count:
            sampling_strategy[cls] = max_count

    if not sampling_strategy:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])

    # Use safe k_neighbors
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
        X_resampled, y_resampled = smote.fit_resample(embeddings, labels)

        # Extract only new samples
        new_emb = X_resampled[len(embeddings):]
        new_labels = y_resampled[len(embeddings):]
        return new_emb, new_labels

    except Exception as e:
        print(f"    SMOTE error: {e}")
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    synth_emb: np.ndarray,
    synth_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray
) -> Dict:
    """Train classifier and evaluate. Returns macro_f1, weighted_f1, per_class_f1."""
    if len(synth_emb) > 0:
        aug_emb = np.vstack([train_emb, synth_emb])
        aug_labels = np.concatenate([train_labels, synth_labels])
    else:
        aug_emb = train_emb
        aug_labels = train_labels

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)

    preds = clf.predict(test_emb)
    all_classes = sorted(set(test_labels))

    macro_f1 = f1_score(test_labels, preds, average='macro')
    weighted_f1 = f1_score(test_labels, preds, average='weighted')
    per_class = f1_score(test_labels, preds, average=None, labels=all_classes)
    per_class_f1 = {cls: float(f) for cls, f in zip(all_classes, per_class)}

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1
    }


# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

def run_baseline(train_emb, train_labels, test_emb, test_labels, seed) -> Dict:
    """Condition: No augmentation."""
    empty = np.array([]).reshape(0, train_emb.shape[1])
    return evaluate(train_emb, train_labels, empty, np.array([]), test_emb, test_labels)


def run_smote_only(train_emb, train_labels, test_emb, test_labels, seed) -> Dict:
    """Condition: SMOTE-only (standard approach)."""
    smote_emb, smote_labels = apply_smote_global(train_emb, train_labels, seed)

    result = evaluate(train_emb, train_labels, smote_emb, smote_labels, test_emb, test_labels)
    result["total_smote_samples"] = len(smote_emb)
    return result


def run_selective_llm(
    provider,
    filter_obj: FilterCascade,
    embed_model: SentenceTransformer,
    train_texts: List[str],
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    size_threshold: int,
    target_size: int,
    seed: int,
    apply_smote: bool = True
) -> Dict:
    """
    Condition: Selective LLM augmentation (with or without SMOTE).

    1. Identify small classes (below threshold)
    2. For each small class, generate LLM samples to reach target_size
    3. Optionally apply SMOTE globally on (real + LLM-augmented)
    """
    all_classes = sorted(set(train_labels))
    small_classes, large_classes = identify_small_classes(train_labels, size_threshold)
    counts = Counter(train_labels)

    print(f"    Small classes (< {size_threshold}): {small_classes}")

    # Step 1: Generate LLM samples for small classes only
    all_llm_emb = []
    all_llm_labels = []
    total_llm_calls = 0
    total_llm_samples = 0
    acceptance_rates = {}

    for cls in small_classes:
        cls_count = counts[cls]
        n_needed = max(0, target_size - cls_count)

        if n_needed == 0:
            print(f"      {cls}: {cls_count} samples, already >= target {target_size}, skipping")
            continue

        cls_texts = [t for t, l in zip(train_texts, list(train_labels)) if l == cls]

        print(f"      {cls}: {cls_count} → {target_size} (need {n_needed} LLM samples)...")

        result = generate_llm_for_class(
            provider=provider,
            filter_obj=filter_obj,
            class_name=cls,
            class_texts=cls_texts,
            target_n=n_needed,
            real_embeddings=train_emb,
            real_labels=train_labels,
            embed_model=embed_model,
            all_classes=all_classes
        )

        llm_emb = result["embeddings"]
        total_llm_calls += result["llm_calls"]
        total_llm_samples += len(llm_emb)
        acceptance_rates[cls] = result["acceptance_rate"]

        if len(llm_emb) > 0:
            all_llm_emb.append(llm_emb)
            all_llm_labels.extend([cls] * len(llm_emb))
            print(f"        Generated {len(llm_emb)} samples ({result['llm_calls']} calls, "
                  f"accept: {result['acceptance_rate']*100:.1f}%)")
        else:
            print(f"        WARNING: No samples generated for {cls}")

    # Step 2: Merge LLM samples into training data
    if all_llm_emb:
        llm_emb_combined = np.vstack(all_llm_emb)
        llm_labels_combined = np.array(all_llm_labels)
        augmented_emb = np.vstack([train_emb, llm_emb_combined])
        augmented_labels = np.concatenate([train_labels, llm_labels_combined])
    else:
        augmented_emb = train_emb
        augmented_labels = train_labels
        llm_emb_combined = np.array([]).reshape(0, train_emb.shape[1])
        llm_labels_combined = np.array([])

    # Step 3: Optionally apply SMOTE globally
    total_smote = 0
    if apply_smote:
        smote_emb, smote_labels = apply_smote_global(augmented_emb, augmented_labels, seed)
        total_smote = len(smote_emb)

        # Evaluate with all synthetic data (LLM + SMOTE)
        all_synth_emb = []
        all_synth_labels = []
        if len(llm_emb_combined) > 0:
            all_synth_emb.append(llm_emb_combined)
            all_synth_labels.extend(list(llm_labels_combined))
        if len(smote_emb) > 0:
            all_synth_emb.append(smote_emb)
            all_synth_labels.extend(list(smote_labels))

        if all_synth_emb:
            synth_emb = np.vstack(all_synth_emb)
            synth_labels = np.array(all_synth_labels)
        else:
            synth_emb = np.array([]).reshape(0, train_emb.shape[1])
            synth_labels = np.array([])
    else:
        # LLM only, no SMOTE
        synth_emb = llm_emb_combined
        synth_labels = llm_labels_combined

    eval_result = evaluate(train_emb, train_labels, synth_emb, synth_labels, test_emb, test_labels)
    eval_result["total_llm_calls"] = total_llm_calls
    eval_result["total_llm_samples"] = total_llm_samples
    eval_result["total_smote_samples"] = total_smote
    eval_result["acceptance_rates"] = acceptance_rates
    eval_result["classes_augmented"] = small_classes

    return eval_result


# ============================================================================
# SUMMARY & RESULTS
# ============================================================================

def compute_class_group_f1(
    per_class_f1: Dict[str, float],
    small_classes: List[str],
    large_classes: List[str]
) -> Tuple[float, float]:
    """Compute average F1 for small and large class groups."""
    small_f1s = [per_class_f1.get(c, 0.0) for c in small_classes]
    large_f1s = [per_class_f1.get(c, 0.0) for c in large_classes]

    small_avg = np.mean(small_f1s) if small_f1s else 0.0
    large_avg = np.mean(large_f1s) if large_f1s else 0.0
    return float(small_avg), float(large_avg)


def print_summary(all_results: List[Dict], smote_f1_by_seed: Dict[int, Dict]):
    """Print final summary table."""
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    # Group results by method
    methods = {}
    for r in all_results:
        method_key = r["method"]
        if r.get("size_threshold"):
            method_key += f"_t{r['size_threshold']}_s{r['target_size']}"

        if method_key not in methods:
            methods[method_key] = []
        methods[method_key].append(r)

    print(f"\n{'Method':<35s} {'Macro F1':>10s} {'Δ Baseline':>12s} {'Small F1':>10s} {'Large F1':>10s} {'LLM Calls':>10s}")
    print("-" * 90)

    for method_key, results in sorted(methods.items()):
        avg_macro = np.mean([r["macro_f1"] for r in results])
        avg_baseline = np.mean([r["baseline_f1"] for r in results])
        avg_delta = (avg_macro - avg_baseline) * 100
        avg_small = np.mean([r["small_class_avg_f1"] for r in results])
        avg_large = np.mean([r["large_class_avg_f1"] for r in results])
        avg_calls = np.mean([r.get("total_llm_calls", 0) for r in results])

        print(f"  {method_key:<33s} {avg_macro:>10.4f} {avg_delta:>+11.2f}pp {avg_small:>10.4f} {avg_large:>10.4f} {avg_calls:>10.1f}")

    # Per-class breakdown for the best selective method
    print("\n" + "-" * 90)
    print("PER-CLASS F1 BREAKDOWN (averaged across seeds)")
    print("-" * 90)

    # Find all unique per-class results
    baseline_pclass = {}
    smote_pclass = {}
    selective_pclass = {}

    for r in all_results:
        pcf = r["per_class_f1"]
        if r["method"] == "baseline":
            for c, f in pcf.items():
                baseline_pclass.setdefault(c, []).append(f)
        elif r["method"] == "smote_only":
            for c, f in pcf.items():
                smote_pclass.setdefault(c, []).append(f)
        elif r["method"] == "llm_selective_smote":
            for c, f in pcf.items():
                selective_pclass.setdefault(c, []).append(f)

    if baseline_pclass and smote_pclass:
        all_classes = sorted(baseline_pclass.keys())
        counts_str = {c: "" for c in all_classes}

        print(f"\n  {'Class':<8s} {'Baseline':>10s} {'SMOTE':>10s} {'Best Selective':>15s} {'Δ vs SMOTE':>12s}")
        print("  " + "-" * 58)

        for cls in all_classes:
            b = np.mean(baseline_pclass.get(cls, [0]))
            s = np.mean(smote_pclass.get(cls, [0]))
            sel_values = selective_pclass.get(cls, [])
            sel = np.mean(sel_values) if sel_values else s
            delta = (sel - s) * 100

            # Mark classes that were LLM-augmented
            augmented_sets = [r["classes_augmented"] for r in all_results if r.get("classes_augmented")]
            marker = " *" if augmented_sets and cls in augmented_sets[0] else ""
            print(f"  {cls:<8s} {b:>10.4f} {s:>10.4f} {sel:>15.4f} {delta:>+11.2f}pp{marker}")


def save_results(all_results: List[Dict], config: Dict):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "experiment": "selective_llm_mbti",
        "timestamp": timestamp,
        "config": config,
        "results": all_results
    }

    # Save timestamped
    path = RESULTS_DIR / f"selective_llm_mbti_{timestamp}.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save latest
    latest_path = RESULTS_DIR / "latest_summary.json"
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("SELECTIVE LLM AUGMENTATION FOR MBTI CLASSIFICATION")
    print("=" * 70)

    # 1. Load MBTI data
    print(f"\nLoading data from {DATA_PATH}...")
    texts, labels = load_mbti_data(DATA_PATH)
    print(f"  Total samples: {len(texts)}, Classes: {len(set(labels))}")
    print_class_distribution(labels, "Full Dataset")

    # 2. Initialize models
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    filter_obj = FilterCascade(**LLM_FILTER)

    all_results = []
    smote_f1_by_seed = {}

    config = {
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "filter": LLM_FILTER,
        "prompt_technique": "negative",
        "n_shot": N_SHOT,
        "batch_size": BATCH_SIZE,
        "oversample_factor": OVERSAMPLE_FACTOR,
        "size_thresholds": SIZE_THRESHOLDS,
        "target_sizes": TARGET_SIZES,
        "seeds": SEEDS,
        "test_size": TEST_SIZE,
    }

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*70}")

        # 3. Stratified train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=seed, stratify=labels
        )
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        print_class_distribution(train_labels, "Train Split")

        # 4. Compute embeddings
        print("\n  Computing embeddings...")
        train_emb = embed_model.encode(train_texts, show_progress_bar=True, batch_size=256)
        test_emb = embed_model.encode(test_texts, show_progress_bar=True, batch_size=256)

        # 5. Baseline (no augmentation)
        print("\n  [1/4] Running BASELINE (no augmentation)...")
        baseline_result = run_baseline(train_emb, train_labels, test_emb, test_labels, seed)
        baseline_f1 = baseline_result["macro_f1"]
        print(f"    Macro F1: {baseline_f1:.4f}")

        # Compute small/large F1 for reference (using threshold=100)
        small_cls_100, large_cls_100 = identify_small_classes(train_labels, 100)
        small_avg, large_avg = compute_class_group_f1(baseline_result["per_class_f1"], small_cls_100, large_cls_100)

        all_results.append({
            "method": "baseline",
            "size_threshold": None,
            "target_size": None,
            "seed": seed,
            "macro_f1": baseline_f1,
            "weighted_f1": baseline_result["weighted_f1"],
            "baseline_f1": baseline_f1,
            "delta_vs_baseline": 0.0,
            "per_class_f1": baseline_result["per_class_f1"],
            "small_class_avg_f1": small_avg,
            "large_class_avg_f1": large_avg,
            "classes_augmented": [],
            "total_llm_calls": 0,
            "total_llm_samples": 0,
            "total_smote_samples": 0,
        })

        # 6. SMOTE only
        print("\n  [2/4] Running SMOTE ONLY...")
        smote_result = run_smote_only(train_emb, train_labels, test_emb, test_labels, seed)
        smote_f1 = smote_result["macro_f1"]
        smote_delta = (smote_f1 - baseline_f1) * 100
        print(f"    Macro F1: {smote_f1:.4f} ({smote_delta:+.2f}pp vs baseline)")

        small_avg, large_avg = compute_class_group_f1(smote_result["per_class_f1"], small_cls_100, large_cls_100)
        smote_f1_by_seed[seed] = smote_result

        all_results.append({
            "method": "smote_only",
            "size_threshold": None,
            "target_size": None,
            "seed": seed,
            "macro_f1": smote_f1,
            "weighted_f1": smote_result["weighted_f1"],
            "baseline_f1": baseline_f1,
            "delta_vs_baseline": smote_delta,
            "per_class_f1": smote_result["per_class_f1"],
            "small_class_avg_f1": small_avg,
            "large_class_avg_f1": large_avg,
            "classes_augmented": [],
            "total_llm_calls": 0,
            "total_llm_samples": 0,
            "total_smote_samples": smote_result.get("total_smote_samples", 0),
        })

        # 7. Selective LLM + SMOTE (vary threshold and target)
        run_idx = 0
        total_selective_runs = len(SIZE_THRESHOLDS) * len(TARGET_SIZES) * 2  # x2 for with/without SMOTE

        for threshold in SIZE_THRESHOLDS:
            for target_size in TARGET_SIZES:
                small_classes, large_classes = identify_small_classes(train_labels, threshold)

                for apply_smote in [True, False]:
                    run_idx += 1
                    method = "llm_selective_smote" if apply_smote else "llm_selective_only"
                    label = f"{method} t={threshold} s={target_size}"
                    print(f"\n  [3-4/{total_selective_runs}] Running {label} (run {run_idx}/{total_selective_runs})...")

                    sel_result = run_selective_llm(
                        provider=provider,
                        filter_obj=filter_obj,
                        embed_model=embed_model,
                        train_texts=train_texts,
                        train_emb=train_emb,
                        train_labels=train_labels,
                        test_emb=test_emb,
                        test_labels=test_labels,
                        size_threshold=threshold,
                        target_size=target_size,
                        seed=seed,
                        apply_smote=apply_smote
                    )

                    sel_f1 = sel_result["macro_f1"]
                    sel_delta = (sel_f1 - baseline_f1) * 100
                    smote_delta_pp = (sel_f1 - smote_f1) * 100

                    small_avg, large_avg = compute_class_group_f1(
                        sel_result["per_class_f1"], small_classes, large_classes
                    )

                    print(f"    Macro F1: {sel_f1:.4f} ({sel_delta:+.2f}pp vs baseline, "
                          f"{smote_delta_pp:+.2f}pp vs SMOTE)")
                    print(f"    Small-class avg F1: {small_avg:.4f}, Large-class avg F1: {large_avg:.4f}")
                    print(f"    LLM calls: {sel_result['total_llm_calls']}, "
                          f"LLM samples: {sel_result['total_llm_samples']}, "
                          f"SMOTE samples: {sel_result['total_smote_samples']}")

                    all_results.append({
                        "method": method,
                        "size_threshold": threshold,
                        "target_size": target_size,
                        "seed": seed,
                        "macro_f1": sel_f1,
                        "weighted_f1": sel_result["weighted_f1"],
                        "baseline_f1": baseline_f1,
                        "delta_vs_baseline": sel_delta,
                        "per_class_f1": sel_result["per_class_f1"],
                        "small_class_avg_f1": small_avg,
                        "large_class_avg_f1": large_avg,
                        "classes_augmented": sel_result["classes_augmented"],
                        "total_llm_calls": sel_result["total_llm_calls"],
                        "total_llm_samples": sel_result["total_llm_samples"],
                        "total_smote_samples": sel_result["total_smote_samples"],
                        "acceptance_rates": sel_result.get("acceptance_rates", {}),
                    })

        # Save incrementally after each seed
        save_results(all_results, config)

    # Final summary
    elapsed = (time.time() - start_time) / 60
    print_summary(all_results, smote_f1_by_seed)

    print(f"\nTotal time: {elapsed:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
