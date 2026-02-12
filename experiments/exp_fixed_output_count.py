#!/usr/bin/env python3
"""
Fixed Output Count Experiment

Compares filters with the SAME number of valid synthetic samples.
Stricter filters need more LLM calls to reach the target count.

This isolates the effect of filter QUALITY from sample QUANTITY.

Key metrics:
- F1 score with exactly N samples per class (N fixed for all filters)
- LLM calls needed to reach N valid samples
- Acceptance rate per filter
- Efficiency: F1 improvement / LLM calls
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter, CombinedGeometricFilter
from core.filter_cascade import FilterCascade
from core.embedding_guided_sampler import EmbeddingGuidedSampler

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "fixed_output_count"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SAMPLES_PER_CLASS = 50
MAX_LLM_CALLS_PER_CLASS = 300
BATCH_SIZE = 25
N_SHOT = 25
EARLY_STOP_ACCEPTANCE = 0.02  # 2%

# Filters to compare
FILTERS = [
    # Control - no geometric filter
    {"name": "none", "type": "none", "params": {}},

    # LOF variants
    {"name": "lof_relaxed", "type": "lof", "params": {"n_neighbors": 10, "threshold": -0.5}},
    {"name": "lof_strict", "type": "lof", "params": {"n_neighbors": 10, "threshold": 0.0}},

    # Cascade levels
    {"name": "cascade_l1", "type": "cascade", "params": {"filter_level": 1, "k_neighbors": 10}},
    {"name": "cascade_l2", "type": "cascade", "params": {"filter_level": 2, "k_neighbors": 10}},
    {"name": "cascade_full", "type": "cascade", "params": {"filter_level": 4, "k_neighbors": 10}},

    # Combined (very restrictive)
    {"name": "combined", "type": "combined", "params": {"lof_threshold": 0.0, "sim_threshold": 0.5}},
]

# Datasets to test
DATASETS = [
    "20newsgroups_10shot",
    "20newsgroups_25shot",
    "sms_spam_10shot",
    "sms_spam_25shot",
    "hate_speech_davidson_10shot",
    "hate_speech_davidson_25shot",
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GenerationResult:
    """Result of generating samples for one class."""
    valid_embeddings: np.ndarray
    valid_texts: List[str]
    llm_calls: int
    total_generated: int
    acceptance_rate: float
    status: str  # SUCCESS, MAX_CALLS_REACHED, LOW_ACCEPTANCE


@dataclass
class FilterExperimentResult:
    """Result for one filter on one dataset."""
    filter_name: str
    filter_type: str
    filter_params: Dict
    f1_score: float
    f1_delta_vs_baseline: float
    total_llm_calls: int
    avg_acceptance_rate: float
    efficiency_score: float  # F1_delta / llm_calls * 1000
    per_class_stats: Dict[str, Dict]
    all_reached_target: bool


@dataclass
class DatasetResult:
    """Results for one dataset."""
    dataset: str
    n_classes: int
    baseline_f1: float
    filter_results: List[FilterExperimentResult]
    timestamp: str


# ============================================================================
# FILTER FACTORY
# ============================================================================

def create_filter(filter_config: Dict):
    """Create a filter instance from config."""
    filter_type = filter_config["type"]
    params = filter_config["params"]

    if filter_type == "none":
        return None
    elif filter_type == "lof":
        return LOFFilter(**params)
    elif filter_type == "combined":
        return CombinedGeometricFilter(**params)
    elif filter_type == "cascade":
        return FilterCascade(**params)
    elif filter_type == "embedding_guided":
        return EmbeddingGuidedSampler(**params)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


# ============================================================================
# DATASET-SPECIFIC PROMPT CONFIGURATION
# ============================================================================

DATASET_PROMPTS = {
    "20newsgroups": {
        "domain": "newsgroup posts and online forum discussions",
        "task_context": "text classification of newsgroup messages into topic categories",
        "text_type": "newsgroup posts or forum messages",
        "style_notes": "Use informal but articulate writing typical of online discussions. "
                       "Include opinions, questions, or information sharing as seen in forums.",
        "length_guidance": "Each post should be 2-5 sentences, similar to typical forum replies.",
        "class_descriptions": {
            "sci.space": "discussions about space exploration, astronomy, NASA, rockets, and space science",
            "rec.sport.baseball": "discussions about baseball games, teams, players, statistics, and MLB",
            "comp.graphics": "discussions about computer graphics, rendering, 3D modeling, and image processing",
            "talk.politics.misc": "discussions about political topics, government, policies, and current events",
        }
    },
    "sms_spam": {
        "domain": "SMS text messages",
        "task_context": "spam detection in mobile phone text messages",
        "text_type": "SMS text messages",
        "style_notes": "Use casual, abbreviated language typical of text messages. "
                       "Spam messages often contain urgency, offers, or requests for action. "
                       "Ham (non-spam) messages are normal personal communications.",
        "length_guidance": "Each message should be 1-3 sentences, typical SMS length (under 160 characters preferred).",
        "class_descriptions": {
            "spam": "unsolicited promotional messages, scams, phishing attempts, or commercial offers",
            "ham": "legitimate personal messages between friends, family, or acquaintances",
        }
    },
    "hate_speech_davidson": {
        "domain": "social media posts (Twitter)",
        "task_context": "hate speech and offensive language detection on social media",
        "text_type": "tweets or social media posts",
        "style_notes": "Use social media language patterns including hashtags and mentions where appropriate. "
                       "Maintain the linguistic patterns of each category without being gratuitously offensive.",
        "length_guidance": "Each post should be 1-2 sentences, typical tweet length.",
        "class_descriptions": {
            "hate_speech": "content expressing hatred toward a group based on race, religion, ethnicity, gender, or other characteristics",
            "offensive_language": "content containing profanity, insults, or crude language but not targeting specific groups",
            "neither": "neutral or benign social media content without offensive elements",
        }
    },
    "ag_news": {
        "domain": "news article headlines and descriptions",
        "task_context": "news categorization by topic",
        "text_type": "news headlines with brief descriptions",
        "style_notes": "Use journalistic, objective language typical of news agencies. "
                       "Include factual information, names, places, and events.",
        "length_guidance": "Each item should be 1-2 sentences in news headline style.",
        "class_descriptions": {
            "World": "international news, foreign affairs, global events, and diplomacy",
            "Sports": "sports news, games, athletes, tournaments, and athletic events",
            "Business": "business news, markets, companies, economics, and finance",
            "Sci/Tech": "science and technology news, innovations, research, and tech companies",
        }
    }
}


def get_dataset_base_name(dataset_name: str) -> str:
    """Extract base dataset name (without shot suffix)."""
    for base in DATASET_PROMPTS.keys():
        if dataset_name.startswith(base):
            return base
    return dataset_name


def create_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str
) -> str:
    """
    Create a robust generation prompt using prompt engineering best practices.

    Techniques used (based on Lee Boonstra's Prompt Engineering guide):
    1. Role definition - Clear expert persona
    2. Context setting - Domain-specific background
    3. Task specification - Explicit instructions
    4. Format constraints - Output structure
    5. Negative prompting - What to avoid
    6. Few-shot examples - Real samples
    7. Output framing - Clear delimiters
    """
    base_dataset = get_dataset_base_name(dataset_name)
    config = DATASET_PROMPTS.get(base_dataset, {})

    # Get dataset-specific information
    domain = config.get("domain", "text documents")
    task_context = config.get("task_context", "text classification")
    text_type = config.get("text_type", "text samples")
    style_notes = config.get("style_notes", "Match the style of the provided examples.")
    length_guidance = config.get("length_guidance", "Match the length of the provided examples.")
    class_descriptions = config.get("class_descriptions", {})

    # Get class-specific description if available
    class_desc = class_descriptions.get(class_name, f"content belonging to the '{class_name}' category")

    # Select and format examples
    selected_examples = examples[:N_SHOT]

    # Calculate average length for guidance
    avg_length = sum(len(ex.split()) for ex in selected_examples) / len(selected_examples)

    examples_text = "\n---\n".join([
        f"[Example {i+1}]\n{ex[:600]}"
        for i, ex in enumerate(selected_examples)
    ])

    prompt = f"""# ROLE
You are a specialist data augmentation system for {task_context}. Your expertise is generating realistic {text_type} that are indistinguishable from authentic human-written content.

# CONTEXT
- Domain: {domain}
- Target class: "{class_name}"
- Class definition: {class_desc}
- Purpose: Generate training data for a machine learning classifier

# REFERENCE EXAMPLES
The following are {len(selected_examples)} authentic examples from the "{class_name}" class. Study their style, vocabulary, tone, and structure carefully:

{examples_text}

# TASK
Generate exactly {n_generate} NEW and UNIQUE {text_type} that belong to the "{class_name}" class.

# REQUIREMENTS
1. **Authenticity**: Each generated text must be realistic and could plausibly be written by a human
2. **Style matching**: {style_notes}
3. **Length**: {length_guidance} (reference examples average ~{int(avg_length)} words)
4. **Diversity**: Each generated example should be distinct - vary topics, perspectives, and vocabulary
5. **Class consistency**: All examples must clearly belong to "{class_name}" category

# CONSTRAINTS - DO NOT:
- Use emojis or emoticons (unless present in reference examples)
- Generate numbered lists or bullet points
- Include meta-commentary like "Here's an example" or "This is a sample"
- Copy or closely paraphrase the reference examples
- Generate text significantly shorter or longer than the reference examples
- Include any prefixes, labels, or formatting markers
- Generate placeholder text or templates

# OUTPUT FORMAT
- Generate exactly {n_generate} {text_type}, one per line
- Do not number them or add prefixes
- Each line should contain only the generated text
- Generate all {n_generate} examples now:"""

    return prompt


# ============================================================================
# LLM GENERATION
# ============================================================================

def generate_llm_batch(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    model: SentenceTransformer,
    dataset_name: str = "unknown"
) -> Tuple[np.ndarray, List[str]]:
    """Generate a batch of samples using LLM with dataset-specific prompts."""
    prompt = create_prompt(class_name, class_texts, n_generate, dataset_name)

    try:
        messages = [{"role": "user", "content": prompt}]
        response, _ = provider.generate(messages, temperature=0.8, max_tokens=4000)

        # Parse response
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        generated = []
        for line in lines:
            clean = line.lstrip('0123456789.-):* ')
            if len(clean) > 10:
                generated.append(clean)

        if not generated:
            return np.array([]).reshape(0, 768), []

        # Embed
        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated

    except Exception as e:
        print(f"        Error generating: {e}")
        return np.array([]).reshape(0, 768), []


# ============================================================================
# FILTER APPLICATION (adapted from exp_filter_comparison.py)
# ============================================================================

def apply_filter(
    filter_obj,
    filter_type: str,
    candidate_embeddings: np.ndarray,
    candidate_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_class: str,
    target_n: int
) -> Tuple[np.ndarray, List[str], Dict]:
    """Apply filter and return filtered samples."""
    if len(candidate_embeddings) == 0:
        return np.array([]).reshape(0, 768), [], {"n_candidates": 0, "n_selected": 0}

    # No filter - return all (will be truncated later)
    if filter_obj is None or filter_type == "none":
        return candidate_embeddings, candidate_texts, {
            "n_candidates": len(candidate_embeddings),
            "n_selected": len(candidate_embeddings),
            "method": "none",
            "pct_accepted": 100.0
        }

    # LOF Filter
    if filter_type == "lof":
        filtered_emb, mask, scores = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )
        passed_indices = np.where(mask)[0]
        return (
            filtered_emb,
            [candidate_texts[i] for i in passed_indices],
            {
                "n_candidates": len(candidate_embeddings),
                "n_passed_filter": int(mask.sum()),
                "pct_accepted": 100 * mask.sum() / len(mask),
                "mean_lof_score": float(scores.mean()) if len(scores) > 0 else 0.0,
            }
        )

    # Combined Geometric Filter
    if filter_type == "combined":
        filtered_emb, mask, stats = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )
        passed_indices = np.where(mask)[0]
        return (
            filtered_emb,
            [candidate_texts[i] for i in passed_indices],
            stats
        )

    # Filter Cascade
    if filter_type == "cascade":
        filtered_emb, avg_quality, details = filter_obj.filter_samples(
            candidates=candidate_embeddings,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=target_class,
            target_count=target_n
        )

        # Get indices by score
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]
        anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)

        scores, _ = filter_obj.compute_quality_scores(
            candidate_embeddings, anchor, real_embeddings, real_labels, target_class
        )
        top_idx = np.argsort(scores)[-len(filtered_emb):]

        return (
            filtered_emb,
            [candidate_texts[i] for i in top_idx],
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(filtered_emb),
                "avg_quality": avg_quality,
                "pct_accepted": 100 * len(filtered_emb) / max(1, len(candidate_embeddings)),
            }
        )

    # Embedding Guided Sampler
    if filter_type == "embedding_guided":
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]

        selected_texts, selected_embs, scores = filter_obj.select_samples(
            candidate_embeddings,
            candidate_texts,
            class_embs,
            target_n,
            class_label=target_class
        )

        return (
            selected_embs if len(selected_embs) > 0 else np.array([]).reshape(0, candidate_embeddings.shape[1]),
            selected_texts,
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(selected_texts),
                "pct_accepted": 100 * len(selected_texts) / max(1, len(candidate_embeddings)),
            }
        )

    raise ValueError(f"Unknown filter type: {filter_type}")


# ============================================================================
# CORE: GENERATE UNTIL N VALID
# ============================================================================

def generate_for_none_filter(
    provider,
    class_name: str,
    class_texts: List[str],
    target_n: int,
    model: SentenceTransformer,
    dataset_name: str
) -> GenerationResult:
    """
    For 'none' filter: generate target_n * 1.5, select random.
    Simulates "generate more without geometric filtering".
    """
    oversample = 1.5
    n_generate = int(target_n * oversample)

    gen_emb, gen_texts = generate_llm_batch(
        provider, class_name, class_texts, n_generate, model, dataset_name
    )

    # Random selection to target_n
    if len(gen_emb) > target_n:
        indices = np.random.choice(len(gen_emb), target_n, replace=False)
        return GenerationResult(
            valid_embeddings=gen_emb[indices],
            valid_texts=[gen_texts[i] for i in indices],
            llm_calls=1,
            total_generated=len(gen_emb),
            acceptance_rate=1.0,
            status="SUCCESS"
        )

    return GenerationResult(
        valid_embeddings=gen_emb,
        valid_texts=gen_texts,
        llm_calls=1,
        total_generated=len(gen_emb),
        acceptance_rate=1.0,
        status="SUCCESS" if len(gen_emb) >= target_n else "INSUFFICIENT"
    )


def generate_until_n_valid(
    provider,
    filter_obj,
    filter_type: str,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    model: SentenceTransformer,
    dataset_name: str
) -> GenerationResult:
    """
    Generate iteratively until we have exactly target_n valid samples.
    Uses pool-based approach for efficiency.
    """
    # Special case: no filter
    if filter_type == "none" or filter_obj is None:
        return generate_for_none_filter(
            provider, class_name, class_texts, target_n, model, dataset_name
        )

    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while True:
        # Generate batch
        batch_emb, batch_texts = generate_llm_batch(
            provider, class_name, class_texts, BATCH_SIZE, model, dataset_name
        )
        llm_calls += 1

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        # Check if we have anything in pool
        if not pool_embeddings:
            if llm_calls >= MAX_LLM_CALLS_PER_CLASS:
                return GenerationResult(
                    valid_embeddings=np.array([]).reshape(0, 768),
                    valid_texts=[],
                    llm_calls=llm_calls,
                    total_generated=0,
                    acceptance_rate=0.0,
                    status="MAX_CALLS_REACHED"
                )
            continue

        # Combine pool
        pool_arr = np.vstack(pool_embeddings)

        # Apply filter to entire pool
        filtered_emb, filtered_texts, stats = apply_filter(
            filter_obj, filter_type,
            pool_arr, pool_texts,
            real_embeddings, real_labels,
            class_name, target_n
        )

        n_valid = len(filtered_emb)
        acceptance_rate = n_valid / len(pool_arr) if len(pool_arr) > 0 else 0

        # Check success
        if n_valid >= target_n:
            # Truncate to exactly target_n
            return GenerationResult(
                valid_embeddings=filtered_emb[:target_n],
                valid_texts=filtered_texts[:target_n],
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="SUCCESS"
            )

        # Check max calls
        if llm_calls >= MAX_LLM_CALLS_PER_CLASS:
            return GenerationResult(
                valid_embeddings=filtered_emb,
                valid_texts=filtered_texts,
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="MAX_CALLS_REACHED"
            )

        # Check low acceptance (early stop)
        if len(pool_arr) > 100 and acceptance_rate < EARLY_STOP_ACCEPTANCE:
            return GenerationResult(
                valid_embeddings=filtered_emb,
                valid_texts=filtered_texts,
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="LOW_ACCEPTANCE"
            )

        # Progress indicator
        if llm_calls % 5 == 0:
            print(f"          ... {llm_calls} calls, {n_valid}/{target_n} valid ({acceptance_rate*100:.1f}%)")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_augmented(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    synthetic_embeddings: np.ndarray,
    synthetic_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str]
) -> float:
    """Train classifier with augmented data and return F1."""
    # Combine original + synthetic
    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = list(train_labels) + list(synthetic_labels)

    # Train and evaluate
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_embeddings, aug_labels)
    pred = clf.predict(test_embeddings)

    return f1_score(test_labels, pred, average='macro')


def compute_baseline_f1(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str]
) -> float:
    """Compute baseline F1 without augmentation."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, train_labels)
    pred = clf.predict(test_embeddings)
    return f1_score(test_labels, pred, average='macro')


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_for_filter(
    filter_config: Dict,
    train_texts: List[str],
    train_labels: List[str],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: List[str],
    baseline_f1: float,
    provider,
    model: SentenceTransformer,
    dataset_name: str
) -> FilterExperimentResult:
    """Run experiment for one filter."""
    filter_name = filter_config["name"]
    filter_type = filter_config["type"]
    filter_params = filter_config["params"]

    print(f"    Filter: {filter_name}")

    # Create filter
    filter_obj = create_filter(filter_config)

    # Generate for each class
    unique_classes = list(set(train_labels))
    train_labels_arr = np.array(train_labels)

    all_synthetic_emb = []
    all_synthetic_labels = []
    total_llm_calls = 0
    per_class_stats = {}
    all_reached_target = True

    for cls in unique_classes:
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        print(f"      Class: {cls} ({len(cls_texts)} real samples)")

        result = generate_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            filter_type=filter_type,
            target_n=TARGET_SAMPLES_PER_CLASS,
            class_name=cls,
            class_texts=cls_texts,
            real_embeddings=train_embeddings,
            real_labels=train_labels_arr,
            model=model,
            dataset_name=dataset_name
        )

        print(f"        -> {len(result.valid_embeddings)}/{TARGET_SAMPLES_PER_CLASS} valid, "
              f"{result.llm_calls} calls, {result.acceptance_rate*100:.1f}% accept, "
              f"status: {result.status}")

        if len(result.valid_embeddings) > 0:
            all_synthetic_emb.append(result.valid_embeddings)
            all_synthetic_labels.extend([cls] * len(result.valid_embeddings))

        total_llm_calls += result.llm_calls

        per_class_stats[cls] = {
            "n_valid": len(result.valid_embeddings),
            "llm_calls": result.llm_calls,
            "total_generated": result.total_generated,
            "acceptance_rate": result.acceptance_rate,
            "status": result.status
        }

        if result.status != "SUCCESS":
            all_reached_target = False

    # Combine synthetic data
    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])

    # Evaluate
    if len(synthetic_embeddings) > 0:
        f1 = evaluate_augmented(
            train_embeddings, train_labels,
            synthetic_embeddings, all_synthetic_labels,
            test_embeddings, test_labels
        )
    else:
        f1 = baseline_f1

    f1_delta = (f1 - baseline_f1) * 100  # percentage points
    avg_acceptance = np.mean([s["acceptance_rate"] for s in per_class_stats.values()])

    # Efficiency: F1 improvement per LLM call (scaled by 1000)
    efficiency = (f1_delta / max(1, total_llm_calls)) * 1000

    print(f"      => F1: {f1:.4f} ({f1_delta:+.2f}pp), "
          f"Total calls: {total_llm_calls}, Efficiency: {efficiency:.2f}")

    return FilterExperimentResult(
        filter_name=filter_name,
        filter_type=filter_type,
        filter_params=filter_params,
        f1_score=float(f1),
        f1_delta_vs_baseline=float(f1_delta),
        total_llm_calls=total_llm_calls,
        avg_acceptance_rate=float(avg_acceptance),
        efficiency_score=float(efficiency),
        per_class_stats=per_class_stats,
        all_reached_target=all_reached_target
    )


def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a benchmark dataset."""
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return (
        data['train_texts'],
        data['train_labels'],
        data['test_texts'],
        data['test_labels']
    )


def run_dataset_experiment(
    dataset_name: str,
    provider,
    model: SentenceTransformer
) -> DatasetResult:
    """Run all filter experiments on one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"  Classes: {set(train_labels)}")
    print(f"  Distribution: {dict(Counter(train_labels))}")

    # Embed
    print("  Embedding texts...")
    train_embeddings = model.encode(train_texts, show_progress_bar=False)
    test_embeddings = model.encode(test_texts, show_progress_bar=False)

    # Baseline
    baseline_f1 = compute_baseline_f1(
        train_embeddings, train_labels, test_embeddings, test_labels
    )
    print(f"  Baseline F1: {baseline_f1:.4f}")

    # Run each filter
    filter_results = []
    for filter_config in FILTERS:
        result = run_experiment_for_filter(
            filter_config,
            train_texts, train_labels, train_embeddings,
            test_embeddings, test_labels,
            baseline_f1, provider, model,
            dataset_name
        )
        filter_results.append(result)

    return DatasetResult(
        dataset=dataset_name,
        n_classes=len(set(train_labels)),
        baseline_f1=float(baseline_f1),
        filter_results=filter_results,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(all_results: List[DatasetResult]):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("FIXED OUTPUT COUNT EXPERIMENT - SUMMARY")
    print("="*80)
    print(f"Target samples per class: {TARGET_SAMPLES_PER_CLASS}")
    print(f"Datasets tested: {len(all_results)}")

    # Aggregate by filter
    filter_aggregates = {}
    for filter_config in FILTERS:
        filter_name = filter_config["name"]
        filter_aggregates[filter_name] = {
            "f1_deltas": [],
            "llm_calls": [],
            "acceptance_rates": [],
            "efficiencies": [],
            "success_count": 0,
            "total_count": 0
        }

    for dataset_result in all_results:
        for fr in dataset_result.filter_results:
            agg = filter_aggregates[fr.filter_name]
            agg["f1_deltas"].append(fr.f1_delta_vs_baseline)
            agg["llm_calls"].append(fr.total_llm_calls)
            agg["acceptance_rates"].append(fr.avg_acceptance_rate)
            agg["efficiencies"].append(fr.efficiency_score)
            agg["total_count"] += 1
            if fr.all_reached_target:
                agg["success_count"] += 1

    # Print comparison table
    print("\n" + "-"*80)
    print("FILTER COMPARISON (averaged across all datasets)")
    print("-"*80)
    print(f"\n{'Filter':<15} {'F1 Delta':>10} {'LLM Calls':>12} {'Accept%':>10} {'Efficiency':>12} {'Success':>10}")
    print("-" * 70)

    # Sort by F1 delta
    sorted_filters = sorted(
        filter_aggregates.items(),
        key=lambda x: np.mean(x[1]["f1_deltas"]) if x[1]["f1_deltas"] else 0,
        reverse=True
    )

    for filter_name, agg in sorted_filters:
        if not agg["f1_deltas"]:
            continue
        mean_delta = np.mean(agg["f1_deltas"])
        mean_calls = np.mean(agg["llm_calls"])
        mean_accept = np.mean(agg["acceptance_rates"]) * 100
        mean_eff = np.mean(agg["efficiencies"])
        success_rate = agg["success_count"] / agg["total_count"] * 100

        print(f"{filter_name:<15} {mean_delta:>+10.2f}pp {mean_calls:>12.0f} {mean_accept:>9.1f}% {mean_eff:>12.2f} {success_rate:>9.0f}%")

    # Per-dataset results
    print("\n" + "-"*80)
    print("PER-DATASET RESULTS")
    print("-"*80)

    for dataset_result in all_results:
        print(f"\n{dataset_result.dataset}:")
        print(f"  Baseline F1: {dataset_result.baseline_f1:.4f}")
        print(f"  {'Filter':<15} {'F1':>8} {'Delta':>10} {'Calls':>8} {'Efficiency':>10}")

        sorted_results = sorted(
            dataset_result.filter_results,
            key=lambda x: x.f1_delta_vs_baseline,
            reverse=True
        )

        for fr in sorted_results:
            status = "*" if not fr.all_reached_target else ""
            print(f"  {fr.filter_name:<15} {fr.f1_score:>8.4f} {fr.f1_delta_vs_baseline:>+10.2f}pp {fr.total_llm_calls:>8} {fr.efficiency_score:>10.2f}{status}")

    # Save results
    output_data = {
        "config": {
            "target_samples_per_class": TARGET_SAMPLES_PER_CLASS,
            "max_llm_calls_per_class": MAX_LLM_CALLS_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "n_shot": N_SHOT,
            "early_stop_acceptance": EARLY_STOP_ACCEPTANCE
        },
        "results": [
            {
                "dataset": dr.dataset,
                "n_classes": dr.n_classes,
                "baseline_f1": dr.baseline_f1,
                "timestamp": dr.timestamp,
                "filter_results": [asdict(fr) for fr in dr.filter_results]
            }
            for dr in all_results
        ],
        "summary": {
            filter_name: {
                "mean_f1_delta": float(np.mean(agg["f1_deltas"])) if agg["f1_deltas"] else 0,
                "mean_llm_calls": float(np.mean(agg["llm_calls"])) if agg["llm_calls"] else 0,
                "mean_acceptance_rate": float(np.mean(agg["acceptance_rates"])) if agg["acceptance_rates"] else 0,
                "mean_efficiency": float(np.mean(agg["efficiencies"])) if agg["efficiencies"] else 0,
                "success_rate": agg["success_count"] / agg["total_count"] if agg["total_count"] > 0 else 0
            }
            for filter_name, agg in filter_aggregates.items()
        }
    }

    output_path = RESULTS_DIR / "experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("="*80)
    print("FIXED OUTPUT COUNT EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Target samples per class: {TARGET_SAMPLES_PER_CLASS}")
    print(f"  Max LLM calls per class: {MAX_LLM_CALLS_PER_CLASS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  N-shot: {N_SHOT}")
    print(f"  Filters: {len(FILTERS)}")
    print(f"  Datasets: {len(DATASETS)}")

    # Initialize
    print("\nLoading model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-3-flash-preview")

    # Run experiments
    all_results = []

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        try:
            result = run_dataset_experiment(dataset_name, provider, model)
            all_results.append(result)

            # Save intermediate results
            intermediate_path = RESULTS_DIR / "intermediate_results.json"
            with open(intermediate_path, 'w') as f:
                json.dump({
                    "completed_datasets": [r.dataset for r in all_results],
                    "results": [asdict(r) for r in all_results]
                }, f, indent=2, default=str)

        except Exception as e:
            print(f"\n  Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    if all_results:
        generate_summary_report(all_results)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
