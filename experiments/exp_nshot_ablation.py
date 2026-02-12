#!/usr/bin/env python3
"""
N-Shot Ablation Experiment

Tests different numbers of examples in the LLM prompt to find optimal n-shot depth.
More examples may improve quality but also constrain diversity.

Tests: 5, 10, 15, 25, 50 examples in prompt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# Import prompt components
from exp_fixed_output_count import DATASET_PROMPTS, get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "nshot_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SHOTS = [5, 10, 15, 25, 50]
# Use 25shot datasets since we need up to 50 examples
DATASETS = ["sms_spam_25shot", "20newsgroups_25shot"]

SYNTHETIC_PER_CLASS = 50
TEMPERATURE = 0.8
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25

# Filter config (best performing from previous experiments)
LLM_FILTER = {"filter_level": 1, "k_neighbors": 10}

# LLM Provider
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NshotResult:
    """Result for one n-shot value on one dataset."""
    dataset: str
    n_shot: int
    baseline_f1: float
    llm_f1: float
    llm_delta: float
    llm_calls: int
    acceptance_rate: float
    n_synthetic: int


# ============================================================================
# PROMPT GENERATION (modified for variable n_shot)
# ============================================================================

def create_prompt_with_nshot(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    n_shot: int
) -> str:
    """
    Create prompt with specific number of examples.
    Adapted from exp_fixed_output_count.create_prompt.
    """
    base_dataset = get_dataset_base_name(dataset_name)
    config = DATASET_PROMPTS.get(base_dataset, {})

    domain = config.get("domain", "text documents")
    task_context = config.get("task_context", "text classification")
    text_type = config.get("text_type", "text samples")
    style_notes = config.get("style_notes", "Match the style of the examples")
    length_guidance = config.get("length_guidance", "Match the length of examples")
    class_descriptions = config.get("class_descriptions", {})

    class_desc = class_descriptions.get(class_name, f"content belonging to '{class_name}'")

    # Select examples (limited by n_shot)
    selected_examples = examples[:n_shot]
    avg_length = sum(len(ex.split()) for ex in selected_examples) / max(len(selected_examples), 1)

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
# GENERATION FUNCTIONS
# ============================================================================

def generate_llm_batch_with_nshot(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    model: SentenceTransformer,
    dataset_name: str,
    n_shot: int
) -> Tuple[np.ndarray, List[str]]:
    """Generate batch with specific n-shot count."""
    prompt = create_prompt_with_nshot(
        class_name, class_texts, n_generate, dataset_name, n_shot
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        response, _ = provider.generate(
            messages,
            temperature=TEMPERATURE,
            max_tokens=4000
        )
    except Exception as e:
        print(f"    LLM error: {e}")
        return np.array([]).reshape(0, 768), []

    # Parse response
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 10:
            generated.append(clean)

    if not generated:
        return np.array([]).reshape(0, 768), []

    embeddings = model.encode(generated, show_progress_bar=False)
    return embeddings, generated


def generate_until_n_valid(
    provider,
    filter_obj: FilterCascade,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    model: SentenceTransformer,
    dataset_name: str,
    n_shot: int
) -> Dict:
    """Generate until we have target_n valid samples."""
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while llm_calls < MAX_LLM_CALLS_PER_CLASS:
        batch_emb, batch_texts = generate_llm_batch_with_nshot(
            provider, class_name, class_texts, BATCH_SIZE, model, dataset_name, n_shot
        )
        llm_calls += 1

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        if not pool_embeddings:
            continue

        pool_arr = np.vstack(pool_embeddings)

        # Apply filter
        class_mask = real_labels == class_name
        if not class_mask.any():
            if len(pool_arr) >= target_n:
                return {
                    "embeddings": pool_arr[:target_n],
                    "texts": pool_texts[:target_n],
                    "llm_calls": llm_calls,
                    "acceptance_rate": 1.0,
                    "status": "SUCCESS"
                }
            continue

        filtered_emb, _, _ = filter_obj.filter_samples(
            candidates=pool_arr,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=class_name,
            target_count=target_n
        )

        n_valid = len(filtered_emb)
        acceptance_rate = n_valid / len(pool_arr) if len(pool_arr) > 0 else 0

        if n_valid >= target_n:
            class_embs = real_embeddings[class_mask]
            anchor = class_embs.mean(axis=0)
            scores, _ = filter_obj.compute_quality_scores(
                pool_arr, anchor, real_embeddings, real_labels, class_name
            )
            top_idx = np.argsort(scores)[-target_n:]

            return {
                "embeddings": pool_arr[top_idx],
                "texts": [pool_texts[i] for i in top_idx],
                "llm_calls": llm_calls,
                "acceptance_rate": acceptance_rate,
                "status": "SUCCESS"
            }

    # Max calls reached
    if pool_embeddings:
        pool_arr = np.vstack(pool_embeddings)
        return {
            "embeddings": pool_arr[:target_n] if len(pool_arr) >= target_n else pool_arr,
            "texts": pool_texts[:target_n] if len(pool_texts) >= target_n else pool_texts,
            "llm_calls": llm_calls,
            "acceptance_rate": len(pool_arr) / (llm_calls * BATCH_SIZE) if llm_calls > 0 else 0,
            "status": "MAX_CALLS"
        }

    return {
        "embeddings": np.array([]).reshape(0, 768),
        "texts": [],
        "llm_calls": llm_calls,
        "acceptance_rate": 0,
        "status": "NO_DATA"
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


def evaluate_with_llm(
    train_emb, train_labels, test_emb, test_labels,
    synth_emb, synth_labels
) -> float:
    """Evaluate with LLM augmentation."""
    if len(synth_emb) == 0:
        return evaluate_baseline(train_emb, train_labels, test_emb, test_labels)

    aug_emb = np.vstack([train_emb, synth_emb])
    aug_labels = list(train_labels) + list(synth_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_nshot_experiment(
    dataset_name: str,
    n_shot: int,
    model: SentenceTransformer,
    provider
) -> NshotResult:
    """Run experiment for one dataset and n-shot value."""
    print(f"\n  N-Shot: {n_shot}")
    print(f"  {'-'*40}")

    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels"]
    test_texts = data["test_texts"]
    test_labels = data["test_labels"]
    classes = list(set(train_labels))

    # Check if we have enough examples
    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        if len(cls_texts) < n_shot:
            print(f"    Warning: class '{cls}' has only {len(cls_texts)} examples, need {n_shot}")

    # Embed
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(test_texts, show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    # Baseline
    baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, test_labels)
    print(f"    Baseline F1: {baseline_f1:.4f}")

    # Generate with LLM
    filter_obj = FilterCascade(**LLM_FILTER)
    all_synth_emb = []
    all_synth_labels = []
    total_calls = 0
    acceptance_rates = []

    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        # Use min of n_shot and available texts
        actual_n_shot = min(n_shot, len(cls_texts))

        result = generate_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            target_n=SYNTHETIC_PER_CLASS,
            class_name=cls,
            class_texts=cls_texts,  # Pass all, function will use n_shot
            real_embeddings=train_emb,
            real_labels=train_labels_arr,
            model=model,
            dataset_name=dataset_name,
            n_shot=actual_n_shot
        )

        total_calls += result["llm_calls"]
        acceptance_rates.append(result["acceptance_rate"])

        if len(result["embeddings"]) > 0:
            all_synth_emb.append(result["embeddings"])
            all_synth_labels.extend([cls] * len(result["embeddings"]))

    # Evaluate
    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
    else:
        synth_emb = np.array([]).reshape(0, 768)

    llm_f1 = evaluate_with_llm(
        train_emb, train_labels_arr, test_emb, test_labels,
        synth_emb, all_synth_labels
    )
    llm_delta = (llm_f1 - baseline_f1) * 100
    avg_acceptance = np.mean(acceptance_rates) if acceptance_rates else 0

    print(f"    LLM F1: {llm_f1:.4f} ({llm_delta:+.2f}pp)")
    print(f"    LLM calls: {total_calls}, acceptance: {avg_acceptance*100:.1f}%")

    return NshotResult(
        dataset=dataset_name,
        n_shot=n_shot,
        baseline_f1=baseline_f1,
        llm_f1=llm_f1,
        llm_delta=llm_delta,
        llm_calls=total_calls,
        acceptance_rate=avg_acceptance,
        n_synthetic=len(synth_emb)
    )


def main():
    print("="*70)
    print("N-SHOT ABLATION EXPERIMENT")
    print("="*70)
    print(f"\nConfig:")
    print(f"  N-Shot values: {N_SHOTS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Synthetic per class: {SYNTHETIC_PER_CLASS}")
    print(f"  Temperature: {TEMPERATURE}")

    # Initialize
    print("\nLoading models...")
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print("="*70)

        for n_shot in N_SHOTS:
            try:
                result = run_nshot_experiment(dataset_name, n_shot, model, provider)
                all_results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[NshotResult]):
    """Print summary of results."""
    print("\n" + "="*70)
    print("SUMMARY: N-SHOT ABLATION")
    print("="*70)

    for dataset in DATASETS:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{dataset}:")
        print(f"{'N-Shot':<8} {'F1':<10} {'Delta':<12} {'Calls':<8} {'Accept%':<10}")
        print("-" * 50)

        baseline = dataset_results[0].baseline_f1
        print(f"{'BASE':<8} {baseline:<10.4f} {'-':<12} {'-':<8} {'-':<10}")

        for r in sorted(dataset_results, key=lambda x: x.n_shot):
            print(f"{r.n_shot:<8} {r.llm_f1:<10.4f} {r.llm_delta:+10.2f}pp "
                  f"{r.llm_calls:<8} {r.acceptance_rate*100:<9.1f}%")

        # Best n-shot
        best = max(dataset_results, key=lambda x: x.llm_f1)
        print(f"\n  BEST: n_shot={best.n_shot} with F1={best.llm_f1:.4f} ({best.llm_delta:+.2f}pp)")


def save_results(all_results: List[NshotResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"nshot_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "n_shots": N_SHOTS,
            "datasets": DATASETS,
            "synthetic_per_class": SYNTHETIC_PER_CLASS,
            "temperature": TEMPERATURE,
            "llm_filter": LLM_FILTER,
            "llm_model": LLM_MODEL,
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Latest summary
    summary_file = RESULTS_DIR / "latest_summary.json"
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
