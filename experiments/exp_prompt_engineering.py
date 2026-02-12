#!/usr/bin/env python3
"""
Prompt Engineering Experiment

Tests different prompting techniques for LLM data generation:
1. Baseline: Simple prompt (current)
2. Chain-of-Thought: Analyze before generating
3. Negative: Include what NOT to generate
4. Multi-class: Include examples from other classes for contrast
5. Persona: Strong domain expert persona
6. Structured: Explicit format constraints

Hypothesis: Better prompts lead to higher quality synthetic data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, asdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# Import dataset prompts for domain info
from exp_fixed_output_count import DATASET_PROMPTS, get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_engineering"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Datasets
DATASETS = ["sms_spam_10shot", "20newsgroups_10shot", "hate_speech_davidson_10shot"]

# Generation config
SYNTHETIC_PER_CLASS = 50
N_SHOT = 25
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25

# Filter config
LLM_FILTER = {"filter_level": 1, "k_neighbors": 10}

# LLM Provider
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"


# ============================================================================
# PROMPT TECHNIQUES
# ============================================================================

def create_baseline_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Baseline simple prompt."""
    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    return f"""Generate {n_generate} NEW and DIVERSE examples for the class: {class_name}

Here are {len(examples[:N_SHOT])} real examples of this class:
{examples_text}

Generate {n_generate} NEW examples that:
- Are similar in style and content to the examples above
- Are diverse (not repetitive)
- Are realistic and natural

Output ONLY the generated examples, one per line, numbered 1-{n_generate}."""


def create_cot_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Chain-of-thought prompt: Analyze first, then generate."""
    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    return f"""You are generating synthetic training data for text classification.

Class: {class_name}

Here are {len(examples[:N_SHOT])} real examples:
{examples_text}

STEP 1: First, analyze the key characteristics of this class:
- What vocabulary is commonly used?
- What is the typical length and structure?
- What distinguishes this class from others?

STEP 2: Based on your analysis, generate {n_generate} NEW diverse examples that:
- Match the identified characteristics
- Maintain variety in content while preserving class identity
- Are realistic and natural sounding

Output format:
[ANALYSIS]
(Your brief analysis here)

[GENERATED EXAMPLES]
1. ...
2. ...
(etc.)

Generate exactly {n_generate} examples."""


def create_negative_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Negative prompting: Include what NOT to generate."""
    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    # Get other classes for negative examples
    other_classes = []
    if all_data:
        all_classes = list(set(all_data["train_labels"]))
        other_classes = [c for c in all_classes if c != class_name]

    negative_section = ""
    if other_classes:
        negative_section = f"""
DO NOT generate examples that could belong to these other classes: {', '.join(other_classes)}
"""

    return f"""Generate {n_generate} NEW examples for class: {class_name}

POSITIVE EXAMPLES (what to generate like):
{examples_text}

REQUIREMENTS:
- Generate text that CLEARLY belongs to class "{class_name}"
- Maintain the style and vocabulary of the positive examples
- Create diverse variations, not copies
{negative_section}
DO NOT:
- Generate generic or vague text
- Copy examples exactly
- Create text that is ambiguous about its class
- Include metadata, labels, or explanations

Output ONLY {n_generate} generated examples, numbered 1-{n_generate}, one per line."""


def create_multiclass_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Multi-class context: Show examples from other classes for contrast."""
    examples_text = "\n".join(f"  {i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    # Get examples from other classes
    contrast_section = ""
    if all_data:
        train_texts = all_data["train_texts"]
        train_labels = all_data["train_labels"]
        all_classes = list(set(train_labels))
        other_classes = [c for c in all_classes if c != class_name]

        contrast_parts = []
        for other_cls in other_classes[:3]:  # Max 3 other classes
            other_examples = [t for t, l in zip(train_texts, train_labels) if l == other_cls][:3]
            if other_examples:
                other_text = "\n".join(f"    - {ex[:100]}..." if len(ex) > 100 else f"    - {ex}"
                                      for ex in other_examples)
                contrast_parts.append(f"  {other_cls}:\n{other_text}")

        if contrast_parts:
            contrast_section = f"""
FOR CONTRAST - Examples from OTHER classes (DO NOT generate like these):
{chr(10).join(contrast_parts)}
"""

    return f"""You are classifying text into these classes: {', '.join(list(set(all_data["train_labels"])) if all_data else [class_name])}

TARGET CLASS: {class_name}

Examples of {class_name} (GENERATE LIKE THESE):
{examples_text}
{contrast_section}
Generate {n_generate} NEW examples that:
- CLEARLY belong to class "{class_name}"
- Are DISTINCT from the other classes shown above
- Maintain the vocabulary and style of the target class
- Are diverse and natural

Output {n_generate} examples, numbered 1-{n_generate}:"""


def create_persona_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Strong domain expert persona."""
    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    # Get domain info
    base_name = get_dataset_base_name(dataset_name)
    domain_info = DATASET_PROMPTS.get(base_name, {})
    domain = domain_info.get("domain", "text classification")

    return f"""You are a world-class expert in {domain} with 20+ years of experience.
You have analyzed millions of texts and can perfectly identify and generate authentic examples.

Your expertise allows you to:
- Understand subtle patterns that distinguish different classes
- Generate highly realistic synthetic examples
- Maintain perfect class consistency while maximizing diversity

TASK: Generate {n_generate} authentic examples of class "{class_name}"

Reference examples from real data:
{examples_text}

As an expert, generate {n_generate} NEW examples that:
- Are indistinguishable from real data
- Perfectly match the class "{class_name}"
- Show natural variation while maintaining authenticity

Output your {n_generate} expert-generated examples, numbered 1-{n_generate}:"""


def create_structured_prompt(
    class_name: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str,
    all_data: Dict = None
) -> str:
    """Structured format with explicit constraints."""
    examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:N_SHOT]))

    # Calculate average length
    avg_len = int(np.mean([len(ex.split()) for ex in examples[:N_SHOT]]))
    min_len = max(5, avg_len - 10)
    max_len = avg_len + 20

    return f"""=== TEXT GENERATION TASK ===

CLASS: {class_name}
COUNT: {n_generate} examples
FORMAT: One example per line, numbered

=== REFERENCE EXAMPLES ===
{examples_text}

=== CONSTRAINTS ===
- Length: {min_len}-{max_len} words per example
- Style: Match the reference examples
- Diversity: Each example must be unique
- Quality: Natural, realistic language only

=== OUTPUT FORMAT ===
1. [First generated example]
2. [Second generated example]
...
{n_generate}. [Last generated example]

=== GENERATE NOW ==="""


# Prompt techniques registry
PROMPT_TECHNIQUES = {
    "baseline": create_baseline_prompt,
    "cot": create_cot_prompt,
    "negative": create_negative_prompt,
    "multiclass": create_multiclass_prompt,
    "persona": create_persona_prompt,
    "structured": create_structured_prompt,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PromptResult:
    """Result for one prompt technique on one dataset."""
    dataset: str
    technique: str
    baseline_f1: float
    llm_f1: float
    delta: float
    llm_calls: int
    acceptance_rate: float
    n_synthetic: int


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_with_prompt(
    provider,
    prompt_fn: Callable,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    embed_model: SentenceTransformer,
    dataset_name: str,
    all_data: Dict
) -> Tuple[np.ndarray, List[str]]:
    """Generate samples using a specific prompt technique."""
    prompt = prompt_fn(class_name, class_texts, n_generate, dataset_name, all_data)
    messages = [{"role": "user", "content": prompt}]

    try:
        response, _ = provider.generate(messages, temperature=1.0, max_tokens=4000)
    except Exception as e:
        print(f"      LLM error: {e}")
        return np.array([]).reshape(0, 768), []

    # Parse response - handle CoT format specially
    lines = response.strip().split('\n')

    # For CoT, look for [GENERATED EXAMPLES] section
    if prompt_fn == create_cot_prompt:
        in_examples = False
        filtered_lines = []
        for line in lines:
            if "[GENERATED EXAMPLES]" in line.upper():
                in_examples = True
                continue
            if in_examples and line.strip():
                filtered_lines.append(line.strip())
        lines = filtered_lines if filtered_lines else lines

    # Parse examples
    generated = []
    for line in lines:
        clean = line.strip().lstrip('0123456789.-):* []')
        if len(clean) > 10 and not clean.startswith('[') and not clean.startswith('ANALYSIS'):
            generated.append(clean)

    if not generated:
        return np.array([]).reshape(0, 768), []

    embeddings = embed_model.encode(generated, show_progress_bar=False)
    return embeddings, generated


def generate_until_n_valid(
    provider,
    filter_obj: FilterCascade,
    prompt_fn: Callable,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    embed_model: SentenceTransformer,
    dataset_name: str,
    all_data: Dict
) -> Dict:
    """Generate until we have target_n valid samples."""
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while llm_calls < MAX_LLM_CALLS_PER_CLASS:
        batch_emb, batch_texts = generate_with_prompt(
            provider, prompt_fn, class_name, class_texts,
            BATCH_SIZE, embed_model, dataset_name, all_data
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
                    "llm_calls": llm_calls,
                    "acceptance_rate": 1.0
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
                "llm_calls": llm_calls,
                "acceptance_rate": acceptance_rate
            }

    # Max calls reached
    if pool_embeddings:
        pool_arr = np.vstack(pool_embeddings)
        return {
            "embeddings": pool_arr[:target_n] if len(pool_arr) >= target_n else pool_arr,
            "llm_calls": llm_calls,
            "acceptance_rate": len(pool_arr) / (llm_calls * BATCH_SIZE) if llm_calls > 0 else 0
        }

    return {
        "embeddings": np.array([]).reshape(0, 768),
        "llm_calls": llm_calls,
        "acceptance_rate": 0
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


def evaluate_augmented(
    train_emb, train_labels, test_emb, test_labels,
    synth_emb, synth_labels
) -> float:
    """Evaluate with augmented data."""
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

def run_technique_experiment(
    dataset_name: str,
    technique_name: str,
    prompt_fn: Callable,
    embed_model: SentenceTransformer,
    provider,
    filter_obj: FilterCascade
) -> PromptResult:
    """Run experiment for one technique on one dataset."""
    print(f"    Technique: {technique_name}")

    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    classes = list(set(train_labels))

    # Embed
    train_emb = embed_model.encode(train_texts, show_progress_bar=False)
    test_emb = embed_model.encode(data["test_texts"], show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    # Baseline
    baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, test_labels)

    # Generate with technique
    all_synth_emb = []
    all_synth_labels = []
    total_calls = 0
    acceptance_rates = []

    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        n_shot_actual = min(N_SHOT, len(cls_texts))

        result = generate_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            prompt_fn=prompt_fn,
            target_n=SYNTHETIC_PER_CLASS,
            class_name=cls,
            class_texts=cls_texts[:n_shot_actual],
            real_embeddings=train_emb,
            real_labels=train_labels_arr,
            embed_model=embed_model,
            dataset_name=dataset_name,
            all_data=data
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

    llm_f1 = evaluate_augmented(
        train_emb, train_labels_arr, test_emb, test_labels,
        synth_emb, all_synth_labels
    )
    delta = (llm_f1 - baseline_f1) * 100
    avg_acceptance = np.mean(acceptance_rates) if acceptance_rates else 0

    print(f"      F1: {llm_f1:.4f} ({delta:+.2f}pp), Calls: {total_calls}, Accept: {avg_acceptance*100:.1f}%")

    return PromptResult(
        dataset=dataset_name,
        technique=technique_name,
        baseline_f1=baseline_f1,
        llm_f1=llm_f1,
        delta=delta,
        llm_calls=total_calls,
        acceptance_rate=avg_acceptance,
        n_synthetic=len(synth_emb)
    )


def main():
    print("=" * 70)
    print("PROMPT ENGINEERING EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Techniques: {list(PROMPT_TECHNIQUES.keys())}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Synthetic per class: {SYNTHETIC_PER_CLASS}")

    # Initialize
    print("\nLoading models...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    filter_obj = FilterCascade(**LLM_FILTER)

    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_name}")
        print("=" * 60)

        for technique_name, prompt_fn in PROMPT_TECHNIQUES.items():
            try:
                result = run_technique_experiment(
                    dataset_name, technique_name, prompt_fn,
                    embed_model, provider, filter_obj
                )
                all_results.append(result)
            except Exception as e:
                print(f"      ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[PromptResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: PROMPT ENGINEERING")
    print("=" * 70)

    for dataset in DATASETS:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{dataset}:")
        print(f"{'Technique':<15} {'F1':<10} {'Delta':<12} {'Calls':<8} {'Accept%':<10}")
        print("-" * 60)

        baseline = dataset_results[0].baseline_f1
        print(f"{'BASELINE':<15} {baseline:<10.4f} {'-':<12} {'-':<8} {'-':<10}")

        for r in sorted(dataset_results, key=lambda x: x.delta, reverse=True):
            print(f"{r.technique:<15} {r.llm_f1:<10.4f} {r.delta:+10.2f}pp "
                  f"{r.llm_calls:<8} {r.acceptance_rate*100:<9.1f}%")

    # Overall ranking
    print("\n" + "-" * 70)
    print("OVERALL RANKING (by average delta):")
    print("-" * 70)

    technique_deltas = {}
    for technique in PROMPT_TECHNIQUES.keys():
        tech_results = [r for r in all_results if r.technique == technique]
        if tech_results:
            technique_deltas[technique] = np.mean([r.delta for r in tech_results])

    for technique, avg_delta in sorted(technique_deltas.items(), key=lambda x: x[1], reverse=True):
        tech_results = [r for r in all_results if r.technique == technique]
        avg_calls = np.mean([r.llm_calls for r in tech_results])
        avg_accept = np.mean([r.acceptance_rate for r in tech_results])
        print(f"{technique:<15} Avg delta: {avg_delta:+.2f}pp, Calls: {avg_calls:.1f}, Accept: {avg_accept*100:.1f}%")


def save_results(all_results: List[PromptResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"prompt_engineering_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "techniques": list(PROMPT_TECHNIQUES.keys()),
            "datasets": DATASETS,
            "synthetic_per_class": SYNTHETIC_PER_CLASS,
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
