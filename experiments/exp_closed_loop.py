#!/usr/bin/env python3
"""
Closed-Loop Regeneration Experiment

Tests whether iterative feedback (diagnose rejections -> improve prompt -> regenerate)
improves LLM-augmented classification over single-pass generation.

Pipeline per class:
  1. GENERATE: LLM generates batch of candidates
  2. EMBED:    SentenceTransformer encodes candidates
  3. FILTER:   Threshold-based geometric filter (cascade_l1 or LOF)
  4. DIAGNOSE: RejectionAnalyzer classifies why samples were rejected
  5. IMPROVE:  PromptImprover modifies prompt based on dominant failure
  6. REPEAT:   Until target reached, budget exhausted, or convergence

Key research question:
  Does iterative prompt improvement via geometric feedback
  outperform single-pass generation with the same filter?

Baselines:
  - No augmentation (real data only)
  - Single-pass with same filter (generate 1.5x, filter once)
  - SMOTE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter
from core.filter_cascade import FilterCascade
from core.rejection_analyzer import RejectionAnalyzer, BatchDiagnosis
from core.prompt_improver import PromptImprover

# Import prompt template and dataset config from existing experiment
from exp_fixed_output_count import (
    DATASET_PROMPTS, get_dataset_base_name, create_prompt
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "closed_loop"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_PER_CLASS = 50
MAX_ITERATIONS = 5
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25
N_SHOT = 25
INITIAL_TEMPERATURE = 0.8

DATASETS = [
    "sms_spam_10shot",
    "sms_spam_25shot",
    "sms_spam_50shot",
    "20newsgroups_10shot",
    "20newsgroups_25shot",
    "20newsgroups_50shot",
    "hate_speech_davidson_10shot",
    "hate_speech_davidson_25shot",
    "hate_speech_davidson_50shot",
    "ag_news_synthetic_10shot",
    "ag_news_synthetic_25shot",
    "ag_news_synthetic_50shot",
]

FILTERS = [
    {"name": "cascade_l1", "type": "cascade", "params": {"filter_level": 1, "k_neighbors": 10}},
    {"name": "lof", "type": "lof", "params": {"n_neighbors": 10, "threshold": 0.0}},
]

SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class IterationMetrics:
    """Metrics for one iteration of the closed loop."""
    iteration: int
    n_generated: int
    n_accepted: int
    n_rejected: int
    acceptance_rate: float
    dominant_failure: str
    rejection_distribution: Dict[str, int]
    diversity_ratio: float
    mean_severity: float
    temperature: float
    modifications_applied: List[str]
    cumulative_accepted: int
    cumulative_llm_calls: int

    def to_dict(self):
        return asdict(self)


@dataclass
class ClosedLoopResult:
    """Result for one full experiment configuration."""
    dataset: str
    filter_name: str
    filter_params: Dict
    seed: int
    # F1 scores
    baseline_f1: float
    single_pass_f1: float
    closed_loop_f1: float
    smote_f1: float
    # Deltas
    delta_vs_baseline: float
    delta_vs_single_pass: float
    delta_vs_smote: float
    # Loop metrics
    n_iterations_per_class: Dict[str, int]
    total_llm_calls: int
    convergence_reasons: Dict[str, str]
    # Trajectories (per class)
    iteration_metrics: Dict[str, List[Dict]]
    # Meta
    timestamp: str

    def to_dict(self):
        return asdict(self)


# ============================================================================
# HELPERS
# ============================================================================

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a benchmark dataset."""
    path = DATA_DIR / f"{dataset_name}.json"
    if not path.exists():
        return None, None, None, None
    with open(path) as f:
        data = json.load(f)
    return (
        data['train_texts'], data['train_labels'],
        data['test_texts'], data['test_labels']
    )


def parse_llm_response(response: str) -> List[str]:
    """Parse LLM response into individual text samples."""
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 10:
            generated.append(clean)
    return generated


def evaluate_f1(
    train_emb: np.ndarray,
    train_labels: List[str],
    synth_emb: np.ndarray,
    synth_labels: List[str],
    test_emb: np.ndarray,
    test_labels: List[str]
) -> float:
    """Train LogisticRegression on train+synth data and return macro F1."""
    if len(synth_emb) > 0:
        aug_emb = np.vstack([train_emb, synth_emb])
        aug_labels = list(train_labels) + list(synth_labels)
    else:
        aug_emb = train_emb
        aug_labels = list(train_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    pred = clf.predict(test_emb)
    return float(f1_score(test_labels, pred, average='macro'))


def compute_baseline_f1(train_emb, train_labels, test_emb, test_labels) -> float:
    """Baseline F1 without augmentation."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    pred = clf.predict(test_emb)
    return float(f1_score(test_labels, pred, average='macro'))


def generate_smote_samples(real_embeddings: np.ndarray, n_generate: int, k_neighbors: int = 5) -> np.ndarray:
    """Generate SMOTE samples for a single class."""
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    X = np.vstack([real_embeddings, np.random.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy}, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        class0_indices = np.where(y_res == 0)[0]
        new_indices = class0_indices[n_base:]
        return X_res[new_indices][:n_generate]
    except Exception as e:
        print(f"        SMOTE error: {e}")
        return np.array([]).reshape(0, real_embeddings.shape[1])


def run_smote_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Run SMOTE baseline and return F1."""
    unique_classes = list(set(train_labels))
    labels_arr = np.array(train_labels)
    all_synth_emb = []
    all_synth_labels = []
    for cls in unique_classes:
        cls_emb = train_emb[labels_arr == cls]
        smote_emb = generate_smote_samples(cls_emb, TARGET_PER_CLASS)
        if len(smote_emb) > 0:
            all_synth_emb.append(smote_emb)
            all_synth_labels.extend([cls] * len(smote_emb))
    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
    else:
        synth_emb = np.array([]).reshape(0, train_emb.shape[1])
    return evaluate_f1(train_emb, train_labels, synth_emb, all_synth_labels, test_emb, test_labels)


# ============================================================================
# THRESHOLD-BASED FILTERING (adapted from rank-based for the loop)
# ============================================================================

def compute_cascade_threshold(real_embeddings: np.ndarray, real_labels: np.ndarray, target_class: str) -> float:
    """
    Compute a distance-score threshold from real data distribution.
    Cascade level=1 uses: score = 1 - (dist / max_dist).
    We set threshold = mean_score - 1*std_score of real samples.
    """
    class_mask = real_labels == target_class
    class_embs = real_embeddings[class_mask]
    if len(class_embs) < 2:
        return 0.0
    centroid = class_embs.mean(axis=0)
    dists = np.linalg.norm(class_embs - centroid, axis=1)
    max_dist = np.max(dists) + 1e-6
    scores = 1 - (dists / max_dist)
    return float(np.mean(scores) - np.std(scores))


def apply_threshold_filter(
    filter_type: str,
    candidate_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_class: str,
    cascade_threshold: float = 0.0,
    lof_filter: Optional[LOFFilter] = None,
    cascade: Optional[FilterCascade] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply threshold-based filtering. Returns (accepted_mask, scores).

    For cascade: compute distance scores, keep samples above threshold.
    For LOF: use LOF decision_function, keep samples above 0.0.
    """
    if len(candidate_embeddings) == 0:
        return np.array([], dtype=bool), np.array([])

    if filter_type == "cascade":
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]
        anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)

        scores, _ = cascade.compute_quality_scores(
            candidate_embeddings, anchor, real_embeddings, real_labels, target_class
        )
        accepted_mask = scores >= cascade_threshold
        return accepted_mask, scores

    elif filter_type == "lof":
        _, mask, scores = lof_filter.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )
        return mask, scores

    else:
        return np.ones(len(candidate_embeddings), dtype=bool), np.ones(len(candidate_embeddings))


# ============================================================================
# CLOSED-LOOP GENERATOR
# ============================================================================

class ClosedLoopGenerator:
    """
    Closed-loop generation: generate -> filter -> diagnose -> improve -> repeat.
    """

    def __init__(
        self,
        provider,
        model: SentenceTransformer,
        filter_name: str,
        filter_type: str,
        filter_params: Dict,
        dataset_name: str,
        max_iterations: int = MAX_ITERATIONS,
        target_per_class: int = TARGET_PER_CLASS,
        batch_size: int = BATCH_SIZE,
        n_shot: int = N_SHOT,
        initial_temperature: float = INITIAL_TEMPERATURE,
    ):
        self.provider = provider
        self.model = model
        self.filter_name = filter_name
        self.filter_type = filter_type
        self.filter_params = filter_params
        self.dataset_name = dataset_name
        self.max_iterations = max_iterations
        self.target_per_class = target_per_class
        self.batch_size = batch_size
        self.n_shot = n_shot
        self.initial_temperature = initial_temperature

        # Create filter object
        if filter_type == "cascade":
            self.cascade = FilterCascade(**filter_params)
            self.lof_filter = None
        elif filter_type == "lof":
            self.lof_filter = LOFFilter(**filter_params)
            self.cascade = None
        else:
            self.cascade = None
            self.lof_filter = None

    def generate_for_class(
        self,
        class_name: str,
        class_texts: List[str],
        real_embeddings: np.ndarray,
        real_labels: np.ndarray,
        analyzer: RejectionAnalyzer,
        improver: PromptImprover,
    ) -> Tuple[np.ndarray, List[str], List[IterationMetrics], str]:
        """
        Run the closed loop for one class.

        Returns:
            (accepted_embeddings, accepted_texts, iteration_metrics, convergence_reason)
        """

        # Compute threshold for cascade
        cascade_threshold = 0.0
        if self.filter_type == "cascade":
            cascade_threshold = compute_cascade_threshold(
                real_embeddings, real_labels, class_name
            )

        # Initialize loop state
        accepted_embeddings = []
        accepted_texts = []
        iteration_metrics = []
        total_llm_calls = 0
        temperature = self.initial_temperature
        prev_acceptance_rates = []

        # Create initial prompt
        selected_examples = class_texts[:self.n_shot]
        prompt = create_prompt(class_name, selected_examples, self.batch_size, self.dataset_name)

        for iteration in range(self.max_iterations):
            # 1. GENERATE
            try:
                messages = [{"role": "user", "content": prompt}]
                response, _ = self.provider.generate(messages, temperature=temperature, max_tokens=4000)
                generated_texts = parse_llm_response(response)
            except Exception as e:
                print(f"          Error generating: {e}")
                generated_texts = []

            total_llm_calls += 1

            if not generated_texts:
                # Empty generation, record and continue
                metrics = IterationMetrics(
                    iteration=iteration, n_generated=0, n_accepted=0,
                    n_rejected=0, acceptance_rate=0.0,
                    dominant_failure="EMPTY_GENERATION",
                    rejection_distribution={}, diversity_ratio=0.0,
                    mean_severity=0.0, temperature=temperature,
                    modifications_applied=[],
                    cumulative_accepted=len(accepted_texts),
                    cumulative_llm_calls=total_llm_calls
                )
                iteration_metrics.append(metrics)
                continue

            # 2. EMBED
            candidate_embeddings = self.model.encode(generated_texts, show_progress_bar=False)

            # 3. FILTER (threshold-based)
            accepted_mask, scores = apply_threshold_filter(
                self.filter_type, candidate_embeddings,
                real_embeddings, real_labels, class_name,
                cascade_threshold=cascade_threshold,
                lof_filter=self.lof_filter,
                cascade=self.cascade
            )

            # 4. DIAGNOSE
            batch_diagnosis = analyzer.analyze_batch(
                candidate_embeddings, accepted_mask, class_name
            )

            # 5. ACCUMULATE accepted samples
            if accepted_mask.any():
                new_accepted_emb = candidate_embeddings[accepted_mask]
                new_accepted_texts = [generated_texts[i] for i in range(len(generated_texts)) if accepted_mask[i]]
                accepted_embeddings.append(new_accepted_emb)
                accepted_texts.extend(new_accepted_texts)

            cumulative_accepted = sum(len(e) for e in accepted_embeddings) if accepted_embeddings else 0

            # 6. IMPROVE prompt based on diagnosis
            new_examples = improver.select_examples_for_iteration(
                class_name, batch_diagnosis, self.n_shot
            )
            if not new_examples:
                new_examples = class_texts[:self.n_shot]

            # Rebuild base prompt with new examples, then apply modifications
            base_prompt = create_prompt(class_name, new_examples, self.batch_size, self.dataset_name)
            improved_prompt, new_temperature, modifications = improver.improve_prompt(
                base_prompt, batch_diagnosis, class_name, iteration, temperature
            )

            mod_descriptions = [m.content[:80] for m in modifications]

            # Record metrics
            metrics = IterationMetrics(
                iteration=iteration,
                n_generated=len(generated_texts),
                n_accepted=int(accepted_mask.sum()),
                n_rejected=int((~accepted_mask).sum()),
                acceptance_rate=batch_diagnosis.acceptance_rate,
                dominant_failure=batch_diagnosis.dominant_failure,
                rejection_distribution=batch_diagnosis.rejection_distribution,
                diversity_ratio=batch_diagnosis.diversity_ratio,
                mean_severity=batch_diagnosis.mean_severity,
                temperature=temperature,
                modifications_applied=mod_descriptions,
                cumulative_accepted=cumulative_accepted,
                cumulative_llm_calls=total_llm_calls
            )
            iteration_metrics.append(metrics)

            # Update state for next iteration
            prompt = improved_prompt
            temperature = new_temperature

            print(f"          Iter {iteration}: {int(accepted_mask.sum())}/{len(generated_texts)} accepted "
                  f"({batch_diagnosis.acceptance_rate*100:.0f}%), "
                  f"dominant: {batch_diagnosis.dominant_failure}, "
                  f"cumul: {cumulative_accepted}/{self.target_per_class}", flush=True)

            # 7. CHECK CONVERGENCE
            # Target reached
            if cumulative_accepted >= self.target_per_class:
                # Truncate to target
                all_emb = np.vstack(accepted_embeddings)[:self.target_per_class]
                all_texts = accepted_texts[:self.target_per_class]
                return all_emb, all_texts, iteration_metrics, "TARGET_REACHED"

            # Budget exhausted
            if total_llm_calls >= MAX_LLM_CALLS_PER_CLASS:
                break

            # Acceptance rate plateau
            prev_acceptance_rates.append(batch_diagnosis.acceptance_rate)
            if len(prev_acceptance_rates) >= 3:
                recent = prev_acceptance_rates[-2:]
                if abs(recent[-1] - recent[-2]) < 0.01:
                    break

        # Finished loop without reaching target
        if accepted_embeddings:
            all_emb = np.vstack(accepted_embeddings)[:self.target_per_class]
            all_texts = accepted_texts[:self.target_per_class]
        else:
            all_emb = np.array([]).reshape(0, 768)
            all_texts = []

        reason = "MAX_ITERATIONS" if total_llm_calls < MAX_LLM_CALLS_PER_CLASS else "BUDGET_EXHAUSTED"
        if len(prev_acceptance_rates) >= 3 and abs(prev_acceptance_rates[-1] - prev_acceptance_rates[-2]) < 0.01:
            reason = "ACCEPTANCE_PLATEAU"

        return all_emb, all_texts, iteration_metrics, reason

    def generate_for_dataset(
        self,
        train_texts: List[str],
        train_labels: List[str],
        train_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, List[str], Dict[str, List[IterationMetrics]], Dict[str, str], int]:
        """
        Run closed-loop generation for all classes.

        Returns:
            (all_synth_emb, all_synth_labels, per_class_metrics, convergence_reasons, total_calls)
        """
        unique_classes = list(set(train_labels))
        labels_arr = np.array(train_labels)

        # Initialize analyzer and improver once with full dataset
        analyzer = RejectionAnalyzer(train_embeddings, labels_arr)
        improver = PromptImprover(
            real_texts=train_texts,
            real_labels=train_labels,
            real_embeddings=train_embeddings,
            dataset_name=self.dataset_name
        )

        all_synth_emb = []
        all_synth_labels = []
        per_class_metrics = {}
        convergence_reasons = {}
        total_calls = 0

        for cls in unique_classes:
            cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
            print(f"      Class: {cls} ({len(cls_texts)} real samples)")

            emb, texts, metrics, reason = self.generate_for_class(
                cls, cls_texts, train_embeddings, labels_arr, analyzer, improver
            )

            if len(emb) > 0:
                all_synth_emb.append(emb)
                all_synth_labels.extend([cls] * len(emb))

            per_class_metrics[cls] = [m.to_dict() for m in metrics]
            convergence_reasons[cls] = reason
            total_calls += sum(1 for m in metrics if m.n_generated > 0 or m.n_generated == 0)

            print(f"        -> {len(emb)} samples, {len(metrics)} iterations, reason: {reason}")

        if all_synth_emb:
            synth_emb = np.vstack(all_synth_emb)
        else:
            synth_emb = np.array([]).reshape(0, train_embeddings.shape[1])

        return synth_emb, all_synth_labels, per_class_metrics, convergence_reasons, total_calls


# ============================================================================
# SINGLE-PASS BASELINE (for comparison)
# ============================================================================

def run_single_pass(
    provider,
    model: SentenceTransformer,
    filter_name: str,
    filter_type: str,
    filter_params: Dict,
    train_texts: List[str],
    train_labels: List[str],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: List[str],
    dataset_name: str,
) -> float:
    """
    Single-pass: generate 1.5x target, filter once, evaluate.
    This is the existing approach.
    """
    unique_classes = list(set(train_labels))
    labels_arr = np.array(train_labels)

    # Create filter
    if filter_type == "cascade":
        filter_obj = FilterCascade(**filter_params)
    elif filter_type == "lof":
        filter_obj = LOFFilter(**filter_params)
    else:
        filter_obj = None

    all_synth_emb = []
    all_synth_labels = []

    for cls in unique_classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        n_generate = int(TARGET_PER_CLASS * 1.5)

        # Generate
        prompt = create_prompt(cls, cls_texts[:N_SHOT], n_generate, dataset_name)
        try:
            messages = [{"role": "user", "content": prompt}]
            response, _ = provider.generate(messages, temperature=INITIAL_TEMPERATURE, max_tokens=4000)
            generated_texts = parse_llm_response(response)
        except Exception as e:
            print(f"        Single-pass error: {e}")
            generated_texts = []

        if not generated_texts:
            continue

        candidate_emb = model.encode(generated_texts, show_progress_bar=False)

        # Filter
        if filter_type == "cascade" and filter_obj:
            filtered_emb, _, _ = filter_obj.filter_samples(
                candidate_emb, train_embeddings, labels_arr, cls, TARGET_PER_CLASS
            )
        elif filter_type == "lof" and filter_obj:
            filtered_emb, mask, _ = filter_obj.filter(
                candidate_emb, train_embeddings, labels_arr, cls
            )
            if len(filtered_emb) > TARGET_PER_CLASS:
                filtered_emb = filtered_emb[:TARGET_PER_CLASS]
        else:
            filtered_emb = candidate_emb[:TARGET_PER_CLASS]

        if len(filtered_emb) > 0:
            all_synth_emb.append(filtered_emb)
            all_synth_labels.extend([cls] * len(filtered_emb))

    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
    else:
        synth_emb = np.array([]).reshape(0, train_embeddings.shape[1])

    return evaluate_f1(train_embeddings, train_labels, synth_emb, all_synth_labels, test_embeddings, test_labels)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(
    dataset_name: str,
    filter_config: Dict,
    seed: int,
    provider,
    model: SentenceTransformer,
) -> Optional[ClosedLoopResult]:
    """Run one full experiment configuration."""
    np.random.seed(seed)

    # Load dataset
    train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
    if train_texts is None:
        print(f"  Dataset {dataset_name} not found, skipping.")
        return None

    print(f"\n  Dataset: {dataset_name}, Filter: {filter_config['name']}, Seed: {seed}")
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: {set(train_labels)}")

    # Embed
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(test_texts, show_progress_bar=False)

    # 1. Baseline (no augmentation)
    baseline_f1 = compute_baseline_f1(train_emb, train_labels, test_emb, test_labels)
    print(f"    Baseline F1: {baseline_f1:.4f}")

    # 2. SMOTE baseline
    smote_f1 = run_smote_baseline(train_emb, train_labels, test_emb, test_labels)
    print(f"    SMOTE F1:    {smote_f1:.4f} ({(smote_f1-baseline_f1)*100:+.2f}pp)")

    # 3. Single-pass baseline
    print(f"    Running single-pass ({filter_config['name']})...")
    single_pass_f1 = run_single_pass(
        provider, model,
        filter_config["name"], filter_config["type"], filter_config["params"],
        train_texts, train_labels, train_emb, test_emb, test_labels,
        dataset_name
    )
    print(f"    Single-pass F1: {single_pass_f1:.4f} ({(single_pass_f1-baseline_f1)*100:+.2f}pp)")

    # 4. Closed-loop
    print(f"    Running closed-loop ({filter_config['name']})...")
    generator = ClosedLoopGenerator(
        provider=provider,
        model=model,
        filter_name=filter_config["name"],
        filter_type=filter_config["type"],
        filter_params=filter_config["params"],
        dataset_name=dataset_name,
    )

    synth_emb, synth_labels, per_class_metrics, convergence_reasons, total_calls = \
        generator.generate_for_dataset(train_texts, train_labels, train_emb)

    closed_loop_f1 = evaluate_f1(train_emb, train_labels, synth_emb, synth_labels, test_emb, test_labels)
    print(f"    Closed-loop F1: {closed_loop_f1:.4f} ({(closed_loop_f1-baseline_f1)*100:+.2f}pp)")
    print(f"    vs single-pass: {(closed_loop_f1-single_pass_f1)*100:+.2f}pp")
    print(f"    vs SMOTE:       {(closed_loop_f1-smote_f1)*100:+.2f}pp")

    # Compute iterations per class
    n_iters = {cls: len(metrics) for cls, metrics in per_class_metrics.items()}

    return ClosedLoopResult(
        dataset=dataset_name,
        filter_name=filter_config["name"],
        filter_params=filter_config["params"],
        seed=seed,
        baseline_f1=baseline_f1,
        single_pass_f1=single_pass_f1,
        closed_loop_f1=closed_loop_f1,
        smote_f1=smote_f1,
        delta_vs_baseline=(closed_loop_f1 - baseline_f1) * 100,
        delta_vs_single_pass=(closed_loop_f1 - single_pass_f1) * 100,
        delta_vs_smote=(closed_loop_f1 - smote_f1) * 100,
        n_iterations_per_class=n_iters,
        total_llm_calls=total_calls,
        convergence_reasons=convergence_reasons,
        iteration_metrics=per_class_metrics,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary(results: List[ClosedLoopResult]):
    """Print and save summary report."""
    print("\n" + "=" * 80)
    print("CLOSED-LOOP REGENERATION EXPERIMENT - SUMMARY")
    print("=" * 80)

    # Key metric: closed-loop vs single-pass
    deltas_vs_sp = [r.delta_vs_single_pass for r in results]
    deltas_vs_smote = [r.delta_vs_smote for r in results]
    wins_vs_sp = sum(1 for d in deltas_vs_sp if d > 0)

    print(f"\nTotal configurations: {len(results)}")
    print(f"\nClosed-Loop vs Single-Pass:")
    print(f"  Mean delta:  {np.mean(deltas_vs_sp):+.2f}pp")
    print(f"  Std:         {np.std(deltas_vs_sp):.2f}pp")
    print(f"  Win rate:    {wins_vs_sp}/{len(results)} ({100*wins_vs_sp/len(results):.1f}%)")
    print(f"  Best:        {max(deltas_vs_sp):+.2f}pp")
    print(f"  Worst:       {min(deltas_vs_sp):+.2f}pp")

    print(f"\nClosed-Loop vs SMOTE:")
    wins_vs_smote = sum(1 for d in deltas_vs_smote if d > 0)
    print(f"  Mean delta:  {np.mean(deltas_vs_smote):+.2f}pp")
    print(f"  Win rate:    {wins_vs_smote}/{len(results)} ({100*wins_vs_smote/len(results):.1f}%)")

    # Per-filter breakdown
    print(f"\n{'Filter':<15} {'Mean vs SP':>12} {'Mean vs SMOTE':>15} {'Win% vs SP':>12}")
    print("-" * 55)
    for fc in FILTERS:
        fn = fc["name"]
        filter_results = [r for r in results if r.filter_name == fn]
        if filter_results:
            sp = [r.delta_vs_single_pass for r in filter_results]
            sm = [r.delta_vs_smote for r in filter_results]
            wins = sum(1 for d in sp if d > 0)
            print(f"{fn:<15} {np.mean(sp):>+12.2f}pp {np.mean(sm):>+15.2f}pp {100*wins/len(sp):>11.1f}%")

    # Per-dataset breakdown
    print(f"\n{'Dataset':<30} {'vs SP':>10} {'vs SMOTE':>10} {'Calls':>8}")
    print("-" * 60)
    for ds in DATASETS:
        ds_results = [r for r in results if r.dataset == ds]
        if ds_results:
            sp = np.mean([r.delta_vs_single_pass for r in ds_results])
            sm = np.mean([r.delta_vs_smote for r in ds_results])
            calls = np.mean([r.total_llm_calls for r in ds_results])
            print(f"{ds:<30} {sp:>+10.2f}pp {sm:>+10.2f}pp {calls:>8.0f}")

    # Convergence analysis
    print("\nConvergence Reasons:")
    all_reasons = {}
    for r in results:
        for cls, reason in r.convergence_reasons.items():
            all_reasons[reason] = all_reasons.get(reason, 0) + 1
    for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Save results
    output = {
        "config": {
            "target_per_class": TARGET_PER_CLASS,
            "max_iterations": MAX_ITERATIONS,
            "max_llm_calls_per_class": MAX_LLM_CALLS_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "n_shot": N_SHOT,
            "initial_temperature": INITIAL_TEMPERATURE,
            "filters": FILTERS,
            "seeds": SEEDS,
        },
        "summary": {
            "n_configurations": len(results),
            "mean_delta_vs_single_pass": float(np.mean(deltas_vs_sp)),
            "mean_delta_vs_smote": float(np.mean(deltas_vs_smote)),
            "win_rate_vs_single_pass": wins_vs_sp / len(results) if results else 0,
            "win_rate_vs_smote": wins_vs_smote / len(results) if results else 0,
        },
        "results": [r.to_dict() for r in results]
    }

    output_path = RESULTS_DIR / "experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 80)
    print("CLOSED-LOOP REGENERATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Target per class: {TARGET_PER_CLASS}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Max LLM calls/class: {MAX_LLM_CALLS_PER_CLASS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  N-shot: {N_SHOT}")
    print(f"  Temperature: {INITIAL_TEMPERATURE}")
    print(f"  Filters: {[f['name'] for f in FILTERS]}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total configs: {len(DATASETS) * len(FILTERS) * len(SEEDS)}")

    # Initialize
    print("\nInitializing LLM provider...")
    provider = create_provider("google", "gemini-3-flash-preview")

    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    # Run experiments
    all_results = []

    for dataset_name in DATASETS:
        for filter_config in FILTERS:
            for seed in SEEDS:
                result = run_experiment(
                    dataset_name, filter_config, seed, provider, model
                )
                if result:
                    all_results.append(result)

                    # Save incrementally
                    if len(all_results) % 5 == 0:
                        partial_path = RESULTS_DIR / "partial_results.json"
                        with open(partial_path, 'w') as f:
                            json.dump(
                                [r.to_dict() for r in all_results],
                                f, indent=2, default=str
                            )

    # Summary
    if all_results:
        generate_summary(all_results)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
