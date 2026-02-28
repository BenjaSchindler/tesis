#!/usr/bin/env python3
"""
Progressive Waves Experiment

Generates synthetic samples in 4 progressive waves, each with increasing
prompt strictness, filter thresholds, and example precision. Early waves
prioritize diversity (broad coverage), later waves prioritize precision
(core distribution samples).

Key differences from existing experiments:
- vs Curriculum: generates NEW batches per wave with different prompts (not just top-K% of one batch)
- vs Closed-loop: proactive design per wave (not reactive to rejection diagnostics)
- vs Single-pass: multiple "zoom levels" from periphery to core

Configuration: 21 datasets × 5 strategies × 3 classifiers × 3 seeds = 945 evaluations + baselines
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import hashlib
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats

from core.filter_cascade import FilterCascade
from core.llm_providers import create_provider
from exp_fixed_output_count import (
    DATASET_PROMPTS, get_dataset_base_name, create_prompt
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = PROJECT_ROOT / "results" / "progressive_waves"
OUTPUT_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Tables"
FIGURE_DIR = PROJECT_ROOT / "Escrito_Tesis" / "Figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
CANDIDATES_PER_WAVE = 75
N_WAVES = 4
N_SHOT = 25
SEEDS = [42, 123, 456]

# Filter: cascade_l1 (proven best)
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

# Soft weighting (proven optimal)
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

# Wave definitions
WAVE_CONFIGS = [
    {
        "name": "exploratory",
        "llm_temperature": 0.9,
        "threshold_k": -2.0,  # mean - 2*std (lenient)
        "example_strategy": "random",
        "prompt_injection": (
            "# GENERATION STRATEGY\n"
            "Prioritize MAXIMUM DIVERSITY in your outputs. Explore different sub-topics, "
            "perspectives, writing styles, and vocabulary choices within the \"{class_name}\" "
            "category. Each generated text should cover a DIFFERENT aspect or angle of "
            "this class than the others.\n"
        ),
    },
    {
        "name": "moderate",
        "llm_temperature": 0.7,
        "threshold_k": -1.0,  # mean - 1*std
        "example_strategy": "mixed",
        "prompt_injection": (
            "# GENERATION STRATEGY\n"
            "Focus on matching the STYLE, THEMES, and VOCABULARY of the reference examples. "
            "Your outputs should sound like they come from the same source as the examples "
            "above. Pay close attention to the tone, sentence structure, and word choices "
            "used in the references.\n"
        ),
    },
    {
        "name": "focused",
        "llm_temperature": 0.5,
        "threshold_k": 0.0,  # mean (strict)
        "example_strategy": "centroid_nearest",
        "prompt_injection": (
            "# GENERATION STRATEGY\n"
            "Stay VERY CLOSE to the vocabulary, sentence structure, and content patterns "
            "of the reference examples. Do NOT introduce topics, jargon, or writing patterns "
            "not present in the examples. Each generated text should feel like a natural "
            "extension of the existing dataset.\n"
        ),
    },
    {
        "name": "ultra_precise",
        "llm_temperature": 0.3,
        "threshold_k": 0.5,  # mean + 0.5*std (very strict)
        "example_strategy": "top5_nearest",
        "prompt_injection": (
            "# GENERATION STRATEGY\n"
            "Your outputs must be PRACTICALLY INDISTINGUISHABLE from the reference examples. "
            "Replicate the exact register, typical sentence length, vocabulary density, and "
            "topical focus of the references. A human expert should not be able to tell your "
            "outputs apart from the real examples. Subtle variations are preferred over "
            "creative departures.\n"
        ),
    },
]

COMBINATION_STRATEGIES = [
    "accumulate",
    "best_wave",
    "progressive_replace",
    "weighted_blend",
    "all_equal_weight",
]

DATASETS = [
    "sms_spam_10shot", "sms_spam_25shot", "sms_spam_50shot",
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "20newsgroups_10shot", "20newsgroups_25shot", "20newsgroups_50shot",
    "ag_news_10shot", "ag_news_25shot", "ag_news_50shot",
    "emotion_10shot", "emotion_25shot", "emotion_50shot",
    "dbpedia14_10shot", "dbpedia14_25shot", "dbpedia14_50shot",
    "20newsgroups_20class_10shot", "20newsgroups_20class_25shot", "20newsgroups_20class_50shot",
]

DATASET_N_CLASSES = {
    "sms_spam": 2, "hate_speech_davidson": 3, "20newsgroups": 4,
    "ag_news": 4, "emotion": 6, "dbpedia14": 14, "20newsgroups_20class": 20,
}

CLASSIFIERS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "svc_linear": lambda seed: SVC(kernel="linear", random_state=seed),
    "ridge": lambda seed: RidgeClassifier(alpha=1.0),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset(name):
    with open(DATA_DIR / f"{name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_dataset_base(name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if name.startswith(base + "_"):
            return base
    return name


def parse_n_shot(ds_name):
    for part in ds_name.split("_"):
        if "shot" in part:
            return int(part.replace("shot", ""))
    return 10


def parse_llm_response(response):
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 10:
            generated.append(clean)
    return generated


def generate_smote_samples(real_embeddings, n_generate, seed=42, k_neighbors=5):
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    rng = np.random.RandomState(seed)
    X = np.vstack([real_embeddings, rng.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res[np.where(y_res == 0)[0][n_base:]][:n_generate]
    except Exception:
        return np.array([]).reshape(0, real_embeddings.shape[1])


def normalize_scores(raw_scores):
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    return np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / TEMPERATURE)


def compute_paired_statistics(method_f1s, baseline_f1s, n_comparisons=5):
    method_f1s = np.array(method_f1s)
    baseline_f1s = np.array(baseline_f1s)
    n = len(method_f1s)
    deltas = method_f1s - baseline_f1s
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1) if n > 1 else 0.0
    se = delta_std / np.sqrt(n) if n > 1 else 0.0

    if n > 1 and se > 0:
        ci_t = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
    else:
        ci_t = (delta_mean, delta_mean)

    rng = np.random.RandomState(42)
    boot = [np.mean(deltas[rng.choice(n, n, replace=True)]) for _ in range(1000)]
    ci_boot = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    if n > 1 and delta_std > 0:
        t_stat, p_val = stats.ttest_rel(method_f1s, baseline_f1s)
    else:
        t_stat, p_val = 0.0, 1.0

    cohen_d = delta_mean / (delta_std + 1e-10) if delta_std > 0 else 0.0
    bonf_p = min(p_val * n_comparisons, 1.0)

    return {
        "delta_mean_pp": float(delta_mean * 100),
        "delta_std_pp": float(delta_std * 100),
        "ci_95_t": (float(ci_t[0] * 100), float(ci_t[1] * 100)),
        "ci_95_bootstrap": (float(ci_boot[0] * 100), float(ci_boot[1] * 100)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "bonferroni_p": float(bonf_p),
        "cohen_d": float(cohen_d),
        "significant_005": bool(p_val < 0.05),
        "significant_bonferroni": bool(bonf_p < 0.05),
        "win_rate": float(np.mean(deltas > 0)),
    }


# ============================================================================
# WAVE-SPECIFIC FUNCTIONS
# ============================================================================

def get_wave_cache_key(dataset, class_name, n_shot, n_generate, wave_index):
    key_str = f"progressive_waves_v1_{dataset}_{class_name}_{n_shot}_{n_generate}_wave{wave_index}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def select_examples_for_wave(class_texts, class_embeddings, centroid, strategy, n_shot, rng):
    """Select reference examples based on wave's example strategy."""
    n = min(n_shot, len(class_texts))
    if n == len(class_texts):
        return class_texts[:n]

    if strategy == "random":
        idx = rng.choice(len(class_texts), n, replace=False)
        return [class_texts[i] for i in idx]

    elif strategy == "mixed":
        dists = np.linalg.norm(class_embeddings - centroid, axis=1)
        n_near = n // 2
        n_rand = n - n_near
        near_idx = list(np.argsort(dists)[:n_near])
        remaining = [i for i in range(len(class_texts)) if i not in near_idx]
        rand_idx = list(rng.choice(remaining, min(n_rand, len(remaining)), replace=False))
        idx = near_idx + rand_idx
        return [class_texts[i] for i in idx]

    elif strategy == "centroid_nearest":
        dists = np.linalg.norm(class_embeddings - centroid, axis=1)
        idx = np.argsort(dists)[:n]
        return [class_texts[i] for i in idx]

    elif strategy == "top5_nearest":
        dists = np.linalg.norm(class_embeddings - centroid, axis=1)
        idx = np.argsort(dists)[:min(5, n)]
        return [class_texts[i] for i in idx]

    # Fallback
    return class_texts[:n]


def create_wave_prompt(class_name, examples, n_generate, dataset_name, wave_config):
    """Build prompt with wave-specific generation strategy injection."""
    base_prompt = create_prompt(class_name, examples, n_generate, dataset_name)
    injection = wave_config["prompt_injection"].format(class_name=class_name)

    # Inject before CONSTRAINTS section (same pattern as prompt_improver.py)
    if "# CONSTRAINTS" in base_prompt:
        parts = base_prompt.split("# CONSTRAINTS", 1)
        return parts[0] + injection + "\n# CONSTRAINTS" + parts[1]
    elif "# OUTPUT FORMAT" in base_prompt:
        parts = base_prompt.split("# OUTPUT FORMAT", 1)
        return parts[0] + injection + "\n# OUTPUT FORMAT" + parts[1]
    else:
        return base_prompt + "\n" + injection


def compute_wave_threshold(real_embeddings, real_labels, target_class, cascade, threshold_k):
    """Compute per-wave acceptance threshold from real data score distribution."""
    labels_arr = np.array(real_labels)
    class_mask = labels_arr == target_class
    class_embs = real_embeddings[class_mask]
    if len(class_embs) < 2:
        return 0.0, 0.0, 0.0

    anchor = class_embs.mean(axis=0)
    scores, _ = cascade.compute_quality_scores(
        class_embs, anchor, real_embeddings, labels_arr, target_class
    )
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    threshold = mean_score + threshold_k * std_score
    return threshold, mean_score, std_score


def generate_wave(
    provider, model, cascade,
    wave_index, wave_config,
    class_name, class_texts, class_embeddings,
    real_embeddings, real_labels, labels_arr,
    dataset_name, n_shot, threshold, std_real_score
):
    """Generate and filter samples for one wave, one class. Includes adaptive fallback."""
    centroid = class_embeddings.mean(axis=0)
    rng = np.random.RandomState(42 + wave_index)

    # 1. Select examples per wave strategy
    examples = select_examples_for_wave(
        class_texts, class_embeddings, centroid,
        wave_config["example_strategy"], n_shot, rng
    )

    # 2. Build wave-specific prompt
    prompt = create_wave_prompt(
        class_name, examples, CANDIDATES_PER_WAVE, dataset_name, wave_config
    )

    # 3. Check cache
    cache_key = get_wave_cache_key(dataset_name, class_name, n_shot, CANDIDATES_PER_WAVE, wave_index)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        gen_texts = cached.get("texts", [])
    else:
        # Generate via LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response, _ = provider.generate(
                messages, temperature=wave_config["llm_temperature"], max_tokens=4000
            )
            gen_texts = parse_llm_response(response)
        except Exception as e:
            print(f"          Wave {wave_index} error: {e}")
            # Retry once after delay
            try:
                time.sleep(5)
                messages = [{"role": "user", "content": prompt}]
                response, _ = provider.generate(
                    messages, temperature=wave_config["llm_temperature"], max_tokens=4000
                )
                gen_texts = parse_llm_response(response)
            except Exception as e2:
                print(f"          Wave {wave_index} retry failed: {e2}")
                gen_texts = []

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "dataset": dataset_name, "class_name": class_name,
                "wave_index": wave_index, "wave_name": wave_config["name"],
                "n_shot": n_shot, "n_generate": CANDIDATES_PER_WAVE,
                "texts": gen_texts, "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    if not gen_texts:
        return np.array([]).reshape(0, 768), np.array([]), {
            "wave_index": wave_index, "wave_name": wave_config["name"],
            "n_generated": 0, "n_accepted": 0, "acceptance_rate": 0.0,
            "mean_score_all": 0.0, "mean_score_accepted": 0.0,
            "threshold": threshold, "fallback_applied": False,
        }

    # 4. Embed
    gen_emb = model.encode(gen_texts, show_progress_bar=False)

    # 5. Score with cascade_l1
    anchor = centroid
    scores, _ = cascade.compute_quality_scores(
        gen_emb, anchor, real_embeddings, labels_arr, class_name
    )

    # 6. Apply threshold filter
    accepted_mask = scores >= threshold
    fallback_applied = False
    original_threshold = threshold
    final_threshold = threshold

    # 7. Adaptive fallback: if 0 accepted, relax by 0.5*std and retry once
    if not accepted_mask.any() and len(gen_texts) > 0 and std_real_score > 0:
        fallback_threshold = threshold - 0.5 * std_real_score
        accepted_mask = scores >= fallback_threshold
        fallback_applied = True
        final_threshold = fallback_threshold

    accepted_emb = gen_emb[accepted_mask]
    accepted_scores = scores[accepted_mask]

    wave_meta = {
        "wave_index": wave_index,
        "wave_name": wave_config["name"],
        "n_generated": len(gen_texts),
        "n_accepted": int(accepted_mask.sum()),
        "acceptance_rate": float(accepted_mask.mean()),
        "mean_score_all": float(scores.mean()),
        "mean_score_accepted": float(accepted_scores.mean()) if len(accepted_scores) > 0 else 0.0,
        "threshold": final_threshold,
        "original_threshold": original_threshold,
        "fallback_applied": fallback_applied,
    }

    return accepted_emb, accepted_scores, wave_meta


# ============================================================================
# COMBINATION STRATEGIES
# ============================================================================

def combine_accumulate(wave_pools, target_n):
    """Pool all waves, rank by wave_weight * quality, select top-N."""
    WAVE_WEIGHTS = {0: 0.6, 1: 0.8, 2: 1.0, 3: 1.2}
    all_emb, all_scores, all_wave_idx = [], [], []
    for w_idx in sorted(wave_pools.keys()):
        emb, scores = wave_pools[w_idx]
        if len(emb) == 0:
            continue
        composite = WAVE_WEIGHTS.get(w_idx, 1.0) * scores
        all_emb.append(emb)
        all_scores.append(composite)
        all_wave_idx.extend([w_idx] * len(emb))
    if not all_emb:
        return np.array([]).reshape(0, 768), np.array([]), []
    all_emb = np.vstack(all_emb)
    all_scores = np.concatenate(all_scores)
    n = min(target_n, len(all_emb))
    top_idx = np.argsort(all_scores)[-n:]
    return all_emb[top_idx], all_scores[top_idx], [all_wave_idx[i] for i in top_idx]


def combine_best_wave(wave_pools, target_n):
    """Pick the single wave with highest mean quality score."""
    best_wave, best_mean = -1, -1.0
    for w_idx, (emb, scores) in wave_pools.items():
        if len(scores) > 0 and float(scores.mean()) > best_mean:
            best_mean = float(scores.mean())
            best_wave = w_idx
    if best_wave < 0:
        return np.array([]).reshape(0, 768), np.array([]), []
    emb, scores = wave_pools[best_wave]
    n = min(target_n, len(emb))
    top_idx = np.argsort(scores)[-n:]
    return emb[top_idx], scores[top_idx], [best_wave] * n


def combine_progressive_replace(wave_pools, target_n):
    """Pool everything, take top-N by raw quality score (later waves naturally score higher)."""
    all_emb, all_scores, all_wave_idx = [], [], []
    for w_idx in sorted(wave_pools.keys()):
        emb, scores = wave_pools[w_idx]
        if len(emb) == 0:
            continue
        all_emb.append(emb)
        all_scores.append(scores)
        all_wave_idx.extend([w_idx] * len(emb))
    if not all_emb:
        return np.array([]).reshape(0, 768), np.array([]), []
    all_emb = np.vstack(all_emb)
    all_scores = np.concatenate(all_scores)
    n = min(target_n, len(all_emb))
    top_idx = np.argsort(all_scores)[-n:]
    return all_emb[top_idx], all_scores[top_idx], [all_wave_idx[i] for i in top_idx]


def combine_weighted_blend(wave_pools, target_n):
    """Fixed quota per wave: 20%, 25%, 30%, 25%."""
    BLEND_FRACTIONS = {0: 0.20, 1: 0.25, 2: 0.30, 3: 0.25}
    all_emb, all_scores, all_waves = [], [], []
    for w_idx in sorted(wave_pools.keys()):
        emb, scores = wave_pools[w_idx]
        if len(emb) == 0:
            continue
        quota = max(1, int(target_n * BLEND_FRACTIONS.get(w_idx, 0.25)))
        n_take = min(quota, len(emb))
        top_idx = np.argsort(scores)[-n_take:]
        all_emb.append(emb[top_idx])
        # Weight by (wave_index + 1) * quality
        weighted = (w_idx + 1) * scores[top_idx]
        all_scores.append(weighted)
        all_waves.extend([w_idx] * n_take)
    if not all_emb:
        return np.array([]).reshape(0, 768), np.array([]), []
    combined_emb = np.vstack(all_emb)
    combined_scores = np.concatenate(all_scores)
    if len(combined_emb) > target_n:
        top_idx = np.argsort(combined_scores)[-target_n:]
        return combined_emb[top_idx], combined_scores[top_idx], [all_waves[i] for i in top_idx]
    return combined_emb, combined_scores, all_waves


def combine_all_equal_weight(wave_pools, target_n):
    """Pool everything, take top-N by raw score, uniform weight (control)."""
    all_emb, all_scores = [], []
    for w_idx in sorted(wave_pools.keys()):
        emb, scores = wave_pools[w_idx]
        if len(emb) == 0:
            continue
        all_emb.append(emb)
        all_scores.append(scores)
    if not all_emb:
        return np.array([]).reshape(0, 768), np.array([]), []
    all_emb = np.vstack(all_emb)
    all_scores = np.concatenate(all_scores)
    n = min(target_n, len(all_emb))
    top_idx = np.argsort(all_scores)[-n:]
    return all_emb[top_idx], np.ones(n), []


COMBINE_FNS = {
    "accumulate": combine_accumulate,
    "best_wave": combine_best_wave,
    "progressive_replace": combine_progressive_replace,
    "weighted_blend": combine_weighted_blend,
    "all_equal_weight": combine_all_equal_weight,
}


# ============================================================================
# SINGLE-PASS BASELINE (standard approach for comparison)
# ============================================================================

def load_single_pass_cached(dataset_name, class_name, n_shot, model, cascade,
                            real_embeddings, labels_arr):
    """Load cached single-pass generations and apply cascade_l1 top-N with soft weighting."""
    n_generate = int(N_SYNTHETIC_PER_CLASS * 3.0)  # 3x oversample
    cache_key = hashlib.md5(
        f"{dataset_name}_{class_name}_{n_shot}_{n_generate}".encode()
    ).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return np.array([]).reshape(0, 768), np.array([])

    with open(cache_file) as f:
        cached = json.load(f)
    gen_texts = cached.get("texts", [])
    if not gen_texts:
        return np.array([]).reshape(0, 768), np.array([])

    gen_emb = model.encode(gen_texts, show_progress_bar=False)
    class_mask = labels_arr == class_name
    class_embs = real_embeddings[class_mask]
    if len(class_embs) == 0:
        return gen_emb[:N_SYNTHETIC_PER_CLASS], np.ones(min(N_SYNTHETIC_PER_CLASS, len(gen_emb)))

    anchor = class_embs.mean(axis=0)
    scores, _ = cascade.compute_quality_scores(
        gen_emb, anchor, real_embeddings, labels_arr, class_name
    )

    # Top-N selection
    n_select = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
    top_idx = np.argsort(scores)[-n_select:]
    selected_emb = gen_emb[top_idx]
    selected_scores = scores[top_idx]
    weights = normalize_scores(selected_scores)

    return selected_emb, weights


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("PROGRESSIVE WAVES EXPERIMENT")
    print("=" * 70)
    print(f"\nWaves: {N_WAVES}")
    print(f"Candidates/wave: {CANDIDATES_PER_WAVE}")
    print(f"Target/class: {N_SYNTHETIC_PER_CLASS}")
    print(f"Strategies: {COMBINATION_STRATEGIES}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"Seeds: {SEEDS}")
    n_evals = len(DATASETS) * len(COMBINATION_STRATEGIES) * len(CLASSIFIERS) * len(SEEDS)
    print(f"Total evaluations: {n_evals} + baselines")

    print("\nInitializing LLM provider...")
    provider = create_provider("google", "gemini-3-flash-preview")

    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    cascade = FilterCascade(**FILTER_CONFIG)

    all_results = []
    wave_analysis = []  # Per-wave diagnostics
    n_done = 0

    for ds_idx, ds_name in enumerate(DATASETS):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(DATASETS)}] {ds_name}")
        print(f"{'='*60}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(ds_name)
        ds_base = get_dataset_base(ds_name)
        n_shot = parse_n_shot(ds_name)
        n_classes = DATASET_N_CLASSES.get(ds_base, len(set(train_labels)))

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        unique_classes = sorted(set(train_labels))
        labels_arr = np.array(train_labels)

        # ---- PHASE 1: Generate all waves for all classes ----
        print(f"\n  Phase 1: Generating {N_WAVES} waves for {len(unique_classes)} classes...")
        wave_pools_per_class = {}  # cls -> {wave_idx: (emb, scores)}
        wave_meta_per_class = {}

        for cls in unique_classes:
            cls_mask = labels_arr == cls
            cls_emb = train_emb[cls_mask]
            cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]

            wave_pools = {}
            wave_metas = []

            for w_idx, w_config in enumerate(WAVE_CONFIGS):
                threshold, mean_real, std_real = compute_wave_threshold(
                    train_emb, train_labels, cls, cascade, w_config["threshold_k"]
                )
                accepted_emb, accepted_scores, meta = generate_wave(
                    provider, model, cascade,
                    w_idx, w_config, cls, cls_texts, cls_emb,
                    train_emb, train_labels, labels_arr,
                    ds_name, n_shot, threshold, std_real
                )
                wave_pools[w_idx] = (accepted_emb, accepted_scores)
                wave_metas.append(meta)

                status = "OK" if meta["n_accepted"] > 0 else "EMPTY"
                fb = " [FALLBACK]" if meta.get("fallback_applied") else ""
                print(f"    {cls} wave {w_idx} ({w_config['name']}): "
                      f"{meta['n_accepted']}/{meta['n_generated']} accepted "
                      f"({meta['acceptance_rate']*100:.0f}%) {status}{fb}")

            wave_pools_per_class[cls] = wave_pools
            wave_meta_per_class[cls] = wave_metas

        # Save wave analysis
        for cls in unique_classes:
            for meta in wave_meta_per_class[cls]:
                wave_analysis.append({
                    "dataset": ds_name, "class": cls, **meta
                })

        # ---- PHASE 2: Evaluate all combinations ----
        print(f"\n  Phase 2: Evaluating {len(COMBINATION_STRATEGIES)} strategies × "
              f"{len(CLASSIFIERS)} classifiers × {len(SEEDS)} seeds...")

        for seed in SEEDS:
            rng_seed = np.random.RandomState(seed)

            # --- Baseline: no augmentation ---
            for clf_name, clf_factory in CLASSIFIERS.items():
                clf = clf_factory(seed)
                clf.fit(train_emb, train_labels)
                f1_noaug = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                all_results.append({
                    "dataset": ds_name, "dataset_base": ds_base,
                    "n_classes": n_classes, "n_shot": n_shot,
                    "method": "no_augmentation", "strategy": "none",
                    "classifier": clf_name, "seed": seed,
                    "f1": f1_noaug, "n_synthetic": 0,
                })

            # --- Baseline: SMOTE ---
            smote_embs, smote_labels = [], []
            for cls in unique_classes:
                cls_emb = train_emb[labels_arr == cls]
                s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                if len(s) > 0:
                    smote_embs.append(s)
                    smote_labels.extend([cls] * len(s))

            for clf_name, clf_factory in CLASSIFIERS.items():
                if smote_labels:
                    aug_emb = np.vstack([train_emb, np.vstack(smote_embs)])
                    aug_lab = list(train_labels) + smote_labels
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab)
                    f1_smote = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                else:
                    f1_smote = f1_noaug

                all_results.append({
                    "dataset": ds_name, "dataset_base": ds_base,
                    "n_classes": n_classes, "n_shot": n_shot,
                    "method": "smote", "strategy": "none",
                    "classifier": clf_name, "seed": seed,
                    "f1": f1_smote, "n_synthetic": sum(len(e) for e in smote_embs) if smote_embs else 0,
                })

            # --- Baseline: single_pass (standard cascade_l1 top-N + soft weight) ---
            sp_embs, sp_labels, sp_weights = [], [], []
            for cls in unique_classes:
                emb, weights = load_single_pass_cached(
                    ds_name, cls, n_shot, model, cascade, train_emb, labels_arr
                )
                if len(emb) > 0:
                    sp_embs.append(emb)
                    sp_labels.extend([cls] * len(emb))
                    sp_weights.append(weights)

            for clf_name, clf_factory in CLASSIFIERS.items():
                if sp_embs:
                    syn_emb = np.vstack(sp_embs)
                    syn_w = np.concatenate(sp_weights)
                    aug_emb = np.vstack([train_emb, syn_emb])
                    aug_lab = list(train_labels) + sp_labels
                    sw = np.concatenate([np.ones(len(train_emb)), syn_w])
                    clf = clf_factory(seed)
                    clf.fit(aug_emb, aug_lab, sample_weight=sw)
                    f1_sp = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                else:
                    clf = clf_factory(seed)
                    clf.fit(train_emb, train_labels)
                    f1_sp = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                all_results.append({
                    "dataset": ds_name, "dataset_base": ds_base,
                    "n_classes": n_classes, "n_shot": n_shot,
                    "method": "single_pass", "strategy": "none",
                    "classifier": clf_name, "seed": seed,
                    "f1": f1_sp, "n_synthetic": sum(len(e) for e in sp_embs) if sp_embs else 0,
                })

            # --- Progressive Waves: all combination strategies ---
            for strategy_name in COMBINATION_STRATEGIES:
                combined_emb_all, combined_labels, combined_weights = [], [], []
                wave_provenance_all = []

                for cls in unique_classes:
                    emb, scores, wave_prov = COMBINE_FNS[strategy_name](
                        wave_pools_per_class[cls], N_SYNTHETIC_PER_CLASS
                    )
                    if len(emb) > 0:
                        if strategy_name == "all_equal_weight":
                            weights = np.ones(len(emb))
                        else:
                            weights = normalize_scores(scores)
                        combined_emb_all.append(emb)
                        combined_labels.extend([cls] * len(emb))
                        combined_weights.append(weights)
                        wave_provenance_all.extend(wave_prov)

                for clf_name, clf_factory in CLASSIFIERS.items():
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"    Progress: {n_done}/{n_evals} ({100*n_done/n_evals:.0f}%)")

                    if combined_emb_all:
                        syn_emb = np.vstack(combined_emb_all)
                        syn_w = np.concatenate(combined_weights)
                        aug_emb = np.vstack([train_emb, syn_emb])
                        aug_lab = list(train_labels) + combined_labels
                        sw = np.concatenate([np.ones(len(train_emb)), syn_w])
                        clf = clf_factory(seed)
                        clf.fit(aug_emb, aug_lab, sample_weight=sw)
                        f1_pw = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))
                    else:
                        clf = clf_factory(seed)
                        clf.fit(train_emb, train_labels)
                        f1_pw = float(f1_score(test_labels, clf.predict(test_emb), average="macro"))

                    # Count wave provenance
                    prov_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                    for w in wave_provenance_all:
                        prov_counts[w] = prov_counts.get(w, 0) + 1

                    all_results.append({
                        "dataset": ds_name, "dataset_base": ds_base,
                        "n_classes": n_classes, "n_shot": n_shot,
                        "method": "progressive_waves", "strategy": strategy_name,
                        "classifier": clf_name, "seed": seed,
                        "f1": f1_pw,
                        "n_synthetic": len(combined_labels) if combined_emb_all else 0,
                        "wave_provenance": prov_counts,
                    })

        # Incremental save
        if (ds_idx + 1) % 3 == 0:
            partial_path = RESULTS_DIR / "partial_results.json"
            with open(partial_path, "w") as f:
                json.dump({"results": all_results, "wave_analysis": wave_analysis}, f, indent=2)
            print(f"  Partial save: {len(all_results)} results")

    # ---- Save final results ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_synthetic_per_class": N_SYNTHETIC_PER_CLASS,
            "candidates_per_wave": CANDIDATES_PER_WAVE,
            "n_waves": N_WAVES,
            "seeds": SEEDS,
            "filter": FILTER_CONFIG,
            "wave_configs": [{k: v for k, v in wc.items() if k != "prompt_injection"}
                            for wc in WAVE_CONFIGS],
            "combination_strategies": COMBINATION_STRATEGIES,
        },
        "n_results": len(all_results),
        "results": all_results,
        "wave_analysis": wave_analysis,
    }

    out_path = RESULTS_DIR / "progressive_waves_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total results: {len(all_results)}")

    # Generate report
    generate_report(all_results)
    generate_wave_analysis_table(wave_analysis)
    generate_figures(all_results, wave_analysis)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results):
    """Statistical comparison of strategies vs SMOTE."""
    print("\n" + "=" * 80)
    print("PROGRESSIVE WAVES — STATISTICAL REPORT")
    print("=" * 80)

    # Collect per-(dataset, classifier, seed) F1s
    smote_f1s = {}
    method_f1s = defaultdict(dict)

    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "smote":
            smote_f1s[key] = r["f1"]
        elif r["method"] == "progressive_waves":
            method_f1s[r["strategy"]][key] = r["f1"]
        elif r["method"] == "single_pass":
            method_f1s["single_pass"][key] = r["f1"]
        elif r["method"] == "no_augmentation":
            method_f1s["no_augmentation"][key] = r["f1"]

    n_comparisons = len(COMBINATION_STRATEGIES) + 1  # strategies + single_pass

    # Header
    print(f"\n{'Method':<25} {'Delta vs SMOTE':>15} {'Win Rate':>10} {'p-value':>10} {'Bonf. p':>10} {'d':>8}")
    print("-" * 80)

    all_stats = {}
    for method_name in ["single_pass"] + COMBINATION_STRATEGIES:
        if method_name not in method_f1s:
            continue
        common_keys = sorted(set(smote_f1s.keys()) & set(method_f1s[method_name].keys()))
        if not common_keys:
            continue
        m_f1 = [method_f1s[method_name][k] for k in common_keys]
        s_f1 = [smote_f1s[k] for k in common_keys]
        st = compute_paired_statistics(m_f1, s_f1, n_comparisons)
        all_stats[method_name] = st

        sig = "***" if st["significant_bonferroni"] else ("*" if st["significant_005"] else "")
        print(f"{method_name:<25} {st['delta_mean_pp']:>+10.2f}pp    {st['win_rate']:>8.1%} "
              f"   {st['p_value']:>10.4f} {st['bonferroni_p']:>10.4f} {st['cohen_d']:>8.2f} {sig}")

    # Per-dataset breakdown for best strategy
    if all_stats:
        best_strategy = max(all_stats, key=lambda k: all_stats[k]["delta_mean_pp"])
        print(f"\nBest strategy: {best_strategy} ({all_stats[best_strategy]['delta_mean_pp']:+.2f}pp)")

        print(f"\n{'Dataset':<35} {'Delta':>8} {'Win%':>8}")
        print("-" * 55)

        ds_deltas = defaultdict(list)
        for r in results:
            if r["method"] == "progressive_waves" and r["strategy"] == best_strategy:
                key = (r["dataset"], r["classifier"], r["seed"])
                if key in smote_f1s:
                    ds_deltas[r["dataset"]].append(r["f1"] - smote_f1s[key])

        for ds in sorted(ds_deltas.keys()):
            d = np.array(ds_deltas[ds])
            print(f"  {ds:<33} {np.mean(d)*100:>+8.2f}pp {np.mean(d > 0):>7.1%}")

    # Per n-shot breakdown
    print(f"\n{'N-shot':<10} ", end="")
    for sname in ["single_pass"] + COMBINATION_STRATEGIES:
        print(f"{sname:<18}", end="")
    print()
    print("-" * (10 + 18 * (1 + len(COMBINATION_STRATEGIES))))

    for nshot in [10, 25, 50]:
        print(f"  {nshot:<8} ", end="")
        for method_name in ["single_pass"] + COMBINATION_STRATEGIES:
            if method_name not in method_f1s:
                print(f"{'N/A':<18}", end="")
                continue
            shot_keys = [k for k in smote_f1s if str(nshot) + "shot" in k[0]]
            common = sorted(set(shot_keys) & set(method_f1s[method_name].keys()))
            if common:
                deltas = [method_f1s[method_name][k] - smote_f1s[k] for k in common]
                print(f"{np.mean(deltas)*100:>+6.2f}pp ({np.mean(np.array(deltas)>0)*100:3.0f}%) ", end="")
            else:
                print(f"{'N/A':<18}", end="")
        print()

    # Save stats
    stats_path = RESULTS_DIR / "statistical_report.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


def generate_wave_analysis_table(wave_analysis):
    """Generate per-wave analysis table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate by wave
    wave_stats = defaultdict(lambda: {"n_gen": [], "n_acc": [], "acc_rate": [],
                                       "mean_score": [], "fallback": 0, "total": 0})
    for wa in wave_analysis:
        w = wa["wave_index"]
        wave_stats[w]["n_gen"].append(wa["n_generated"])
        wave_stats[w]["n_acc"].append(wa["n_accepted"])
        wave_stats[w]["acc_rate"].append(wa["acceptance_rate"])
        wave_stats[w]["mean_score"].append(wa["mean_score_accepted"])
        wave_stats[w]["total"] += 1
        if wa.get("fallback_applied"):
            wave_stats[w]["fallback"] += 1

    print("\n" + "-" * 70)
    print("WAVE ANALYSIS")
    print("-" * 70)
    print(f"{'Wave':<20} {'Gen':>6} {'Acc':>6} {'Rate':>8} {'Score':>8} {'Fallback':>10}")
    print("-" * 70)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{An\'alisis por oleada: tasa de aceptaci\'on y calidad media "
                 r"de las muestras aceptadas. El umbral se deriva de los scores de datos reales.}")
    lines.append(r"\label{tab:progressive_waves_per_wave}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Oleada} & \textbf{Nombre} & \textbf{Generados} & "
                 r"\textbf{Aceptados} & \textbf{Tasa (\%)} & \textbf{Score medio} \\")
    lines.append(r"\midrule")

    for w_idx in range(N_WAVES):
        ws = wave_stats[w_idx]
        name = WAVE_CONFIGS[w_idx]["name"]
        avg_gen = np.mean(ws["n_gen"])
        avg_acc = np.mean(ws["n_acc"])
        avg_rate = np.mean(ws["acc_rate"]) * 100
        avg_score = np.mean([s for s in ws["mean_score"] if s > 0]) if any(s > 0 for s in ws["mean_score"]) else 0
        fb_pct = ws["fallback"] / ws["total"] * 100 if ws["total"] > 0 else 0

        print(f"  {w_idx} ({name:<14}) {avg_gen:>6.0f} {avg_acc:>6.1f} {avg_rate:>7.1f}% "
              f"{avg_score:>8.3f} {fb_pct:>9.1f}%")
        lines.append(f"{w_idx} & {name} & {avg_gen:.0f} & {avg_acc:.1f} & "
                     f"{avg_rate:.1f} & {avg_score:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    path = OUTPUT_DIR / "tab_progressive_waves_per_wave.tex"
    path.write_text(table)
    print(f"\nTable saved: {path}")


def generate_figures(results, wave_analysis):
    """Generate visualization figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Wave contribution stacked bar ----
    # Count wave provenance across all strategies
    strategy_provenance = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    for r in results:
        if r["method"] == "progressive_waves" and "wave_provenance" in r:
            for w_str, count in r["wave_provenance"].items():
                w = int(w_str) if isinstance(w_str, str) else w_str
                strategy_provenance[r["strategy"]][w] += count

    if strategy_provenance:
        fig, ax = plt.subplots(figsize=(10, 5))
        strategies = list(strategy_provenance.keys())
        x = np.arange(len(strategies))
        width = 0.6

        wave_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
        wave_names = [wc["name"] for wc in WAVE_CONFIGS]

        bottoms = np.zeros(len(strategies))
        for w_idx in range(N_WAVES):
            vals = []
            for s in strategies:
                total = sum(strategy_provenance[s].values())
                vals.append(strategy_provenance[s][w_idx] / total * 100 if total > 0 else 0)
            ax.bar(x, vals, width, bottom=bottoms, label=f"Oleada {w_idx}: {wave_names[w_idx]}",
                   color=wave_colors[w_idx])
            bottoms += np.array(vals)

        ax.set_ylabel("Contribucion al pool final (%)", fontsize=11)
        ax.set_xlabel("Estrategia de combinacion", fontsize=11)
        ax.set_title("Contribucion de cada oleada al pool final", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        path = FIGURE_DIR / "fig_wave_contribution.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {path}")

    # ---- Figure 2: Strategy comparison bar chart ----
    smote_f1s = {}
    method_f1s = defaultdict(dict)
    for r in results:
        key = (r["dataset"], r["classifier"], r["seed"])
        if r["method"] == "smote":
            smote_f1s[key] = r["f1"]
        elif r["method"] == "progressive_waves":
            method_f1s[r["strategy"]][key] = r["f1"]
        elif r["method"] == "single_pass":
            method_f1s["single_pass"][key] = r["f1"]

    methods_to_plot = ["single_pass"] + COMBINATION_STRATEGIES
    deltas = []
    errs = []
    labels = []

    for m in methods_to_plot:
        if m not in method_f1s:
            continue
        common = sorted(set(smote_f1s.keys()) & set(method_f1s[m].keys()))
        if not common:
            continue
        d = [method_f1s[m][k] - smote_f1s[k] for k in common]
        deltas.append(np.mean(d) * 100)
        errs.append(np.std(d) / np.sqrt(len(d)) * 100)
        labels.append(m)

    if deltas:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#888888"] + ["#2196F3"] * len(COMBINATION_STRATEGIES)
        x = np.arange(len(labels))
        bars = ax.bar(x, deltas, yerr=errs, capsize=4, color=colors[:len(labels)], edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, label="SMOTE baseline")
        ax.set_ylabel("Delta F1 vs SMOTE (pp)", fontsize=11)
        ax.set_xlabel("Metodo", fontsize=11)
        ax.set_title("Oleadas Progresivas: Comparacion de Estrategias", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:+.2f}", ha="center", va="bottom", fontsize=8)

        path = FIGURE_DIR / "fig_progressive_waves_comparison.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {path}")

    # ---- Figure 3: Per-wave acceptance rate and quality ----
    wave_rates = defaultdict(list)
    wave_scores = defaultdict(list)
    for wa in wave_analysis:
        wave_rates[wa["wave_index"]].append(wa["acceptance_rate"])
        if wa["mean_score_accepted"] > 0:
            wave_scores[wa["wave_index"]].append(wa["mean_score_accepted"])

    if wave_rates:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        waves_x = sorted(wave_rates.keys())
        rates_mean = [np.mean(wave_rates[w]) * 100 for w in waves_x]
        rates_se = [np.std(wave_rates[w]) / np.sqrt(len(wave_rates[w])) * 100 for w in waves_x]
        wave_labels = [WAVE_CONFIGS[w]["name"] for w in waves_x]

        ax1.bar(waves_x, rates_mean, yerr=rates_se, capsize=4,
                color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
        ax1.set_xlabel("Oleada", fontsize=11)
        ax1.set_ylabel("Tasa de aceptacion (%)", fontsize=11)
        ax1.set_title("Tasa de aceptacion por oleada", fontsize=12)
        ax1.set_xticks(waves_x)
        ax1.set_xticklabels(wave_labels, fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        scores_mean = [np.mean(wave_scores[w]) if wave_scores[w] else 0 for w in waves_x]
        scores_se = [np.std(wave_scores[w]) / np.sqrt(len(wave_scores[w])) if wave_scores[w] else 0 for w in waves_x]

        ax2.bar(waves_x, scores_mean, yerr=scores_se, capsize=4,
                color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
        ax2.set_xlabel("Oleada", fontsize=11)
        ax2.set_ylabel("Score medio de aceptados", fontsize=11)
        ax2.set_title("Calidad media por oleada", fontsize=12)
        ax2.set_xticks(waves_x)
        ax2.set_xticklabels(wave_labels, fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path = FIGURE_DIR / "fig_wave_diagnostics.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
