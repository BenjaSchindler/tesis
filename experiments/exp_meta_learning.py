#!/usr/bin/env python3
"""
Meta-Learning Selector Experiment

Predicts the optimal filter configuration from dataset characteristics.
Uses 1210 completed experiments as training data.

Three strategies tested with Leave-One-Dataset-Out (LODO) evaluation:
A) Per-instance regression (meta-features + config-features -> delta_vs_smote)
B) Ranking-based portfolio (average rank across training datasets)
C) Rule-based decision tree (shallow interpretable rules)

Two baselines:
1) Always cascade level=1, llm_pct=100, n_shot=10 (current best static rec)
2) Random config selection (expected performance)

Evaluation:
- Leave-One-Dataset-Out (LODO) cross-validation (5 folds, excluding incomplete dataset)
- Regret analysis: how much worse than the oracle (best config per dataset)
- Top-K accuracy: is the oracle config in the top K predicted configs?
- Rank correlation (Spearman rho)
- Feature importance analysis (GBR on all data)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "meta_learning"
EXPERIMENT_RESULTS_PATH = PROJECT_ROOT / "results" / "filter_comparison" / "final_results.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets with full experiment coverage (240 each)
DATASETS = [
    "20newsgroups_10shot",
    "20newsgroups_25shot",
    "sms_spam_10shot",
    "sms_spam_25shot",
    "hate_speech_davidson_10shot",
    # hate_speech_davidson_25shot excluded: only 10/240 experiments
]

MIN_EXPERIMENTS_PER_DATASET = 100


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class DatasetMetaFeatures:
    """Meta-features computed from a single dataset."""
    dataset_name: str
    # Basic (4)
    n_train: int
    n_classes: int
    samples_per_class: float
    imbalance_ratio: float
    # Text (5)
    avg_text_length: float
    text_length_std: float
    vocab_size: int
    type_token_ratio: float
    avg_text_length_chars: float
    # Embedding-based (5)
    avg_centroid_distance: float
    avg_intra_class_distance: float
    class_separation_ratio: float
    embedding_spread: float
    max_class_overlap: float
    # Density (2)
    avg_knn_distance_k5: float
    density_variance: float
    # Context (1)
    n_shot: int

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.n_train,
            self.n_classes,
            self.samples_per_class,
            self.imbalance_ratio,
            self.avg_text_length,
            self.text_length_std,
            self.vocab_size,
            self.type_token_ratio,
            self.avg_text_length_chars,
            self.avg_centroid_distance,
            self.avg_intra_class_distance,
            self.class_separation_ratio,
            self.embedding_spread,
            self.max_class_overlap,
            self.avg_knn_distance_k5,
            self.density_variance,
            self.n_shot,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "n_train", "n_classes", "samples_per_class", "imbalance_ratio",
            "avg_text_length", "text_length_std", "vocab_size", "type_token_ratio",
            "avg_text_length_chars",
            "avg_centroid_distance", "avg_intra_class_distance",
            "class_separation_ratio", "embedding_spread", "max_class_overlap",
            "avg_knn_distance_k5", "density_variance", "n_shot",
        ]


@dataclass
class ConfigFeatures:
    """Features encoding a filter configuration."""
    is_none: int
    is_lof: int
    is_cascade: int
    is_combined: int
    is_embedding_guided: int
    llm_pct: float
    n_shot_config: float
    lof_n_neighbors: float
    lof_threshold: float
    cascade_filter_level: float
    combined_sim_threshold: float
    eg_coverage_weight: float

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.is_none, self.is_lof, self.is_cascade,
            self.is_combined, self.is_embedding_guided,
            self.llm_pct, self.n_shot_config,
            self.lof_n_neighbors, self.lof_threshold,
            self.cascade_filter_level, self.combined_sim_threshold,
            self.eg_coverage_weight,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "is_none", "is_lof", "is_cascade", "is_combined", "is_embedding_guided",
            "llm_pct", "n_shot_config",
            "lof_n_neighbors", "lof_threshold",
            "cascade_filter_level", "combined_sim_threshold",
            "eg_coverage_weight",
        ]


@dataclass
class LODOResult:
    """Result of one LODO fold."""
    held_out_dataset: str
    strategy: str
    predicted_best_config: Dict[str, Any]
    predicted_delta_vs_smote: float
    actual_best_config: Dict[str, Any]
    actual_best_delta: float
    actual_delta_of_prediction: float
    regret: float
    regret_pct: float
    top_k_accuracy: Dict[int, bool]
    rank_correlation: float


# ==============================================================================
# Step 1: Meta-Feature Extraction
# ==============================================================================

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a benchmark dataset from JSON."""
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def extract_text_features(texts: List[str]) -> Dict[str, float]:
    """Extract text-based meta-features."""
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return {"avg_text_length": 0, "text_length_std": 0,
                "avg_text_length_chars": 0, "vocab_size": 0, "type_token_ratio": 0}

    word_lengths = [len(t.split()) for t in valid_texts]
    char_lengths = [len(t) for t in valid_texts]

    all_tokens = []
    for t in valid_texts:
        all_tokens.extend(t.lower().split())

    vocab = set(all_tokens)
    total_tokens = len(all_tokens)

    return {
        "avg_text_length": float(np.mean(word_lengths)),
        "text_length_std": float(np.std(word_lengths)),
        "avg_text_length_chars": float(np.mean(char_lengths)),
        "vocab_size": len(vocab),
        "type_token_ratio": len(vocab) / max(1, total_tokens),
    }


def extract_embedding_features(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Extract embedding-based meta-features."""
    unique_classes = np.unique(labels)

    # Class centroids and intra-class distances
    centroids = []
    intra_distances = []
    for cls in unique_classes:
        cls_embs = embeddings[labels == cls]
        centroid = cls_embs.mean(axis=0)
        centroids.append(centroid)
        dists = np.linalg.norm(cls_embs - centroid, axis=1)
        intra_distances.append(dists.mean())

    centroids = np.array(centroids)
    avg_intra = float(np.mean(intra_distances))

    # Inter-class distances
    if len(centroids) > 1:
        centroid_dists = cdist(centroids, centroids, metric="euclidean")
        upper = centroid_dists[np.triu_indices_from(centroid_dists, k=1)]
        avg_inter = float(upper.mean())
        min_inter = float(upper.min())
    else:
        avg_inter = 0.0
        min_inter = 0.0

    # Embedding spread
    embedding_spread = float(np.mean(np.std(embeddings, axis=0)))

    # KNN density
    k = min(5, len(embeddings) - 1)
    if k > 0:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embeddings)
        knn_dists, _ = nn.kneighbors(embeddings)
        avg_knn_dist = float(knn_dists.mean())
        density_variance = float(knn_dists.mean(axis=1).var())
    else:
        avg_knn_dist = 0.0
        density_variance = 0.0

    separation_ratio = avg_inter / max(avg_intra, 1e-6)

    return {
        "avg_centroid_distance": avg_inter,
        "avg_intra_class_distance": avg_intra,
        "class_separation_ratio": separation_ratio,
        "embedding_spread": embedding_spread,
        "max_class_overlap": min_inter,
        "avg_knn_distance_k5": avg_knn_dist,
        "density_variance": density_variance,
    }


def compute_all_meta_features(model: SentenceTransformer) -> Dict[str, DatasetMetaFeatures]:
    """Compute meta-features for all datasets."""
    meta_features = {}

    for dataset_name in DATASETS:
        print(f"  Computing meta-features for {dataset_name}...")
        train_texts, train_labels, _, _ = load_dataset(dataset_name)
        labels_arr = np.array(train_labels)

        # Basic
        n_train = len(train_texts)
        class_counts = Counter(train_labels)
        n_classes = len(class_counts)
        samples_per_class = n_train / n_classes
        imbalance_ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)

        # Text
        text_feats = extract_text_features(train_texts)

        # Embeddings
        valid_texts = [t if t.strip() else "empty" for t in train_texts]
        embeddings = model.encode(valid_texts, show_progress_bar=False)
        emb_feats = extract_embedding_features(embeddings, labels_arr)

        # N-shot from name
        n_shot = 10 if "10shot" in dataset_name else 25

        meta_features[dataset_name] = DatasetMetaFeatures(
            dataset_name=dataset_name,
            n_train=n_train,
            n_classes=n_classes,
            samples_per_class=samples_per_class,
            imbalance_ratio=imbalance_ratio,
            avg_text_length=text_feats["avg_text_length"],
            text_length_std=text_feats["text_length_std"],
            vocab_size=text_feats["vocab_size"],
            type_token_ratio=text_feats["type_token_ratio"],
            avg_text_length_chars=text_feats["avg_text_length_chars"],
            avg_centroid_distance=emb_feats["avg_centroid_distance"],
            avg_intra_class_distance=emb_feats["avg_intra_class_distance"],
            class_separation_ratio=emb_feats["class_separation_ratio"],
            embedding_spread=emb_feats["embedding_spread"],
            max_class_overlap=emb_feats["max_class_overlap"],
            avg_knn_distance_k5=emb_feats["avg_knn_distance_k5"],
            density_variance=emb_feats["density_variance"],
            n_shot=n_shot,
        )

    return meta_features


# ==============================================================================
# Step 2: Config Feature Encoding
# ==============================================================================

def encode_config(experiment: Dict) -> ConfigFeatures:
    """Encode an experiment's config as numeric features."""
    ft = experiment["filter_type"]
    params = experiment.get("filter_params", {})

    return ConfigFeatures(
        is_none=int(ft == "none"),
        is_lof=int(ft == "lof"),
        is_cascade=int(ft == "cascade"),
        is_combined=int(ft == "combined"),
        is_embedding_guided=int(ft == "embedding_guided"),
        llm_pct=experiment["llm_pct"] / 100.0,
        n_shot_config=experiment["n_shot"] / 50.0,
        lof_n_neighbors=params.get("n_neighbors", 0) / 20.0,
        lof_threshold=params.get("threshold", 0),
        cascade_filter_level=params.get("filter_level", 0) / 4.0,
        combined_sim_threshold=params.get("sim_threshold", 0),
        eg_coverage_weight=params.get("coverage_weight", 0),
    )


def config_to_key(experiment: Dict) -> str:
    """Create a unique string key for a configuration."""
    return json.dumps({
        "filter_type": experiment["filter_type"],
        "filter_params": experiment.get("filter_params", {}),
        "llm_pct": experiment["llm_pct"],
        "n_shot": experiment["n_shot"],
    }, sort_keys=True)


def config_summary(experiment: Dict) -> Dict[str, Any]:
    """Extract config dict for reporting."""
    return {
        "filter_type": experiment["filter_type"],
        "filter_params": experiment.get("filter_params", {}),
        "llm_pct": experiment["llm_pct"],
        "n_shot": experiment["n_shot"],
    }


# ==============================================================================
# Step 3A: Strategy A — Per-Instance Regression
# ==============================================================================

def strategy_regression_lodo(
    experiments: List[Dict],
    meta_features: Dict[str, DatasetMetaFeatures],
    datasets: List[str],
) -> List[LODOResult]:
    """Train regression on (meta_features || config_features) -> delta_vs_smote."""
    results = []

    for held_out in datasets:
        train_exps = [e for e in experiments if e["dataset"] != held_out and e["dataset"] in meta_features]
        test_exps = [e for e in experiments if e["dataset"] == held_out]
        if not test_exps:
            continue

        # Build feature matrices
        X_train, y_train = [], []
        for exp in train_exps:
            meta_vec = meta_features[exp["dataset"]].to_feature_vector()
            config_vec = encode_config(exp).to_feature_vector()
            X_train.append(np.concatenate([meta_vec, config_vec]))
            y_train.append(exp["delta_vs_smote"])

        X_test, y_test, test_configs = [], [], []
        for exp in test_exps:
            if held_out not in meta_features:
                continue
            meta_vec = meta_features[held_out].to_feature_vector()
            config_vec = encode_config(exp).to_feature_vector()
            X_test.append(np.concatenate([meta_vec, config_vec]))
            y_test.append(exp["delta_vs_smote"])
            test_configs.append(exp)

        if not X_test:
            continue

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Normalize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Try multiple regressors, pick best on training MAE
        best_model, best_name = None, ""
        best_score = -float("inf")

        regressors = [
            ("ridge_a1", Ridge(alpha=1.0)),
            ("ridge_a10", Ridge(alpha=10.0)),
            ("ridge_a100", Ridge(alpha=100.0)),
            ("lasso_a01", Lasso(alpha=0.1, max_iter=5000)),
            ("rf_shallow", RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)),
            ("gbr_shallow", GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)),
        ]

        for name, model in regressors:
            model.fit(X_train_s, y_train)
            score = -mean_absolute_error(y_train, model.predict(X_train_s))
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        # Predict
        y_pred = best_model.predict(X_test_s)
        pred_best_idx = np.argmax(y_pred)
        actual_best_idx = np.argmax(y_test)

        actual_delta_of_pred = y_test[pred_best_idx]
        actual_best_delta = y_test[actual_best_idx]
        regret = actual_best_delta - actual_delta_of_pred

        # Top-K accuracy
        pred_ranking = np.argsort(y_pred)[::-1]
        actual_best_key = config_to_key(test_configs[actual_best_idx])
        top_k_acc = {}
        for k in [1, 3, 5, 10, 20]:
            top_k_keys = {config_to_key(test_configs[i]) for i in pred_ranking[:k]}
            top_k_acc[k] = actual_best_key in top_k_keys

        # Rank correlation
        rho, _ = spearmanr(y_pred, y_test)

        results.append(LODOResult(
            held_out_dataset=held_out,
            strategy=f"regression_{best_name}",
            predicted_best_config=config_summary(test_configs[pred_best_idx]),
            predicted_delta_vs_smote=float(y_pred[pred_best_idx]),
            actual_best_config=config_summary(test_configs[actual_best_idx]),
            actual_best_delta=float(actual_best_delta),
            actual_delta_of_prediction=float(actual_delta_of_pred),
            regret=float(regret),
            regret_pct=float(regret / max(abs(actual_best_delta), 0.01) * 100),
            top_k_accuracy=top_k_acc,
            rank_correlation=float(rho) if not np.isnan(rho) else 0.0,
        ))

    return results


# ==============================================================================
# Step 3B: Strategy B — Average-Rank Portfolio
# ==============================================================================

def strategy_portfolio_lodo(
    experiments: List[Dict],
    datasets: List[str],
) -> List[LODOResult]:
    """Recommend config with best average rank across training datasets."""
    results = []

    # Build per-dataset scores
    dataset_scores = defaultdict(dict)
    config_key_to_exp = {}
    for exp in experiments:
        if exp["dataset"] not in datasets:
            continue
        key = config_to_key(exp)
        dataset_scores[exp["dataset"]][key] = exp["delta_vs_smote"]
        config_key_to_exp[key] = exp

    for held_out in datasets:
        train_ds = [d for d in datasets if d != held_out]

        # Rank configs per training dataset
        config_ranks = defaultdict(list)
        for ds in train_ds:
            scores = dataset_scores.get(ds, {})
            if not scores:
                continue
            sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            for rank, key in enumerate(sorted_keys, 1):
                config_ranks[key].append(rank)

        # Average rank (require tested on at least half of training datasets)
        min_required = max(1, len(train_ds) // 2)
        avg_ranks = {k: np.mean(v) for k, v in config_ranks.items() if len(v) >= min_required}
        if not avg_ranks:
            continue

        best_key = min(avg_ranks, key=avg_ranks.get)
        predicted_exp = config_key_to_exp[best_key]

        # Evaluate on held-out
        held_scores = dataset_scores.get(held_out, {})
        if not held_scores or best_key not in held_scores:
            continue

        actual_best_key = max(held_scores, key=held_scores.get)
        actual_best_delta = held_scores[actual_best_key]
        predicted_delta = held_scores[best_key]
        regret = actual_best_delta - predicted_delta

        # Top-K
        pred_sorted = sorted(avg_ranks.keys(), key=lambda k: avg_ranks[k])
        top_k_acc = {}
        for k in [1, 3, 5, 10, 20]:
            top_k_acc[k] = actual_best_key in set(pred_sorted[:k])

        # Rank correlation
        common_keys = sorted(set(avg_ranks.keys()) & set(held_scores.keys()))
        if len(common_keys) > 2:
            pred_order = [avg_ranks[k] for k in common_keys]
            actual_order = [-held_scores[k] for k in common_keys]
            rho, _ = spearmanr(pred_order, actual_order)
        else:
            rho = 0.0

        results.append(LODOResult(
            held_out_dataset=held_out,
            strategy="portfolio_avg_rank",
            predicted_best_config=config_summary(predicted_exp),
            predicted_delta_vs_smote=float(predicted_delta),
            actual_best_config=config_summary(config_key_to_exp[actual_best_key]),
            actual_best_delta=float(actual_best_delta),
            actual_delta_of_prediction=float(predicted_delta),
            regret=float(regret),
            regret_pct=float(regret / max(abs(actual_best_delta), 0.01) * 100),
            top_k_accuracy=top_k_acc,
            rank_correlation=float(rho) if not np.isnan(rho) else 0.0,
        ))

    return results


# ==============================================================================
# Step 3C: Strategy C — Rule-Based Decision Tree
# ==============================================================================

def strategy_rule_based_lodo(
    experiments: List[Dict],
    meta_features: Dict[str, DatasetMetaFeatures],
    datasets: List[str],
) -> Tuple[List[LODOResult], List[str]]:
    """Shallow decision tree on meta-features -> best filter type, then portfolio for params."""
    results = []
    tree_rules_log = []

    # Pre-compute best filter type per dataset
    dataset_best_filter = {}
    for ds in datasets:
        ds_exps = [e for e in experiments if e["dataset"] == ds]
        if ds_exps:
            best = max(ds_exps, key=lambda e: e["delta_vs_smote"])
            dataset_best_filter[ds] = best["filter_type"]

    # Build config key lookup
    config_key_to_exp = {}
    for exp in experiments:
        config_key_to_exp[config_to_key(exp)] = exp

    for held_out in datasets:
        train_ds = [d for d in datasets if d != held_out and d in meta_features and d in dataset_best_filter]
        if len(train_ds) < 2:
            continue

        # Train: meta_features -> best_filter_type
        X_tree = np.array([meta_features[d].to_feature_vector() for d in train_ds])
        y_labels = [dataset_best_filter[d] for d in train_ds]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)

        tree = DecisionTreeClassifier(max_depth=2, random_state=42)
        tree.fit(X_tree, y_encoded)

        # Predict filter type
        X_held = meta_features[held_out].to_feature_vector().reshape(1, -1)
        pred_encoded = tree.predict(X_held)[0]
        predicted_filter = le.inverse_transform([pred_encoded])[0]

        # Log tree rules
        rules = export_text(tree, feature_names=DatasetMetaFeatures.feature_names(), max_depth=2)
        tree_rules_log.append(f"--- LODO fold: held_out={held_out} ---\n{rules}\n"
                              f"Predicted filter: {predicted_filter}\n"
                              f"Actual best filter: {dataset_best_filter.get(held_out, '?')}\n")

        # Portfolio within predicted filter type
        config_ranks = defaultdict(list)
        for ds in train_ds:
            ds_exps = sorted(
                [e for e in experiments if e["dataset"] == ds and e["filter_type"] == predicted_filter],
                key=lambda e: e["delta_vs_smote"], reverse=True,
            )
            for rank, exp in enumerate(ds_exps, 1):
                config_ranks[config_to_key(exp)].append(rank)

        if not config_ranks:
            # Fallback: use all filter types with portfolio
            for ds in train_ds:
                ds_exps = sorted(
                    [e for e in experiments if e["dataset"] == ds],
                    key=lambda e: e["delta_vs_smote"], reverse=True,
                )
                for rank, exp in enumerate(ds_exps, 1):
                    config_ranks[config_to_key(exp)].append(rank)

        avg_ranks = {k: np.mean(v) for k, v in config_ranks.items()}
        best_key = min(avg_ranks, key=avg_ranks.get)
        predicted_exp = config_key_to_exp.get(best_key)
        if not predicted_exp:
            continue

        # Evaluate
        held_exps = [e for e in experiments if e["dataset"] == held_out]
        if not held_exps:
            continue
        actual_best = max(held_exps, key=lambda e: e["delta_vs_smote"])

        held_scores = {config_to_key(e): e["delta_vs_smote"] for e in held_exps}
        predicted_delta = held_scores.get(best_key, 0.0)
        regret = actual_best["delta_vs_smote"] - predicted_delta

        # Top-K
        pred_sorted = sorted(avg_ranks.keys(), key=lambda k: avg_ranks[k])
        actual_best_key = config_to_key(actual_best)
        top_k_acc = {}
        for k in [1, 3, 5, 10, 20]:
            top_k_acc[k] = actual_best_key in set(pred_sorted[:k])

        # Rank correlation
        common_keys = sorted(set(avg_ranks.keys()) & set(held_scores.keys()))
        if len(common_keys) > 2:
            pred_order = [avg_ranks[k] for k in common_keys]
            actual_order = [-held_scores[k] for k in common_keys]
            rho, _ = spearmanr(pred_order, actual_order)
        else:
            rho = 0.0

        results.append(LODOResult(
            held_out_dataset=held_out,
            strategy=f"rule_based_{predicted_filter}",
            predicted_best_config=config_summary(predicted_exp),
            predicted_delta_vs_smote=float(predicted_delta),
            actual_best_config=config_summary(actual_best),
            actual_best_delta=float(actual_best["delta_vs_smote"]),
            actual_delta_of_prediction=float(predicted_delta),
            regret=float(regret),
            regret_pct=float(regret / max(abs(actual_best["delta_vs_smote"]), 0.01) * 100),
            top_k_accuracy=top_k_acc,
            rank_correlation=float(rho) if not np.isnan(rho) else 0.0,
        ))

    return results, tree_rules_log


# ==============================================================================
# Step 3 Baselines
# ==============================================================================

def baseline_always_cascade_l1(
    experiments: List[Dict],
    datasets: List[str],
) -> List[LODOResult]:
    """Baseline: always recommend cascade level=1, llm_pct=100, n_shot=10."""
    fixed_config = {
        "filter_type": "cascade",
        "filter_params": {"filter_level": 1, "k_neighbors": 10},
        "llm_pct": 100,
        "n_shot": 10,
    }
    fixed_key = json.dumps(fixed_config, sort_keys=True)

    results = []
    for ds in datasets:
        ds_exps = [e for e in experiments if e["dataset"] == ds]
        if not ds_exps:
            continue

        scores = {config_to_key(e): e["delta_vs_smote"] for e in ds_exps}
        actual_best_key = max(scores, key=scores.get)
        actual_best = max(ds_exps, key=lambda e: e["delta_vs_smote"])

        predicted_delta = scores.get(fixed_key, 0.0)
        regret = actual_best["delta_vs_smote"] - predicted_delta

        top_k_acc = {k: (actual_best_key == fixed_key) for k in [1, 3, 5, 10, 20]}

        results.append(LODOResult(
            held_out_dataset=ds,
            strategy="baseline_always_cascade_l1",
            predicted_best_config=fixed_config,
            predicted_delta_vs_smote=float(predicted_delta),
            actual_best_config=config_summary(actual_best),
            actual_best_delta=float(actual_best["delta_vs_smote"]),
            actual_delta_of_prediction=float(predicted_delta),
            regret=float(regret),
            regret_pct=float(regret / max(abs(actual_best["delta_vs_smote"]), 0.01) * 100),
            top_k_accuracy=top_k_acc,
            rank_correlation=0.0,
        ))

    return results


def baseline_random(
    experiments: List[Dict],
    datasets: List[str],
) -> List[LODOResult]:
    """Baseline: expected performance of random config selection."""
    results = []
    for ds in datasets:
        ds_exps = [e for e in experiments if e["dataset"] == ds]
        if not ds_exps:
            continue

        actual_best = max(ds_exps, key=lambda e: e["delta_vs_smote"])
        deltas = [e["delta_vs_smote"] for e in ds_exps]
        avg_delta = float(np.mean(deltas))
        regret = actual_best["delta_vs_smote"] - avg_delta

        results.append(LODOResult(
            held_out_dataset=ds,
            strategy="baseline_random",
            predicted_best_config={"random": True},
            predicted_delta_vs_smote=avg_delta,
            actual_best_config=config_summary(actual_best),
            actual_best_delta=float(actual_best["delta_vs_smote"]),
            actual_delta_of_prediction=avg_delta,
            regret=float(regret),
            regret_pct=float(regret / max(abs(actual_best["delta_vs_smote"]), 0.01) * 100),
            top_k_accuracy={1: False, 3: False, 5: False, 10: False, 20: False},
            rank_correlation=0.0,
        ))

    return results


# ==============================================================================
# Step 5: Feature Importance
# ==============================================================================

def compute_feature_importance(
    experiments: List[Dict],
    meta_features: Dict[str, DatasetMetaFeatures],
) -> Dict[str, float]:
    """Train GBR on all data and report feature importances."""
    X, y = [], []
    for exp in experiments:
        ds = exp["dataset"]
        if ds not in meta_features:
            continue
        meta_vec = meta_features[ds].to_feature_vector()
        config_vec = encode_config(exp).to_feature_vector()
        X.append(np.concatenate([meta_vec, config_vec]))
        y.append(exp["delta_vs_smote"])

    X, y = np.array(X), np.array(y)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)

    all_names = DatasetMetaFeatures.feature_names() + ConfigFeatures.feature_names()
    importances = dict(zip(all_names, model.feature_importances_))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


# ==============================================================================
# Step 6: Report Generation
# ==============================================================================

def generate_report(
    all_results: Dict[str, List[LODOResult]],
    meta_features: Dict[str, DatasetMetaFeatures],
    feature_importances: Dict[str, float],
    tree_rules: List[str],
    datasets: List[str],
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("META-LEARNING SELECTOR EXPERIMENT — ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Date: {datetime.now().isoformat()}")
    lines.append(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    lines.append(f"Strategies: {list(all_results.keys())}")
    lines.append("")

    # Summary table
    lines.append("-" * 90)
    lines.append(f"{'Strategy':<30} {'Avg Regret':>12} {'Avg Delta':>12} {'Top-1':>8} {'Top-5':>8} {'Avg rho':>10}")
    lines.append("-" * 90)

    for name, res_list in all_results.items():
        if not res_list:
            continue
        avg_regret = np.mean([r.regret for r in res_list])
        avg_delta = np.mean([r.actual_delta_of_prediction for r in res_list])
        top1 = np.mean([r.top_k_accuracy.get(1, False) for r in res_list]) * 100
        top5 = np.mean([r.top_k_accuracy.get(5, False) for r in res_list]) * 100
        avg_rho = np.mean([r.rank_correlation for r in res_list])
        lines.append(
            f"{name:<30} {avg_regret:>+12.3f}pp {avg_delta:>+12.3f}pp "
            f"{top1:>7.0f}% {top5:>7.0f}% {avg_rho:>10.3f}"
        )

    # Per-dataset breakdown
    lines.append("")
    lines.append("=" * 80)
    lines.append("PER-DATASET BREAKDOWN")
    lines.append("=" * 80)

    for ds in datasets:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Dataset: {ds}")
        lines.append(f"{'─' * 60}")
        for name, res_list in all_results.items():
            ds_res = [r for r in res_list if r.held_out_dataset == ds]
            for r in ds_res:
                lines.append(
                    f"  {name:<25} delta={r.actual_delta_of_prediction:+.3f}pp  "
                    f"regret={r.regret:.3f}pp  top1={'Y' if r.top_k_accuracy.get(1) else 'N'}  "
                    f"rho={r.rank_correlation:.3f}"
                )
                lines.append(f"    Predicted: {r.predicted_best_config}")
                lines.append(f"    Oracle:    {r.actual_best_config} (delta={r.actual_best_delta:+.3f}pp)")

    # Feature importance
    lines.append("")
    lines.append("=" * 80)
    lines.append("FEATURE IMPORTANCE (GBR on all 1200 experiments)")
    lines.append("=" * 80)
    for feat, imp in list(feature_importances.items())[:15]:
        bar = "#" * int(imp * 200)
        lines.append(f"  {feat:<30} {imp:.4f}  {bar}")

    # Decision tree rules
    if tree_rules:
        lines.append("")
        lines.append("=" * 80)
        lines.append("DECISION TREE RULES (Strategy C)")
        lines.append("=" * 80)
        for rule_block in tree_rules:
            lines.append(rule_block)

    # Meta-features table
    lines.append("")
    lines.append("=" * 80)
    lines.append("DATASET META-FEATURES")
    lines.append("=" * 80)
    for ds_name, mf in meta_features.items():
        lines.append(f"\n{ds_name}:")
        lines.append(f"  n_train={mf.n_train}, n_classes={mf.n_classes}, spc={mf.samples_per_class:.0f}")
        lines.append(f"  text: avg_len={mf.avg_text_length:.1f} words, vocab={mf.vocab_size}, ttr={mf.type_token_ratio:.3f}")
        lines.append(f"  embedding: separation={mf.class_separation_ratio:.3f}, spread={mf.embedding_spread:.4f}, overlap={mf.max_class_overlap:.4f}")
        lines.append(f"  density: knn5={mf.avg_knn_distance_k5:.4f}, var={mf.density_variance:.6f}")

    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("META-LEARNING SELECTOR EXPERIMENT")
    print("=" * 80)

    # 1. Load experiment results
    print("\n[1/6] Loading experiment results...")
    with open(EXPERIMENT_RESULTS_PATH) as f:
        data = json.load(f)
    all_experiments = data["results"]
    print(f"  Loaded {len(all_experiments)} total experiments")

    # Filter to datasets with sufficient coverage
    dataset_counts = Counter(e["dataset"] for e in all_experiments)
    valid_datasets = [d for d in DATASETS if dataset_counts.get(d, 0) >= MIN_EXPERIMENTS_PER_DATASET]
    experiments = [e for e in all_experiments if e["dataset"] in valid_datasets]
    print(f"  Using {len(valid_datasets)} datasets with >= {MIN_EXPERIMENTS_PER_DATASET} experiments:")
    for d in valid_datasets:
        print(f"    {d}: {dataset_counts[d]} experiments")

    # 2. Compute meta-features
    print(f"\n[2/6] Computing meta-features ({len(valid_datasets)} datasets)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    meta_features = compute_all_meta_features(model)
    del model  # free GPU memory

    mf_data = {k: asdict(v) for k, v in meta_features.items()}
    with open(RESULTS_DIR / "meta_features.json", "w") as f:
        json.dump(mf_data, f, indent=2)
    print("  Saved to results/meta_learning/meta_features.json")

    # 3. Run strategies
    all_results = {}

    print(f"\n[3/6] Strategy A: Per-Instance Regression (LODO, {len(valid_datasets)} folds)...")
    regression_results = strategy_regression_lodo(experiments, meta_features, valid_datasets)
    all_results["regression"] = regression_results
    for r in regression_results:
        print(f"  {r.held_out_dataset}: regret={r.regret:.3f}pp  rho={r.rank_correlation:.3f}  model={r.strategy}")

    print(f"\n[4/6] Strategy B: Portfolio Ranking (LODO, {len(valid_datasets)} folds)...")
    portfolio_results = strategy_portfolio_lodo(experiments, valid_datasets)
    all_results["portfolio"] = portfolio_results
    for r in portfolio_results:
        print(f"  {r.held_out_dataset}: regret={r.regret:.3f}pp  rho={r.rank_correlation:.3f}")

    print(f"\n[5/6] Strategy C: Rule-Based Tree (LODO, {len(valid_datasets)} folds)...")
    rule_results, tree_rules = strategy_rule_based_lodo(experiments, meta_features, valid_datasets)
    all_results["rule_based"] = rule_results
    for r in rule_results:
        print(f"  {r.held_out_dataset}: regret={r.regret:.3f}pp  rho={r.rank_correlation:.3f}  filter={r.strategy}")

    # Baselines
    print("\n  Baseline: Always cascade_l1...")
    cascade_results = baseline_always_cascade_l1(experiments, valid_datasets)
    all_results["baseline_cascade_l1"] = cascade_results
    for r in cascade_results:
        print(f"  {r.held_out_dataset}: regret={r.regret:.3f}pp")

    print("  Baseline: Random selection...")
    random_results = baseline_random(experiments, valid_datasets)
    all_results["baseline_random"] = random_results
    for r in random_results:
        print(f"  {r.held_out_dataset}: regret={r.regret:.3f}pp")

    # 4. Feature importance
    print("\n[6/6] Feature importance analysis...")
    feature_importances = compute_feature_importance(experiments, meta_features)
    print("  Top 10:")
    for feat, imp in list(feature_importances.items())[:10]:
        print(f"    {feat:<30} {imp:.4f}")

    # 5. Generate report
    report = generate_report(all_results, meta_features, feature_importances, tree_rules, valid_datasets)

    with open(RESULTS_DIR / "analysis_report.txt", "w") as f:
        f.write(report)
    print(f"\n  Report saved to results/meta_learning/analysis_report.txt")

    # 6. Save JSON results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments_used": len(experiments),
        "n_datasets": len(valid_datasets),
        "datasets": valid_datasets,
        "strategies": {},
    }
    for strategy_name, res_list in all_results.items():
        results_data["strategies"][strategy_name] = {
            "results": [asdict(r) for r in res_list],
            "avg_regret": float(np.mean([r.regret for r in res_list])) if res_list else None,
            "avg_delta_achieved": float(np.mean([r.actual_delta_of_prediction for r in res_list])) if res_list else None,
            "avg_rank_correlation": float(np.mean([r.rank_correlation for r in res_list])) if res_list else None,
        }
    results_data["feature_importances"] = feature_importances
    results_data["tree_rules"] = tree_rules

    with open(RESULTS_DIR / "lodo_summary.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"  Results saved to results/meta_learning/lodo_summary.json")

    # Print report to console
    print("\n")
    print(report)


if __name__ == "__main__":
    main()
