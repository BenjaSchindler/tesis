#!/usr/bin/env python3
"""
Classifier Tuning Experiment — Can we recover RF? Does XGBoost work better?

Re-uses cached LLM generations from exp_thesis_final. Only varies the classifier:
  - random_forest (default thesis config)
  - random_forest_tuned (depth-limited, balanced, more trees)
  - xgboost (sensible defaults)
  - xgboost_tuned (regularised for small data)

Compares vs SMOTE baseline across {no_augmentation, smote, soft_weighted}.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# Reuse thesis_final infrastructure
from exp_thesis_final import (
    DATA_DIR, CACHE_DIR, DATASET_N_CLASSES,
    N_SYNTHETIC_PER_CLASS, OVERSAMPLE_FACTOR,
    FILTER_CONFIG, NORMALIZATION, TEMPERATURE, MIN_WEIGHT,
    load_dataset, get_dataset_base,
    generate_llm_samples_cached, generate_smote_samples,
    normalize_scores,
)
from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade


# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "classifier_tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Subset of datasets focused on where SOFT WEIGHTING matters most
DATASETS = [
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "20newsgroups_10shot",         "20newsgroups_25shot",         "20newsgroups_50shot",
    "emotion_10shot",              "emotion_25shot",              "emotion_50shot",
    "dbpedia14_10shot",            "dbpedia14_25shot",            "dbpedia14_50shot",
]

SEEDS = [42, 123, 456, 789, 1011]
METHODS = ["no_augmentation", "smote", "soft_weighted"]

CLASSIFIERS = ["random_forest", "random_forest_tuned", "xgboost", "xgboost_tuned"]


def create_classifier(name: str, seed: int, n_classes: int):
    if name == "random_forest":
        # Same as thesis_final default
        return RandomForestClassifier(n_estimators=100, random_state=seed)

    if name == "random_forest_tuned":
        # Depth-limited, balanced, more trees
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    if name == "xgboost":
        # Reasonable defaults
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss",
            tree_method="hist",
        )

    if name == "xgboost_tuned":
        # Regularised for small datasets
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss",
            tree_method="hist",
        )

    raise ValueError(name)


# ============================================================================
# DATACLASS
# ============================================================================
@dataclass
class TuningResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    classifier: str
    augmentation_method: str
    seed: int
    f1_macro: float
    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# LABEL ENCODER for XGBoost (needs integer labels)
# ============================================================================
class LabelMapper:
    def __init__(self, classes):
        self.classes = sorted(set(classes))
        self.to_int = {c: i for i, c in enumerate(self.classes)}
        self.to_str = {i: c for c, i in self.to_int.items()}

    def encode(self, labels):
        return np.array([self.to_int[c] for c in labels])

    def decode(self, ints):
        return np.array([self.to_str[i] for i in ints])


def fit_classifier(clf, X, y, sample_weight=None):
    if isinstance(clf, XGBClassifier):
        # XGBoost needs integer labels
        mapper = LabelMapper(y)
        y_int = mapper.encode(y)
        if sample_weight is not None:
            clf.fit(X, y_int, sample_weight=sample_weight)
        else:
            clf.fit(X, y_int)
        return clf, mapper
    if sample_weight is not None:
        clf.fit(X, y, sample_weight=sample_weight)
    else:
        clf.fit(X, y)
    return clf, None


def predict_classifier(clf, X, mapper):
    pred = clf.predict(X)
    if mapper is not None:
        pred = mapper.decode(pred)
    return pred


# ============================================================================
# PIPELINE
# ============================================================================
def precompute_augmentations(
    dataset_name, train_texts, train_labels, train_embeddings,
    n_shot, model, provider, seed
):
    """Pre-compute SMOTE + LLM cascade scores for one (dataset, seed)."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    dim = train_embeddings.shape[1]

    # SMOTE
    smote_embs, smote_labels = [], []
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
        if len(s) > 0:
            smote_embs.append(s)
            smote_labels.extend([cls] * len(s))
    smote_data = {
        "embeddings": np.vstack(smote_embs) if smote_embs else np.zeros((0, dim)),
        "labels": smote_labels,
    }

    # LLM + cascade (cached)
    cascade = FilterCascade(**FILTER_CONFIG)
    llm_data = {}
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
        gen_emb, _ = generate_llm_samples_cached(
            provider, dataset_name, cls, cls_texts, n_gen, n_shot, model
        )
        if len(gen_emb) == 0:
            llm_data[cls] = None
            continue
        anchor = cls_emb.mean(axis=0) if len(cls_emb) > 0 else train_embeddings.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_embeddings, labels_arr, cls)
        weights = normalize_scores(scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        llm_data[cls] = {
            "top_emb": gen_emb[top_idx],
            "top_weights": weights[top_idx],
        }

    return {"smote": smote_data, "llm": llm_data}


def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, aug_method, seed,
    train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
):
    if aug_method == "no_augmentation":
        clf = create_classifier(classifier_name, seed, n_classes)
        clf, mapper = fit_classifier(clf, train_embeddings, train_labels)
        pred = predict_classifier(clf, test_embeddings, mapper)
        f1 = f1_score(test_labels, pred, average="macro")
        return TuningResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0.0, weight_std=0.0,
            timestamp=datetime.now().isoformat(),
        )

    if aug_method == "smote":
        method_data = aug_data["smote"]
        if len(method_data["labels"]) == 0:
            return run_single_config(
                dataset_name, dataset_base, n_classes, n_shot,
                classifier_name, "no_augmentation", seed,
                train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
            )
        aug_emb = np.vstack([train_embeddings, method_data["embeddings"]])
        aug_labels = list(train_labels) + method_data["labels"]
        clf = create_classifier(classifier_name, seed, n_classes)
        clf, mapper = fit_classifier(clf, aug_emb, aug_labels)
        pred = predict_classifier(clf, test_embeddings, mapper)
        f1 = f1_score(test_labels, pred, average="macro")
        return TuningResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings),
            n_synthetic_samples=len(method_data["labels"]),
            weight_mean=1.0, weight_std=0.0,
            timestamp=datetime.now().isoformat(),
        )

    # soft_weighted
    llm_data = aug_data["llm"]
    syn_embs, syn_labels, syn_weights = [], [], []
    for cls in sorted(set(train_labels)):
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        syn_embs.append(ld["top_emb"])
        syn_labels.extend([cls] * len(ld["top_emb"]))
        syn_weights.append(ld["top_weights"])

    if not syn_embs:
        return run_single_config(
            dataset_name, dataset_base, n_classes, n_shot,
            classifier_name, "no_augmentation", seed,
            train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
        )

    synthetic_emb = np.vstack(syn_embs)
    synthetic_weights = np.concatenate(syn_weights)
    aug_emb = np.vstack([train_embeddings, synthetic_emb])
    aug_labels = list(train_labels) + syn_labels
    sample_weights = np.concatenate([np.ones(len(train_embeddings)), synthetic_weights])

    clf = create_classifier(classifier_name, seed, n_classes)
    clf, mapper = fit_classifier(clf, aug_emb, aug_labels, sample_weight=sample_weights)
    pred = predict_classifier(clf, test_embeddings, mapper)
    f1 = f1_score(test_labels, pred, average="macro")
    return TuningResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, augmentation_method=aug_method,
        seed=seed, f1_macro=float(f1),
        n_real_samples=len(train_embeddings),
        n_synthetic_samples=len(synthetic_emb),
        weight_mean=float(synthetic_weights.mean()),
        weight_std=float(synthetic_weights.std()),
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    provider = create_provider("google", "gemini-3-flash-preview")

    all_results = []
    total_configs = len(DATASETS) * len(SEEDS) * len(CLASSIFIERS) * len(METHODS)
    print(f"Total runs: {total_configs}\n")

    run_idx = 0
    for dataset_name in DATASETS:
        ds_base = get_dataset_base(dataset_name)
        n_classes = DATASET_N_CLASSES[ds_base]
        n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

        print(f"\n=== {dataset_name} (classes={n_classes}, n_shot={n_shot}) ===")
        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)

        # Encode once per dataset (cached by sentence-transformers under the hood for repeated text)
        print(f"  Encoding {len(train_texts)} train / {len(test_texts)} test texts...")
        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)

        for seed in SEEDS:
            print(f"  Seed {seed}: building aug_data...", end=" ", flush=True)
            aug_data = precompute_augmentations(
                dataset_name, train_texts, train_labels, train_emb, n_shot,
                model, provider, seed
            )
            print("done.")

            for clf_name in CLASSIFIERS:
                for method in METHODS:
                    run_idx += 1
                    try:
                        result = run_single_config(
                            dataset_name, ds_base, n_classes, n_shot,
                            clf_name, method, seed,
                            train_emb, train_labels, test_emb, test_labels, aug_data,
                        )
                        all_results.append(asdict(result))
                    except Exception as e:
                        print(f"    [{run_idx}/{total_configs}] {clf_name}/{method}/s{seed} FAILED: {e}")

            # Periodic save
            if run_idx % 60 == 0:
                with open(RESULTS_DIR / "partial_results.json", "w") as f:
                    json.dump({"results": all_results, "run_idx": run_idx}, f)
                print(f"  [checkpoint @ {run_idx}/{total_configs}]")

    # Final save
    out = {"timestamp": datetime.now().isoformat(),
           "n_results": len(all_results),
           "results": all_results}
    with open(RESULTS_DIR / "final_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] {len(all_results)} results saved to {RESULTS_DIR/'final_results.json'}")


if __name__ == "__main__":
    main()
