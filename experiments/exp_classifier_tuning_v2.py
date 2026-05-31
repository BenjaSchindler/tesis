#!/usr/bin/env python3
"""
Classifier Tuning Experiment v2 — Rigorous validation of RF/XGBoost + soft weighting.

Addresses three committee-grade concerns left open by v1
(exp_classifier_tuning.py):

  1. ATTRIBUTION ABLATION (with-filter vs without-filter vs no-augmentation).
     v1 only had {no_augmentation, smote, soft_weighted}, so it could not rule out
     "the gain is just LLM data, not the filter". v2 adds the full chain:

        no_augmentation  -> floor
        smote            -> classic synthetic baseline
        llm_unfiltered   -> LLM samples, RANDOM selection, uniform weight  (LLM effect)
        binary_filter    -> LLM + geometric filter (top-N), uniform weight (FILTER effect)
        soft_weighted    -> LLM + filter + quality sample_weight           (WEIGHT effect)

     Decomposition:
        llm_unfiltered - no_aug        = does generating with the LLM help?
        binary_filter  - llm_unfiltered = does FILTERING help?   <-- the thesis
        soft_weighted  - binary_filter  = does soft weighting add over filtering?

  2. NO TUNING ON TEST. v1's "tuned" configs were hand-picked, which invites the
     attack "you tuned XGBoost on the test set". v2 selects hyperparameters via
     internal stratified k-fold CV on the (augmented) TRAINING set only. The test
     set is touched exactly once, after selection. Selection is run per method, so
     every method gets its own fairly-tuned classifier.

  3. REAL CROSS-SEED VARIANCE. RF and XGBoost are stochastic (bootstrap / subsample /
     colsample), so unlike the deterministic linear classifiers with a cached LLM,
     the cross-seed CIs here are genuine — this directly answers the caching-variance
     limitation. The seed varies the classifier (and XGB subsampling).

Re-uses the cached LLM generations from exp_thesis_final (no new API calls).
Writes to results/classifier_tuning_v2/ so it never collides with the v1 run.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import json
import numpy as np
from itertools import product
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, asdict
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# Reuse thesis_final infrastructure (same cache, same SMOTE, same cascade config)
from exp_thesis_final import (
    DATASET_N_CLASSES,
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
RESULTS_DIR = Path(__file__).parent.parent / "results" / "classifier_tuning_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "20newsgroups_10shot",         "20newsgroups_25shot",         "20newsgroups_50shot",
    "emotion_10shot",              "emotion_25shot",              "emotion_50shot",
    "dbpedia14_10shot",            "dbpedia14_25shot",            "dbpedia14_50shot",
]

SEEDS = [42, 123, 456, 789, 1011]

# Full attribution chain — the two middle conditions are what v1 was missing.
METHODS = ["no_augmentation", "smote", "llm_unfiltered", "binary_filter", "soft_weighted"]

# default = fixed sensible config (matches v1's untuned baseline).
# *_cv   = hyperparameters selected by internal CV on the TRAIN set only.
CLASSIFIERS = ["random_forest", "random_forest_cv", "xgboost", "xgboost_cv"]

CV_FOLDS = 3  # capped down to the smallest class count when needed
CV_JOBS = max(1, (os.cpu_count() or 4) - 4)  # parallelize the CV grid across cores; leave a few free


# ============================================================================
# CLASSIFIER FACTORIES + CV GRIDS
# ============================================================================
def _xgb_objective(n_classes):
    return "multi:softprob" if n_classes > 2 else "binary:logistic"


def make_rf(params, seed, n_classes):
    # NOTE: NO regularization is injected here. The fixed "random_forest" MUST match
    # v1 (exp_thesis_final.py:174) EXACTLY — RandomForestClassifier(n_estimators=100,
    # random_state=seed), i.e. sklearn defaults (class_weight=None, max_depth=None) —
    # so the v1 "RF collapse with soft weighting" is reproduced faithfully. Any
    # regularization (class_weight, bounded depth, ...) is introduced ONLY through the
    # CV grid, so random_forest_cv can *discover* whether it rescues RF.
    base = dict(random_state=seed, n_jobs=1)  # 1 thread/fit — the CV grid parallelizes across combos instead
    base.update(params)
    return RandomForestClassifier(**base)


def make_xgb(params, seed, n_classes):
    # NOTE: subsample/colsample are NOT forced here, so the fixed "xgboost" is a
    # vanilla baseline (subsample=colsample=1.0). Row/column subsampling is a CV grid
    # axis, so xgboost_cv can discover whether it helps in low-resource. eval_metric is
    # logloss for binary / mlogloss for multiclass (only matters if early stopping is added).
    base = dict(
        random_state=seed, n_jobs=1, verbosity=0,   # 1 thread/fit; CV grid parallelizes across combos
        objective=_xgb_objective(n_classes),
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        tree_method="hist",
    )
    base.update(params)
    return XGBClassifier(**base)


# Fixed (untuned) configs. These MUST reproduce v1 EXACTLY so the v1 baselines are
# comparable: plain sklearn RF (the one that "collapsed" with soft weighting) and a
# vanilla XGB (subsample=colsample=1.0). Do NOT add regularization here — it belongs
# in the CV grids below.
RF_DEFAULT_PARAMS = dict(n_estimators=100)  # else sklearn defaults: class_weight=None, max_depth=None
XGB_DEFAULT_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1)  # subsample/colsample=1.0 (vanilla)

# CV grids. The regularization axes that the fixed defaults LACK live here, so the
# *_cv variants answer "can honest tuning rescue the classifier?".
RF_GRID = {
    "n_estimators": [300],
    "max_depth": [6, 10, 15],                        # bounded — fixed RF already covers depth=None (the collapse)
    "min_samples_leaf": [1, 3],
    "class_weight": [None, "balanced_subsample"],    # let CV discover class balancing
}
# -> 1*3*2*2 = 12 combos
XGB_GRID = {
    "n_estimators": [100],                            # 100 trees is ample for 30-300 samples (and cheaper)
    "learning_rate": [0.1],
    "max_depth": [3, 5],                              # swept
    "min_child_weight": [1, 3],                       # key low-resource overfitting knob (was missing in v1 grid)
    "subsample": [0.8, 1.0],                          # row subsampling — swept
    "colsample_bytree": [0.8],                        # column subsampling regularized (768-dim); fixed to bound runtime
}
# -> 1*1*2*2*2*1 = 8 combos
# Grids kept lean on purpose: XGBoost multiclass on 768-dim embeddings costs ~1-6s/fit
# (scales with n_classes); the swept axes are the regularizers that actually matter in
# low-resource. Wider grids were benchmarked at ~10x the wall-clock for negligible gain.

CLASSIFIER_SPEC = {
    "random_forest":     ("rf",  None,    RF_DEFAULT_PARAMS),
    "random_forest_cv":  ("rf",  RF_GRID, None),
    "xgboost":           ("xgb", None,    XGB_DEFAULT_PARAMS),
    "xgboost_cv":        ("xgb", XGB_GRID, None),
}


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
        mapper = LabelMapper(y)
        y_int = mapper.encode(y)
        clf.fit(X, y_int, sample_weight=sample_weight)
        return clf, mapper
    clf.fit(X, y, sample_weight=sample_weight)
    return clf, None


def predict_classifier(clf, X, mapper):
    pred = clf.predict(X)
    if mapper is not None:
        pred = mapper.decode(pred)
    return pred


# ============================================================================
# HYPERPARAMETER SELECTION VIA INTERNAL CV (never touches the test set)
# ============================================================================
def _score_combo(make, params, splits, X, y_arr, sample_weight, seed, n_classes):
    """CV score for ONE hyperparameter combo — the unit of parallel work."""
    fold_scores = []
    for tr_idx, va_idx in splits:
        clf = make(params, seed, n_classes)
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        clf_f, mapper = fit_classifier(clf, X[tr_idx], y_arr[tr_idx], sample_weight=sw)
        pred = predict_classifier(clf_f, X[va_idx], mapper)
        fold_scores.append(f1_score(y_arr[va_idx], pred, average="macro"))
    return float(np.mean(fold_scores))


def select_params_cv(family, grid, X, y, sample_weight, seed, n_classes):
    """Stratified k-fold CV on (augmented) TRAIN data to pick hyperparameters.

    The combo grid is swept IN PARALLEL across CPU cores (each fit is single-threaded
    — see n_jobs=1 in make_rf/make_xgb — so parallelism lives at the combo level via
    threads, the right granularity for many tiny fits; sklearn/XGBoost release the GIL
    during fit). Results are IDENTICAL to a serial sweep: folds are precomputed once
    and shared, combo order is preserved by joblib, and argmax returns the FIRST max —
    matching the serial `if m > best_score` tie-break. The test set is never seen here.
    """
    make = make_rf if family == "rf" else make_xgb
    y_arr = np.array(y)

    min_count = min(Counter(y_arr).values())
    folds = max(2, min(CV_FOLDS, min_count))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y_arr))   # fixed folds, shared by every combo

    keys = list(grid.keys())
    combos = [dict(zip(keys, c)) for c in product(*grid.values())]
    scores = Parallel(n_jobs=CV_JOBS, prefer="threads")(
        delayed(_score_combo)(make, p, splits, X, y_arr, sample_weight, seed, n_classes)
        for p in combos
    )
    best_i = int(np.argmax(scores))      # first max == serial tie-break
    return combos[best_i], float(scores[best_i])


def build_and_fit(classifier_name, X, y, sample_weight, seed, n_classes):
    """Resolve a classifier (fixed or CV-selected), fit on full aug-train, return fitted."""
    family, grid, fixed = CLASSIFIER_SPEC[classifier_name]
    if grid is None:
        params, cv_f1 = fixed, None
    else:
        params, cv_f1 = select_params_cv(family, grid, X, y, sample_weight, seed, n_classes)
    make = make_rf if family == "rf" else make_xgb
    clf = make(params, seed, n_classes)
    clf, mapper = fit_classifier(clf, X, y, sample_weight=sample_weight)
    return clf, mapper, params, cv_f1


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
    cv_f1: float            # internal CV F1 of the selected config (-1 if fixed)
    chosen_params: str      # json string of selected hyperparameters
    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# AUGMENTATION PRECOMPUTE (per dataset+seed) — stores the FULL candidate pool
# ============================================================================
def precompute_augmentations(
    dataset_name, train_texts, train_labels, train_embeddings,
    n_shot, model, provider, seed
):
    """SMOTE samples + full LLM candidate pool with cascade scores/weights.

    Unlike v1 we keep the *entire* generated pool (+ scores + weights) so the
    per-method selection (random / top-N / weighted) happens downstream.
    """
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    dim = train_embeddings.shape[1]

    # --- SMOTE ---
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

    # --- LLM + cascade (cached) ---
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
        llm_data[cls] = {
            "gen_emb": gen_emb,            # full pool
            "scores": scores,             # cascade quality scores
            "weights": weights,           # normalized soft weights
        }

    return {"smote": smote_data, "llm": llm_data}


def _select_llm_samples(llm_data, classes, mode, seed):
    """Build synthetic (emb, label, weight) arrays for a given LLM-based mode.

      mode='unfiltered' : random target_n from the pool, weight = 1.0
      mode='binary'     : top target_n by cascade score, weight = 1.0
      mode='soft'       : top target_n by cascade score, weight = quality weight
    """
    rng = np.random.default_rng(seed)
    syn_embs, syn_labels, syn_weights = [], [], []
    for cls in classes:
        ld = llm_data.get(cls)
        if ld is None:
            continue
        gen_emb = ld["gen_emb"]
        scores = ld["scores"]
        weights = ld["weights"]
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))

        if mode == "unfiltered":
            idx = rng.choice(len(gen_emb), size=target_n, replace=False)
            w = np.ones(target_n)
        elif mode == "binary":
            idx = np.argsort(scores)[-target_n:]
            w = np.ones(target_n)
        elif mode == "soft":
            idx = np.argsort(scores)[-target_n:]
            w = weights[idx]
        else:
            raise ValueError(mode)

        syn_embs.append(gen_emb[idx])
        syn_labels.extend([cls] * target_n)
        syn_weights.append(w)

    if not syn_embs:
        return None
    return np.vstack(syn_embs), syn_labels, np.concatenate(syn_weights)


# ============================================================================
# SINGLE CONFIG
# ============================================================================
def _result(dataset_name, dataset_base, n_classes, n_shot, classifier_name,
            aug_method, seed, f1, cv_f1, params, n_real, n_syn, w_mean, w_std):
    return TuningResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, augmentation_method=aug_method, seed=seed,
        f1_macro=float(f1),
        cv_f1=float(cv_f1) if cv_f1 is not None else -1.0,
        chosen_params=json.dumps(params),
        n_real_samples=int(n_real), n_synthetic_samples=int(n_syn),
        weight_mean=float(w_mean), weight_std=float(w_std),
        timestamp=datetime.now().isoformat(),
    )


def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, aug_method, seed,
    train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
):
    classes = sorted(set(train_labels))

    # Assemble augmented training set + sample weights per method.
    if aug_method == "no_augmentation":
        X, y, sw = train_embeddings, list(train_labels), None
        n_syn, w_mean, w_std = 0, 0.0, 0.0

    elif aug_method == "smote":
        md = aug_data["smote"]
        if len(md["labels"]) == 0:
            return run_single_config(dataset_name, dataset_base, n_classes, n_shot,
                                     classifier_name, "no_augmentation", seed,
                                     train_embeddings, train_labels,
                                     test_embeddings, test_labels, aug_data)
        X = np.vstack([train_embeddings, md["embeddings"]])
        y = list(train_labels) + md["labels"]
        sw = None  # SMOTE samples get standard uniform treatment
        n_syn, w_mean, w_std = len(md["labels"]), 1.0, 0.0

    else:
        mode = {"llm_unfiltered": "unfiltered",
                "binary_filter": "binary",
                "soft_weighted": "soft"}[aug_method]
        sel = _select_llm_samples(aug_data["llm"], classes, mode, seed)
        if sel is None:
            return run_single_config(dataset_name, dataset_base, n_classes, n_shot,
                                     classifier_name, "no_augmentation", seed,
                                     train_embeddings, train_labels,
                                     test_embeddings, test_labels, aug_data)
        syn_emb, syn_labels, syn_w = sel
        X = np.vstack([train_embeddings, syn_emb])
        y = list(train_labels) + syn_labels
        sw = np.concatenate([np.ones(len(train_embeddings)), syn_w])
        n_syn, w_mean, w_std = len(syn_emb), float(syn_w.mean()), float(syn_w.std())

    clf, mapper, params, cv_f1 = build_and_fit(
        classifier_name, X, np.array(y), sw, seed, n_classes
    )
    pred = predict_classifier(clf, test_embeddings, mapper)
    f1 = f1_score(test_labels, pred, average="macro")

    return _result(dataset_name, dataset_base, n_classes, n_shot, classifier_name,
                   aug_method, seed, f1, cv_f1, params,
                   len(train_embeddings), n_syn, w_mean, w_std)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    provider = create_provider("google", "gemini-3-flash-preview")

    all_results = []
    total = len(DATASETS) * len(SEEDS) * len(CLASSIFIERS) * len(METHODS)
    print(f"Total runs: {total}")
    print(f"Methods: {METHODS}")
    print(f"Classifiers: {CLASSIFIERS}\n")

    run_idx = 0
    for dataset_name in DATASETS:
        ds_base = get_dataset_base(dataset_name)
        n_classes = DATASET_N_CLASSES[ds_base]
        n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

        print(f"\n=== {dataset_name} (classes={n_classes}, n_shot={n_shot}) ===")
        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)

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
                        print(f"    [{run_idx}/{total}] {clf_name}/{method}/s{seed} FAILED: {e}")

            with open(RESULTS_DIR / "partial_results.json", "w") as f:
                json.dump({"results": all_results, "run_idx": run_idx}, f)
            print(f"  [checkpoint @ {run_idx}/{total}]")

    out = {"timestamp": datetime.now().isoformat(),
           "n_results": len(all_results),
           "config": {"datasets": DATASETS, "seeds": SEEDS,
                      "methods": METHODS, "classifiers": CLASSIFIERS,
                      "cv_folds": CV_FOLDS},
           "results": all_results}
    with open(RESULTS_DIR / "final_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] {len(all_results)} results -> {RESULTS_DIR/'final_results.json'}")


if __name__ == "__main__":
    main()
