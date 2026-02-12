#!/usr/bin/env python3
"""
Download and prepare CLINC150, BANKING77, and TREC datasets
in the same JSON format used by exp_thesis_final.py.

Creates 10/25/50-shot versions + full test set for each.
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def save_dataset(name, train_texts, train_labels, test_texts, test_labels):
    """Save dataset in standard JSON format."""
    data = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path.name}: {len(train_texts)} train, {len(test_texts)} test")


def create_few_shot(train_texts, train_labels, n_shot, seed=RANDOM_SEED):
    """Sample n_shot examples per class."""
    rng = random.Random(seed)
    by_class = {}
    for text, label in zip(train_texts, train_labels):
        by_class.setdefault(label, []).append(text)

    shot_texts, shot_labels = [], []
    for label in sorted(by_class.keys()):
        examples = by_class[label]
        sampled = rng.sample(examples, min(n_shot, len(examples)))
        shot_texts.extend(sampled)
        shot_labels.extend([label] * len(sampled))

    return shot_texts, shot_labels


# ============================================================================
# 1. CLINC150 — 150 intent classes
# ============================================================================

def download_clinc150():
    print("\n1. CLINC150 (150 intents)")
    print("-" * 50)
    from datasets import load_dataset

    ds = load_dataset("clinc_oos", "small")

    # Get intent names
    intent_names = ds["train"].features["intent"].names

    # Filter out "oos" (out-of-scope, intent=42)
    oos_idx = intent_names.index("oos") if "oos" in intent_names else -1

    train_texts, train_labels = [], []
    for row in ds["train"]:
        if row["intent"] == oos_idx:
            continue
        train_texts.append(row["text"])
        train_labels.append(intent_names[row["intent"]])

    test_texts, test_labels = [], []
    for row in ds["test"]:
        if row["intent"] == oos_idx:
            continue
        test_texts.append(row["text"])
        test_labels.append(intent_names[row["intent"]])

    n_classes = len(set(train_labels))
    print(f"  Full: {len(train_texts)} train, {len(test_texts)} test, {n_classes} classes")
    counts = Counter(train_labels)
    min_count = min(counts.values())
    max_count = max(counts.values())
    print(f"  Samples/class: min={min_count}, max={max_count}")

    # Save full
    save_dataset("clinc150_full", train_texts, train_labels, test_texts, test_labels)

    # Few-shot versions
    for n_shot in [10, 25, 50]:
        shot_texts, shot_labels = create_few_shot(train_texts, train_labels, n_shot)
        save_dataset(f"clinc150_{n_shot}shot", shot_texts, shot_labels, test_texts, test_labels)

    return True


# ============================================================================
# 2. BANKING77 — 77 banking intent classes
# ============================================================================

def download_banking77():
    print("\n2. BANKING77 (77 intents)")
    print("-" * 50)
    from datasets import load_dataset

    ds = load_dataset("banking77")
    label_names = ds["train"].features["label"].names

    train_texts = [row["text"] for row in ds["train"]]
    train_labels = [label_names[row["label"]] for row in ds["train"]]
    test_texts = [row["text"] for row in ds["test"]]
    test_labels = [label_names[row["label"]] for row in ds["test"]]

    n_classes = len(set(train_labels))
    print(f"  Full: {len(train_texts)} train, {len(test_texts)} test, {n_classes} classes")
    counts = Counter(train_labels)
    min_count = min(counts.values())
    max_count = max(counts.values())
    print(f"  Samples/class: min={min_count}, max={max_count}")

    save_dataset("banking77_full", train_texts, train_labels, test_texts, test_labels)

    for n_shot in [10, 25, 50]:
        shot_texts, shot_labels = create_few_shot(train_texts, train_labels, n_shot)
        save_dataset(f"banking77_{n_shot}shot", shot_texts, shot_labels, test_texts, test_labels)

    return True


# ============================================================================
# 3. TREC — 6 question type classes
# ============================================================================

def download_trec():
    print("\n3. TREC-6 (6 question types)")
    print("-" * 50)
    from datasets import load_dataset

    ds = load_dataset("trec", revision="refs/convert/parquet")

    label_names = ds["train"].features["coarse_label"].names

    train_texts = [row["text"] for row in ds["train"]]
    train_labels = [label_names[row["coarse_label"]] for row in ds["train"]]
    test_texts = [row["text"] for row in ds["test"]]
    test_labels = [label_names[row["coarse_label"]] for row in ds["test"]]

    n_classes = len(set(train_labels))
    print(f"  Full: {len(train_texts)} train, {len(test_texts)} test, {n_classes} classes")
    counts = Counter(train_labels)
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls}: {cnt}")

    save_dataset("trec6_full", train_texts, train_labels, test_texts, test_labels)

    for n_shot in [10, 25, 50]:
        shot_texts, shot_labels = create_few_shot(train_texts, train_labels, n_shot)
        save_dataset(f"trec6_{n_shot}shot", shot_texts, shot_labels, test_texts, test_labels)

    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING NEW DATASETS")
    print("=" * 60)

    results = {}
    for name, fn in [("CLINC150", download_clinc150),
                     ("BANKING77", download_banking77),
                     ("TREC-6", download_trec)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)
