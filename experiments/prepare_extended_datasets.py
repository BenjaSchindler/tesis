#!/usr/bin/env python3
"""
Prepare extended datasets for the soft weighting robustness experiment.

Creates low-resource splits (10/25/50-shot) for:
1. Real AG News (from existing ag_news_full.json)
2. 20newsgroups full 20 classes (from sklearn)
3. Emotion (6 classes, from dair-ai GitHub)
4. DBpedia-14 (14 classes, from HuggingFace)
"""

import json
import numpy as np
import requests
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_SHOTS = [10, 25, 50]


def create_low_resource_splits(
    train_texts, train_labels, test_texts, test_labels,
    dataset_name, n_shots=N_SHOTS, max_test_per_class=None
):
    """Create n-shot splits from a full dataset."""
    np.random.seed(42)

    # Optionally cap test set per class
    if max_test_per_class is not None:
        capped_texts, capped_labels = [], []
        class_counts = {}
        for t, l in zip(test_texts, test_labels):
            class_counts.setdefault(l, 0)
            if class_counts[l] < max_test_per_class:
                capped_texts.append(t)
                capped_labels.append(l)
                class_counts[l] += 1
        test_texts, test_labels = capped_texts, capped_labels

    for n_per_class in n_shots:
        subset_texts, subset_labels = [], []

        for cls in sorted(set(train_labels)):
            cls_indices = [i for i, l in enumerate(train_labels) if l == cls]
            n_select = min(n_per_class, len(cls_indices))
            if n_select > 0:
                selected = np.random.choice(cls_indices, n_select, replace=False)
                for idx in selected:
                    subset_texts.append(train_texts[idx])
                    subset_labels.append(train_labels[idx])

        data = {
            "train_texts": subset_texts,
            "train_labels": subset_labels,
            "test_texts": test_texts,
            "test_labels": test_labels,
            "n_per_class": n_per_class,
            "n_train": len(subset_texts),
            "n_test": len(test_texts),
            "n_classes": len(set(subset_labels)),
            "class_distribution": dict(Counter(subset_labels)),
        }

        path = DATA_DIR / f"{dataset_name}_{n_per_class}shot.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"    {n_per_class}-shot: {len(subset_texts)} train, {len(test_texts)} test")


# ============================================================================
# 1. Real AG News splits
# ============================================================================

def create_ag_news_splits():
    """Create low-resource splits from existing ag_news_full.json."""
    print("\n1. AG News (4 classes) — from existing full data")
    print("-" * 50)

    full_path = DATA_DIR / "ag_news_full.json"
    if not full_path.exists():
        print("   ERROR: ag_news_full.json not found!")
        return False

    with open(full_path) as f:
        data = json.load(f)

    print(f"   Full data: {len(data['train_texts'])} train, {len(data['test_texts'])} test")
    print(f"   Classes: {sorted(set(data['train_labels']))}")

    create_low_resource_splits(
        data["train_texts"], data["train_labels"],
        data["test_texts"], data["test_labels"],
        "ag_news", max_test_per_class=500,
    )
    return True


# ============================================================================
# 2. 20newsgroups full 20 classes
# ============================================================================

def download_20newsgroups_20class():
    """Download full 20-class 20newsgroups from sklearn."""
    print("\n2. 20newsgroups (20 classes) — from sklearn")
    print("-" * 50)

    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

    train_texts = [t.strip() for t in train.data]
    train_labels = [train.target_names[i] for i in train.target]
    test_texts = [t.strip() for t in test.data]
    test_labels = [test.target_names[i] for i in test.target]

    # Filter out empty texts
    valid_train = [(t, l) for t, l in zip(train_texts, train_labels) if len(t) > 20]
    valid_test = [(t, l) for t, l in zip(test_texts, test_labels) if len(t) > 20]
    train_texts, train_labels = zip(*valid_train)
    test_texts, test_labels = zip(*valid_test)
    train_texts, train_labels = list(train_texts), list(train_labels)
    test_texts, test_labels = list(test_texts), list(test_labels)

    print(f"   Full data: {len(train_texts)} train, {len(test_texts)} test")
    print(f"   Classes ({len(set(train_labels))}): {sorted(set(train_labels))[:5]}... ")

    # Save full version
    full_data = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_classes": len(set(train_labels)),
        "class_distribution": dict(Counter(train_labels)),
    }
    with open(DATA_DIR / "20newsgroups_20class_full.json", "w") as f:
        json.dump(full_data, f, indent=2)

    # Create splits (cap test at 100 per class = 2000 total)
    create_low_resource_splits(
        train_texts, train_labels, test_texts, test_labels,
        "20newsgroups_20class", max_test_per_class=100,
    )
    return True


# ============================================================================
# 3. Emotion dataset (6 classes)
# ============================================================================

def download_emotion():
    """Download emotion dataset from dair-ai GitHub or HuggingFace."""
    print("\n3. Emotion (6 classes)")
    print("-" * 50)

    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    train_texts, train_labels, test_texts, test_labels = None, None, None, None

    # Try GitHub first
    base_url = "https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/data"

    def parse_emotion_file(url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None, None
            texts, labels = [], []
            for line in response.text.strip().split("\n"):
                if ";" in line:
                    text, label = line.rsplit(";", 1)
                    if len(text.strip()) > 10:
                        texts.append(text.strip())
                        labels.append(label.strip())
            return texts, labels
        except Exception:
            return None, None

    train_texts, train_labels = parse_emotion_file(f"{base_url}/train.txt")
    test_texts, test_labels = parse_emotion_file(f"{base_url}/test.txt")

    # Fallback: HuggingFace datasets
    if not train_texts:
        print("   GitHub failed, trying HuggingFace datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("dair-ai/emotion")
            train_texts = list(ds["train"]["text"])
            train_labels = [label_map[l] for l in ds["train"]["label"]]
            test_texts = list(ds["test"]["text"])
            test_labels = [label_map[l] for l in ds["test"]["label"]]
        except Exception as e:
            print(f"   HuggingFace also failed: {e}")
            return False

    if not train_texts:
        print("   ERROR: Failed to download emotion dataset!")
        return False

    print(f"   Full data: {len(train_texts)} train, {len(test_texts)} test")
    print(f"   Classes: {sorted(set(train_labels))}")

    # Save full version
    full_data = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_classes": len(set(train_labels)),
        "class_distribution": dict(Counter(train_labels)),
    }
    with open(DATA_DIR / "emotion_full.json", "w") as f:
        json.dump(full_data, f, indent=2)

    # Create splits (cap test at 500 total for consistency)
    create_low_resource_splits(
        train_texts, train_labels, test_texts, test_labels,
        "emotion", max_test_per_class=200,
    )
    return True


# ============================================================================
# 4. DBpedia-14 (14 classes)
# ============================================================================

def download_dbpedia14():
    """Download DBpedia-14 from HuggingFace datasets."""
    print("\n4. DBpedia-14 (14 classes) — from HuggingFace")
    print("-" * 50)

    from datasets import load_dataset

    ds = load_dataset("fancyzhx/dbpedia_14")

    label_names = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
    ]

    def extract_split(split):
        texts, labels = [], []
        for row in split:
            text = row["content"][:512]  # Truncate to 512 chars
            if len(text) > 20:
                texts.append(text)
                labels.append(label_names[row["label"]])
        return texts, labels

    train_texts, train_labels = extract_split(ds["train"])
    test_texts, test_labels = extract_split(ds["test"])

    print(f"   Full data: {len(train_texts)} train, {len(test_texts)} test")
    print(f"   Classes ({len(set(train_labels))}): {sorted(set(train_labels))[:5]}...")

    # Subsample full version to keep file size reasonable (2K per class train)
    np.random.seed(42)
    sub_train_texts, sub_train_labels = [], []
    for cls in sorted(set(train_labels)):
        cls_idx = [i for i, l in enumerate(train_labels) if l == cls]
        selected = np.random.choice(cls_idx, min(2000, len(cls_idx)), replace=False)
        for idx in selected:
            sub_train_texts.append(train_texts[idx])
            sub_train_labels.append(train_labels[idx])

    full_data = {
        "train_texts": sub_train_texts,
        "train_labels": sub_train_labels,
        "test_texts": test_texts[:7000],
        "test_labels": test_labels[:7000],
        "n_train": len(sub_train_texts),
        "n_test": min(7000, len(test_texts)),
        "n_classes": len(set(sub_train_labels)),
        "class_distribution": dict(Counter(sub_train_labels)),
    }
    with open(DATA_DIR / "dbpedia14_full.json", "w") as f:
        json.dump(full_data, f, indent=2)

    # Create splits (cap test at 500 per class)
    create_low_resource_splits(
        sub_train_texts, sub_train_labels,
        test_texts, test_labels,
        "dbpedia14", max_test_per_class=500,
    )
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("PREPARING EXTENDED DATASETS")
    print("=" * 60)

    results = {}
    results["ag_news"] = create_ag_news_splits()
    results["20newsgroups_20class"] = download_20newsgroups_20class()
    results["emotion"] = download_emotion()
    results["dbpedia14"] = download_dbpedia14()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    # Verify all expected files exist
    print("\nVerifying files...")
    expected = []
    for ds in ["ag_news", "20newsgroups_20class", "emotion", "dbpedia14"]:
        for n in N_SHOTS:
            expected.append(f"{ds}_{n}shot.json")

    missing = [f for f in expected if not (DATA_DIR / f).exists()]
    if missing:
        print(f"  MISSING: {missing}")
    else:
        print(f"  All {len(expected)} files created successfully!")


if __name__ == "__main__":
    main()
