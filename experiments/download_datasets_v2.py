#!/usr/bin/env python3
"""
Download benchmark datasets using direct HTTP downloads.
Fallback approach when HuggingFace datasets library has issues.
"""

import os
import json
import requests
import numpy as np
from pathlib import Path
from collections import Counter
from io import StringIO
import csv

# Create data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_from_github():
    """Download datasets from GitHub repos with raw CSV files."""
    print("="*60)
    print("Downloading datasets from GitHub...")
    print("="*60)

    datasets = {}

    # 1. Emotion dataset from dair-ai
    print("\n  Downloading Emotion dataset...")
    try:
        base_url = "https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/data"
        train_url = f"{base_url}/train.txt"
        test_url = f"{base_url}/test.txt"
        val_url = f"{base_url}/val.txt"

        def parse_emotion_file(url):
            response = requests.get(url)
            if response.status_code != 200:
                return None, None
            texts = []
            labels = []
            for line in response.text.strip().split('\n'):
                if ';' in line:
                    text, label = line.rsplit(';', 1)
                    texts.append(text)
                    labels.append(label)
            return texts, labels

        train_texts, train_labels = parse_emotion_file(train_url)
        test_texts, test_labels = parse_emotion_file(test_url)

        if train_texts:
            datasets['emotion'] = {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts,
                'test_labels': test_labels,
                'n_train': len(train_texts),
                'n_test': len(test_texts),
                'n_classes': len(set(train_labels)),
                'class_distribution': dict(Counter(train_labels))
            }
            print(f"    Train: {len(train_texts)}, Test: {len(test_texts)}")
            print(f"    Classes: {set(train_labels)}")
    except Exception as e:
        print(f"    Error: {e}")

    # 2. AG News from Kaggle (simplified version)
    print("\n  Creating AG News synthetic (for testing pipeline)...")
    # We'll create a small synthetic version based on the actual format
    ag_news_labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    ag_news_examples = {
        'World': [
            "The United Nations announced new peacekeeping measures today.",
            "International leaders met to discuss climate change policies.",
            "A major earthquake struck the region causing significant damage.",
            "Diplomatic tensions rose between neighboring countries.",
            "Humanitarian aid was delivered to refugee camps.",
        ],
        'Sports': [
            "The championship game ended in a stunning overtime victory.",
            "Star athlete breaks world record in 100m dash.",
            "The team announced a major trade deal today.",
            "Olympic qualifiers begin next month in the capital.",
            "Coach resigns after disappointing season performance.",
        ],
        'Business': [
            "Stock markets rallied on positive earnings reports.",
            "Major tech company announces quarterly profits.",
            "Federal Reserve considers interest rate changes.",
            "Merger deal worth billions announced today.",
            "Unemployment rates dropped to historic lows.",
        ],
        'Sci/Tech': [
            "Scientists discover new exoplanet in habitable zone.",
            "Tech giant unveils revolutionary AI technology.",
            "Breakthrough in quantum computing achieved.",
            "New smartphone features cutting-edge processors.",
            "Research reveals promising cancer treatment.",
        ],
    }

    # Create synthetic AG News dataset
    np.random.seed(42)
    ag_train_texts = []
    ag_train_labels = []
    for label, examples in ag_news_examples.items():
        for ex in examples * 10:  # Repeat to get more samples
            ag_train_texts.append(ex)
            ag_train_labels.append(label)

    # Shuffle
    indices = np.random.permutation(len(ag_train_texts))
    ag_train_texts = [ag_train_texts[i] for i in indices]
    ag_train_labels = [ag_train_labels[i] for i in indices]

    datasets['ag_news_synthetic'] = {
        'train_texts': ag_train_texts,
        'train_labels': ag_train_labels,
        'test_texts': ag_train_texts[:50],  # Use some for test
        'test_labels': ag_train_labels[:50],
        'n_train': len(ag_train_texts),
        'n_test': 50,
        'n_classes': 4,
        'class_distribution': dict(Counter(ag_train_labels)),
        'note': 'Synthetic dataset for pipeline testing'
    }
    print(f"    Created synthetic AG News: {len(ag_train_texts)} samples")

    # 3. Hate Speech Dataset from Davidson et al.
    print("\n  Downloading Davidson Hate Speech dataset...")
    try:
        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        response = requests.get(url)
        if response.status_code == 200:
            from io import StringIO
            import pandas as pd
            df = pd.read_csv(StringIO(response.text))

            # Map classes
            class_map = {0: 'hate_speech', 1: 'offensive', 2: 'neither'}
            texts = df['tweet'].astype(str).tolist()
            labels = [class_map[c] for c in df['class'].tolist()]

            datasets['hate_speech_davidson'] = {
                'train_texts': texts[:20000],
                'train_labels': labels[:20000],
                'test_texts': texts[20000:],
                'test_labels': labels[20000:],
                'n_train': 20000,
                'n_test': len(texts) - 20000,
                'n_classes': 3,
                'class_distribution': dict(Counter(labels[:20000]))
            }
            print(f"    Loaded: {len(texts)} samples")
            print(f"    Distribution: {dict(Counter(labels))}")
    except Exception as e:
        print(f"    Error: {e}")

    return datasets


def create_low_resource_splits(datasets):
    """Create low-resource splits for each dataset."""
    print("\n" + "="*60)
    print("Creating low-resource splits...")
    print("="*60)

    np.random.seed(42)

    for dataset_name, data in datasets.items():
        print(f"\n  {dataset_name}:")
        texts = data['train_texts']
        labels = data['train_labels']

        for n_per_class in [10, 25, 50]:
            subset_texts = []
            subset_labels = []

            for cls in set(labels):
                cls_indices = [i for i, l in enumerate(labels) if l == cls]
                n_select = min(n_per_class, len(cls_indices))
                if n_select > 0:
                    selected = np.random.choice(cls_indices, n_select, replace=False)
                    for idx in selected:
                        subset_texts.append(texts[idx])
                        subset_labels.append(labels[idx])

            subset_data = {
                'train_texts': subset_texts,
                'train_labels': subset_labels,
                'test_texts': data['test_texts'][:500],  # Limit test
                'test_labels': data['test_labels'][:500],
                'n_per_class': n_per_class,
                'n_train': len(subset_texts),
                'n_test': min(500, len(data['test_texts'])),
                'n_classes': len(set(subset_labels)),
                'class_distribution': dict(Counter(subset_labels))
            }

            output_path = DATA_DIR / f"{dataset_name}_{n_per_class}shot.json"
            with open(output_path, 'w') as f:
                json.dump(subset_data, f, indent=2)

            print(f"    {n_per_class}-shot: {len(subset_texts)} samples")


def save_full_datasets(datasets):
    """Save full versions of all datasets."""
    print("\n" + "="*60)
    print("Saving full datasets...")
    print("="*60)

    for name, data in datasets.items():
        output_path = DATA_DIR / f"{name}_full.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {data['n_train']} train, {data['n_test']} test")


def summarize():
    """Print summary of all downloaded datasets."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    for file in sorted(DATA_DIR.glob("*.json")):
        with open(file) as f:
            data = json.load(f)

        name = file.stem
        n_train = data.get('n_train', 'N/A')
        n_test = data.get('n_test', 'N/A')
        n_classes = data.get('n_classes', 'N/A')
        dist = data.get('class_distribution', {})

        print(f"\n  {name}:")
        print(f"    Train: {n_train}, Test: {n_test}, Classes: {n_classes}")
        if dist:
            print(f"    Distribution: {dist}")


if __name__ == "__main__":
    print("Downloading benchmark datasets (direct HTTP method)\n")

    # Download
    datasets = download_from_github()

    if datasets:
        # Create low-resource splits
        create_low_resource_splits(datasets)

        # Save full datasets
        save_full_datasets(datasets)

        # Summary
        summarize()

    print("\n" + "="*60)
    print("DONE! Datasets saved to:", DATA_DIR)
    print("="*60)
