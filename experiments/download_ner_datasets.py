#!/usr/bin/env python3
"""
Download NER benchmark datasets and create low-resource splits.

Datasets:
- CoNLL-2003: PER, ORG, LOC, MISC (gold standard NER benchmark)
- WikiANN (en): PER, ORG, LOC (large, diverse)
- Few-NERD: 8 coarse entity types (designed for few-shot NER)

Low-resource splits: 10, 25, 50 sentences per entity type.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "ner"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PARSING UTILITIES
# ============================================================================

def bio_tags_to_entity_types(ner_tags):
    """Extract unique entity types from BIO tags (strip B-/I- prefixes)."""
    types = set()
    for tag in ner_tags:
        if tag.startswith("B-") or tag.startswith("I-"):
            types.add(tag[2:])
    return types


def sentence_has_entity_type(ner_tags, entity_type):
    """Check if a sentence contains at least one entity of the given type."""
    return any(
        tag == f"B-{entity_type}" or tag == f"I-{entity_type}"
        for tag in ner_tags
    )


def get_dominant_entity_type(ner_tags):
    """Get the most frequent entity type in a sentence (by B- tag count)."""
    type_counts = Counter()
    for tag in ner_tags:
        if tag.startswith("B-"):
            type_counts[tag[2:]] += 1
    if not type_counts:
        return "O"
    return type_counts.most_common(1)[0][0]


# ============================================================================
# DATASET LOADERS
# ============================================================================

def load_multinerd():
    """Load MultiNERD (English) NER dataset via streaming from HuggingFace.

    MultiNERD is a multi-genre NER dataset with 15 fine-grained entity types.
    We use 5 common types: PER, LOC, ORG, DIS(ease), EVE(nt) -> mapped to
    coarse BIO tags. Used as CoNLL-2003 replacement when auth is unavailable.

    Tag mapping (from dataset inspection):
    0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC,
    7=B-ANIM, 8=I-ANIM, 9=B-BIO, 10=I-BIO, 11=B-CEL, 12=I-CEL,
    13=B-DIS, 14=I-DIS, 15=B-EVE, 16=I-EVE, 17=B-FOOD, 18=I-FOOD,
    19=B-INST, 20=I-INST, 21=B-MEDIA, 22=I-MEDIA, 23=B-PLANT, 24=I-PLANT,
    25=B-MYTH, 26=I-MYTH, 27=B-TIME, 28=I-TIME, 29=B-VEHI, 30=I-VEHI
    """
    print("\n  Loading MultiNERD (English)...")

    TAG_MAP = [
        "O",
        "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
        "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
        "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
        "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-PLANT", "I-PLANT",
        "B-MYTH", "I-MYTH", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
    ]

    # Focus on 5 main entity types for manageable experiment
    KEEP_TYPES = {"PER", "ORG", "LOC", "DIS", "EVE"}

    def convert_and_filter(sample):
        """Convert integer tags to BIO strings, keep only KEEP_TYPES."""
        tokens = sample["tokens"]
        int_tags = sample["ner_tags"]
        bio_tags = []
        for t in int_tags:
            if t < len(TAG_MAP):
                tag = TAG_MAP[t]
            else:
                tag = "O"
            # Filter to kept types
            if tag != "O":
                etype = tag[2:]  # strip B-/I-
                if etype not in KEEP_TYPES:
                    tag = "O"
            bio_tags.append(tag)
        return {"tokens": tokens, "ner_tags": bio_tags}

    # Load via streaming, collect English only
    MAX_TRAIN = 20000
    MAX_TEST = 2000

    train_sentences = []
    test_sentences = []

    print("    Loading train split (streaming, English only)...")
    for sample in load_dataset("Babelscape/multinerd", split="train", streaming=True):
        if sample.get("lang") == "en":
            converted = convert_and_filter(sample)
            # Only keep sentences with at least one entity
            if any(t.startswith("B-") for t in converted["ner_tags"]):
                train_sentences.append(converted)
        if len(train_sentences) >= MAX_TRAIN:
            break

    print("    Loading test split (streaming, English only)...")
    for sample in load_dataset("Babelscape/multinerd", split="test", streaming=True):
        if sample.get("lang") == "en":
            converted = convert_and_filter(sample)
            test_sentences.append(converted)
        if len(test_sentences) >= MAX_TEST:
            break

    entity_types = sorted(KEEP_TYPES)

    print(f"    Train: {len(train_sentences)} sentences, Test: {len(test_sentences)} sentences")
    print(f"    Entity types: {entity_types}")

    for etype in entity_types:
        count = sum(1 for s in train_sentences if sentence_has_entity_type(s["ner_tags"], etype))
        print(f"    Sentences with {etype}: {count}")

    return train_sentences, test_sentences, entity_types


def load_wikiann():
    """Load WikiANN (English) NER dataset from HuggingFace."""
    print("\n  Loading WikiANN (en)...")

    TAG_MAP = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    ds = load_dataset("wikiann", "en")

    def convert_split(split):
        sentences = []
        for example in split:
            tokens = example["tokens"]
            tags = [TAG_MAP[t] for t in example["ner_tags"]]
            if len(tokens) > 0:
                sentences.append({"tokens": tokens, "ner_tags": tags})
        return sentences

    train = convert_split(ds["train"])
    test = convert_split(ds["test"])
    entity_types = ["PER", "ORG", "LOC"]

    print(f"    Train: {len(train)} sentences, Test: {len(test)} sentences")
    print(f"    Entity types: {entity_types}")

    for etype in entity_types:
        count = sum(1 for s in train if sentence_has_entity_type(s["ner_tags"], etype))
        print(f"    Sentences with {etype}: {count}")

    return train, test, entity_types


def load_few_nerd():
    """Load Few-NERD (coarse) NER dataset from HuggingFace.

    Few-NERD uses flat integer tags (not BIO) for coarse types:
    0=O, 1=art, 2=building, 3=event, 4=location, 5=organization, 6=other, 7=person, 8=product

    We convert these to BIO format by detecting entity spans (consecutive non-O tokens
    of the same type get B- for the first, I- for the rest).
    """
    print("\n  Loading Few-NERD...")

    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")

    # Coarse tag names from the ClassLabel feature
    COARSE_NAMES = ["O", "art", "building", "event", "location",
                    "organization", "other", "person", "product"]

    def convert_split(split):
        sentences = []
        for example in split:
            tokens = example["tokens"]
            int_tags = example["ner_tags"]  # integers

            # Convert flat integer tags to BIO format
            bio_tags = []
            prev_type = "O"
            for tag_idx in int_tags:
                coarse_type = COARSE_NAMES[tag_idx]
                if coarse_type == "O":
                    bio_tags.append("O")
                    prev_type = "O"
                elif coarse_type != prev_type:
                    # New entity span
                    bio_tags.append(f"B-{coarse_type}")
                    prev_type = coarse_type
                else:
                    # Continuation of same entity type
                    bio_tags.append(f"I-{coarse_type}")

            if len(tokens) > 0:
                sentences.append({"tokens": tokens, "ner_tags": bio_tags})
        return sentences

    train = convert_split(ds["train"])
    test = convert_split(ds["test"])

    # Discover entity types from data
    all_types = set()
    for s in train:
        all_types.update(bio_tags_to_entity_types(s["ner_tags"]))
    entity_types = sorted(all_types)

    print(f"    Train: {len(train)} sentences, Test: {len(test)} sentences")
    print(f"    Entity types: {entity_types}")

    for etype in entity_types:
        count = sum(1 for s in train if sentence_has_entity_type(s["ner_tags"], etype))
        print(f"    Sentences with {etype}: {count}")

    return train, test, entity_types


# ============================================================================
# LOW-RESOURCE SPLIT CREATION
# ============================================================================

def create_ner_low_resource_splits(
    train_sentences, test_sentences, entity_types, dataset_name
):
    """Create low-resource NER splits (N sentences per entity type).

    Strategy:
    - For each n_shot value (10, 25, 50), select sentences such that
      each entity type has at least n_shot sentences containing it.
    - A sentence containing both PER and ORG counts toward both quotas.
    - Greedy: prioritize sentences that cover the most under-quota types.
    """
    print(f"\n  Creating low-resource splits for {dataset_name}...")
    np.random.seed(42)

    # Limit test set to 500 sentences for speed
    test_subset = test_sentences[:500]

    for n_per_type in [10, 25, 50]:
        selected_indices = set()
        type_counts = {etype: 0 for etype in entity_types}

        # Index: for each entity type, which sentences contain it?
        type_to_sentences = defaultdict(list)
        for i, sent in enumerate(train_sentences):
            for etype in entity_types:
                if sentence_has_entity_type(sent["ner_tags"], etype):
                    type_to_sentences[etype].append(i)

        # Shuffle sentence indices per type for randomness
        for etype in entity_types:
            indices = type_to_sentences[etype]
            np.random.shuffle(indices)

        # Greedy selection: pick sentences that fill the most quotas
        max_iterations = len(train_sentences)
        iteration = 0

        while not all(c >= n_per_type for c in type_counts.values()) and iteration < max_iterations:
            # Find entity types still under quota
            under_quota = [et for et in entity_types if type_counts[et] < n_per_type]
            if not under_quota:
                break

            # Score each candidate sentence by how many under-quota types it covers
            best_idx = None
            best_score = 0

            # Sample candidates from under-quota types
            candidates = set()
            for etype in under_quota:
                for idx in type_to_sentences[etype]:
                    if idx not in selected_indices:
                        candidates.add(idx)

            for idx in candidates:
                sent = train_sentences[idx]
                score = sum(
                    1 for et in under_quota
                    if sentence_has_entity_type(sent["ner_tags"], et)
                    and type_counts[et] < n_per_type
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected_indices.add(best_idx)
            sent = train_sentences[best_idx]
            for etype in entity_types:
                if sentence_has_entity_type(sent["ner_tags"], etype):
                    type_counts[etype] += 1

            iteration += 1

        # Collect selected sentences
        selected = [train_sentences[i] for i in sorted(selected_indices)]

        # Build metadata
        type_distribution = {}
        for etype in entity_types:
            count = sum(1 for s in selected if sentence_has_entity_type(s["ner_tags"], etype))
            type_distribution[etype] = count

        output = {
            "train_sentences": selected,
            "test_sentences": test_subset,
            "entity_types": entity_types,
            "metadata": {
                "source": dataset_name,
                "n_shot": n_per_type,
                "n_train": len(selected),
                "n_test": len(test_subset),
                "entity_type_counts": type_distribution
            }
        }

        filename = f"{dataset_name}_{n_per_type}shot.json"
        output_path = DATA_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"    {n_per_type}-shot: {len(selected)} sentences, types: {type_distribution}")

    # Save full dataset too
    full_output = {
        "train_sentences": train_sentences,
        "test_sentences": test_subset,
        "entity_types": entity_types,
        "metadata": {
            "source": dataset_name,
            "n_shot": "full",
            "n_train": len(train_sentences),
            "n_test": len(test_subset)
        }
    }
    full_path = DATA_DIR / f"{dataset_name}_full.json"
    with open(full_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"    full: {len(train_sentences)} sentences")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("NER DATASET DOWNLOADER")
    print("=" * 60)

    # MultiNERD (English) — CoNLL-2003 replacement
    try:
        train, test, types = load_multinerd()
        create_ner_low_resource_splits(train, test, types, "multinerd")
    except Exception as e:
        print(f"  Error loading MultiNERD: {e}")
        import traceback; traceback.print_exc()

    # WikiANN
    try:
        train, test, types = load_wikiann()
        create_ner_low_resource_splits(train, test, types, "wikiann")
    except Exception as e:
        print(f"  Error loading WikiANN: {e}")
        import traceback; traceback.print_exc()

    # Few-NERD
    try:
        train, test, types = load_few_nerd()
        create_ner_low_resource_splits(train, test, types, "fewnerd")
    except Exception as e:
        print(f"  Error loading Few-NERD: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for f in sorted(DATA_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get("metadata", {})
        print(f"  {f.name}: {meta.get('n_train', '?')} train, {meta.get('n_test', '?')} test")

    print(f"\nDatasets saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
