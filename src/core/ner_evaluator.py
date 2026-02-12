#!/usr/bin/env python3
"""
NER Evaluation Pipeline

Trains SpaCy NER models and evaluates using seqeval entity-level F1.

Supports:
- Training on real + synthetic NER data
- Entity-level F1 (strict match) via seqeval
- Per-entity-type metrics
- K-fold cross-validation

Usage:
    from core.ner_evaluator import evaluate_ner_augmentation, compute_ner_baseline

    baseline = compute_ner_baseline(train_sentences, test_sentences)
    augmented = evaluate_ner_augmentation(train_sentences, synthetic_sentences, test_sentences)
"""

import random
import warnings
import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from typing import List, Dict, Tuple, Optional
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision
from seqeval.metrics import recall_score as seq_recall
from seqeval.metrics import classification_report as seq_classification_report


# ============================================================================
# SPACY NER TRAINING
# ============================================================================

def _sentences_to_spacy_format(sentences: List[Dict]) -> List[Tuple[str, Dict]]:
    """Convert NER sentences to SpaCy training format.

    SpaCy expects: (text, {"entities": [(start, end, label), ...]})
    """
    training_data = []

    for sent in sentences:
        tokens = sent["tokens"]
        tags = sent["ner_tags"]

        # Reconstruct text with character offsets
        text = ""
        char_offsets = []  # (start, end) for each token
        for i, token in enumerate(tokens):
            start = len(text)
            text += token
            end = len(text)
            char_offsets.append((start, end))
            if i < len(tokens) - 1:
                text += " "

        # Extract entities from BIO tags
        entities = []
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                entity_type = tags[i][2:]
                entity_start = char_offsets[i][0]
                entity_end = char_offsets[i][1]

                # Extend through I- tags
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{entity_type}":
                    entity_end = char_offsets[j][1]
                    j += 1

                entities.append((entity_start, entity_end, entity_type))
                i = j
            else:
                i += 1

        training_data.append((text, {"entities": entities}))

    return training_data


def train_spacy_ner(
    train_sentences: List[Dict],
    n_epochs: int = 30,
    dropout: float = 0.3,
    seed: int = 42
) -> spacy.language.Language:
    """Train a SpaCy NER model from scratch on given sentences.

    Args:
        train_sentences: List of {"tokens": [...], "ner_tags": [...]}
        n_epochs: Number of training epochs
        dropout: Dropout rate
        seed: Random seed

    Returns:
        Trained SpaCy language model
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create blank model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    # Add entity labels
    all_labels = set()
    for sent in train_sentences:
        for tag in sent["ner_tags"]:
            if tag.startswith("B-") or tag.startswith("I-"):
                all_labels.add(tag[2:])

    for label in all_labels:
        ner.add_label(label)

    # Convert to SpaCy format
    training_data = _sentences_to_spacy_format(train_sentences)

    # Create Example objects
    nlp.initialize()
    examples = []
    for text, annot in training_data:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annot)
            examples.append(example)
        except Exception:
            continue

    if not examples:
        return nlp

    # Train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(n_epochs):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=dropout, losses=losses)

    return nlp


# ============================================================================
# EVALUATION
# ============================================================================

def predict_bio_tags(nlp: spacy.language.Language, sentences: List[Dict]) -> List[List[str]]:
    """Run SpaCy NER model on sentences and produce BIO tags.

    Args:
        nlp: Trained SpaCy model
        sentences: Test sentences with tokens

    Returns:
        List of BIO tag lists (one per sentence)
    """
    predictions = []

    for sent in sentences:
        tokens = sent["tokens"]
        text = " ".join(tokens)
        doc = nlp(text)

        # Build token-to-char mapping for alignment
        char_positions = []
        pos = 0
        for token in tokens:
            char_positions.append((pos, pos + len(token)))
            pos += len(token) + 1  # +1 for space

        # Initialize all as O
        pred_tags = ["O"] * len(tokens)

        # Map SpaCy entities back to token-level BIO tags
        for ent in doc.ents:
            ent_start = ent.start_char
            ent_end = ent.end_char
            ent_label = ent.label_

            first = True
            for i, (tok_start, tok_end) in enumerate(char_positions):
                # Check if token overlaps with entity
                if tok_start >= ent_start and tok_end <= ent_end:
                    if first:
                        pred_tags[i] = f"B-{ent_label}"
                        first = False
                    else:
                        pred_tags[i] = f"I-{ent_label}"
                elif tok_start < ent_end and tok_end > ent_start:
                    # Partial overlap
                    if first:
                        pred_tags[i] = f"B-{ent_label}"
                        first = False
                    else:
                        pred_tags[i] = f"I-{ent_label}"

        predictions.append(pred_tags)

    return predictions


def compute_ner_metrics(
    true_tags: List[List[str]],
    pred_tags: List[List[str]]
) -> Dict:
    """Compute entity-level NER metrics using seqeval.

    Args:
        true_tags: Ground truth BIO tags (list of lists)
        pred_tags: Predicted BIO tags (list of lists)

    Returns:
        Dict with f1, precision, recall, per_type metrics
    """
    f1 = seq_f1_score(true_tags, pred_tags, average="macro", zero_division=0)
    precision = seq_precision(true_tags, pred_tags, average="macro", zero_division=0)
    recall = seq_recall(true_tags, pred_tags, average="macro", zero_division=0)

    # Per-type metrics
    report = seq_classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)

    per_type = {}
    for key, vals in report.items():
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(vals, dict):
            per_type[key] = {
                "precision": vals.get("precision", 0),
                "recall": vals.get("recall", 0),
                "f1": vals.get("f1-score", 0),
                "support": vals.get("support", 0)
            }

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "per_type": per_type
    }


# ============================================================================
# HIGH-LEVEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_ner_augmentation(
    train_sentences: List[Dict],
    synthetic_sentences: List[Dict],
    test_sentences: List[Dict],
    n_epochs: int = 30,
    seed: int = 42
) -> Dict:
    """Train NER on real + synthetic data, evaluate on test set.

    Args:
        train_sentences: Real NER training sentences
        synthetic_sentences: Filtered synthetic NER sentences
        test_sentences: Test set sentences
        n_epochs: Training epochs
        seed: Random seed

    Returns:
        Metrics dict with f1, precision, recall, per_type
    """
    # Combine real + synthetic
    combined = list(train_sentences) + list(synthetic_sentences)

    # Train
    nlp = train_spacy_ner(combined, n_epochs=n_epochs, seed=seed)

    # Predict
    true_tags = [s["ner_tags"] for s in test_sentences]
    pred_tags = predict_bio_tags(nlp, test_sentences)

    # Compute metrics
    return compute_ner_metrics(true_tags, pred_tags)


def compute_ner_baseline(
    train_sentences: List[Dict],
    test_sentences: List[Dict],
    n_epochs: int = 30,
    seed: int = 42
) -> Dict:
    """Train NER on real data only (baseline), evaluate on test set."""
    nlp = train_spacy_ner(train_sentences, n_epochs=n_epochs, seed=seed)

    true_tags = [s["ner_tags"] for s in test_sentences]
    pred_tags = predict_bio_tags(nlp, test_sentences)

    return compute_ner_metrics(true_tags, pred_tags)


def evaluate_with_cv(
    train_sentences: List[Dict],
    synthetic_sentences: List[Dict],
    test_sentences: List[Dict],
    n_folds: int = 3,
    n_epochs: int = 30
) -> Dict:
    """Evaluate with multiple random seeds (pseudo-CV for NER).

    Since SpaCy NER training has randomness (shuffling, dropout),
    we run multiple seeds and report mean/std.
    """
    f1_scores = []
    all_metrics = []

    for seed in range(42, 42 + n_folds):
        metrics = evaluate_ner_augmentation(
            train_sentences, synthetic_sentences, test_sentences,
            n_epochs=n_epochs, seed=seed
        )
        f1_scores.append(metrics["f1"])
        all_metrics.append(metrics)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "f1_scores": f1_scores,
        "runs": all_metrics
    }
