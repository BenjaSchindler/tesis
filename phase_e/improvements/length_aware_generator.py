#!/usr/bin/env python3
"""
Length-Aware Synthetic Text Generator for SMOTE-LLM.

Problem: Synthetic texts are ~93% shorter than real texts (31-50 words vs 500 words).
Solution: Calculate target length per class and enforce in prompts + post-validation.

Usage:
    from length_aware_generator import LengthStats, get_length_enhanced_prompt

    # Calculate stats
    stats = LengthStats.from_dataframe(df, text_col="text", label_col="label")

    # Get length-aware prompt
    prompt = get_length_enhanced_prompt(
        base_prompt=original_prompt,
        target_class="ISFJ",
        length_stats=stats
    )

Author: Phase E Improvements
Date: 2024-11-30
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClassLengthStats:
    """Length statistics for a single class."""
    class_name: str
    mean_words: float
    std_words: float
    median_words: float
    p25_words: float  # 25th percentile
    p75_words: float  # 75th percentile
    min_words: float
    max_words: float
    sample_count: int

    @property
    def target_words(self) -> int:
        """Target word count for synthetic generation."""
        return int(self.median_words)

    @property
    def min_acceptable(self) -> int:
        """Minimum acceptable word count (25th percentile)."""
        return max(50, int(self.p25_words * 0.8))  # At least 50 words

    @property
    def max_acceptable(self) -> int:
        """Maximum acceptable word count (75th percentile + margin)."""
        return int(self.p75_words * 1.2)

    def is_acceptable_length(self, word_count: int) -> bool:
        """Check if a word count is acceptable for this class."""
        return self.min_acceptable <= word_count <= self.max_acceptable

    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "mean_words": float(round(self.mean_words, 1)),
            "std_words": float(round(self.std_words, 1)),
            "median_words": float(round(self.median_words, 1)),
            "p25_words": float(round(self.p25_words, 1)),
            "p75_words": float(round(self.p75_words, 1)),
            "min_words": float(round(self.min_words, 1)),
            "max_words": float(round(self.max_words, 1)),
            "sample_count": int(self.sample_count),
            "target_words": int(self.target_words),
            "min_acceptable": int(self.min_acceptable),
            "max_acceptable": int(self.max_acceptable),
        }


@dataclass
class LengthStats:
    """Length statistics for all classes in a dataset."""
    class_stats: Dict[str, ClassLengthStats] = field(default_factory=dict)
    global_mean: float = 0.0
    global_std: float = 0.0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label"
    ) -> "LengthStats":
        """Calculate length statistics from a DataFrame."""
        stats = cls()

        # Calculate word counts
        df = df.copy()
        df["_word_count"] = df[text_col].apply(lambda x: len(str(x).split()))

        # Global stats
        stats.global_mean = df["_word_count"].mean()
        stats.global_std = df["_word_count"].std()

        # Per-class stats
        for class_name in df[label_col].unique():
            class_df = df[df[label_col] == class_name]
            word_counts = class_df["_word_count"].values

            stats.class_stats[class_name] = ClassLengthStats(
                class_name=class_name,
                mean_words=np.mean(word_counts),
                std_words=np.std(word_counts),
                median_words=np.median(word_counts),
                p25_words=np.percentile(word_counts, 25),
                p75_words=np.percentile(word_counts, 75),
                min_words=np.min(word_counts),
                max_words=np.max(word_counts),
                sample_count=len(word_counts),
            )

        return stats

    def get_class_stats(self, class_name: str) -> Optional[ClassLengthStats]:
        """Get stats for a specific class."""
        return self.class_stats.get(class_name)

    def save(self, path: str) -> None:
        """Save stats to JSON file."""
        data = {
            "global_mean": round(self.global_mean, 1),
            "global_std": round(self.global_std, 1),
            "classes": {k: v.to_dict() for k, v in self.class_stats.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LengthStats":
        """Load stats from JSON file."""
        with open(path) as f:
            data = json.load(f)

        stats = cls()
        stats.global_mean = data["global_mean"]
        stats.global_std = data["global_std"]

        for class_name, class_data in data["classes"].items():
            stats.class_stats[class_name] = ClassLengthStats(
                class_name=class_data["class_name"],
                mean_words=class_data["mean_words"],
                std_words=class_data["std_words"],
                median_words=class_data["median_words"],
                p25_words=class_data["p25_words"],
                p75_words=class_data["p75_words"],
                min_words=class_data["min_words"],
                max_words=class_data["max_words"],
                sample_count=class_data["sample_count"],
            )

        return stats

    def print_summary(self) -> None:
        """Print a summary of length statistics."""
        print("\n" + "=" * 70)
        print("  LENGTH STATISTICS PER CLASS")
        print("=" * 70)
        print(f"\nGlobal: mean={self.global_mean:.0f} words, std={self.global_std:.0f}")
        print(f"\n{'Class':<8} {'Mean':>8} {'Median':>8} {'P25-P75':>15} {'Target':>8} {'Acceptable':>18}")
        print("-" * 70)

        for class_name in sorted(self.class_stats.keys()):
            s = self.class_stats[class_name]
            print(f"{class_name:<8} {s.mean_words:>8.0f} {s.median_words:>8.0f} "
                  f"{s.p25_words:>6.0f}-{s.p75_words:<6.0f} {s.target_words:>8} "
                  f"{s.min_acceptable:>6}-{s.max_acceptable:<6}")
        print()


def get_length_instruction(
    class_stats: ClassLengthStats,
    mode: str = "strict"  # "strict", "range", "approximate"
) -> str:
    """
    Generate a length instruction for the prompt.

    Args:
        class_stats: Statistics for the target class
        mode: How strict the length requirement should be
            - "strict": Exact target with small tolerance
            - "range": Acceptable range (p25-p75)
            - "approximate": Approximate target

    Returns:
        String instruction to add to the prompt
    """
    target = class_stats.target_words
    min_acc = class_stats.min_acceptable
    max_acc = class_stats.max_acceptable

    if mode == "strict":
        return (
            f"\n\nIMPORTANT LENGTH REQUIREMENT: "
            f"Each generated text MUST be approximately {target} words long "
            f"(minimum {min_acc} words, maximum {max_acc} words). "
            f"This matches the typical length of real {class_stats.class_name} texts. "
            f"Short responses will be rejected."
        )
    elif mode == "range":
        return (
            f"\n\nLENGTH REQUIREMENT: "
            f"Each text should be between {min_acc} and {max_acc} words. "
            f"Target approximately {target} words per text. "
            f"Maintain the detailed, elaborate style of real {class_stats.class_name} posts."
        )
    else:  # approximate
        return (
            f"\n\nNote: Aim for texts around {target} words long, "
            f"similar to authentic {class_stats.class_name} writing."
        )


def get_length_enhanced_prompt(
    base_prompt: str,
    target_class: str,
    length_stats: LengthStats,
    mode: str = "strict"
) -> str:
    """
    Enhance a prompt with length requirements.

    Args:
        base_prompt: The original generation prompt
        target_class: The MBTI class being generated
        length_stats: Pre-calculated length statistics
        mode: Length enforcement mode

    Returns:
        Enhanced prompt with length instructions
    """
    class_stats = length_stats.get_class_stats(target_class)

    if class_stats is None:
        print(f"  [Warning] No length stats for class {target_class}, using global average")
        # Use global average as fallback
        target = int(length_stats.global_mean)
        instruction = (
            f"\n\nLENGTH REQUIREMENT: Each text should be approximately {target} words. "
            f"Match the detailed style of real MBTI posts."
        )
        return base_prompt + instruction

    instruction = get_length_instruction(class_stats, mode)
    return base_prompt + instruction


def validate_synthetic_length(
    synthetic_text: str,
    target_class: str,
    length_stats: LengthStats,
    strict: bool = True
) -> Tuple[bool, int, str]:
    """
    Validate that a synthetic text meets length requirements.

    Args:
        synthetic_text: The generated text
        target_class: The MBTI class
        length_stats: Pre-calculated length statistics
        strict: If True, reject texts outside acceptable range

    Returns:
        Tuple of (is_valid, word_count, reason)
    """
    word_count = len(synthetic_text.split())
    class_stats = length_stats.get_class_stats(target_class)

    if class_stats is None:
        return True, word_count, "no_stats_available"

    if word_count < class_stats.min_acceptable:
        return False, word_count, f"too_short (min={class_stats.min_acceptable})"

    if strict and word_count > class_stats.max_acceptable:
        return False, word_count, f"too_long (max={class_stats.max_acceptable})"

    return True, word_count, "ok"


def filter_synthetics_by_length(
    synthetics: List[Dict],
    length_stats: LengthStats,
    text_key: str = "text",
    label_key: str = "label",
    strict: bool = True
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Filter synthetic samples by length requirements.

    Args:
        synthetics: List of synthetic samples with text and label
        length_stats: Pre-calculated length statistics
        text_key: Key for text in each sample dict
        label_key: Key for label in each sample dict
        strict: If True, apply strict length filtering

    Returns:
        Tuple of (filtered_synthetics, rejection_stats)
    """
    accepted = []
    rejection_stats = {"too_short": 0, "too_long": 0, "accepted": 0}

    for sample in synthetics:
        text = sample[text_key]
        label = sample[label_key]

        is_valid, word_count, reason = validate_synthetic_length(
            text, label, length_stats, strict
        )

        if is_valid:
            sample["_word_count"] = word_count
            accepted.append(sample)
            rejection_stats["accepted"] += 1
        elif "too_short" in reason:
            rejection_stats["too_short"] += 1
        elif "too_long" in reason:
            rejection_stats["too_long"] += 1

    return accepted, rejection_stats


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Length-Aware Generator Demo")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    sample_data = {
        "text": [
            " ".join(["word"] * np.random.randint(400, 600))  # ~500 words
            for _ in range(100)
        ],
        "label": ["ISFJ"] * 30 + ["ISTJ"] * 30 + ["INFJ"] * 40,
    }
    df = pd.DataFrame(sample_data)

    # Vary lengths by class
    df.loc[df["label"] == "ISFJ", "text"] = df.loc[df["label"] == "ISFJ", "text"].apply(
        lambda x: " ".join(["word"] * np.random.randint(450, 550))
    )
    df.loc[df["label"] == "ISTJ", "text"] = df.loc[df["label"] == "ISTJ", "text"].apply(
        lambda x: " ".join(["word"] * np.random.randint(480, 520))
    )

    # Calculate stats
    stats = LengthStats.from_dataframe(df)
    stats.print_summary()

    # Test prompt enhancement
    base_prompt = "Generate 3 MBTI personality texts for class ISFJ."
    enhanced = get_length_enhanced_prompt(base_prompt, "ISFJ", stats, mode="strict")

    print("\n--- Original Prompt ---")
    print(base_prompt)
    print("\n--- Enhanced Prompt ---")
    print(enhanced)

    # Test length validation
    print("\n--- Length Validation Tests ---")
    test_texts = [
        ("ISFJ", "Short text with only a few words."),  # Too short
        ("ISFJ", " ".join(["word"] * 500)),  # Good
        ("ISFJ", " ".join(["word"] * 1000)),  # Too long
    ]

    for label, text in test_texts:
        is_valid, wc, reason = validate_synthetic_length(text, label, stats)
        status = "PASS" if is_valid else "FAIL"
        print(f"  [{status}] {label}: {wc} words - {reason}")

    # Save/load test
    stats.save("/tmp/test_length_stats.json")
    loaded = LengthStats.load("/tmp/test_length_stats.json")
    print(f"\n  Save/Load test: {len(loaded.class_stats)} classes loaded")

    print("\n  Demo completed successfully!")
