#!/usr/bin/env python3
"""
Phase F Analysis: Per-class impact across configurations
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

# MBTI classes
CLASSES = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
           'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

def load_synth_by_class(config, seed=42):
    """Load synthetic counts by class for a config."""
    synth_path = RESULTS_DIR / f"{config}_s{seed}_synth.csv"
    if not synth_path.exists():
        return {}

    df = pd.read_csv(synth_path)
    return df['label'].value_counts().to_dict()

def load_all_configs():
    """Find all configs with synthetic data."""
    configs = set()
    for f in RESULTS_DIR.glob("*_s42_synth.csv"):
        config = f.stem.replace("_s42_synth", "")
        configs.add(config)
    return sorted(configs)

def analyze_class_coverage():
    """Analyze which classes get synthetics in which configs."""
    configs = load_all_configs()

    # Matrix: config x class -> count
    matrix = {}
    for config in configs:
        matrix[config] = load_synth_by_class(config)

    print("\n" + "="*100)
    print("PER-CLASS SYNTHETIC DISTRIBUTION")
    print("="*100)

    # Print header
    print(f"\n{'Config':<25}", end="")
    for cls in CLASSES:
        print(f"{cls:>6}", end="")
    print(f"{'Total':>8}")
    print("-"*100)

    # Print each config
    class_totals = defaultdict(int)
    for config in configs:
        print(f"{config:<25}", end="")
        total = 0
        for cls in CLASSES:
            count = matrix[config].get(cls, 0)
            class_totals[cls] += count
            total += count
            if count > 0:
                print(f"{count:>6}", end="")
            else:
                print(f"{'·':>6}", end="")
        print(f"{total:>8}")

    # Print totals
    print("-"*100)
    print(f"{'TOTAL':<25}", end="")
    grand_total = 0
    for cls in CLASSES:
        print(f"{class_totals[cls]:>6}", end="")
        grand_total += class_totals[cls]
    print(f"{grand_total:>8}")

    # Classes with zero synthetics
    zero_classes = [cls for cls in CLASSES if class_totals[cls] == 0]
    low_classes = [cls for cls in CLASSES if 0 < class_totals[cls] < 20]
    high_classes = [cls for cls in CLASSES if class_totals[cls] >= 50]

    print("\n" + "="*100)
    print("CLASS COVERAGE SUMMARY")
    print("="*100)
    print(f"\nClasses with ZERO synthetics: {zero_classes if zero_classes else 'None'}")
    print(f"Classes with LOW synthetics (<20): {low_classes}")
    print(f"Classes with HIGH synthetics (>=50): {high_classes}")

    return matrix, class_totals

def analyze_config_contribution():
    """Analyze which configs contribute unique synthetics."""
    configs = load_all_configs()

    print("\n" + "="*100)
    print("CONFIG CONTRIBUTION TO CLASSES")
    print("="*100)

    # For each class, which configs contribute?
    class_contributors = defaultdict(list)

    for config in configs:
        synth = load_synth_by_class(config)
        for cls, count in synth.items():
            if count > 0:
                class_contributors[cls].append((config, count))

    print(f"\n{'Class':<8} {'#Configs':>10} {'Total Synth':>12} {'Contributors':<60}")
    print("-"*100)

    for cls in CLASSES:
        contributors = class_contributors.get(cls, [])
        n_configs = len(contributors)
        total = sum(c[1] for c in contributors)

        if contributors:
            contrib_str = ", ".join([f"{c[0][:12]}({c[1]})" for c in sorted(contributors, key=lambda x: -x[1])[:4]])
        else:
            contrib_str = "None"

        status = "✓" if total > 0 else "✗"
        print(f"{cls:<8} {n_configs:>10} {total:>12} {contrib_str:<60} {status}")

def analyze_ensemble_diversity():
    """Analyze how ensembles combine different class contributions."""
    print("\n" + "="*100)
    print("ENSEMBLE DIVERSITY ANALYSIS")
    print("="*100)

    ensembles = {
        "ENS_Top3_G5": ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium"],
        "ENS_Top3": ["CMB3_skip", "CF1_conf_band", "V4_ultra"],
        "ENS_CMB3_V2": ["CMB3_skip", "V2_high_vol"],
        "ENS_CMB3_CF1": ["CMB3_skip", "CF1_conf_band"],
        "ENS_CMB3_G5": ["CMB3_skip", "G5_K25_medium"],
    }

    for ens_name, components in ensembles.items():
        print(f"\n### {ens_name}")
        print(f"Components: {' + '.join(components)}")

        # Combine synth by class
        combined = defaultdict(lambda: defaultdict(int))
        for comp in components:
            synth = load_synth_by_class(comp)
            for cls, count in synth.items():
                combined[cls][comp] = count

        print(f"\n{'Class':<8}", end="")
        for comp in components:
            print(f"{comp[:10]:>12}", end="")
        print(f"{'Total':>10}")
        print("-"*60)

        total_synth = 0
        for cls in CLASSES:
            if any(combined[cls].values()):
                print(f"{cls:<8}", end="")
                cls_total = 0
                for comp in components:
                    val = combined[cls].get(comp, 0)
                    cls_total += val
                    if val > 0:
                        print(f"{val:>12}", end="")
                    else:
                        print(f"{'·':>12}", end="")
                print(f"{cls_total:>10}")
                total_synth += cls_total

        print("-"*60)
        print(f"{'TOTAL':<8}", end="")
        for comp in components:
            comp_total = sum(load_synth_by_class(comp).values())
            print(f"{comp_total:>12}", end="")
        print(f"{total_synth:>10}")

def find_unique_contributors():
    """Find classes where only one config contributes."""
    print("\n" + "="*100)
    print("UNIQUE CONTRIBUTORS (classes with single source)")
    print("="*100)

    configs = load_all_configs()
    class_sources = defaultdict(set)

    for config in configs:
        if config.startswith("ENS_"):
            continue  # Skip ensembles
        synth = load_synth_by_class(config)
        for cls, count in synth.items():
            if count > 0:
                class_sources[cls].add(config)

    print(f"\n{'Class':<8} {'#Sources':>10} {'Sources':<60}")
    print("-"*80)

    for cls in CLASSES:
        sources = class_sources.get(cls, set())
        n_sources = len(sources)
        sources_str = ", ".join(sorted(sources)[:5]) if sources else "None"

        flag = "⚠ SINGLE" if n_sources == 1 else ""
        print(f"{cls:<8} {n_sources:>10} {sources_str:<60} {flag}")

def main():
    print("="*100)
    print("PHASE F: PER-CLASS IMPACT ANALYSIS")
    print("="*100)

    matrix, totals = analyze_class_coverage()
    analyze_config_contribution()
    analyze_ensemble_diversity()
    find_unique_contributors()

    # Save summary
    summary = {
        "class_totals": dict(totals),
        "zero_synth_classes": [c for c in CLASSES if totals[c] == 0],
        "top_beneficiaries": sorted(totals.items(), key=lambda x: -x[1])[:5]
    }

    with open(OUTPUT_DIR / "per_class_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {OUTPUT_DIR / 'per_class_summary.json'}")

if __name__ == "__main__":
    main()
