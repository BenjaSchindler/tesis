#!/usr/bin/env python3
"""
Extract all metrics from Phase E experiments for trend analysis.
Creates unified CSV/JSON datasets for analysis.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def extract_config_info(filename: str) -> dict:
    """Extract config name, seed from filename."""
    # Pattern: ConfigName_sXXX_YYYYMMDD_HHMMSS_metrics.json
    match = re.match(r'([A-Za-z0-9_]+)_s(\d+)_(\d+)_\d+_metrics\.json', filename)
    if match:
        return {
            'config': match.group(1),
            'seed': int(match.group(2)),
            'date': match.group(3)
        }
    # Fallback for other patterns
    match = re.match(r'([A-Za-z0-9_]+)_metrics\.json', filename)
    if match:
        return {'config': match.group(1), 'seed': 0, 'date': ''}
    return {'config': filename, 'seed': 0, 'date': ''}


def extract_metrics_from_json(filepath: Path) -> dict:
    """Extract all relevant metrics from a metrics JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    info = extract_config_info(filepath.name)

    # Basic metrics
    baseline_f1 = data.get('baseline', {}).get('macro_f1', 0)
    augmented_f1 = data.get('augmented', {}).get('macro_f1', 0)
    delta_f1 = augmented_f1 - baseline_f1
    delta_f1_pct = delta_f1 * 100

    # Rejection analysis
    rejection = data.get('rejection_analysis', {})
    total_generated = rejection.get('total_generated', 0)
    total_accepted = rejection.get('total_accepted', 0)
    acceptance_rate = rejection.get('overall_acceptance_rate', 0)
    rejection_reasons = rejection.get('rejection_reasons', {})

    # Aggregate quality metrics from per_class_quality
    per_class_quality = data.get('per_class_quality', {})

    all_similarities_centroid = []
    all_similarities_anchor = []
    all_anchor_purities = []
    all_anchor_qualities = []
    all_classifier_confs = []
    all_token_counts = []

    per_class_data = []

    for class_name, class_data in per_class_quality.items():
        samples = class_data.get('per_sample', [])
        rejection_stats = class_data.get('rejection_stats', {})

        for sample in samples:
            if sample.get('similarity_to_centroid'):
                all_similarities_centroid.append(sample['similarity_to_centroid'])
            if sample.get('similarity_to_anchor'):
                all_similarities_anchor.append(sample['similarity_to_anchor'])
            if sample.get('anchor_purity'):
                all_anchor_purities.append(sample['anchor_purity'])
            if sample.get('anchor_quality'):
                all_anchor_qualities.append(sample['anchor_quality'])
            if sample.get('classifier_confidence'):
                all_classifier_confs.append(sample['classifier_confidence'])
            if sample.get('token_count'):
                all_token_counts.append(sample['token_count'])

        # Per-class data
        baseline_report = data.get('baseline', {}).get('report', {})
        augmented_report = data.get('augmented', {}).get('report', {})

        class_baseline_f1 = baseline_report.get(class_name, {}).get('f1-score', 0)
        class_augmented_f1 = augmented_report.get(class_name, {}).get('f1-score', 0)
        class_support = baseline_report.get(class_name, {}).get('support', 0)

        per_class_data.append({
            'class': class_name,
            'baseline_f1': class_baseline_f1,
            'augmented_f1': class_augmented_f1,
            'delta_f1': class_augmented_f1 - class_baseline_f1,
            'support': class_support,
            'n_accepted': class_data.get('total_accepted', 0),
            'n_candidates': class_data.get('total_candidates', 0),
            'acceptance_rate': class_data.get('acceptance_rate', 0),
            'rejected_knn': rejection_stats.get('knn', 0),
            'rejected_classifier': rejection_stats.get('classifier', 0),
            'rejected_similarity': rejection_stats.get('similarity', 0),
            'rejected_repel': rejection_stats.get('repel', 0),
        })

    # Synthetic data details
    synthetic_data = data.get('synthetic_data', {})
    per_class_details = synthetic_data.get('per_class_details', {})

    avg_quality_score = 0
    avg_anchor_purity_synth = 0
    avg_anchor_cohesion = 0
    n_classes = 0

    for class_name, details in per_class_details.items():
        if details.get('quality_score'):
            avg_quality_score += details['quality_score']
            n_classes += 1
        if details.get('anchor_purity'):
            avg_anchor_purity_synth += details['anchor_purity']
        if details.get('anchor_cohesion'):
            avg_anchor_cohesion += details['anchor_cohesion']

    if n_classes > 0:
        avg_quality_score /= n_classes
        avg_anchor_purity_synth /= n_classes
        avg_anchor_cohesion /= n_classes

    # Compute averages
    result = {
        **info,
        'baseline_f1': baseline_f1,
        'augmented_f1': augmented_f1,
        'delta_f1': delta_f1,
        'delta_f1_pct': delta_f1_pct,

        # Rejection metrics
        'total_generated': total_generated,
        'total_accepted': total_accepted,
        'acceptance_rate': acceptance_rate,
        'rejected_knn': rejection_reasons.get('knn', 0),
        'rejected_classifier': rejection_reasons.get('classifier', 0),
        'rejected_similarity': rejection_reasons.get('similarity', 0),
        'rejected_repel': rejection_reasons.get('repel', 0),

        # Quality metrics (averaged)
        'avg_similarity_centroid': sum(all_similarities_centroid) / len(all_similarities_centroid) if all_similarities_centroid else 0,
        'avg_similarity_anchor': sum(all_similarities_anchor) / len(all_similarities_anchor) if all_similarities_anchor else 0,
        'avg_anchor_purity': sum(all_anchor_purities) / len(all_anchor_purities) if all_anchor_purities else 0,
        'avg_anchor_quality': sum(all_anchor_qualities) / len(all_anchor_qualities) if all_anchor_qualities else 0,
        'avg_classifier_conf': sum(all_classifier_confs) / len(all_classifier_confs) if all_classifier_confs else 0,
        'avg_token_count': sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0,

        # Synthetic data quality
        'avg_quality_score': avg_quality_score,
        'avg_anchor_purity_synth': avg_anchor_purity_synth,
        'avg_anchor_cohesion': avg_anchor_cohesion,

        # Per class data
        'per_class': per_class_data,
    }

    return result


def extract_all_baseline_class_f1(data: dict) -> dict:
    """Extract per-class F1 from baseline report."""
    report = data.get('baseline', {}).get('report', {})
    class_f1 = {}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            class_f1[class_name] = metrics['f1-score']
    return class_f1


def get_config_group(config: str) -> str:
    """Get the experiment group (A, B, C, etc.) from config name."""
    if config.startswith('A'):
        return 'A_reasoning'
    elif config.startswith('B'):
        return 'B_ip_scaling'
    elif config.startswith('C'):
        return 'C_volume'
    elif config.startswith('D'):
        return 'D_filters'
    elif config.startswith('E'):
        return 'E_minority'
    elif config.startswith('F'):
        return 'F_length'
    elif config.startswith('G'):
        return 'G_combo'
    elif config.startswith('H'):
        return 'H_experimental'
    else:
        return 'other'


def main():
    results_dir = Path('/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/small_datasets_tests/results')
    output_dir = Path('/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/analysis')

    # Find all metrics files
    metrics_files = list(results_dir.glob('*_metrics.json'))
    print(f"Found {len(metrics_files)} metrics files")

    # Extract all data
    all_experiments = []
    all_per_class = []

    for filepath in sorted(metrics_files):
        try:
            metrics = extract_metrics_from_json(filepath)

            # Add group
            metrics['group'] = get_config_group(metrics['config'])

            # Store per-class data separately
            per_class = metrics.pop('per_class')
            for pc in per_class:
                pc['config'] = metrics['config']
                pc['seed'] = metrics['seed']
                pc['group'] = metrics['group']
                pc['exp_delta_f1'] = metrics['delta_f1_pct']
                all_per_class.append(pc)

            all_experiments.append(metrics)

        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")

    # Create DataFrames
    df_experiments = pd.DataFrame(all_experiments)
    df_per_class = pd.DataFrame(all_per_class)

    # Sort by delta
    df_experiments = df_experiments.sort_values('delta_f1_pct', ascending=False)

    # Save to CSV
    df_experiments.to_csv(output_dir / 'experiments_data.csv', index=False)
    df_per_class.to_csv(output_dir / 'per_class_data.csv', index=False)

    # Save to JSON for detailed access
    with open(output_dir / 'experiments_data.json', 'w') as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nExtracted {len(all_experiments)} experiments")
    print(f"Extracted {len(all_per_class)} per-class records")
    print(f"\nSaved to:")
    print(f"  - {output_dir / 'experiments_data.csv'}")
    print(f"  - {output_dir / 'per_class_data.csv'}")
    print(f"  - {output_dir / 'experiments_data.json'}")

    # Quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)

    print(f"\nTotal experiments: {len(df_experiments)}")
    print(f"Improved: {len(df_experiments[df_experiments['delta_f1_pct'] > 0])}")
    print(f"Worsened: {len(df_experiments[df_experiments['delta_f1_pct'] < 0])}")
    print(f"Average delta: {df_experiments['delta_f1_pct'].mean():.3f}%")

    print("\nBy group:")
    group_stats = df_experiments.groupby('group')['delta_f1_pct'].agg(['mean', 'std', 'count'])
    print(group_stats.sort_values('mean', ascending=False).to_string())

    return df_experiments, df_per_class


if __name__ == '__main__':
    main()
