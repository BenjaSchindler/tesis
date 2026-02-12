#!/usr/bin/env python3
"""
Generate publication-quality visualizations for embedding ablation study.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_data():
    """Load and parse results."""
    results_path = Path("/home/benja/Desktop/Tesis/filters/results/embedding_ablation/final_results.json")
    with open(results_path) as f:
        data = json.load(f)
    return data['results']

def extract_paired_deltas(results, model, nshot=None, dataset=None):
    """Extract paired SMOTE vs soft_weighted deltas."""
    filtered = results
    if model:
        filtered = [r for r in filtered if r['embedding_model'] == model]
    if nshot is not None:
        filtered = [r for r in filtered if r['n_shot'] == nshot]
    if dataset:
        filtered = [r for r in filtered if r['dataset'] == dataset]

    paired_data = defaultdict(lambda: {'smote': None, 'soft': None})
    for r in filtered:
        key = (r['dataset'], r['n_shot'], r['seed'])
        if r['method'] == 'smote':
            paired_data[key]['smote'] = r['f1_macro']
        elif r['method'] == 'soft_weighted':
            paired_data[key]['soft'] = r['f1_macro']

    deltas = []
    smote_vals = []
    soft_vals = []
    for vals in paired_data.values():
        if vals['smote'] is not None and vals['soft'] is not None:
            deltas.append(vals['soft'] - vals['smote'])
            smote_vals.append(vals['smote'])
            soft_vals.append(vals['soft'])

    return np.array(deltas), np.array(smote_vals), np.array(soft_vals)

def plot_overall_comparison(results, output_dir):
    """Figure 1: Overall performance by model."""
    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']
    stats_data = []

    for model in models:
        deltas, smote, soft = extract_paired_deltas(results, model)
        mean_delta = np.mean(deltas)
        ci_95 = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
        stats_data.append({
            'model': model,
            'delta': mean_delta,
            'ci_lower': mean_delta - ci_95,
            'ci_upper': mean_delta + ci_95
        })

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(models))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = ax.bar(x_pos, [s['delta']*100 for s in stats_data], color=colors, alpha=0.7, edgecolor='black')

    # Error bars
    yerr = [[s['delta']*100 - s['ci_lower']*100 for s in stats_data],
            [s['ci_upper']*100 - s['delta']*100 for s in stats_data]]
    ax.errorbar(x_pos, [s['delta']*100 for s in stats_data], yerr=yerr,
                fmt='none', ecolor='black', capsize=5, capthick=2)

    # Highlight best
    best_idx = np.argmax([s['delta'] for s in stats_data])
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Embedding Model', fontweight='bold')
    ax.set_ylabel('Mean Δ F1 vs SMOTE (pp)', fontweight='bold')
    ax.set_title('Geometric Filtering Improvement Across Embedding Models\n(n=63 paired comparisons per model)',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('-', '\n') for m in models])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max([s['delta']*100 for s in stats_data]) * 1.2)

    # Add dimension labels
    dims = [768, 1024, 1024, 384]
    for i, (bar, dim) in enumerate(zip(bars, dims)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{dim}d',
                ha='center', va='bottom', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig1_overall_comparison.png'}")
    plt.close()

def plot_nshot_breakdown(results, output_dir):
    """Figure 2: N-shot breakdown by model."""
    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']
    nshots = [10, 25, 50]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(nshots))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, model in enumerate(models):
        means = []
        errs = []
        for nshot in nshots:
            deltas, _, _ = extract_paired_deltas(results, model, nshot=nshot)
            mean_delta = np.mean(deltas)
            ci_95 = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
            means.append(mean_delta * 100)
            errs.append(ci_95 * 100)

        ax.bar(x + i*width, means, width, label=model, color=colors[i],
               alpha=0.7, edgecolor='black')
        ax.errorbar(x + i*width, means, yerr=errs, fmt='none',
                   ecolor='black', capsize=3, alpha=0.7)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('N-shot', fontweight='bold')
    ax.set_ylabel('Mean Δ F1 vs SMOTE (pp)', fontweight='bold')
    ax.set_title('Geometric Filtering Gains by N-Shot and Model\n(Error bars: 95% CI)',
                 fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{n}-shot' for n in nshots])
    ax.legend(title='Embedding Model', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_nshot_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig2_nshot_breakdown.png'}")
    plt.close()

def plot_dimensionality_analysis(results, output_dir):
    """Figure 3: Dimensionality vs performance."""
    dim_map = {
        'mpnet-base': 768,
        'bge-large': 1024,
        'e5-large': 1024,
        'bge-small': 384
    }

    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']
    dims = [dim_map[m] for m in models]
    deltas = []
    colors_map = {384: '#d62728', 768: '#1f77b4', 1024: '#ff7f0e'}
    colors = [colors_map[d] for d in dims]

    for model in models:
        delta, _, _ = extract_paired_deltas(results, model)
        deltas.append(np.mean(delta) * 100)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(dims, deltas, s=300, c=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add model labels
    for model, dim, delta in zip(models, dims, deltas):
        ax.annotate(model, (dim, delta), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Add trend line (to show no clear trend)
    z = np.polyfit(dims, deltas, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(dims)-50, max(dims)+50, 100)
    ax.plot(x_trend, p(x_trend), "--", color='gray', alpha=0.5, linewidth=2,
           label=f'Linear fit (slope={z[0]:.4f})')

    ax.set_xlabel('Embedding Dimension', fontweight='bold')
    ax.set_ylabel('Mean Δ F1 vs SMOTE (pp)', fontweight='bold')
    ax.set_title('Dimensionality vs Filtering Effectiveness\n(No monotonic relationship)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim(300, 1100)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_dimensionality.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig3_dimensionality.png'}")
    plt.close()

def plot_dataset_heatmap(results, output_dir):
    """Figure 4: Dataset-model heatmap."""
    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']
    datasets = sorted(set(r['dataset'] for r in results))

    matrix = np.zeros((len(datasets), len(models)))

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            deltas, _, _ = extract_paired_deltas(results, model, dataset=dataset)
            matrix[i, j] = np.mean(deltas) * 100

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Δ F1 vs SMOTE (pp)', rotation=-90, va="bottom", fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels([d.replace('_', ' ') for d in datasets])

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=7)

    ax.set_title('Dataset-Model Performance Heatmap\n(Δ F1 vs SMOTE in pp)',
                 fontweight='bold', pad=20)
    ax.set_xlabel('Embedding Model', fontweight='bold')
    ax.set_ylabel('Dataset', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig4_dataset_heatmap.png'}")
    plt.close()

def plot_win_rate_comparison(results, output_dir):
    """Figure 5: Win rate by model."""
    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']
    win_rates = []

    for model in models:
        deltas, _, _ = extract_paired_deltas(results, model)
        wins = np.sum(deltas > 0)
        total = len(deltas)
        win_rate = 100 * wins / total
        win_rates.append(win_rate)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.barh(models, win_rates, color=colors, alpha=0.7, edgecolor='black')

    # Highlight best
    best_idx = np.argmax(win_rates)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline')
    ax.set_xlabel('Win Rate (%)', fontweight='bold')
    ax.set_ylabel('Embedding Model', fontweight='bold')
    ax.set_title('Win Rate: Soft Weighting > SMOTE\n(out of 63 comparisons per model)',
                 fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')

    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_win_rates.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig5_win_rates.png'}")
    plt.close()

def plot_consistency_analysis(results, output_dir):
    """Figure 6: Cross-model consistency by dataset-nshot."""
    datasets = sorted(set(r['dataset'] for r in results))
    nshots = [10, 25, 50]
    models = ['mpnet-base', 'bge-large', 'e5-large', 'bge-small']

    consistency_data = []

    for dataset in datasets:
        for nshot in nshots:
            wins = 0
            for model in models:
                deltas, _, _ = extract_paired_deltas(results, model, nshot=nshot, dataset=dataset)
                if len(deltas) > 0 and np.mean(deltas) > 0:
                    wins += 1

            consistency_data.append({
                'dataset': dataset,
                'nshot': nshot,
                'wins': wins,
                'label': f'{dataset}_{nshot}shot'
            })

    # Group by nshot
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    for ax_idx, nshot in enumerate(nshots):
        subset = [d for d in consistency_data if d['nshot'] == nshot]
        datasets_subset = [d['dataset'].replace('_', ' ') for d in subset]
        wins_subset = [d['wins'] for d in subset]

        colors = ['red' if w == 0 else 'orange' if w <= 2 else 'yellow' if w == 3 else 'green'
                 for w in wins_subset]

        y_pos = np.arange(len(datasets_subset))
        axes[ax_idx].barh(y_pos, wins_subset, color=colors, alpha=0.7, edgecolor='black')
        axes[ax_idx].set_xlabel('Models Improved', fontweight='bold')
        axes[ax_idx].set_title(f'{nshot}-shot', fontweight='bold')
        axes[ax_idx].set_xlim(0, 4)
        axes[ax_idx].set_xticks([0, 1, 2, 3, 4])
        axes[ax_idx].axvline(2, color='black', linestyle='--', alpha=0.3, linewidth=1)
        axes[ax_idx].grid(axis='x', alpha=0.3)

        if ax_idx == 0:
            axes[ax_idx].set_yticks(y_pos)
            axes[ax_idx].set_yticklabels(datasets_subset, fontsize=8)
        else:
            axes[ax_idx].set_yticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='All 4 models improve'),
        Patch(facecolor='yellow', alpha=0.7, label='3 models improve'),
        Patch(facecolor='orange', alpha=0.7, label='1-2 models improve'),
        Patch(facecolor='red', alpha=0.7, label='No models improve')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Cross-Model Consistency by Dataset and N-Shot\n', fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / 'fig6_consistency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig6_consistency.png'}")
    plt.close()

def main():
    results = load_data()
    output_dir = Path("/home/benja/Desktop/Tesis/filters/results/embedding_ablation/figures")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating visualizations for embedding ablation study...")
    print(f"Total results loaded: {len(results)}")
    print()

    plot_overall_comparison(results, output_dir)
    plot_nshot_breakdown(results, output_dir)
    plot_dimensionality_analysis(results, output_dir)
    plot_dataset_heatmap(results, output_dir)
    plot_win_rate_comparison(results, output_dir)
    plot_consistency_analysis(results, output_dir)

    print()
    print("=" * 80)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
