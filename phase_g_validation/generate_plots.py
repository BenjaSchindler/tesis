#!/usr/bin/env python3
"""
Generate plots and visualizations for Phase G Validation results
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

RESULTS_FILE = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results/FULL_SUMMARY.json")
PLOTS_DIR = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/plots")
PLOTS_DIR.mkdir(exist_ok=True)

def load_results():
    """Load compiled results"""
    with open(RESULTS_FILE) as f:
        return json.load(f)

def plot_top10_configs(results):
    """Plot top 10 configurations by delta %"""
    # Flatten all configs
    all_configs = []
    for category, configs in results.items():
        for cfg in configs:
            cfg["category"] = category
            all_configs.append(cfg)

    # Sort and get top 10
    top10 = sorted(all_configs, key=lambda x: x["delta_pct"], reverse=True)[:10]

    # Prepare data
    names = [c["config"] for c in top10]
    deltas = [c["delta_pct"] for c in top10]
    colors = ['green' if c["significant"] else 'gray' for c in top10]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), deltas, color=colors)

    # Customize
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('F1 Delta (%)')
    ax.set_title('Top 10 Configurations - Phase G Validation', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        ax.text(delta + 0.1, i, f'{delta:+.2f}%', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Statistically significant (p<0.05)'),
        Patch(facecolor='gray', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'top10_configs.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'top10_configs.png'}")
    plt.close()

def plot_wave_comparison(results):
    """Compare average delta % across waves"""
    wave_data = {}

    for wave_num in range(1, 10):
        wave_key = f"wave{wave_num}"
        if wave_key in results and results[wave_key]:
            deltas = [c["delta_pct"] for c in results[wave_key]]
            wave_data[f"Wave {wave_num}"] = np.mean(deltas)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    waves = list(wave_data.keys())
    avg_deltas = list(wave_data.values())

    bars = ax.bar(range(len(waves)), avg_deltas, color='steelblue', alpha=0.7)

    # Customize
    ax.set_xticks(range(len(waves)))
    ax.set_xticklabels(waves, rotation=45, ha='right')
    ax.set_ylabel('Average F1 Delta (%)')
    ax.set_title('Average Improvement by Wave', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Add value labels
    for bar, delta in zip(bars, avg_deltas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{delta:+.2f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'wave_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'wave_comparison.png'}")
    plt.close()

def plot_rare_class_heatmap(results):
    """Heatmap of rare class improvements"""
    # Collect rare class configs
    rare_configs = []
    for category, configs in results.items():
        for cfg in configs:
            per_class = cfg.get("per_class_delta", {})
            esfj = per_class.get("ESFJ", 0)
            esfp = per_class.get("ESFP", 0)
            estj = per_class.get("ESTJ", 0)

            if esfj != 0 or esfp != 0 or estj != 0:
                rare_configs.append({
                    "config": cfg["config"],
                    "ESFJ": esfj,
                    "ESFP": esfp,
                    "ESTJ": estj
                })

    if not rare_configs:
        print("No rare class improvements found")
        return

    # Sort by ESFJ delta
    rare_configs = sorted(rare_configs, key=lambda x: x["ESFJ"], reverse=True)[:15]

    # Prepare data
    configs = [c["config"] for c in rare_configs]
    data = np.array([[c["ESFJ"], c["ESFP"], c["ESTJ"]] for c in rare_configs])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.15)

    # Set ticks
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['ESFJ', 'ESFP', 'ESTJ'])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Delta', rotation=270, labelpad=20)

    # Add values
    for i in range(len(configs)):
        for j in range(3):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)

    ax.set_title('Rare Class Improvements - Top 15 Configs', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rare_class_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'rare_class_heatmap.png'}")
    plt.close()

def plot_pvalue_distribution(results):
    """Distribution of p-values"""
    p_values = []
    deltas = []
    significant = []

    for category, configs in results.items():
        for cfg in configs:
            p = cfg.get("p_value", 1.0)
            if not np.isnan(p):
                p_values.append(p)
                deltas.append(cfg["delta_pct"])
                significant.append(cfg.get("significant", False))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: delta vs p-value
    colors = ['green' if s else 'red' for s in significant]
    ax1.scatter(p_values, deltas, c=colors, alpha=0.6, s=50)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.axvline(0.05, color='red', linewidth=1.5, linestyle='--', label='α=0.05')
    ax1.set_xlabel('p-value')
    ax1.set_ylabel('F1 Delta (%)')
    ax1.set_title('Statistical Significance Analysis', fontweight='bold')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Histogram of p-values
    ax2.hist([p for p, s in zip(p_values, significant) if s],
             bins=20, alpha=0.7, label='Significant (p<0.05)', color='green')
    ax2.hist([p for p, s in zip(p_values, significant) if not s],
             bins=20, alpha=0.7, label='Not Significant', color='red')
    ax2.axvline(0.05, color='red', linewidth=1.5, linestyle='--', label='α=0.05')
    ax2.set_xlabel('p-value')
    ax2.set_ylabel('Count')
    ax2.set_title('P-Value Distribution', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pvalue_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'pvalue_analysis.png'}")
    plt.close()

def plot_multiclassifier_comparison():
    """Compare multi-classifier results"""
    # Load exp14b results (manually since JSON had error)
    classifiers = ["LogisticRegression", "MLP_256_128", "MLP_512_256_128", "XGBoost", "LightGBM"]
    baselines = [0.2272, 0.2273, 0.2075, 0.1788, 0.1677]
    augmented = [0.2308, 0.2306, 0.2333, 0.1745, 0.1667]
    deltas = [1.61, 1.47, 12.41, -2.41, -0.63]
    significant = [False, False, True, False, False]

    # Rare class deltas
    esfj_deltas = [0.0266, 0.1123, 0.1242, -0.0267, -0.0415]
    esfp_deltas = [0.0068, 0.0000, 0.0000, 0.0000, 0.0000]
    estj_deltas = [0.0024, 0.0000, 0.0179, 0.0000, 0.0000]

    # Plot 1: Overall Delta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['green' if s else 'gray' for s in significant]
    bars = ax1.bar(range(len(classifiers)), deltas, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(classifiers)))
    ax1.set_xticklabels(classifiers, rotation=45, ha='right')
    ax1.set_ylabel('F1 Delta (%)')
    ax1.set_title('Multi-Classifier Comparison - Overall F1 Delta', fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2,
                height + (0.5 if height > 0 else -1.5),
                f'{delta:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # Plot 2: Rare class comparison
    x = np.arange(len(classifiers))
    width = 0.25

    bars1 = ax2.bar(x - width, esfj_deltas, width, label='ESFJ', color='steelblue')
    bars2 = ax2.bar(x, esfp_deltas, width, label='ESFP', color='orange')
    bars3 = ax2.bar(x + width, estj_deltas, width, label='ESTJ', color='green')

    ax2.set_xticks(x)
    ax2.set_xticklabels(classifiers, rotation=45, ha='right')
    ax2.set_ylabel('F1 Delta (absolute)')
    ax2.set_title('Rare Class Improvements by Classifier', fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'multiclassifier_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'multiclassifier_comparison.png'}")
    plt.close()

def plot_category_summary(results):
    """Summary by category"""
    categories = ["wave1", "wave2", "wave3", "wave4", "wave5", "wave6",
                  "wave7", "wave9", "component", "rare_class", "ensembles"]

    data = []
    for cat in categories:
        if cat in results and results[cat]:
            configs = results[cat]
            avg_delta = np.mean([c["delta_pct"] for c in configs])
            num_sig = sum(1 for c in configs if c["significant"])
            total = len(configs)
            data.append({
                "category": cat,
                "avg_delta": avg_delta,
                "sig_rate": 100 * num_sig / total if total > 0 else 0,
                "count": total
            })

    # Sort by avg_delta
    data = sorted(data, key=lambda x: x["avg_delta"], reverse=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Average delta
    cats = [d["category"] for d in data]
    deltas = [d["avg_delta"] for d in data]

    bars = ax1.barh(range(len(cats)), deltas, color='teal', alpha=0.7)
    ax1.set_yticks(range(len(cats)))
    ax1.set_yticklabels(cats)
    ax1.set_xlabel('Average F1 Delta (%)')
    ax1.set_title('Average Improvement by Category', fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.invert_yaxis()

    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        ax1.text(delta + 0.1, i, f'{delta:+.2f}%', va='center', fontsize=9)

    # Significance rate
    sig_rates = [d["sig_rate"] for d in data]
    counts = [d["count"] for d in data]

    bars = ax2.barh(range(len(cats)), sig_rates, color='forestgreen', alpha=0.7)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(cats)
    ax2.set_xlabel('Significant Configs (%)')
    ax2.set_title('Statistical Significance Rate by Category', fontweight='bold')
    ax2.set_xlim([0, 105])
    ax2.invert_yaxis()

    for i, (bar, rate, count) in enumerate(zip(bars, sig_rates, counts)):
        ax2.text(rate + 2, i, f'{rate:.0f}% ({count})', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'category_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'category_summary.png'}")
    plt.close()

def main():
    """Generate all plots"""
    print("=" * 60)
    print("Generating Phase G Validation Plots")
    print("=" * 60)

    results = load_results()

    print("\n1. Top 10 Configurations...")
    plot_top10_configs(results)

    print("\n2. Wave Comparison...")
    plot_wave_comparison(results)

    print("\n3. Rare Class Heatmap...")
    plot_rare_class_heatmap(results)

    print("\n4. P-Value Analysis...")
    plot_pvalue_distribution(results)

    print("\n5. Multi-Classifier Comparison...")
    plot_multiclassifier_comparison()

    print("\n6. Category Summary...")
    plot_category_summary(results)

    print("\n" + "=" * 60)
    print(f"✓ All plots saved to: {PLOTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
