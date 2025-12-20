#!/usr/bin/env python3
"""Generate per-class improvement plot for optimal ensemble (ENS_TopG5_Extended)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Output directory
plots_dir = Path(__file__).parent / "plots"

# Per-class data from ENS_TopG5_Extended (Hold-out Correct methodology)
# Source: phase_g/RESULTS_HOLDOUT_CORRECT.md
per_class_data = {
    'ISFJ': {'baseline': 0.057, 'augmented': 0.256, 'delta_pp': 19.9},
    'ESTP': {'baseline': 0.000, 'augmented': 0.190, 'delta_pp': 19.0},
    'ENTJ': {'baseline': 0.080, 'augmented': 0.138, 'delta_pp': 5.8},
    'ENFP': {'baseline': 0.460, 'augmented': 0.462, 'delta_pp': 0.2},
    'INTP': {'baseline': 0.446, 'augmented': 0.447, 'delta_pp': 0.1},
    'ENTP': {'baseline': 0.415, 'augmented': 0.415, 'delta_pp': 0.0},
    'ENFJ': {'baseline': 0.000, 'augmented': 0.000, 'delta_pp': 0.0},
    'ESFJ': {'baseline': 0.000, 'augmented': 0.000, 'delta_pp': 0.0},
    'ESFP': {'baseline': 0.000, 'augmented': 0.000, 'delta_pp': 0.0},
    'ESTJ': {'baseline': 0.000, 'augmented': 0.000, 'delta_pp': 0.0},
    'ISFP': {'baseline': 0.068, 'augmented': 0.068, 'delta_pp': 0.0},
    'ISTJ': {'baseline': 0.047, 'augmented': 0.047, 'delta_pp': 0.0},
    'INFJ': {'baseline': 0.442, 'augmented': 0.441, 'delta_pp': -0.1},
    'INFP': {'baseline': 0.556, 'augmented': 0.555, 'delta_pp': -0.1},
    'ISTP': {'baseline': 0.289, 'augmented': 0.286, 'delta_pp': -0.3},
    'INTJ': {'baseline': 0.210, 'augmented': 0.196, 'delta_pp': -1.4},
}

# Define tiers based on baseline F1
def get_tier(baseline):
    if baseline < 0.20:
        return 'LOW'
    elif baseline < 0.45:
        return 'MID'
    else:
        return 'HIGH'

# Colors
COLORS = {
    'breakthrough': '#28A745',  # Green
    'good': '#2E86AB',          # Blue
    'slight': '#F18F01',        # Orange
    'neutral': '#6C757D',       # Gray
    'negative': '#C73E1D',      # Red
}

def get_color(delta):
    if delta >= 15:
        return COLORS['breakthrough']
    elif delta >= 5:
        return COLORS['good']
    elif delta > 0:
        return COLORS['slight']
    elif delta == 0:
        return COLORS['neutral']
    else:
        return COLORS['negative']


def fig_perclass_improvement():
    """Main per-class improvement plot."""

    # Sort by delta (descending)
    sorted_classes = sorted(per_class_data.items(), key=lambda x: x[1]['delta_pp'], reverse=True)

    classes = [c[0] for c in sorted_classes]
    deltas = [c[1]['delta_pp'] for c in sorted_classes]
    colors = [get_color(d) for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(range(len(classes)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Add tier labels
    for i, (cls, data) in enumerate(sorted_classes):
        tier = get_tier(data['baseline'])
        tier_color = {'LOW': '#e74c3c', 'MID': '#f39c12', 'HIGH': '#27ae60'}[tier]
        ax.text(-0.5, i, tier, ha='right', va='center', fontsize=8,
                color=tier_color, fontweight='bold')

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel(r'$\Delta$ F1 (pp)', fontsize=13)
    ax.set_ylabel('Clase MBTI', fontsize=13)
    ax.set_title('Mejora por Clase - ENS_TopG5_Extended (Hold-out Correcto)',
                 fontsize=14, fontweight='bold')

    # Add values at end of bars
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        width = bar.get_width()
        if val >= 0:
            ax.annotate(f'+{val:.1f}', xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=10,
                       fontweight='bold' if val >= 5 else 'normal')
        else:
            ax.annotate(f'{val:.1f}', xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(-3, 0), textcoords='offset points',
                       ha='right', va='center', fontsize=10, color=COLORS['negative'])

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['breakthrough'], edgecolor='black', label='Breakthrough (>=15 pp)'),
        Patch(facecolor=COLORS['good'], edgecolor='black', label='Buena mejora (5-15 pp)'),
        Patch(facecolor=COLORS['slight'], edgecolor='black', label='Mejora leve (0-5 pp)'),
        Patch(facecolor=COLORS['neutral'], edgecolor='black', label='Sin cambio (0 pp)'),
        Patch(facecolor=COLORS['negative'], edgecolor='black', label='Degradacion (<0 pp)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add summary stats
    avg_low = np.mean([d['delta_pp'] for c, d in per_class_data.items() if get_tier(d['baseline']) == 'LOW'])
    avg_mid = np.mean([d['delta_pp'] for c, d in per_class_data.items() if get_tier(d['baseline']) == 'MID'])
    avg_high = np.mean([d['delta_pp'] for c, d in per_class_data.items() if get_tier(d['baseline']) == 'HIGH'])

    stats_text = (
        f"Promedio por Tier:\n"
        f"  LOW (F1<0.20):  {avg_low:+.2f} pp\n"
        f"  MID (0.20-0.45): {avg_mid:+.2f} pp\n"
        f"  HIGH (>=0.45):  {avg_high:+.2f} pp"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace')

    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(min(deltas) - 2, max(deltas) + 3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'optimal_perclass_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'optimal_perclass_improvement.png', dpi=150, bbox_inches='tight')
    print(f"Saved: optimal_perclass_improvement.pdf/png")


def fig_perclass_baseline_vs_augmented():
    """Side-by-side baseline vs augmented F1 per class."""

    # Sort by baseline (ascending) to show progression
    sorted_classes = sorted(per_class_data.items(), key=lambda x: x[1]['baseline'])

    classes = [c[0] for c in sorted_classes]
    baselines = [c[1]['baseline'] for c in sorted_classes]
    augmented = [c[1]['augmented'] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, baselines, width, label='Baseline', color='#808080', edgecolor='black')
    bars2 = ax.bar(x + width/2, augmented, width, label='Augmented', color='#2ecc71', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=13)
    ax.set_xlabel('Clase MBTI (ordenado por F1 baseline)', fontsize=13)
    ax.set_title('F1 Baseline vs Augmented por Clase - ENS_TopG5_Extended',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)

    # Add delta annotations for significant changes
    for i, (cls, data) in enumerate(sorted_classes):
        delta = data['delta_pp']
        if abs(delta) >= 5:
            mid_y = max(data['baseline'], data['augmented']) + 0.02
            ax.annotate(f'+{delta:.1f}pp', xy=(i, mid_y), ha='center', fontsize=9,
                       fontweight='bold', color='#27ae60' if delta > 0 else '#e74c3c')

    # Add tier separators
    # Find where LOW ends and MID begins
    low_end = sum(1 for c, d in sorted_classes if get_tier(d['baseline']) == 'LOW') - 0.5
    mid_end = low_end + sum(1 for c, d in sorted_classes if get_tier(d['baseline']) == 'MID')

    ax.axvline(x=low_end, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=mid_end, color='orange', linestyle='--', alpha=0.5, linewidth=2)

    ax.text(low_end/2, 0.58, 'LOW\n(F1<0.20)', ha='center', fontsize=10, color='red', alpha=0.7)
    ax.text((low_end + mid_end)/2, 0.58, 'MID\n(0.20-0.45)', ha='center', fontsize=10, color='orange', alpha=0.7)
    ax.text((mid_end + len(classes))/2, 0.58, 'HIGH\n(>=0.45)', ha='center', fontsize=10, color='green', alpha=0.7)

    ax.set_ylim(0, 0.65)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'optimal_perclass_baseline_vs_aug.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'optimal_perclass_baseline_vs_aug.png', dpi=150, bbox_inches='tight')
    print(f"Saved: optimal_perclass_baseline_vs_aug.pdf/png")


def fig_tier_summary():
    """Summary by tier."""

    tiers = {'LOW': [], 'MID': [], 'HIGH': []}
    for cls, data in per_class_data.items():
        tier = get_tier(data['baseline'])
        tiers[tier].append(data['delta_pp'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Average delta by tier
    ax = axes[0]
    tier_names = ['LOW\n(F1<0.20)', 'MID\n(0.20-0.45)', 'HIGH\n(>=0.45)']
    tier_avgs = [np.mean(tiers['LOW']), np.mean(tiers['MID']), np.mean(tiers['HIGH'])]
    tier_colors = ['#e74c3c', '#f39c12', '#27ae60']

    bars = ax.bar(tier_names, tier_avgs, color=tier_colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, tier_avgs):
        ax.annotate(f'{val:+.2f} pp', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5 if val >= 0 else -15), textcoords='offset points',
                   ha='center', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel(r'$\Delta$ F1 Promedio (pp)', fontsize=13)
    ax.set_title('(a) Mejora Promedio por Tier', fontsize=14, fontweight='bold')
    ax.set_ylim(min(tier_avgs) - 1, max(tier_avgs) + 2)

    # Add class count
    for i, tier in enumerate(['LOW', 'MID', 'HIGH']):
        ax.text(i, -0.8, f'n={len(tiers[tier])}', ha='center', fontsize=10, color='gray')

    # Panel B: Distribution of deltas by tier
    ax = axes[1]
    positions = [1, 2, 3]
    bplot = ax.boxplot([tiers['LOW'], tiers['MID'], tiers['HIGH']],
                       positions=positions, patch_artist=True,
                       widths=0.6)

    for patch, color in zip(bplot['boxes'], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.set_xticks(positions)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel(r'$\Delta$ F1 (pp)', fontsize=13)
    ax.set_title('(b) Distribucion de Mejoras por Tier', fontsize=14, fontweight='bold')

    # Add individual points
    for i, (tier, color) in enumerate(zip(['LOW', 'MID', 'HIGH'], tier_colors), 1):
        y = tiers[tier]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, color='black', s=30, zorder=3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'optimal_tier_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'optimal_tier_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: optimal_tier_summary.pdf/png")


def fig_phasef_perclass():
    """Per-class improvement plot for Phase F Optimal (K-fold validation)."""

    # Load Phase F results from tier_impact
    import json
    tier_impact_path = Path(__file__).parent / "results" / "tier_impact" / "tier_impact_results.json"

    with open(tier_impact_path) as f:
        d = json.load(f)

    # Calculate actual deltas in pp
    phasef_data = {}
    for cls in d['baseline_f1s'].keys():
        baseline = d['baseline_f1s'][cls]
        augmented = d['augmented_f1s'][cls]
        delta_pp = (augmented - baseline) * 100
        phasef_data[cls] = {'baseline': baseline, 'augmented': augmented, 'delta_pp': delta_pp}

    # Sort by delta (descending)
    sorted_classes = sorted(phasef_data.items(), key=lambda x: x[1]['delta_pp'], reverse=True)

    classes = [c[0] for c in sorted_classes]
    deltas = [c[1]['delta_pp'] for c in sorted_classes]
    colors = [get_color(d) for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(range(len(classes)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Add tier labels
    for i, (cls, data) in enumerate(sorted_classes):
        tier = get_tier(data['baseline'])
        tier_color = {'LOW': '#e74c3c', 'MID': '#f39c12', 'HIGH': '#27ae60'}[tier]
        ax.text(-0.3, i, tier, ha='right', va='center', fontsize=8,
                color=tier_color, fontweight='bold')

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel(r'$\Delta$ F1 (pp)', fontsize=13)
    ax.set_ylabel('Clase MBTI', fontsize=13)
    ax.set_title('Mejora por Clase - Phase F Optimal (K-fold 5x3)',
                 fontsize=14, fontweight='bold')

    # Add values at end of bars
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        width = bar.get_width()
        if val >= 0:
            ax.annotate(f'+{val:.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=10,
                       fontweight='bold' if val >= 1 else 'normal')
        else:
            ax.annotate(f'{val:.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(-3, 0), textcoords='offset points',
                       ha='right', va='center', fontsize=10, color=COLORS['negative'])

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['good'], edgecolor='black', label='Buena mejora (>=1 pp)'),
        Patch(facecolor=COLORS['slight'], edgecolor='black', label='Mejora leve (0-1 pp)'),
        Patch(facecolor=COLORS['neutral'], edgecolor='black', label='Sin cambio (0 pp)'),
        Patch(facecolor=COLORS['negative'], edgecolor='black', label='Degradacion (<0 pp)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(min(deltas) - 0.5, max(deltas) + 0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / 'phasef_perclass_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'phasef_perclass_improvement.png', dpi=150, bbox_inches='tight')
    print(f"Saved: phasef_perclass_improvement.pdf/png")


def fig_comparison_kfold_vs_holdout():
    """Compare Phase F (K-fold) vs ENS_TopG5_Extended (Hold-out) per class."""

    import json
    tier_impact_path = Path(__file__).parent / "results" / "tier_impact" / "tier_impact_results.json"

    with open(tier_impact_path) as f:
        d = json.load(f)

    # Phase F data
    phasef_data = {}
    for cls in d['baseline_f1s'].keys():
        baseline = d['baseline_f1s'][cls]
        augmented = d['augmented_f1s'][cls]
        delta_pp = (augmented - baseline) * 100
        phasef_data[cls] = delta_pp

    # Get common classes
    common_classes = set(phasef_data.keys()) & set(per_class_data.keys())
    common_classes = sorted(common_classes)

    # Sort by holdout delta
    common_classes = sorted(common_classes, key=lambda c: per_class_data[c]['delta_pp'], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(common_classes))
    width = 0.35

    kfold_deltas = [phasef_data[c] for c in common_classes]
    holdout_deltas = [per_class_data[c]['delta_pp'] for c in common_classes]

    bars1 = ax.bar(x - width/2, kfold_deltas, width, label='Phase F (K-fold)', color='#808080', edgecolor='black')
    bars2 = ax.bar(x + width/2, holdout_deltas, width, label='ENS_TopG5_Extended (Hold-out)', color='#2ecc71', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(common_classes, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel(r'$\Delta$ F1 (pp)', fontsize=13)
    ax.set_xlabel('Clase MBTI', fontsize=13)
    ax.set_title('Comparacion K-fold vs Hold-out Correcto por Clase', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations for big differences
    for i, cls in enumerate(common_classes):
        if holdout_deltas[i] >= 5:
            ax.annotate(f'+{holdout_deltas[i]:.1f}', xy=(i + width/2, holdout_deltas[i]),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_kfold_vs_holdout.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'comparison_kfold_vs_holdout.png', dpi=150, bbox_inches='tight')
    print(f"Saved: comparison_kfold_vs_holdout.pdf/png")


if __name__ == '__main__':
    print("Generating per-class plots for optimal configuration...")
    print(f"Output directory: {plots_dir}\n")

    fig_perclass_improvement()
    fig_perclass_baseline_vs_augmented()
    fig_tier_summary()
    fig_phasef_perclass()
    fig_comparison_kfold_vs_holdout()

    print("\nDone! Generated 5 per-class plots.")
