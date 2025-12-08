#!/bin/bash
# Analyze results from dual GPU run

echo "═══════════════════════════════════════════════════════════"
echo "  Dual GPU Results Analysis"
echo "═══════════════════════════════════════════════════════════"
echo ""

cd results_dual_gpu

# Count results
TOTAL_EXPECTED=11
METRICS_FOUND=$(ls *_metrics.json 2>/dev/null | wc -l)

echo "Results found: $METRICS_FOUND / $TOTAL_EXPECTED"
echo ""

if [ $METRICS_FOUND -eq 0 ]; then
    echo "❌ No results found. Check if experiments finished:"
    echo "   ps aux | grep python3 | grep runner_phase2"
    exit 1
fi

# Statistical analysis
python3 << 'EOF'
import json
import numpy as np
from pathlib import Path

print("═══════════════════════════════════════════════════════════")
print("STATISTICAL ANALYSIS - 11 Seeds")
print("═══════════════════════════════════════════════════════════")
print()

# Load all metrics
results = []
for metrics_file in sorted(Path('.').glob('*_metrics.json')):
    seed = metrics_file.stem.replace('seed', '').replace('_metrics', '')
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    baseline = data['baseline']['macro_f1']
    augmented = data['augmented']['macro_f1']
    delta = data['improvement']['f1_delta_abs']
    delta_pct = data['improvement']['f1_delta_pct']
    synthetics = data['synthetic_data']['accepted_count']

    results.append({
        'seed': seed,
        'baseline': baseline,
        'augmented': augmented,
        'delta': delta,
        'delta_pct': delta_pct,
        'synthetics': synthetics
    })

# Sort by seed
results.sort(key=lambda x: int(x['seed']))

print("Individual Results:")
print(f"{'Seed':<8} | {'Baseline':>10} | {'Augmented':>10} | {'Delta':>10} | {'Delta %':>10} | {'Synth':>6}")
print("-" * 80)

for r in results:
    print(f"{r['seed']:<8} | {r['baseline']:>10.5f} | {r['augmented']:>10.5f} | {r['delta']:>+10.5f} | {r['delta_pct']:>+9.2f}% | {r['synthetics']:>6d}")

print()
print("═══════════════════════════════════════════════════════════")
print("AGGREGATE STATISTICS")
print("═══════════════════════════════════════════════════════════")
print()

deltas = [r['delta'] for r in results]
deltas_pct = [r['delta_pct'] for r in results]

mean_delta = np.mean(deltas)
median_delta = np.median(deltas)
std_delta = np.std(deltas, ddof=1)
min_delta = np.min(deltas)
max_delta = np.max(deltas)

mean_pct = np.mean(deltas_pct)
median_pct = np.median(deltas_pct)
std_pct = np.std(deltas_pct, ddof=1)

# 95% confidence interval
n = len(deltas)
sem = std_delta / np.sqrt(n)
ci_95 = 1.96 * sem

print(f"Sample size:        {n} seeds")
print()
print(f"Mean delta:         {mean_delta:+.5f} ({mean_pct:+.2f}%)")
print(f"Median delta:       {median_delta:+.5f} ({median_pct:+.2f}%)")
print(f"Std deviation:      {std_delta:.5f} ({std_pct:.2f}%)")
print(f"Min delta:          {min_delta:+.5f} ({min(deltas_pct):+.2f}%)")
print(f"Max delta:          {max_delta:+.5f} ({max(deltas_pct):+.2f}%)")
print()
print(f"95% CI:             [{mean_delta - ci_95:+.5f}, {mean_delta + ci_95:+.5f}]")
print()

# Success rate
positive = sum(1 for d in deltas if d > 0)
success_rate = (positive / n) * 100
print(f"Success rate:       {positive}/{n} seeds improved ({success_rate:.1f}%)")
print()

# Compare with targets
target_single = 0.00377  # Phase B single seed
target_5seeds = -0.00119  # Phase C 5 seeds mean

print("═══════════════════════════════════════════════════════════")
print("COMPARISON WITH TARGETS")
print("═══════════════════════════════════════════════════════════")
print()
print(f"Target (Phase B single):     {target_single:+.5f} (+0.83%)")
print(f"Previous (5 seeds):          {target_5seeds:+.5f} (-0.26%)")
print(f"This run (11 seeds):         {mean_delta:+.5f} ({mean_pct:+.2f}%)")
print()

if mean_delta > target_single:
    print(f"✅ BEATS Phase B target by {(mean_delta - target_single):.5f}!")
elif mean_delta > target_5seeds:
    print(f"✅ Better than 5-seed run by {(mean_delta - target_5seeds):.5f}")
    print(f"   ({((mean_delta - target_5seeds) / abs(target_5seeds)) * 100:.1f}% improvement)")
else:
    print(f"❌ Below previous run by {(target_5seeds - mean_delta):.5f}")

print()

# Statistical significance (t-test vs 0)
from scipy import stats
t_stat, p_value = stats.ttest_1samp(deltas, 0)
print("Statistical Significance (vs null hypothesis delta=0):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.6f}")
if p_value < 0.05:
    print(f"  ✅ Statistically significant (p < 0.05)")
else:
    print(f"  ⚠️  Not statistically significant (p >= 0.05)")

print()

# Synthetics stats
synth_mean = np.mean([r['synthetics'] for r in results])
synth_std = np.std([r['synthetics'] for r in results], ddof=1)
print(f"Synthetic samples:  {synth_mean:.1f} ± {synth_std:.1f}")
print()

EOF

echo ""
echo "Results saved in results_dual_gpu/"
echo ""
