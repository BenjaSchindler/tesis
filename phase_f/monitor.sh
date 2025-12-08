#!/bin/bash
# Phase F - Monitor script
# Shows progress of running experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-42 100 123}"
CONFIGS=(V1_baseline V2_high_vol V3_low_vol CF1_conf_band CF2_knn_only CF3_relaxed IP1_2x IP2_3x IP3_skip_high CMB1_balanced CMB2_aggressive CMB3_skip)

clear
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase F Monitor - $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

total=0
completed=0
running=0

for cfg in "${CONFIGS[@]}"; do
    echo -n "  $cfg: "
    for seed in $SEEDS; do
        total=$((total + 1))
        metrics_file="results/${cfg}_s${seed}_metrics.json"
        log_file="results/${cfg}_s${seed}.log"

        if [ -f "$metrics_file" ]; then
            completed=$((completed + 1))
            delta=$(python3 -c "import json; d=json.load(open('$metrics_file')); print(f\"{d.get('delta_f1_pct', 0):+.2f}%\")" 2>/dev/null || echo "?")
            echo -n "✓$seed($delta) "
        elif [ -f "$log_file" ] && pgrep -f "$cfg.*seed.*$seed" > /dev/null 2>&1; then
            running=$((running + 1))
            last_line=$(tail -1 "$log_file" 2>/dev/null | head -c 40)
            echo -n "⏳$seed "
        elif [ -f "$log_file" ]; then
            # Log exists but no metrics - might be failed or still running
            if tail -5 "$log_file" 2>/dev/null | grep -q "Error\|Exception\|Traceback"; then
                echo -n "✗$seed "
            else
                running=$((running + 1))
                echo -n "⏳$seed "
            fi
        else
            echo -n "·$seed "
        fi
    done
    echo ""
done

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  Progress: $completed/$total completed | $running running"
echo "───────────────────────────────────────────────────────────────"

# Show best results so far
echo ""
echo "Best results so far:"
for metrics in results/*_metrics.json; do
    [ -f "$metrics" ] || continue
    python3 -c "
import json
d = json.load(open('$metrics'))
name = '$metrics'.replace('results/', '').replace('_metrics.json', '')
delta = d.get('delta_f1_pct', 0)
print(f'  {name}: {delta:+.4f}%')
" 2>/dev/null
done | sort -t: -k2 -rn | head -5

echo ""
echo "Refresh with: watch -n 30 ./monitor.sh"
