#!/bin/bash
# ============================================================================
# Run ALL PyTorch MLP Phase F experiments in parallel batches
# ============================================================================
# Total: 11 experiments, ~57 configurations
# Estimated time: ~1.5-2 hours with parallelization
# ============================================================================

set -e

cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation

# Create results directory
mkdir -p results/pytorch_phase_f

echo "============================================================"
echo "Phase F Validation with PyTorch MLP"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    exit 1
fi
echo "✓ OPENAI_API_KEY configured"

# Check CUDA
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
echo ""

# ============================================================================
# BATCH 1: Priority experiments (temperature, n-shot, budget)
# ============================================================================
echo "============================================================"
echo "BATCH 1: Temperature, N-shot, Budget (in parallel)"
echo "============================================================"

python3 experiments/exp08_temperature_pytorch.py > results/pytorch_phase_f/log_exp08.txt 2>&1 &
PID1=$!
echo "  Started exp08_temperature (PID $PID1)"

python3 experiments/exp09_nshot_pytorch.py > results/pytorch_phase_f/log_exp09.txt 2>&1 &
PID2=$!
echo "  Started exp09_nshot (PID $PID2)"

python3 experiments/exp07c_budget_pytorch.py > results/pytorch_phase_f/log_exp07c.txt 2>&1 &
PID3=$!
echo "  Started exp07c_budget (PID $PID3)"

echo "  Waiting for Batch 1..."
wait $PID1 $PID2 $PID3
echo "✓ Batch 1 complete"
echo ""

# ============================================================================
# BATCH 2: Filters and clustering
# ============================================================================
echo "============================================================"
echo "BATCH 2: Filter cascade, Clustering, K-neighbors (in parallel)"
echo "============================================================"

python3 experiments/exp04_filter_pytorch.py > results/pytorch_phase_f/log_exp04.txt 2>&1 &
PID1=$!
echo "  Started exp04_filter (PID $PID1)"

python3 experiments/exp01_clustering_pytorch.py > results/pytorch_phase_f/log_exp01.txt 2>&1 &
PID2=$!
echo "  Started exp01_clustering (PID $PID2)"

python3 experiments/exp03_kneighbors_pytorch.py > results/pytorch_phase_f/log_exp03.txt 2>&1 &
PID3=$!
echo "  Started exp03_kneighbors (PID $PID3)"

echo "  Waiting for Batch 2..."
wait $PID1 $PID2 $PID3
echo "✓ Batch 2 complete"
echo ""

# ============================================================================
# BATCH 3: Thresholds and weights
# ============================================================================
echo "============================================================"
echo "BATCH 3: Thresholds, Weights, Weight-by-tier (in parallel)"
echo "============================================================"

python3 experiments/exp05_thresholds_pytorch.py > results/pytorch_phase_f/log_exp05.txt 2>&1 &
PID1=$!
echo "  Started exp05_thresholds (PID $PID1)"

python3 experiments/exp07a_weights_pytorch.py > results/pytorch_phase_f/log_exp07a.txt 2>&1 &
PID2=$!
echo "  Started exp07a_weights (PID $PID2)"

python3 experiments/exp07b_weight_tier_pytorch.py > results/pytorch_phase_f/log_exp07b.txt 2>&1 &
PID3=$!
echo "  Started exp07b_weight_tier (PID $PID3)"

echo "  Waiting for Batch 3..."
wait $PID1 $PID2 $PID3
echo "✓ Batch 3 complete"
echo ""

# ============================================================================
# BATCH 4: Anchor strategies and Tier analysis
# ============================================================================
echo "============================================================"
echo "BATCH 4: Anchor strategies, Tier analysis (in parallel)"
echo "============================================================"

python3 experiments/exp02_anchor_pytorch.py > results/pytorch_phase_f/log_exp02.txt 2>&1 &
PID1=$!
echo "  Started exp02_anchor (PID $PID1)"

python3 experiments/exp06_tier_pytorch.py > results/pytorch_phase_f/log_exp06.txt 2>&1 &
PID2=$!
echo "  Started exp06_tier (PID $PID2)"

echo "  Waiting for Batch 4..."
wait $PID1 $PID2
echo "✓ Batch 4 complete"
echo ""

# ============================================================================
# Compile Summary
# ============================================================================
echo "============================================================"
echo "Compiling Results Summary"
echo "============================================================"

python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

results_dir = Path("results/pytorch_phase_f")
summary = {"experiments": {}, "timestamp": datetime.now().isoformat()}

exp_files = {
    "exp01_clustering": "Clustering (K_max)",
    "exp02_anchor": "Anchor Strategies",
    "exp03_kneighbors": "K-Neighbors",
    "exp04_filter": "Filter Cascade",
    "exp05_thresholds": "Adaptive Thresholds",
    "exp06_tier": "Tier Impact",
    "exp07a_weights": "Synthetic Weights",
    "exp07b_weight_tier": "Weight by Tier",
    "exp07c_budget": "Budget",
    "exp08_temperature": "Temperature",
    "exp09_nshot": "N-shot",
}

for exp_file, exp_name in exp_files.items():
    json_path = results_dir / f"{exp_file}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        # Extract best result
        if "results" in data:
            results = data["results"]
            if results:
                # Find best by delta_pp
                if isinstance(results[0], dict):
                    best = max(results, key=lambda x: x.get("delta_pp", 0))
                    summary["experiments"][exp_name] = {
                        "best_config": best.get("config_value"),
                        "delta_pp": best.get("delta_pp"),
                        "p_value": best.get("p_value"),
                        "significant": best.get("significant"),
                    }
        elif "macro_f1" in data:  # Tier experiment
            summary["experiments"][exp_name] = {
                "delta_pp": data["macro_f1"]["delta_pp"],
                "p_value": data["macro_f1"]["p_value"],
            }
        print(f"  ✓ {exp_name}")
    else:
        print(f"  ✗ {exp_name} (not found)")

# Save summary
with open(results_dir / "SUMMARY_pytorch_phasef.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Summary saved to {results_dir / 'SUMMARY_pytorch_phasef.json'}")
EOF

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results in: results/pytorch_phase_f/"
echo "LaTeX tables in: ../Escrito_Tesis/Tables/"
echo ""

# Print quick summary
echo "Quick Summary:"
echo "============================================================"
for f in results/pytorch_phase_f/exp*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        # Try to extract best delta
        delta=$(python3 -c "
import json
try:
    with open('$f') as f:
        d = json.load(f)
    if 'results' in d and d['results']:
        best = max(d['results'], key=lambda x: x.get('delta_pp', 0))
        print(f\"{best.get('delta_pp', 0):+.2f}\")
    elif 'macro_f1' in d:
        print(f\"{d['macro_f1']['delta_pp']:+.2f}\")
except:
    print('N/A')
" 2>/dev/null)
        printf "  %-25s %s pp\n" "$name" "$delta"
    fi
done
echo "============================================================"
