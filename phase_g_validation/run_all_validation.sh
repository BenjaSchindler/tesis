#!/bin/bash
# Phase G Validation - Run All Experiments
#
# Hardware: RTX 3090 + Ryzen 9 5900x + 32GB RAM + High OpenAI tier
# Estimated time: ~4-5 hours with parallelism
#
# Usage:
#   ./run_all_validation.sh          # Run all experiments in parallel
#   ./run_all_validation.sh --seq    # Run sequentially (debugging)
#   ./run_all_validation.sh exp01    # Run specific experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY not set${NC}"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Phase G Validation - Full Run${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Configuration:"
echo "  - K-Fold: 5 splits x 3 repeats = 15 folds per config"
echo "  - Configs: 27 individual + 5 ensembles"
echo "  - Parallel API calls: 25 concurrent"
echo "  - Hardware: RTX 3090 + Ryzen 9 5900x"
echo ""

# Check for parallel command
HAS_PARALLEL=false
if command -v parallel &> /dev/null; then
    HAS_PARALLEL=true
    echo -e "${GREEN}GNU Parallel found - will use parallel execution${NC}"
else
    echo -e "${YELLOW}GNU Parallel not found - will run sequentially${NC}"
    echo "Install with: sudo apt install parallel"
fi

# Parse arguments
RUN_MODE="parallel"
SPECIFIC_EXP=""

if [ "$1" == "--seq" ]; then
    RUN_MODE="sequential"
elif [ -n "$1" ]; then
    SPECIFIC_EXP="$1"
fi

# Create logs directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/validation_${TIMESTAMP}.log"

echo ""
echo -e "${BLUE}Starting at $(date)${NC}"
echo -e "${BLUE}Log file: ${LOG_FILE}${NC}"
echo ""

# Function to run single experiment
run_exp() {
    local exp=$1
    local exp_name=$(basename "$exp" .py)
    local exp_log="logs/${exp_name}_${TIMESTAMP}.log"

    echo -e "${YELLOW}Starting: ${exp_name}${NC}"
    python3 -u "experiments/${exp}" 2>&1 | tee "$exp_log"
    echo -e "${GREEN}Completed: ${exp_name}${NC}"
}

# Run specific experiment if requested
if [ -n "$SPECIFIC_EXP" ]; then
    echo -e "${BLUE}Running specific experiment: ${SPECIFIC_EXP}${NC}"
    run_exp "${SPECIFIC_EXP}.py"
    exit 0
fi

# Sequential mode
if [ "$RUN_MODE" == "sequential" ] || [ "$HAS_PARALLEL" == "false" ]; then
    echo -e "${BLUE}Running experiments sequentially...${NC}"
    echo ""

    # Phase F components first (foundation)
    run_exp "exp01_phase_f_components.py"

    # Wave experiments
    run_exp "exp02_wave1_gates.py"
    run_exp "exp03_wave2_volume.py"
    run_exp "exp04_wave3_filters.py"
    run_exp "exp05_wave4_targeting.py"
    run_exp "exp06_wave5_prompting.py"
    run_exp "exp07_wave6_temperature.py"
    run_exp "exp08_wave7_yolo.py"
    run_exp "exp09_wave8_models.py"
    run_exp "exp10_wave9_combinations.py"

    # Phase F derived (problem class focus)
    run_exp "exp12_pf_derived.py"

    # Ensembles last
    run_exp "exp11_ensemble_validation.py"

else
    # Parallel mode
    echo -e "${BLUE}Running experiments in parallel (4 jobs max)...${NC}"
    echo ""

    # Batch 1: Phase F components + Wave 1-3 (foundation experiments)
    echo -e "${YELLOW}Batch 1: Foundation experiments${NC}"
    parallel -j 4 --halt soon,fail=1 --progress \
        python3 -u experiments/{}.py ">" logs/{}_${TIMESTAMP}.log 2>&1 ::: \
        exp01_phase_f_components \
        exp02_wave1_gates \
        exp03_wave2_volume \
        exp04_wave3_filters

    echo -e "${GREEN}Batch 1 complete${NC}"

    # Batch 2: Wave 4-7
    echo -e "${YELLOW}Batch 2: Wave experiments 4-7${NC}"
    parallel -j 4 --halt soon,fail=1 --progress \
        python3 -u experiments/{}.py ">" logs/{}_${TIMESTAMP}.log 2>&1 ::: \
        exp05_wave4_targeting \
        exp06_wave5_prompting \
        exp07_wave6_temperature \
        exp08_wave7_yolo

    echo -e "${GREEN}Batch 2 complete${NC}"

    # Batch 3: Wave 8-9 + Phase F derived
    echo -e "${YELLOW}Batch 3: Final wave experiments + PF derived${NC}"
    parallel -j 3 --halt soon,fail=1 --progress \
        python3 -u experiments/{}.py ">" logs/{}_${TIMESTAMP}.log 2>&1 ::: \
        exp09_wave8_models \
        exp10_wave9_combinations \
        exp12_pf_derived

    echo -e "${GREEN}Batch 3 complete${NC}"

    # Ensembles last (requires component synthetics)
    echo -e "${YELLOW}Final: Ensemble validation${NC}"
    python3 -u experiments/exp11_ensemble_validation.py 2>&1 | tee "logs/exp11_ensemble_validation_${TIMESTAMP}.log"
    echo -e "${GREEN}Ensemble validation complete${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   All experiments completed!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results saved in: results/"
echo "Logs saved in: logs/"
echo ""
echo -e "${BLUE}Completed at $(date)${NC}"

# Generate summary
echo ""
echo "Generating summary..."

python3 -c "
import json
from pathlib import Path

results_dir = Path('results')
summary = {'individual': {}, 'ensembles': {}}

# Collect individual results
for wave_dir in results_dir.iterdir():
    if wave_dir.is_dir() and wave_dir.name != 'ensembles':
        for result_file in wave_dir.glob('*_kfold.json'):
            with open(result_file) as f:
                data = json.load(f)
                config = data['config_name']
                summary['individual'][config] = {
                    'delta_pct': data['delta_pct'],
                    'p_value': data['p_value'],
                    'significant': data['significant'],
                    'n_synthetic': data['n_synthetic']
                }

# Collect ensemble results
ensemble_dir = results_dir / 'ensembles'
if ensemble_dir.exists():
    for result_file in ensemble_dir.glob('*_kfold.json'):
        with open(result_file) as f:
            data = json.load(f)
            name = data['config_name']
            summary['ensembles'][name] = {
                'delta_pct': data['delta_pct'],
                'p_value': data['p_value'],
                'significant': data['significant'],
                'n_synthetic': data['n_synthetic']
            }

# Print summary
print()
print('='*70)
print('VALIDATION SUMMARY')
print('='*70)

print()
print('INDIVIDUAL CONFIGS:')
print('-'*70)
for config, stats in sorted(summary['individual'].items(), key=lambda x: -x[1]['delta_pct']):
    sig = '*' if stats['significant'] else ''
    print(f\"  {config:25} delta={stats['delta_pct']:+.2f}% p={stats['p_value']:.4f} {sig}\")

print()
print('ENSEMBLES:')
print('-'*70)
for ens, stats in sorted(summary['ensembles'].items(), key=lambda x: -x[1]['delta_pct']):
    sig = '*' if stats['significant'] else ''
    print(f\"  {ens:25} delta={stats['delta_pct']:+.2f}% p={stats['p_value']:.4f} {sig}\")

# Save summary
with open('results/validation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print('Summary saved to: results/validation_summary.json')
print('='*70)
"
