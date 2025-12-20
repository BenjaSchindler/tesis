#!/bin/bash
# =============================================================================
# RARE_MLP Suite - Parallel Execution Script
# =============================================================================
# Run 25+ RARE_MLP configurations with MLP classifier + SOTA techniques
# Optimized for: Ryzen 9 5900x + 32GB RAM + RTX 3090 + OpenAI Tier 3
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PYTHON=python3
LOG_DIR="$SCRIPT_DIR/logs/rare_mlp"
RESULTS_DIR="$SCRIPT_DIR/results/rare_mlp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "  RARE_MLP Suite - Exhaustive Search for Rare Class Improvement"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Python: $PYTHON"
echo "  Log directory: $LOG_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "\033[0;31m[ERROR] OPENAI_API_KEY not set\033[0m"
    echo "Please set: export OPENAI_API_KEY='your-key'"
    exit 1
else
    echo -e "\033[0;32m[OK] OPENAI_API_KEY configured\033[0m"
fi

# Check Python and dependencies
$PYTHON -c "import sklearn; import numpy; import openai" 2>/dev/null || {
    echo -e "\033[0;31m[ERROR] Missing Python dependencies\033[0m"
    echo "Please install: pip install scikit-learn numpy openai"
    exit 1
}
echo -e "\033[0;32m[OK] Python dependencies available\033[0m"

echo ""
echo "========================================================================"
echo "  RARE_MLP Suite Contents"
echo "========================================================================"
echo ""
echo "Total Configurations: 25+"
echo ""
echo "Categories:"
echo "  A. Architecture Variations     (5 tests): Different MLP sizes"
echo "  B. Oversampling Variations     (3 tests): 20x/30x/50x multipliers"
echo "  C. Component Combinations      (3 tests): Top configs + MLP"
echo "  D. Class-Specific Focus        (4 tests): ESFJ/ESTJ/ESFP targeted"
echo "  E. Weighted/Dedup              (2 tests): Performance weighting"
echo "  F. SOTA - Focal Loss          (4 tests): gamma=2.0/5.0 + class weights"
echo "  G. SOTA - Embedding Mixup      (2 tests): Remix/Intra-class"
echo "  H. SOTA - Contrastive          (1 test):  Embedding refinement"
echo "  I. SOTA - Combined             (2 tests): Full SOTA + kitchen sink"
echo ""
echo "SOTA Techniques Applied:"
echo "  - Focal Loss (Lin et al. 2017): Down-weight easy examples"
echo "  - Class Weights: Inverse frequency weighting"
echo "  - Remix Mixup (Chou et al. 2020): Boost minority in interpolation"
echo "  - Intra-Class Mixup: Expand rare class feature space"
echo "  - Contrastive Refinement: Separate rare classes in embedding space"
echo ""
echo "Estimated Time: ~10 hours"
echo "========================================================================"
echo ""

read -p "Start RARE_MLP Suite? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "========================================================================"
echo "  Starting RARE_MLP Suite"
echo "========================================================================"
echo ""
echo "  Start time: $(date)"
echo "  Log file: $LOG_DIR/rare_mlp_suite_$TIMESTAMP.log"
echo ""

# Run the suite
$PYTHON -u experiments/exp_rare_mlp_suite.py 2>&1 | tee "$LOG_DIR/rare_mlp_suite_$TIMESTAMP.log"

echo ""
echo "========================================================================"
echo "  RARE_MLP Suite Complete!"
echo "========================================================================"
echo ""
echo "  End time: $(date)"
echo "  Results: $RESULTS_DIR/"
echo "  Log: $LOG_DIR/rare_mlp_suite_$TIMESTAMP.log"
echo ""

# Show summary
if [ -f "$RESULTS_DIR/rare_mlp_summary.json" ]; then
    echo "Summary:"
    $PYTHON -c "
import json
with open('$RESULTS_DIR/rare_mlp_summary.json') as f:
    data = json.load(f)
    print(f'  Total configs: {data[\"total_configs\"]}')
    print(f'  Significant: {data[\"significant_count\"]}')
    print()
    print('  Top 5 by Macro-F1:')
    for i, r in enumerate(data['top_10'][:5], 1):
        sig = 'sig' if r['significant'] else 'ns'
        print(f'    {i}. {r[\"name\"]}: {r[\"delta_pct\"]:+.2f}% ({sig})')
    print()
    print(f'  Best ESFJ: {data[\"best_esfj\"][\"name\"]} ({data[\"best_esfj\"][\"delta\"]:+.4f})')
    print(f'  Best ESTJ: {data[\"best_estj\"][\"name\"]} ({data[\"best_estj\"][\"delta\"]:+.4f})')
    print(f'  Best ESFP: {data[\"best_esfp\"][\"name\"]} ({data[\"best_esfp\"][\"delta\"]:+.4f})')
"
fi

echo ""
echo "Done!"
