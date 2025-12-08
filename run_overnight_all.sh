#!/bin/bash
# =============================================================================
# OVERNIGHT RUN - Phase F Experiments + Phase G Nightrun2
# =============================================================================
# Runs both experiment phases sequentially overnight
# Total estimated time: ~4-6 hours
# =============================================================================

set -e

# Verify API key is set
export OPENAI_API_KEY="${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/overnight_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  OVERNIGHT RUN - Phase F + Phase G"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Start time: $(date)"
echo "  Log directory: $LOG_DIR"
echo ""
echo "  Phase F: 8 new experimental configs"
echo "  Phase G: 20 configs (W1-W9)"
echo ""
echo "  Total configs: 28"
echo "  Estimated time: 4-6 hours"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

# =============================================================================
# PHASE F - New Experiments
# =============================================================================

echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE F - New Experiments (8 configs)                              ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Started: $(date)"

cd "$SCRIPT_DIR/phase_f/experiments"

# Run Phase F experiments (with auto-yes to prompts)
echo "y" | bash run_experiments.sh 2>&1 | tee "$LOG_DIR/phase_f_$TIMESTAMP.log"

PHASE_F_STATUS=$?
echo ""
echo "Phase F completed with status: $PHASE_F_STATUS"
echo "Finished: $(date)"

# =============================================================================
# PHASE G - Nightrun2
# =============================================================================

echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE G - Nightrun2 (20 configs)                                   ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Started: $(date)"

cd "$SCRIPT_DIR/phase_g"

bash run_nightrun2.sh 2>&1 | tee "$LOG_DIR/phase_g_$TIMESTAMP.log"

PHASE_G_STATUS=$?
echo ""
echo "Phase G completed with status: $PHASE_G_STATUS"
echo "Finished: $(date)"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  OVERNIGHT RUN COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  End time: $(date)"
echo ""
echo "  Results:"
echo "    Phase F: $SCRIPT_DIR/phase_f/results/"
echo "    Phase G: $SCRIPT_DIR/phase_g/results/"
echo ""
echo "  Logs:"
echo "    Phase F: $LOG_DIR/phase_f_$TIMESTAMP.log"
echo "    Phase G: $LOG_DIR/phase_g_$TIMESTAMP.log"
echo ""

# Quick summary of Phase F results
echo "  === Phase F Quick Summary ==="
if [ -d "$SCRIPT_DIR/phase_f/results" ]; then
    for f in "$SCRIPT_DIR/phase_f/results"/EXP*_s42_kfold_k5.json; do
        if [ -f "$f" ]; then
            cfg=$(basename "$f" _s42_kfold_k5.json)
            delta=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['delta']['mean']*100:+.3f}%\")" 2>/dev/null || echo "ERR")
            echo "    $cfg: $delta"
        fi
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
