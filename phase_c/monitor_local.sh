#!/bin/bash
# Monitor Phase C local execution

LOG_FILE=$(ls -t /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c/phaseC_seed42_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No log file found"
    exit 1
fi

clear
echo "════════════════════════════════════════════════════════"
echo "  Phase C - Local Monitor (with GPU)"
echo "════════════════════════════════════════════════════════"
echo ""
date
echo ""

# Check if process is running
if ps aux | grep "python3.*runner_phase2.py" | grep -v grep > /dev/null; then
    echo "Status: ✅ RUNNING"
    PROC_INFO=$(ps aux | grep "python3.*runner_phase2.py" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, RAM: "$4"%"}')
    echo "$PROC_INFO"
else
    echo "Status: ⚠️  NOT RUNNING (may have completed or failed)"
fi

echo ""
echo "Log file: $(basename $LOG_FILE)"
echo "Log size: $(du -h "$LOG_FILE" | awk '{print $1}')"
echo ""

# Check for output files
echo "Output files:"
if [ -f "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_c/phaseC_seed42_metrics.json" ]; then
    echo "  ✅ phaseC_seed42_metrics.json (COMPLETED!)"
else
    echo "  ⏳ phaseC_seed42_metrics.json (pending)"
fi

if [ -f "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_c/phaseC_seed42_synthetic.csv" ]; then
    SIZE=$(du -h "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_c/phaseC_seed42_synthetic.csv" | awk '{print $1}')
    echo "  ✅ phaseC_seed42_synthetic.csv ($SIZE)"
else
    echo "  ⏳ phaseC_seed42_synthetic.csv (pending)"
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "Last 40 lines of log:"
echo "─────────────────────────────────────────────────────────"
tail -40 "$LOG_FILE"

echo ""
echo "─────────────────────────────────────────────────────────"

# Check for adaptive temperature messages
TEMP_MSGS=$(grep "🌡️" "$LOG_FILE" 2>/dev/null)
if [ -n "$TEMP_MSGS" ]; then
    echo ""
    echo "Adaptive Temperature Adjustments:"
    echo "$TEMP_MSGS"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "Commands:"
echo "  Follow log: tail -f $LOG_FILE"
echo "  Check process: ps aux | grep python3 | grep runner"
echo "  Re-run monitor: ./monitor_local.sh"
echo "════════════════════════════════════════════════════════"
