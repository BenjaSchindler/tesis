#!/bin/bash
# Quick script to check for problematic classes in latest outputs

cd "$(dirname "$0")"

echo "=== Checking for ESFP and ESTJ in recent synthetics ==="
echo ""

for f in $(ls -t results/*_synth.csv 2>/dev/null | head -10); do
    esfp=$(grep -c "ESFP" "$f" 2>/dev/null || echo 0)
    estj=$(grep -c "ESTJ" "$f" 2>/dev/null || echo 0)
    total=$(wc -l < "$f" 2>/dev/null || echo 0)
    fname=$(basename "$f")

    if [ "$esfp" -gt 0 ] || [ "$estj" -gt 0 ]; then
        echo "* $fname: ESFP=$esfp, ESTJ=$estj (total=$total)"
    else
        echo "  $fname: ESFP=$esfp, ESTJ=$estj (total=$total)"
    fi
done

echo ""
echo "=== Distribution of latest synthetic file ==="
latest=$(ls -t results/*_synth.csv 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    echo "File: $latest"
    cut -d',' -f2 "$latest" | sort | uniq -c | sort -rn
fi
