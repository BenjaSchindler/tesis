#!/bin/bash
#
# Collect and Analyze 25-Seed Results
#

set -e

ZONE="us-central1-a"
BATCHES=(batch1 batch2 batch3 batch4 batch5)
RESULTS_DIR="phaseA_25seeds_results"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

declare -A BATCH_SEEDS
BATCH_SEEDS[batch1]="42 100 123 456 789"
BATCH_SEEDS[batch2]="111 222 333 444 555"
BATCH_SEEDS[batch3]="1000 2000 3000 4000 5000"
BATCH_SEEDS[batch4]="7 13 21 37 101"
BATCH_SEEDS[batch5]="1234 2345 3456 4567 5678"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Recopilando Resultados 25-Seed Fase A                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

# Download results from each VM
echo -e "${BLUE}Paso 1/3: Descargando archivos de 5 VMs (25 seeds)...${NC}"
echo ""

# Track VMs that were started temporarily
STARTED_VMS=()

for batch in "${BATCHES[@]}"; do
    VM_NAME="vm-${batch}"
    SEEDS="${BATCH_SEEDS[$batch]}"

    echo -e "${GREEN}Processing $VM_NAME (seeds: $SEEDS)...${NC}"

    # Check VM status
    VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(status)' 2>/dev/null || echo "NOT_FOUND")

    if [ "$VM_STATUS" = "NOT_FOUND" ]; then
        echo -e "  ${RED}⚠ VM not found, skipping${NC}"
        continue
    fi

    # Start VM if it's stopped
    if [ "$VM_STATUS" = "TERMINATED" ]; then
        echo -e "  ${YELLOW}VM is stopped. Starting temporarily to download results...${NC}"
        gcloud compute instances start "$VM_NAME" --zone="$ZONE" --quiet

        # Track this VM to stop it later
        STARTED_VMS+=("$VM_NAME")

        # Wait for VM to be ready
        echo -n "  Waiting for SSH"
        for i in {1..30}; do
            sleep 2
            echo -n "."
            if timeout 5 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="echo ready" &>/dev/null; then
                echo " ✓"
                break
            fi
        done
        echo ""
    fi

    # Create batch directory
    mkdir -p "$batch"

    # Download log
    gcloud compute scp "$VM_NAME:LaptopRuns/nohup_${batch}.log" "$batch/" --zone="$ZONE" --quiet 2>/dev/null || echo "  ⚠ log not found"

    # Download all seed results
    for seed in $SEEDS; do
        echo -n "  seed $seed: "
        gcloud compute scp "$VM_NAME:LaptopRuns/phaseA_seed${seed}_metrics.json" "$batch/" --zone="$ZONE" --quiet 2>/dev/null && echo -n "✓ metrics " || echo -n "✗ "
        gcloud compute scp "$VM_NAME:LaptopRuns/phaseA_seed${seed}_synthetic.csv" "$batch/" --zone="$ZONE" --quiet 2>/dev/null && echo -n "✓ synth " || echo -n "✗ "
        gcloud compute scp "$VM_NAME:LaptopRuns/phaseA_seed${seed}_augmented.csv" "$batch/" --zone="$ZONE" --quiet 2>/dev/null && echo "✓ aug" || echo "✗"
    done
    echo ""
done

# Stop VMs that were started temporarily
if [ ${#STARTED_VMS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Deteniendo VMs que se iniciaron temporalmente...${NC}"
    for vm in "${STARTED_VMS[@]}"; do
        echo -e "  Stopping $vm..."
        gcloud compute instances stop "$vm" --zone="$ZONE" --quiet &
    done
    wait
    echo -e "${GREEN}✓ VMs detenidas (costo ahora: ~\$0.01/hora por disco)${NC}"
fi

echo ""
echo -e "${GREEN}✓ Descarga completada${NC}"
echo ""

# Analyze results
echo -e "${BLUE}Paso 2/3: Analizando 25 resultados...${NC}"
echo ""

python3 << 'PYEOF'
import json
import os
import numpy as np
from glob import glob

# Collect all seeds from all batches
all_seeds = []
results = {}

batches = {
    'batch1': [42, 100, 123, 456, 789],
    'batch2': [111, 222, 333, 444, 555],
    'batch3': [1000, 2000, 3000, 4000, 5000],
    'batch4': [7, 13, 21, 37, 101],
    'batch5': [1234, 2345, 3456, 4567, 5678]
}

print("═" * 80)
print("RESULTADOS FASE A - 25 SEEDS ANALYSIS")
print("═" * 80)
print()

# Load all metrics
found_count = 0
for batch, seeds in batches.items():
    for seed in seeds:
        metrics_file = f"{batch}/phaseA_seed{seed}_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results[seed] = json.load(f)
            all_seeds.append(seed)
            found_count += 1
        else:
            results[seed] = None

print(f"Seeds encontrados: {found_count}/25")
print()

if found_count == 0:
    print("❌ No se encontraron resultados")
    exit(1)

# Group by batch
for batch, seeds in batches.items():
    print(f"─── {batch.upper()} (seeds: {seeds}) ───")
    batch_deltas = []
    for seed in seeds:
        if results[seed]:
            baseline_f1 = results[seed]['baseline']['macro_f1']
            augmented_f1 = results[seed]['augmented']['macro_f1']
            delta_pct = results[seed]['improvement']['f1_delta_pct']
            n_synthetics = results[seed]['synthetic_data']['accepted_count']
            batch_deltas.append(delta_pct)

            status = "✓" if delta_pct > 0 else "✗"
            print(f"  Seed {seed:4d}: Baseline={baseline_f1:.4f}, Aug={augmented_f1:.4f}, "
                  f"Delta={delta_pct:+.3f}%, Synth={n_synthetics:3d} {status}")
        else:
            print(f"  Seed {seed:4d}: ⚠ NO DATA")

    if batch_deltas:
        print(f"  Batch mean: {np.mean(batch_deltas):+.3f}% ± {np.std(batch_deltas):.3f}%")
    print()

# Overall statistics
deltas = []
baseline_f1s = []
augmented_f1s = []
n_synthetics_all = []

for seed in all_seeds:
    if results[seed]:
        baseline_f1 = results[seed]['baseline']['macro_f1']
        augmented_f1 = results[seed]['augmented']['macro_f1']
        delta_pct = results[seed]['improvement']['f1_delta_pct']
        n_synthetics = results[seed]['synthetic_data']['accepted_count']

        baseline_f1s.append(baseline_f1)
        augmented_f1s.append(augmented_f1)
        deltas.append(delta_pct)
        n_synthetics_all.append(n_synthetics)

print("═" * 80)
print("ESTADÍSTICAS AGREGADAS (25 seeds)")
print("═" * 80)
print()

if deltas:
    print(f"  Mean Delta:        {np.mean(deltas):+.4f}% ± {np.std(deltas):.4f}%")
    print(f"  Median Delta:      {np.median(deltas):+.4f}%")
    print(f"  Min Delta:         {np.min(deltas):+.4f}%")
    print(f"  Max Delta:         {np.max(deltas):+.4f}%")
    print(f"  25th percentile:   {np.percentile(deltas, 25):+.4f}%")
    print(f"  75th percentile:   {np.percentile(deltas, 75):+.4f}%")
    print()
    print(f"  Mean Synthetics:   {np.mean(n_synthetics_all):.1f} ± {np.std(n_synthetics_all):.1f}")
    print()
    print(f"  Success Rate:      {sum(1 for d in deltas if d > 0)}/{len(deltas)} seeds improved")
    print()

    # 95% confidence interval for mean
    from scipy import stats
    ci = stats.t.interval(0.95, len(deltas)-1, loc=np.mean(deltas), scale=stats.sem(deltas))
    print(f"  95% CI for mean:   [{ci[0]:+.4f}%, {ci[1]:+.4f}%]")
    print()

    # Check if target is met
    mean_delta = np.mean(deltas)
    if mean_delta >= 1.20:
        print(f"  ✅ META ALCANZADA: {mean_delta:+.4f}% ≥ +1.20% target")
    elif mean_delta >= 1.00:
        print(f"  ⚠ PARCIALMENTE EXITOSO: {mean_delta:+.4f}% (entre +1.00% y +1.20%)")
    else:
        print(f"  ❌ META NO ALCANZADA: {mean_delta:+.4f}% < +1.00%")
    print()

    # Statistical significance test (one-sample t-test against 0)
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    print(f"  Significancia estadística:")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"    ✅ Mejora es estadísticamente significativa (p < 0.05)")
    else:
        print(f"    ⚠ Mejora NO es significativa (p ≥ 0.05)")
    print()

    # Save summary
    summary = {
        'total_seeds': len(deltas),
        'seeds': all_seeds,
        'deltas': deltas,
        'baseline_f1s': baseline_f1s,
        'augmented_f1s': augmented_f1s,
        'n_synthetics': n_synthetics_all,
        'statistics': {
            'mean_delta_pct': float(np.mean(deltas)),
            'std_delta_pct': float(np.std(deltas)),
            'median_delta_pct': float(np.median(deltas)),
            'min_delta_pct': float(np.min(deltas)),
            'max_delta_pct': float(np.max(deltas)),
            'percentile_25': float(np.percentile(deltas, 25)),
            'percentile_75': float(np.percentile(deltas, 75)),
            'ci_95_lower': float(ci[0]),
            'ci_95_upper': float(ci[1]),
            'success_rate': f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    }

    with open('summary_25seeds.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("  ✓ Summary saved to summary_25seeds.json")

print("═" * 80)
PYEOF

echo ""
echo -e "${GREEN}✓ Análisis completado${NC}"
echo ""

# Cleanup VMs (optional)
echo -e "${BLUE}Paso 3/3: Limpieza de VMs${NC}"
echo ""
echo -e "${YELLOW}¿Deseas eliminar las 5 VMs para ahorrar costos? (y/n)${NC}"
read -r CLEANUP

if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
    for batch in "${BATCHES[@]}"; do
        VM_NAME="vm-${batch}"
        echo -e "${YELLOW}Deleting $VM_NAME...${NC}"
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet &
    done
    wait
    echo -e "${GREEN}✓ Todas las VMs eliminadas${NC}"
else
    echo -e "${BLUE}VMs mantenidas. Para eliminarlas después:${NC}"
    echo "  gcloud compute instances delete vm-batch{1,2,3,4,5} --zone=$ZONE"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ ANÁLISIS 25-SEED COMPLETADO                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Resultados en: $(pwd)${NC}"
echo "  - summary_25seeds.json: Resumen estadístico completo"
echo "  - batch*/: Resultados por VM"
echo ""
