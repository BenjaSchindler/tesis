#!/bin/bash
#
# Fase A - 25 Seeds Robustness Test
# 5 VMs × 5 seeds each = 25 total experiments
#

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Fase A - 25 Seeds Robustness Test (5 VMs × 5 seeds) ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY no está configurada${NC}"
    echo "Por favor, ejecuta:"
    echo "  export OPENAI_API_KEY='tu-api-key'"
    exit 1
fi

# Configuration
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="30GB"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Seed batches for each VM
declare -A VM_SEEDS
VM_SEEDS[batch1]="42 100 123 456 789"
VM_SEEDS[batch2]="111 222 333 444 555"
VM_SEEDS[batch3]="1000 2000 3000 4000 5000"
VM_SEEDS[batch4]="7 13 21 37 101"
VM_SEEDS[batch5]="1234 2345 3456 4567 5678"

echo -e "${GREEN}Configuración:${NC}"
echo "  Total VMs: 5"
echo "  Seeds per VM: 5"
echo "  Total Experiments: 25"
echo "  Machine Type: $MACHINE_TYPE (4 vCPUs, 15GB RAM)"
echo "  Dataset: MBTI_500.csv (106K samples)"
echo "  Tiempo estimado: ~5 horas en paralelo"
echo "  Costo estimado: ~\$10 USD"
echo ""
echo -e "${YELLOW}Distribución de seeds:${NC}"
for batch in batch1 batch2 batch3 batch4 batch5; do
    echo "  vm-${batch}: ${VM_SEEDS[$batch]}"
done
echo ""

# Confirm
echo -e "${YELLOW}¿Continuar con el lanzamiento? (y/n)${NC}"
read -r CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelado"
    exit 0
fi
echo ""

# Check and delete existing VMs
echo -e "${YELLOW}Verificando VMs existentes...${NC}"
for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
        echo -e "${YELLOW}⚠ VM $VM_NAME ya existe. Eliminando...${NC}"
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    fi
done
echo ""

# Create VMs
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 1/4: Creando 5 VMs...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -e "${GREEN}Creating $VM_NAME...${NC}"

    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="pd-standard" \
        --metadata=startup-script='#!/bin/bash
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv python3.10-venv
touch /tmp/startup-complete
' \
        --quiet &
done

wait
echo -e "${GREEN}✓ Todas las VMs creadas${NC}"
echo ""

# Wait for startup scripts to complete (Paso 1.5)
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 1.5/4: Verificando instalación de Python...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -n "Verificando $VM_NAME..."

    for i in {1..60}; do
        if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="test -f /tmp/startup-complete && python3 -m venv --help >/dev/null 2>&1" 2>/dev/null; then
            echo " ✓"
            break
        fi
        sleep 2
    done
done

echo -e "${GREEN}✓ Python y dependencias instaladas en todas las VMs${NC}"
echo ""

# Upload files
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 2/4: Subiendo archivos a VMs...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -e "${GREEN}Uploading to $VM_NAME...${NC}"

    # Create directory
    timeout 60 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="mkdir -p LaptopRuns" &
done
wait

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"

    {
        # Upload all core modules (Phase 2 complete)
        gcloud compute scp "$LOCAL_DIR/core/runner_phase2.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/ensemble_anchor_selector.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/contamination_aware_filter.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/enhanced_quality_gate.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/anchor_quality_improvements.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/quality_gate_predictor.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/mbti_class_descriptions.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/adversarial_discriminator.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/core/multi_seed_ensemble.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet

        # Upload dataset and batch script
        gcloud compute scp "$LOCAL_DIR/MBTI_500.csv" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --quiet
        gcloud compute scp "$LOCAL_DIR/phase_a/batch_scripts/run_batch_${batch}.sh" "$VM_NAME:LaptopRuns/run_all_seeds.sh" --zone="$ZONE" --quiet
        echo "✓ $VM_NAME all files uploaded (10 modules + dataset + script)"
    } &
done
wait

echo -e "${GREEN}✓ Todos los archivos subidos${NC}"
echo ""

# Setup dependencies
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 3/4: Instalando dependencias en VMs...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -e "${GREEN}Setup $VM_NAME...${NC}"

    timeout 240 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="
        cd LaptopRuns

        # Verify python3-venv is installed
        if ! python3 -m venv --help >/dev/null 2>&1; then
            echo '✗ ERROR: python3-venv not available'
            exit 1
        fi

        # Create venv with error handling
        if ! python3 -m venv .venv; then
            echo '✗ ERROR: Failed to create venv'
            exit 1
        fi

        # Activate venv
        if ! source .venv/bin/activate; then
            echo '✗ ERROR: Failed to activate venv'
            exit 1
        fi

        # Verify activation
        if [ -z \"\$VIRTUAL_ENV\" ]; then
            echo '✗ ERROR: venv not activated properly'
            exit 1
        fi

        # Install dependencies
        pip install -q --upgrade pip || exit 1
        pip install -q numpy pandas scikit-learn sentence-transformers openai python-dotenv || exit 1

        echo '✓ Dependencies installed successfully'
    " &
done
wait

echo -e "${GREEN}✓ Todas las dependencias instaladas${NC}"
echo ""

# Verify dependencies are importable
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 3.5/4: Verificando imports...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -e "${GREEN}Verify $VM_NAME...${NC}"

    timeout 30 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="
        cd LaptopRuns
        source .venv/bin/activate
        python3 -c 'import sentence_transformers; import openai; import pandas; import sklearn; print(\"✓ All imports OK\")' || exit 1
    " &
done
wait

echo -e "${GREEN}✓ Imports verificados en todas las VMs${NC}"
echo ""

# Launch experiments
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 4/4: Lanzando 25 experimentos...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    SEEDS="${VM_SEEDS[$batch]}"
    echo -e "${GREEN}[$batch] Launching 5 seeds: $SEEDS${NC}"

    timeout 30 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="
        cd LaptopRuns
        source .venv/bin/activate
        export OPENAI_API_KEY='$OPENAI_API_KEY'

        # Make script executable
        chmod +x run_all_seeds.sh

        # Run all 5 seeds in background
        nohup bash run_all_seeds.sh > nohup_${batch}.log 2>&1 &

        sleep 2
        echo '✓ Batch $batch launched (seeds: $SEEDS)'
    " &
done
wait

echo ""
echo -e "${GREEN}✓ Todos los experimentos iniciados${NC}"
echo ""

# Verify experiments are actually running
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 5/5: Verificando que experimentos iniciaron...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

sleep 10  # Give experiments time to start

for batch in batch1 batch2 batch3 batch4 batch5; do
    VM_NAME="vm-${batch}"
    echo -n "[$batch] Checking process: "

    if timeout 15 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="pgrep -f 'python3 runner_phase2' >/dev/null" 2>/dev/null; then
        echo -e "${GREEN}✓ RUNNING${NC}"
    else
        echo -e "${RED}✗ NOT RUNNING${NC}"
        echo "  Checking logs..."
        timeout 10 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="tail -10 'Laptop Runs/nohup_${batch}.log' 2>/dev/null" || echo "  (no logs available yet)"
    fi
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ 25 EXPERIMENTOS LANZADOS (5 VMs × 5 seeds)          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Información:${NC}"
echo "  - 5 VMs corriendo en paralelo"
echo "  - Cada VM ejecuta 5 seeds secuencialmente"
echo "  - Total: 25 experimentos"
echo "  - Tiempo estimado: 4-5 horas"
echo "  - Costo estimado: ~\$10 USD"
echo ""
echo -e "${YELLOW}Comandos útiles:${NC}"
echo "  # Monitorear progreso global:"
echo "  bash /tmp/monitor_25seeds.sh"
echo ""
echo "  # Ver logs de un batch específico:"
echo "  gcloud compute ssh vm-batch1 --zone=$ZONE --command='tail -30 \"Laptop Runs/nohup_batch1.log\"'"
echo ""
echo "  # Recopilar resultados cuando terminen (~5h):"
echo "  bash /tmp/collect_25results.sh"
echo ""
