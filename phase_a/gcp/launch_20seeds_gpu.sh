#!/bin/bash
#
# Fase A - 20 Seeds Robustness Test with GPU
# 4 VMs × 5 seeds each = 20 total experiments
# Using NVIDIA T4 GPUs
#

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Fase A - 20 Seeds GPU Test (4 VMs × 5 seeds)        ║${NC}"
echo -e "${BLUE}║   NVIDIA T4 GPUs - Proyecto: prexx-465515              ║${NC}"
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
PROJECT="prexx-465515"
ACCOUNT="contacto@doctor911.cl"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="50GB"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Seed batches for each VM (4 VMs, 5 seeds each)
declare -A VM_SEEDS
VM_SEEDS[batch1]="42 100 123 456 789"
VM_SEEDS[batch2]="111 222 333 444 555"
VM_SEEDS[batch3]="1000 2000 3000 4000 5000"
VM_SEEDS[batch4]="7 13 21 37 101"

echo -e "${GREEN}Configuración:${NC}"
echo "  Proyecto: $PROJECT"
echo "  Cuenta: $ACCOUNT"
echo "  Total VMs: 4"
echo "  Seeds per VM: 5"
echo "  Total Experiments: 20"
echo "  Machine Type: $MACHINE_TYPE (4 vCPUs, 15GB RAM)"
echo "  GPU: $GPU_TYPE × $GPU_COUNT per VM"
echo "  Dataset: MBTI_500.csv (106K samples)"
echo "  Tiempo estimado: ~2.5 horas en paralelo"
echo "  Costo estimado: ~\$6.75 USD (4 VMs × \$0.54/h × 2.5h)"
echo ""
echo -e "${YELLOW}Distribución de seeds:${NC}"
for batch in batch1 batch2 batch3 batch4; do
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

# Set GCP project and account
echo -e "${YELLOW}Configurando proyecto GCP...${NC}"
gcloud config set project "$PROJECT"
gcloud config set account "$ACCOUNT"
echo ""

# Check and delete existing VMs
echo -e "${YELLOW}Verificando VMs existentes...${NC}"
for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
        echo -e "${YELLOW}⚠ VM $VM_NAME ya existe. Eliminando...${NC}"
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
    fi
done
echo ""

# Create VMs with GPUs
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 1/4: Creando 4 VMs con GPUs T4...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -e "${GREEN}Creating $VM_NAME with T4 GPU...${NC}"

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="pd-standard" \
        --maintenance-policy=TERMINATE \
        --metadata=startup-script='#!/bin/bash
# Update and install Python
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv python3.10-venv

# Install NVIDIA drivers
apt-get install -y -qq ubuntu-drivers-common
ubuntu-drivers autoinstall

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update -qq
apt-get -y -qq install cuda-toolkit-12-3

# Mark startup complete
touch /tmp/startup-complete
' \
        --quiet &
done
wait

echo -e "${GREEN}✓ Todas las VMs creadas${NC}"
echo ""

# Wait for startup scripts and GPU drivers
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 1.5/4: Esperando instalación de drivers GPU...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -e "${GREEN}Esperando $VM_NAME...${NC}"

    # Wait up to 5 minutes for startup
    for i in {1..30}; do
        if timeout 20 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="[ -f /tmp/startup-complete ] && nvidia-smi &>/dev/null" 2>/dev/null; then
            echo -e "${GREEN}✓ $VM_NAME GPU ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}✗ $VM_NAME timeout esperando GPU${NC}"
            exit 1
        fi
        sleep 10
    done
done

echo -e "${GREEN}✓ Drivers GPU instalados en todas las VMs${NC}"
echo ""

# Upload files to VMs
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 2/4: Subiendo archivos a VMs...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -e "${GREEN}Uploading to $VM_NAME...${NC}"

    # Create LaptopRuns directory
    timeout 30 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="mkdir -p LaptopRuns"

    # Upload all core modules (9 files)
    gcloud compute scp "$LOCAL_DIR/core/runner_phase2.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/ensemble_anchor_selector.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/contamination_aware_filter.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/enhanced_quality_gate.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/anchor_quality_improvements.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/quality_gate_predictor.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/mbti_class_descriptions.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/adversarial_discriminator.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/core/multi_seed_ensemble.py" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet

    # Upload dataset and batch script
    gcloud compute scp "$LOCAL_DIR/MBTI_500.csv" "$VM_NAME:LaptopRuns/" --zone="$ZONE" --project="$PROJECT" --quiet
    gcloud compute scp "$LOCAL_DIR/phase_a/batch_scripts/run_batch_${batch}_gpu.sh" "$VM_NAME:LaptopRuns/run_all_seeds.sh" --zone="$ZONE" --project="$PROJECT" --quiet

    echo "✓ $VM_NAME all files uploaded (9 modules + dataset + script)"
done &

wait

echo -e "${GREEN}✓ Todos los archivos subidos${NC}"
echo ""

# Setup dependencies
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 3/4: Instalando dependencias Python + PyTorch...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -e "${GREEN}Setup $VM_NAME...${NC}"

    timeout 300 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="
        cd LaptopRuns

        # Create venv
        if ! python3 -m venv .venv; then
            echo '✗ ERROR: Failed to create venv'
            exit 1
        fi

        # Activate venv
        source .venv/bin/activate

        # Verify activation
        if [ -z \"\$VIRTUAL_ENV\" ]; then
            echo '✗ ERROR: venv not activated properly'
            exit 1
        fi

        # Upgrade pip
        pip install -q --upgrade pip

        # Install PyTorch with CUDA 12.1 support
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

        # Install other dependencies
        pip install -q numpy pandas scikit-learn sentence-transformers openai python-dotenv

        echo '✓ Dependencies + PyTorch CUDA installed successfully'
    " &
done
wait

echo -e "${GREEN}✓ Todas las dependencias instaladas${NC}"
echo ""

# Verify imports and GPU access
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 3.5/4: Verificando PyTorch + GPU...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -e "${GREEN}Verify $VM_NAME...${NC}"

    timeout 30 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="
        cd LaptopRuns
        source .venv/bin/activate
        python3 -c '
import torch
import sentence_transformers
import openai
import pandas
import sklearn

print(\"PyTorch version:\", torch.__version__)
print(\"CUDA available:\", torch.cuda.is_available())
if torch.cuda.is_available():
    print(\"CUDA device:\", torch.cuda.get_device_name(0))
    print(\"✓ All imports OK + GPU accessible\")
else:
    print(\"✗ CUDA not available!\")
    exit(1)
'
    " || echo -e "${RED}✗ $VM_NAME GPU verification failed${NC}"
done

echo -e "${GREEN}✓ PyTorch + GPU verificados en todas las VMs${NC}"
echo ""

# Launch experiments
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Paso 4/4: Lanzando 20 experimentos...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    SEEDS="${VM_SEEDS[$batch]}"
    echo -e "${GREEN}[$batch] Launching 5 seeds: $SEEDS${NC}"

    timeout 30 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="
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

for batch in batch1 batch2 batch3 batch4; do
    VM_NAME="vm-${batch}-gpu"
    echo -n "[$batch] Checking process: "

    if timeout 15 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command="ps aux | grep '[p]ython3.*runner_phase2' >/dev/null" 2>/dev/null; then
        echo -e "${GREEN}✓ RUNNING${NC}"
    else
        echo -e "${RED}✗ NOT RUNNING${NC}"
    fi
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ 20 EXPERIMENTOS LANZADOS (4 VMs × 5 seeds)          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Información:${NC}"
echo "  - 4 VMs con GPU T4 corriendo en paralelo"
echo "  - Cada VM ejecuta 5 seeds secuencialmente"
echo "  - Total: 20 experimentos"
echo "  - Aceleración: ~10-15x vs CPU"
echo "  - Tiempo estimado: 2-3 horas"
echo "  - Costo estimado: ~\$6.75 USD"
echo ""

echo -e "${YELLOW}Comandos útiles:${NC}"
echo "  # Ver logs de un batch específico:"
echo "  gcloud compute ssh vm-batch1-gpu --zone=$ZONE --project=$PROJECT --command='tail -f LaptopRuns/nohup_batch1.log'"
echo ""
echo "  # Verificar GPU usage:"
echo "  gcloud compute ssh vm-batch1-gpu --zone=$ZONE --project=$PROJECT --command='nvidia-smi'"
echo ""
echo "  # Recopilar resultados cuando terminen (~2.5h):"
echo "  bash collect_20results_gpu.sh"
