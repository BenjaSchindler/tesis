# GCP Deployment System for SMOTE-LLM

Sistema genérico y robusto para lanzar experimentos SMOTE-LLM en Google Cloud Platform con GPUs.

## Overview

Este sistema permite lanzar experimentos paralelos en GCP de forma:
- **Robusta**: Manejo automático de paths, API keys, y auto-shutdown
- **Genérica**: Templates adaptables a cualquier configuración experimental
- **Eficiente**: Múltiples VMs en paralelo, auto-shutdown para ahorrar costos
- **Documentada**: Guías completas con troubleshooting y ejemplos

## Quick Start

### Para Experimentos Nuevos

```bash
# 1. Copiar template
cp gcp/launch_experiment_template.sh my_experiment/launch.sh

# 2. Editar configuración
nano my_experiment/launch.sh
# Modificar: EXPERIMENT_NAME, SEEDS, VM_ARGS, etc.

# 3. Configurar API key
export OPENAI_API_KEY='sk-...'

# 4. Lanzar
cd my_experiment
./launch.sh
```

### Para Experimentos Existentes (Phase D)

```bash
# 1. Set API key
export OPENAI_API_KEY='sk-...'

# 2. Launch 4 variants
cd phase_d
./launch_phased_4variants.sh

# 3. Monitor progress (optional)
./monitor_phased.sh

# 4. Collect results (after ~2.5 hours)
./collect_phased_results.sh
```

## Arquitectura

```
Local Machine
    │
    ├─ gcp/
    │  ├─ gcp_toolkit.sh                    # Funciones reutilizables
    │  ├─ launch_experiment_template.sh     # Template genérico
    │  ├─ collect_results_template.sh       # Collector genérico
    │  ├─ EXPERIMENT_GUIDE.md               # Guía completa
    │  └─ README.md                         # Este archivo
    │
    ├─ phase_d/
    │  ├─ launch_phased_4variants.sh        # Launcher específico
    │  ├─ monitor_phased.sh                 # Monitor en tiempo real
    │  ├─ collect_phased_results.sh         # Collector + análisis
    │  └─ QUICK_START.md                    # Guía rápida Phase D
    │
    └──> GCP us-central1-a
         │
         ├─ vm-1 (n1-standard-4 + T4 GPU)
         │  └─ Seeds: 42, 100, 123 (secuencial)
         ├─ vm-2 (n1-standard-4 + T4 GPU)
         │  └─ Seeds: 42, 100, 123 (secuencial)
         ├─ vm-3 (...)
         └─ vm-4 (...)
```

## Documentación Completa

| Archivo | Descripción |
|---------|-------------|
| [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | Guía completa con troubleshooting y ejemplos |
| [launch_experiment_template.sh](launch_experiment_template.sh) | Template genérico para nuevos experimentos |
| [collect_results_template.sh](collect_results_template.sh) | Template genérico para recolectar resultados |
| [gcp_toolkit.sh](gcp_toolkit.sh) | Librería de funciones reutilizables |

## Configuración Estándar de VMs

Las VMs usan la siguiente configuración probada:

- **Machine**: `n1-standard-4` (4 vCPUs, 15GB RAM)
- **GPU**: `nvidia-tesla-t4` (16GB VRAM, ~$0.35/hora)
- **Image**: `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` (Deep Learning VM)
- **Disk**: 100GB SSD (mínimo requerido por la imagen)
- **Zone**: `us-central1-a` (Iowa, USA)

**Ventajas de Deep Learning VM**:
- CUDA 12.8 y drivers NVIDIA pre-instalados
- PyTorch 2.7 pre-instalado
- No requiere tiempo de setup de drivers
- `sudo pip3 install` funciona globalmente

## Costos Estimados

| Configuración | Costo/hora/VM | Tiempo típico | Total (4 VMs × 3 seeds) |
|---------------|---------------|---------------|-------------------------|
| n1-standard-4 | $0.20 | 2.5 horas | $2.00 |
| NVIDIA T4 GPU | $0.35 | 2.5 horas | $3.50 |
| Disco 100GB SSD | $0.02 | 2.5 horas | $0.20 |
| OpenAI API | Variable | - | $5-10 |
| **TOTAL** | **$0.57** | **2.5 horas** | **~$13** |

**Importante**: Si las VMs no se auto-apagan, el costo es ~$2.28/hora extra por cada VM.

## Funciones Principales (gcp_toolkit.sh)

```bash
# Cargar toolkit
source gcp/gcp_toolkit.sh

# Ver todas las funciones disponibles
gcp_toolkit_help

# Crear VMs con GPU en paralelo
gcp_create_gpu_vms vm-1 vm-2 vm-3 vm-4

# Subir archivos específicos
gcp_upload_files_multi "core/runner_phase2.py MBTI_500.csv" vm-1 vm-2

# Instalar dependencias
gcp_setup_vm vm-1 "numpy pandas scikit-learn sentence-transformers openai"

# Monitorear logs
gcp_get_logs vm-1 experiment.log 50

# Eliminar VMs
gcp_delete_vms vm-1 vm-2 vm-3 vm-4
```

## Ejemplos de Uso

### Ejemplo 1: Probar Diferentes Modelos LLM

```bash
EXPERIMENT_NAME="llm-comparison"
NUM_VMS=3
SEEDS=(42)

BASE_ARGS="--data-path MBTI_500.csv --device cuda"

# Diferentes modelos por VM
VM_ARGS[1]="--llm-model gpt-4o-mini"
VM_ARGS[2]="--llm-model gpt-4o"
VM_ARGS[3]="--llm-model gpt-5-mini-2025-08-07"

VM_LABELS[1]="GPT-4o-mini"
VM_LABELS[2]="GPT-4o"
VM_LABELS[3]="GPT-5-mini"
```

### Ejemplo 2: Grid Search de Hyperparámetros

```bash
EXPERIMENT_NAME="hyperparameter-grid"
NUM_VMS=4
SEEDS=(42 100)

# Grid: max-clusters × prompts-per-cluster
VM_ARGS[1]="--max-clusters 2 --prompts-per-cluster 3"
VM_ARGS[2]="--max-clusters 3 --prompts-per-cluster 3"
VM_ARGS[3]="--max-clusters 3 --prompts-per-cluster 5"
VM_ARGS[4]="--max-clusters 5 --prompts-per-cluster 3"
```

### Ejemplo 3: Ablation Study (Phase D)

```bash
EXPERIMENT_NAME="phase-d-ablation"
NUM_VMS=4
SEEDS=(42 100 123)

# Probar features individuales y combinadas
VM_ARGS[1]="--use-contrastive-prompting"
VM_ARGS[2]="--use-focal-loss"
VM_ARGS[3]="--use-two-stage-training"
VM_ARGS[4]="--use-contrastive-prompting --use-focal-loss --use-two-stage-training"

VM_LABELS[1]="Contrastive Only"
VM_LABELS[2]="Focal Loss Only"
VM_LABELS[3]="Two-Stage Only"
VM_LABELS[4]="All Features"
```

Ver más ejemplos en [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

## Problemas Comunes y Soluciones

### 1. Error: `/usr/bin/scp: stat local "core/file.py": No such file or directory`

**Causa**: Rutas relativas no se resuelven correctamente.

**Solución**: El template usa detección automática de project root:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
gcloud compute scp "$PROJECT_ROOT/$file" "$vm":~/ --zone="$GCP_ZONE"
```

### 2. Error: `Bad port '<PUERTO>'` en SSH config

**Causa**: Archivo `~/.ssh/config` tiene placeholders sin reemplazar.

**Solución**:
```bash
# Comentar entradas con placeholders
nano ~/.ssh/config
# Añadir # al inicio de líneas problemáticas
```

### 3. ModuleNotFoundError en la VM

**Causa**: Dependencias no instaladas o instaladas en path incorrecto.

**Solución**: Deep Learning VMs requieren `sudo pip3 install` global:
```bash
sudo pip3 install --upgrade pip
sudo pip3 install numpy pandas scikit-learn sentence-transformers openai tqdm
```

### 4. VMs no se auto-apagan

**Verificar shutdown script**:
```bash
gcloud compute ssh vm-name --zone=us-central1-a --command="ps aux | grep shutdown"
```

**El template usa shutdown condicional**:
```bash
if [ $FAILED_SEEDS -eq ${#SEEDS[@]} ]; then
    sudo shutdown -h now  # Inmediato si todo falló
else
    sudo shutdown -h +2   # 2 min delay si hubo éxito
fi
```

### 5. Imports fallan en runner

**Causa**: Módulos subidos en subdirectorio incorrecto.

**Verificar**:
```bash
gcloud compute ssh vm-name --zone=us-central1-a --command="ls -la ~/"
# Deben estar en ~/ (no en subdirectorios)
```

Ver troubleshooting completo en [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

## Best Practices

1. **Verificar quotas antes de lanzar**:
   ```bash
   gcloud compute project-info describe --project=YOUR_PROJECT
   ```

2. **Usar nombres consistentes**:
   ```bash
   VM_PREFIXES[1]="exp_featureA"  # Resultará en: exp_featureA_seed42_metrics.json
   ```

3. **Monitorear durante ejecución**:
   ```bash
   # Ver procesos corriendo
   for vm in vm-exp-{1..4}; do
       gcloud compute ssh "$vm" --zone=us-central1-a \
         --command="ps aux | grep python3 | grep -v grep"
   done
   ```

4. **Descargar resultados ANTES de eliminar VMs**:
   ```bash
   # Usar collect_results.sh ANTES de gcp_delete_vms
   ```

5. **Configurar alertas de costo**:
   - GCP Console → Billing → Budgets & Alerts
   - Alert at 50%, 90%, 100% of monthly budget

## Workflow Completo

```bash
# 1. Preparación
export OPENAI_API_KEY='sk-...'
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Crear experimento desde template
cp gcp/launch_experiment_template.sh my_exp/launch.sh
nano my_exp/launch.sh  # Editar configuración

# 3. Lanzar
cd my_exp
./launch.sh
# Confirmar con 'y' cuando pregunte

# 4. Monitorear (terminal separada, opcional)
watch -n 60 'gcloud compute instances list --filter="name~my-exp"'

# 5. Recolectar resultados (~2.5 horas después)
./collect_results.sh
# Esto descarga métricas, logs, CSVs y analiza resultados

# 6. Cleanup
# collect_results.sh pregunta si eliminar VMs
# O manualmente: gcloud compute instances delete vm-* --zone=us-central1-a
```

## Estructura de Archivos Recomendada

```
project_root/
├── core/
│   ├── runner_phase2.py          # Runner principal
│   └── *.py                       # Módulos auxiliares
├── gcp/
│   ├── gcp_toolkit.sh             # Funciones reutilizables
│   ├── launch_experiment_template.sh  # Template genérico
│   ├── collect_results_template.sh    # Collector genérico
│   ├── EXPERIMENT_GUIDE.md        # Guía completa
│   └── README.md                  # Este archivo
├── my_experiment/
│   ├── launch_my_exp.sh           # Adaptado del template
│   ├── collect_results.sh         # Adaptado del template
│   ├── monitor.sh                 # Monitor (opcional)
│   └── results/                   # Resultados descargados
│       ├── variant_seedX_metrics.json
│       ├── variant_seedX_synthetic.csv
│       └── variant_seedX.log
└── MBTI_500.csv                   # Dataset
```

## Changelog

- **2025-11-25**: Sistema genérico de templates
  - Creado `launch_experiment_template.sh` para cualquier experimento
  - Creado `collect_results_template.sh` con análisis estadístico
  - Creado `EXPERIMENT_GUIDE.md` con troubleshooting completo
  - Documentados problemas de paths, SSH config, API key injection
  - Ejemplos: Phase D ablation study lanzado exitosamente

- **2025-11**: Deep Learning VM adoption
  - Migrado a imagen `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
  - Auto-shutdown por proceso (no timer fijo)
  - `sudo pip3 install` global para evitar problemas de path

## Recursos

- [GCP Compute Pricing](https://cloud.google.com/compute/pricing)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)
- [Deep Learning VM Images](https://cloud.google.com/deep-learning-vm/docs/images)
- [OpenAI Pricing](https://openai.com/pricing)

## Support

Para problemas no cubiertos en esta documentación:

1. Revisar [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - Troubleshooting completo
2. Verificar logs en la VM: `gcloud compute ssh vm-name --command="tail -100 run.log"`
3. Revisar serial port output: `gcloud compute instances get-serial-port-output vm-name`
4. Consultar documentación de GCP: https://cloud.google.com/compute/docs
