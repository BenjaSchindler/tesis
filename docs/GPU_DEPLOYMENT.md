# Phase A - GPU Deployment Guide

## Configuración Actualizada: 4 VMs × NVIDIA T4

**Proyecto:** `prexx-465515`
**Cuenta:** `contacto@doctor911.cl`
**Fecha:** 2025-11-15

---

## 📊 Especificaciones

### Hardware
- **VMs:** 4 instancias en paralelo
- **Machine Type:** n1-standard-4 (4 vCPUs, 15GB RAM)
- **GPU:** NVIDIA T4 (1 por VM)
- **Boot Disk:** 50GB SSD
- **Zone:** us-central1-a

### Experimentos
- **Total Seeds:** 20 (4 VMs × 5 seeds cada una)
- **Distribución:**
  - vm-batch1-gpu: 42, 100, 123, 456, 789
  - vm-batch2-gpu: 111, 222, 333, 444, 555
  - vm-batch3-gpu: 1000, 2000, 3000, 4000, 5000
  - vm-batch4-gpu: 7, 13, 21, 37, 101

### Software
- **CUDA:** 12.3
- **PyTorch:** Latest con soporte CUDA 12.1
- **Drivers NVIDIA:** Auto-instalados vía ubuntu-drivers
- **Python:** 3.10 con venv

---

## 💰 Costos Estimados

| Componente | Costo/hora | Costo Total (2.5h) |
|------------|------------|-------------------|
| **n1-standard-4** | $0.19 × 4 | $1.90 |
| **NVIDIA T4** | $0.35 × 4 | $3.50 |
| **Boot Disk (50GB)** | $0.01 × 4 | $0.10 |
| **Network** | ~$0.05 | $0.12 |
| **OpenAI API** (gpt-4o-mini) | ~$0.50/exp | $10.00 |
| **TOTAL** | **$2.28/hora** | **~$15.62** |

**Comparación con CPU:**
- CPU (tesis-477120): $0.95/hora × 8+ horas = $7.60+ (sin completar)
- GPU (prexx-465515): $2.28/hora × 2.5 horas = $15.62 (completado)

---

## 🚀 Lanzamiento

### Prerequisitos

1. **Configurar API Key:**
```bash
export OPENAI_API_KEY="sk-proj-..."
```

2. **Navegar al directorio:**
```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/gcp
```

### Ejecutar Deployment

```bash
./launch_20seeds_gpu.sh
```

**Pasos del Script:**
1. ✅ Verificar API key
2. ✅ Configurar proyecto GCP (prexx-465515)
3. ✅ Crear 4 VMs con GPUs T4
4. ✅ Instalar drivers NVIDIA + CUDA 12.3
5. ✅ Subir 9 módulos + dataset + batch scripts
6. ✅ Instalar PyTorch + dependencias
7. ✅ Verificar GPU accessibility
8. ✅ Lanzar 20 experimentos

**Tiempo Total de Setup:** ~10-15 minutos
**Tiempo Total de Ejecución:** ~2.5 horas

---

## 🔍 Monitoreo

### Ver Logs en Tiempo Real
```bash
# Batch 1
gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --project=prexx-465515 \
  --command='tail -f LaptopRuns/nohup_batch1.log'

# Batch 2
gcloud compute ssh vm-batch2-gpu --zone=us-central1-a --project=prexx-465515 \
  --command='tail -f LaptopRuns/nohup_batch2.log'
```

### Verificar GPU Usage
```bash
gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --project=prexx-465515 \
  --command='nvidia-smi'
```

**Salida Esperada:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.3   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   60C    P0    30W /  70W |   8000MiB / 15360MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

### Ver Estado de VMs
```bash
gcloud compute instances list --filter='name~vm-batch.*-gpu' \
  --project=prexx-465515 --format='table(name,status,zone)'
```

### Contar Seeds Completados
```bash
for batch in batch1 batch2 batch3 batch4; do
  echo "[$batch]"
  gcloud compute ssh "vm-$batch-gpu" --zone=us-central1-a --project=prexx-465515 \
    --command="cd LaptopRuns && ls phaseA_gpu_*.json 2>/dev/null | wc -l"
done
```

---

## 📦 Recopilación de Resultados

### Esperar Completación (~2.5 horas)

**Verificar que todas las VMs se apagaron automáticamente:**
```bash
gcloud compute instances list --filter='name~vm-batch.*-gpu' --project=prexx-465515
```

Si todas muestran `TERMINATED`, los experimentos completaron exitosamente.

### Descargar Resultados

**Antes de que se apaguen las VMs, guardar resultados:**

```bash
# Crear directorio local
mkdir -p /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/results_gpu
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/results_gpu

# Descargar de cada VM (mientras estén corriendo)
for batch in batch1 batch2 batch3 batch4; do
  mkdir -p $batch
  gcloud compute scp "vm-$batch-gpu:LaptopRuns/phaseA_gpu_*.json" "$batch/" \
    --zone=us-central1-a --project=prexx-465515
  gcloud compute scp "vm-$batch-gpu:LaptopRuns/phaseA_gpu_*.csv" "$batch/" \
    --zone=us-central1-a --project=prexx-465515
done
```

**Archivos Esperados:** 60 totales
- 20 × `phaseA_gpu_seed*_metrics.json`
- 20 × `phaseA_gpu_seed*_synthetic.csv`
- 20 × `phaseA_gpu_seed*_augmented.csv`

---

## ⚙️ Configuración Phase A (GPU-Optimized)

### Cambios vs CPU

| Parámetro | CPU | GPU | Motivo |
|-----------|-----|-----|--------|
| `--device` | cpu | **cuda** | Usar GPU |
| `--embedding-batch-size` | 64 | **256** | GPU memoria mayor |
| Archivos output | `phaseA_seed*` | `phaseA_gpu_seed*` | Diferenciar |

### ⚠️ Nota sobre F1-Budget Thresholds

**Este run (2025-11-15) usa thresholds 0.35/0.20:**
- HIGH F1: ≥0.35
- MID F1: 0.20-0.35
- LOW F1: <0.20

**Configuración validada (Batch 5, 2025-11-12) usa 0.45/0.20:**
- HIGH F1: ≥0.45 (+1.00% macro F1 alcanzado)

Corriendo con 0.35 para comparar. Si resultados son inferiores, cambiaremos a 0.45.
Ver: [GPU_RUN_2025_11_15.md](GPU_RUN_2025_11_15.md) para detalles.

### Parámetros Completos

```bash
python3 runner_phase2.py \
  --data-path MBTI_500.csv \
  --test-size 0.2 \
  --random-seed $SEED \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --device cuda \
  --embedding-batch-size 256 \
  --llm-model gpt-4o-mini \
  --max-clusters 3 \
  --prompts-per-cluster 3 \
  --prompt-mode mix \
  --use-ensemble-selection \
  --use-val-gating \
  --val-size 0.15 \
  --val-tolerance 0.02 \
  --enable-anchor-gate \
  --anchor-quality-threshold 0.50 \
  --enable-anchor-selection \
  --anchor-selection-ratio 0.8 \
  --anchor-outlier-threshold 1.5 \
  --enable-adaptive-filters \
  --use-class-description \
  --use-f1-budget-scaling \
  --f1-budget-thresholds 0.35 0.20 \
  --f1-budget-multipliers 30 70 100 \
  --similarity-threshold 0.90 \
  --min-classifier-confidence 0.10 \
  --contamination-threshold 0.95 \
  --synthetic-weight 0.5 \
  --synthetic-weight-mode flat \
  --synthetic-output "phaseA_gpu_seed${SEED}_synthetic.csv" \
  --augmented-train-output "phaseA_gpu_seed${SEED}_augmented.csv" \
  --metrics-output "phaseA_gpu_seed${SEED}_metrics.json"
```

---

## 🐛 Troubleshooting

### GPU No Detectada

**Síntoma:**
```
ERROR: nvidia-smi failed - GPU not accessible
```

**Solución:**
```bash
# SSH a la VM
gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --project=prexx-465515

# Verificar instalación de drivers
sudo ubuntu-drivers list
sudo ubuntu-drivers autoinstall

# Reiniciar
sudo reboot
```

### PyTorch No Ve CUDA

**Síntoma:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solución:**
```bash
# Verificar versión de CUDA
nvcc --version  # Debería mostrar 12.3

# Reinstalar PyTorch con CUDA correcta
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### OOM (Out of Memory)

**Síntoma:**
```
CUDA out of memory. Tried to allocate X.XX GiB
```

**Solución:**
```bash
# Reducir batch size en el script
--embedding-batch-size 128  # En vez de 256
```

### Experimento Muy Lento

**Verificar:**
```bash
# ¿Está usando GPU?
nvidia-smi  # GPU-Util debería ser >80%

# ¿Cuello de botella en CPU?
htop  # CPU usage debería ser <100%

# ¿API calls lentas?
grep "API" nohup_batch1.log | tail -20
```

---

## 📊 Resultados Esperados

### Métricas por Seed

Cada `phaseA_gpu_seed*_metrics.json` contendrá:

```json
{
  "random_seed": 42,
  "baseline_f1_macro": 0.XXXX,
  "augmented_f1_macro": 0.XXXX,
  "f1_delta": 0.XXXX,
  "baseline_per_class": {...},
  "augmented_per_class": {...},
  "ensemble_selection": {...},
  "synthetic_stats": {...},
  "gpu_used": true,
  "execution_time_seconds": XXXX
}
```

### Agregación de 20 Seeds

**Comandos de Análisis:**

```python
import json
import numpy as np
from pathlib import Path

# Cargar todos los metrics
metrics = []
for json_file in Path('results_gpu').rglob('*_metrics.json'):
    with open(json_file) as f:
        metrics.append(json.load(f))

# Calcular estadísticas
deltas = [m['f1_delta'] for m in metrics]
print(f"Mean F1 Delta: {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
print(f"Median F1 Delta: {np.median(deltas):.4f}")
print(f"95% CI: [{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}]")
print(f"Success Rate: {sum(d > 0 for d in deltas)}/{len(deltas)} ({100*sum(d > 0 for d in deltas)/len(deltas):.1f}%)")
```

---

## ✅ Checklist Final

- [ ] VMs con GPU T4 creadas exitosamente
- [ ] Drivers NVIDIA instalados (nvidia-smi funciona)
- [ ] PyTorch detecta CUDA (`torch.cuda.is_available() == True`)
- [ ] 20 experimentos lanzados
- [ ] GPU usage >80% durante ejecución
- [ ] 60 archivos de salida generados (20 seeds × 3 archivos)
- [ ] Resultados descargados localmente
- [ ] VMs auto-shutdown ejecutado
- [ ] Análisis estadístico completado

---

**Última Actualización:** 2025-11-15
**Script:** [launch_20seeds_gpu.sh](phase_a/gcp/launch_20seeds_gpu.sh)
**Proyecto GCP:** prexx-465515
