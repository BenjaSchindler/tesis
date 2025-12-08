# Fase B - Adaptive Weighting Experiments

## 🚀 Ejecución Rápida (Recomendado)

### Test Rápido (10K samples, ~10 minutos)
```bash
cd "/home/benja/Desktop/Tesis/Tesis_Limpio/Laptop Runs"

# Configurar API Key (si no está configurada)
export OPENAI_API_KEY="tu-api-key-aqui"

# Ejecutar test rápido
./launch_phaseB_gpu.sh MBTI_10k.csv 42
```

### Experimento Completo (106K samples, ~45-60 minutos)
```bash
# Ejecutar con dataset completo
./launch_phaseB_gpu.sh MBTI_500.csv 42
```

### Múltiples Seeds (para validación estadística)
```bash
# Seed 42
./launch_phaseB_gpu.sh MBTI_500.csv 42

# Seed 100
./launch_phaseB_gpu.sh MBTI_500.csv 100

# Seed 200
./launch_phaseB_gpu.sh MBTI_500.csv 200
```

---

## 📊 Resultados Esperados

### Fase A (Baseline - ya completado)
- **Macro F1:** +1.00% mejora
- **Seed variance:** 93% reducción (54pp → 3.75pp)
- **Configuración:** F1-budget scaling + ensemble selection

### Fase B (Adaptive Weighting - objetivo)
- **Meta:** +1.20% a +1.40% macro F1
- **Mejora clave:** Pesos adaptativos por clase (0.5/0.3/0.1/0.05)
- **Soluciona:** Degradación en clases MID tier

---

## 🔧 Configuración Técnica

### Hardware Optimizado (RTX 3070)
- **GPU:** CUDA habilitado automáticamente
- **VRAM:** 8GB (óptimo para batch_size=128)
- **Embedding Model:** all-mpnet-base-v2 (768 dims, mejor calidad)
- **Velocidad:** ~6x más rápido que CPU

### Parámetros Fase B
```yaml
# Adaptive Weighting (NUEVO en Fase B)
--enable-adaptive-weighting: true
--synthetic-weight: 0.5
--synthetic-weight-mode: flat  # Pesos por clase según F1

# Inherited from Fase A (TIER S)
--use-ensemble-selection: true       # Max(baseline, augmented)
--use-val-gating: true                # Validación-based filtering
--enable-anchor-gate: true            # Quality threshold 0.50
--enable-adaptive-filters: true       # Dynamic thresholds

# Quality Control
--similarity-threshold: 0.90          # Anti-contamination
--contamination-threshold: 0.95       # Strict filtering
--min-classifier-confidence: 0.10     # Quality gate
```

---

## 📁 Archivos Generados

Después de la ejecución, encontrarás:

```
batch5_phaseB_gpu_seed42_synthetic.csv      # Datos sintéticos generados
batch5_phaseB_gpu_seed42_augmented.csv      # Train + sintéticos
batch5_phaseB_gpu_seed42_metrics.json       # Métricas completas
```

### Ver Resultados
```bash
# Ver métricas JSON formateadas
cat batch5_phaseB_gpu_seed42_metrics.json | jq

# Métricas clave
cat batch5_phaseB_gpu_seed42_metrics.json | jq '.macro_f1_augmented, .macro_f1_baseline, .delta_macro_f1'
```

---

## ⚡ Troubleshooting

### GPU no detectada
Si ves "GPU no detectada":
```bash
# Verificar CUDA
nvidia-smi

# Si no funciona, el script usará CPU automáticamente (más lento)
```

### Out of Memory (OOM)
Si la GPU se queda sin memoria:
```bash
# Reducir batch size manualmente
python3 runner_phase2.py \
  --embedding-batch-size 64 \  # En lugar de 128
  ... (resto de parámetros)
```

### API Rate Limit
Si OpenAI API te limita:
```bash
# El script tiene retry automático, pero puedes:
# 1. Usar dataset más pequeño (MBTI_10k.csv)
# 2. Reducir --prompts-per-cluster de 3 a 2
# 3. Esperar unos minutos y reintentar
```

---

## 🎯 Validación de Resultados

### Métricas Clave a Verificar

1. **Delta Macro F1** (principal)
   - Fase A: +1.00%
   - Fase B esperado: +1.20% a +1.40%

2. **MID Tier Classes** (mejora esperada en Fase B)
   - ENTJ: Actualmente -1.91% → Esperado: neutral o positivo
   - ESFJ: Actualmente -0.72% → Esperado: neutral o positivo

3. **LOW/HIGH Tier** (mantener ganancias)
   - LOW: +12.17% promedio (mantener)
   - HIGH: -0.05% (mantener protección)

### Comando de Análisis Rápido
```bash
python3 << 'PYEOF'
import json

with open('batch5_phaseB_gpu_seed42_metrics.json') as f:
    m = json.load(f)

print(f"╔{'═'*50}╗")
print(f"║ {'RESULTADOS FASE B':^48} ║")
print(f"╠{'═'*50}╣")
print(f"║ Baseline F1:    {m['macro_f1_baseline']:6.4f} ({m['macro_f1_baseline']*100:5.2f}%) ║")
print(f"║ Augmented F1:   {m['macro_f1_augmented']:6.4f} ({m['macro_f1_augmented']*100:5.2f}%) ║")
print(f"║ Delta:          {m['delta_macro_f1']:+6.4f} ({m['delta_pct']:+5.2f}%)   ║")
print(f"╚{'═'*50}╝")

if m['delta_pct'] >= 1.20:
    print("\n✓ META ALCANZADA: Delta ≥ +1.20%")
elif m['delta_pct'] >= 1.00:
    print("\n⚠ Bueno pero por debajo de meta (+1.00% a +1.19%)")
else:
    print("\n✗ Por debajo de Fase A (<+1.00%)")
PYEOF
```

---

## 📚 Siguiente Paso: Multi-Seed Validation

Una vez que confirmes que Fase B funciona:

```bash
# Lanzar 3 seeds para validación estadística
for seed in 42 100 200; do
    echo "Ejecutando seed $seed..."
    ./launch_phaseB_gpu.sh MBTI_500.csv $seed
    sleep 5
done

# Analizar varianza entre seeds
python3 analyze_multi_seed.py batch5_phaseB_gpu_seed*_metrics.json
```

---

## 🎓 Documentación Relacionada

- **Presentación Tesis 3:** `../Presentacion_tesis_3/README.md`
- **Fase A Results:** `../Presentacion_tesis_3/05_RESULTADOS/`
- **Best Config:** `../Presentacion_tesis_3/05_RESULTADOS/05_best_config.md`
