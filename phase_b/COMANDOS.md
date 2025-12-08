# 🚀 COMANDOS LISTOS PARA FASE B

## Paso 1: Preparar Entorno

```bash
cd "/home/benja/Desktop/Tesis/Tesis_Limpio/Laptop Runs"

# Instalar dependencia faltante
pip install sentence-transformers

# Configurar API Key de OpenAI
export OPENAI_API_KEY="your-openai-api-key-here"
```

---

## Paso 2: Test Rápido (10K samples, ~10 min)

```bash
# Ejecutar test con dataset pequeño
./launch_phaseB_gpu.sh MBTI_10k.csv 42
```

**Espera:** ~10-15 minutos
**Objetivo:** Verificar que todo funciona antes del experimento completo

---

## Paso 3: Experimento Completo (106K samples, ~45-60 min)

### Opción A: Single Seed (más rápido)
```bash
# Un solo experimento con seed 42
./launch_phaseB_gpu.sh MBTI_500.csv 42
```

### Opción B: Multi-Seed (validación estadística robusta)
```bash
# 3 seeds (suficiente para validación inicial)
for seed in 42 100 200; do
    echo "═══════════════════════════════════════"
    echo "Ejecutando seed $seed..."
    echo "═══════════════════════════════════════"
    ./launch_phaseB_gpu.sh MBTI_500.csv $seed
    sleep 5
done
```

### Opción C: Multi-Seed Completo (10 seeds, máxima robustez)
```bash
# 10 seeds para paper/tesis
for seed in 42 100 200 300 400 500 600 700 800 900; do
    echo "═══════════════════════════════════════"
    echo "Ejecutando seed $seed ($(date '+%H:%M:%S'))..."
    echo "═══════════════════════════════════════"
    ./launch_phaseB_gpu.sh MBTI_500.csv $seed
    sleep 5
done

echo ""
echo "✓ Todos los experimentos completados!"
echo "Tiempo total estimado: 8-10 horas"
```

---

## Paso 4: Analizar Resultados

### Ver Resultados de un Experimento
```bash
# Ver métricas JSON formateadas
cat batch5_phaseB_gpu_seed42_metrics.json | jq

# Ver solo métricas clave
cat batch5_phaseB_gpu_seed42_metrics.json | jq '{
    baseline_f1: .macro_f1_baseline,
    augmented_f1: .macro_f1_augmented,
    delta_pct: .delta_pct,
    selected_model: .selected_model
}'
```

### Análisis Rápido (Python one-liner)
```bash
python3 << 'PYEOF'
import json

with open('batch5_phaseB_gpu_seed42_metrics.json') as f:
    m = json.load(f)

print(f"""
╔═══════════════════════════════════════════════╗
║        RESULTADOS FASE B - SEED 42            ║
╠═══════════════════════════════════════════════╣
║ Baseline F1:    {m['macro_f1_baseline']:.4f} ({m['macro_f1_baseline']*100:.2f}%)   ║
║ Augmented F1:   {m['macro_f1_augmented']:.4f} ({m['macro_f1_augmented']*100:.2f}%)   ║
║ Delta:          {m['delta_macro_f1']:+.4f} ({m['delta_pct']:+.2f}%)      ║
║ Selected:       {m['selected_model']:<30} ║
╚═══════════════════════════════════════════════╝
""")

# Comparar con Fase A
fase_a_delta = 1.00
if m['delta_pct'] >= 1.20:
    print("✓ META FASE B ALCANZADA (≥+1.20%)")
elif m['delta_pct'] >= fase_a_delta:
    print(f"⚠ Mejora respecto Fase A pero por debajo de meta ({m['delta_pct']:+.2f}% vs +1.20%)")
else:
    print(f"✗ Por debajo de Fase A ({m['delta_pct']:+.2f}% < +{fase_a_delta}%)")
PYEOF
```

### Análisis Multi-Seed (si ejecutaste múltiples seeds)
```bash
# Análisis estadístico completo
python3 analyze_multi_seed.py batch5_phaseB_gpu_seed*_metrics.json
```

---

## 📊 Targets y Comparación

| Fase | Macro F1 Delta | Seed Variance | Configuración Principal |
|------|----------------|---------------|------------------------|
| **Fase A** | +1.00% | 3.75pp | F1-budget + ensemble |
| **Fase B (target)** | +1.20% - +1.40% | <5pp | + Adaptive weighting |

---

## 🔍 Verificar que Adaptive Weighting está Activo

```bash
# Buscar en el log que adaptive weighting esté habilitado
grep -i "adaptive" batch5_phaseB_gpu_seed42_synthetic.csv 2>/dev/null || \
grep -i "weight" batch5_phaseB_gpu_seed42_metrics.json | head -5
```

---

## ⚡ Optimizaciones para RTX 3070

### Si tienes OOM (Out of Memory)
```bash
# Reducir batch size
# Editar launch_phaseB_gpu.sh línea BATCH_SIZE=128 → 64
```

### Si quieres MÁS velocidad
```bash
# Aumentar batch size (si tienes VRAM libre)
# Editar launch_phaseB_gpu.sh línea BATCH_SIZE=128 → 192
```

### Monitoreo en Tiempo Real
```bash
# En otra terminal, monitorear uso de GPU
watch -n 2 nvidia-smi

# Ver progreso del experimento (últimas líneas)
tail -f nohup.log  # Si corres en background
```

---

## 📁 Estructura de Archivos Esperada

Después de completar Fase B:

```
Laptop Runs/
├── MBTI_500.csv                                    # Dataset original (106K)
├── MBTI_10k.csv                                    # Dataset test (10K)
├── runner_phase2.py                                # Pipeline principal
├── launch_phaseB_gpu.sh                            # Script de lanzamiento ✓
├── analyze_multi_seed.py                           # Análisis estadístico ✓
├── FASE_B_INSTRUCCIONES.md                         # Instrucciones ✓
├── COMANDOS_FASE_B.md                              # Este archivo ✓
│
├── batch5_phaseB_gpu_seed42_synthetic.csv          # Sintéticos seed 42
├── batch5_phaseB_gpu_seed42_augmented.csv          # Train augmented seed 42
├── batch5_phaseB_gpu_seed42_metrics.json           # Métricas seed 42
│
├── batch5_phaseB_gpu_seed100_*.csv/json            # Seed 100 (si aplica)
├── batch5_phaseB_gpu_seed200_*.csv/json            # Seed 200 (si aplica)
└── ...
```

---

## 🎯 Siguiente Paso: Documentar Resultados

Una vez completado Fase B con resultados positivos:

1. **Actualizar Presentación Tesis 3**
   ```bash
   cd "../Presentacion_tesis_3"
   # Agregar sección 06_FASE_B/
   ```

2. **Generar gráficos comparativos**
   - Fase A vs Fase B macro F1
   - Per-class improvements (especialmente MID tier)
   - Multi-seed variance

3. **Escribir conclusiones**
   - ¿Se alcanzó la meta de +1.20%?
   - ¿Adaptive weighting mejoró MID tier?
   - ¿Varianza bajo control?

---

## ❓ Troubleshooting

### Error: "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="tu-key-aqui"
```

### Error: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Error: "CUDA out of memory"
```bash
# Reducir batch size en launch_phaseB_gpu.sh
BATCH_SIZE=64  # o incluso 32
```

### Proceso parece congelado
```bash
# Normal - el embedding inicial toma tiempo
# Con RTX 3070: ~2-3 minutos para cargar modelo
# Si pasan >10 min sin progreso, puede haber un problema

# Verificar proceso activo
ps aux | grep runner_phase2

# Ver uso de GPU
nvidia-smi
```

---

## 📞 Soporte

Si encuentras problemas:
1. Revisar [FASE_B_INSTRUCCIONES.md](./FASE_B_INSTRUCCIONES.md)
2. Verificar logs: `tail -100 nohup.log`
3. Verificar GPU: `nvidia-smi`
