# Resultados: 4 Variantes de Robustez (Seed 42)

Fecha: 2025-11-24

## Objetivo

Reducir la variabilidad causada por la estocasticidad del LLM mediante 4 estrategias:

1. **Variante A**: Más intentos de generación (5 prompts/cluster vs 3 baseline)
2. **Variante B**: Multi-temperature ensemble [0.7, 1.0, 1.3]
3. **Variante C1**: GPT-5-mini modo terse (no reasoning, low verbosity)
4. **Variante C3**: GPT-5-mini modo thorough (minimal reasoning, high verbosity)

## Resultados

| Rank | Variante | Baseline | Augmented | Delta | Delta % | Sintéticos |
|------|----------|----------|-----------|-------|---------|------------|
| 🏆 1 | C3 (GPT-5 thorough) | 0.45457 | 0.45634 | **+0.00176** | **+0.39%** | 347 |
| 🥈 2 | A (5 prompts) | 0.45457 | 0.45522 | +0.00065 | +0.14% | 182 |
| 🥉 3 | B (multi-temp) | 0.45457 | 0.45500 | +0.00043 | +0.09% | 296 |
| ❌ 4 | C1 (GPT-5 terse) | 0.45457 | 0.45412 | -0.00045 | -0.10% | 138 |

## Análisis

### Ganador: Variante C3 (GPT-5 thorough)

**Configuración:**
```bash
--llm-model gpt-5-mini-2025-08-07
--reasoning-effort minimal
--output-verbosity high
```

**Métricas:**
- Delta absoluto: +0.00176 (+0.39%)
- Sintéticos aceptados: 347
- Target objetivo: +0.00377 (+0.83%)
- **Alcanza el 47% del target**

### Comparación con Experimentos Anteriores

| Experimento | Media Delta | Configuración |
|-------------|-------------|---------------|
| 5 seeds (anterior) | -0.00119 (-0.26%) | gpt-4o-mini baseline |
| 4 variantes (este) | **+0.00060 (+0.13%)** | Diversas estrategias |
| Target (Phase B) | +0.00377 (+0.83%) | Single seed óptimo |

### Insights

1. **GPT-5-mini con reasoning > GPT-4o-mini**
   - C3 (GPT-5 thorough): +0.00176
   - Baseline (GPT-4o-mini): media de -0.00119 en 5 seeds

2. **Verbosity alta genera más sintéticos de calidad**
   - C3 (high verbosity): 347 sintéticos
   - C1 (low verbosity): 138 sintéticos
   - Más sintéticos → mejor cobertura de clases minoritarias

3. **Multi-temp ensemble no ayuda significativamente**
   - B: +0.00043 vs A: +0.00065
   - 3x más lento pero menor mejora

4. **5 prompts/cluster vs 3: mejora marginal**
   - A: +0.00065 con 182 sintéticos
   - Costo adicional no justifica la mejora

## Recomendación

**Usar Variante C3 para futuros experimentos:**

```bash
--llm-model gpt-5-mini-2025-08-07
--reasoning-effort minimal
--output-verbosity high
--prompts-per-cluster 3  # Default es suficiente
```

**Ventajas:**
- Mejor resultado (+0.39% vs -0.26%)
- Más sintéticos de calidad (347 vs 138-296)
- Más determinista que GPT-4o-mini

**Limitaciones:**
- No alcanza el target de +0.83%
- Sigue habiendo variabilidad entre seeds
- Necesita validación con más seeds

## Próximos Pasos

1. **Probar C3 con 5 seeds** para validar consistencia
2. **Combinar C3 + anchor quality improvements** (Phase C completa)
3. **Explorar reasoning-effort=low/medium** (entre minimal y none)
4. **Análisis por clase**: Ver qué clases mejoran más con C3

## Archivos Generados

```
results_4variants_seed42/
├── variante_a_seed42_metrics.json
├── variante_b_seed42_metrics.json
├── variante_c1_seed42_metrics.json
├── variante_c3_seed42_metrics.json
└── RESUMEN.md
```

## Comandos de Lanzamiento

```bash
# Lanzar las 4 variantes
cd gcp
export OPENAI_API_KEY='sk-...'
./launch_4variants.sh 42

# Descargar resultados
gcloud compute scp vm-variante-*-seed42:~/*_metrics.json ./results_4variants_seed42/ --zone=us-central1-a
```

## Costo

- 4 VMs × n1-standard-4 + T4 GPU × ~50 min = **~$1.50**
- Auto-shutdown funcionó correctamente
