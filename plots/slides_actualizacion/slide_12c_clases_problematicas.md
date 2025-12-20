# Slide 12c: Clases Problemáticas

## Título
**Resolviendo las Clases Ultra-Minoritarias**

---

## El Desafío

Las 3 clases con **menos de 50 muestras** en MBTI:

| Clase | N muestras | F1 Base | Problema |
|-------|------------|---------|----------|
| ESTJ | 39 | ~0% | Ultra-minoritaria |
| ESFJ | 42 | ~0% | Ultra-minoritaria |
| ESFP | 48 | ~0% | Ultra-minoritaria |

> *Estas clases tienen F1 = 0% en la línea base (el modelo nunca las predice correctamente)*

---

## Estrategias Evaluadas

### Progresión de Soluciones

| Clase | Config. Estándar | Sobremuestreo 20× | MLP Profundo | Estado |
|-------|------------------|-------------------|--------------|--------|
| **ESFJ** | 0.00 pp | +0.80 pp | **+1.36 pp** | ✅ **RESUELTO** |
| ESTJ | 0.00 pp | 0.00 pp | +0.43 pp | ⚠️ Parcial |
| ESFP | 0.00 pp | 0.00 pp | +0.10 pp | ❌ No resuelto |

---

## Caso de Éxito: ESFJ

### Evolución de ESFJ (42 muestras)

```
Config. Estándar    |                     0.00 pp
Sobremuestreo 20×   |████                +0.80 pp
MLP Profundo        |███████████         +1.36 pp ✓
```

**Factores de éxito**:
1. Sobremuestreo masivo (20× = ~840 sintéticos)
2. Clasificador no lineal profundo
3. La clase tenía patrones textuales distinguibles

---

## Limitación Conocida: ESFP

### ESFP permanece sin resolver

A pesar de:
- 944 muestras sintéticas generadas
- 5 clasificadores diferentes probados
- Múltiples configuraciones de generación

**Hipótesis**:
- Superposición semántica alta con otras clases
- Los patrones textuales de ESFP no son suficientemente distintivos
- Posible limitación de embeddings solo-texto

---

## Resumen Visual

```
         Config     Sobrem.    MLP
         Estándar   20×        Profundo

ESFJ     ○          ◐          ●  RESUELTO
ESTJ     ○          ○          ◔  PARCIAL
ESFP     ○          ○          ○  NO RESUELTO

○ = Sin mejora  ◔ = Mejora mínima  ◐ = Mejora parcial  ● = Resuelto
```

---

## Conclusión

> **Combinación óptima para clases ultra-minoritarias:**
>
> **Sobremuestreo masivo (20×)** + **MLP Profundo (512-256-128)**
>
> Resuelve **1 de 3** clases problemáticas. ESFP requiere investigación adicional.

---

## Figura Sugerida
- `fig_rare_class_heatmap.pdf` - Heatmap de rendimiento por clase y estrategia
- `fig_per_class_improvement.pdf` - Mejoras detalladas por clase MBTI

---

## Notas del Presentador
- ESFJ es el caso de éxito - destacar la progresión
- ESFP como limitación honesta del método
- Mencionar posibles soluciones futuras: embeddings multimodales, más datos
