# Análisis de Tendencias en Generación de Sintéticos

## Resumen Ejecutivo

**Hallazgo clave**: No todas las clases minoritarias se benefician igualmente de los sintéticos. Identificamos 3 patrones claros:

1. **Clases que SIEMPRE mejoran**: ISFP, ISFJ (+3% a +15%)
2. **Clases que SIEMPRE empeoran**: ESFP (-9% a -14%)
3. **Clases inconsistentes**: Dependen de la variante usada

## Clases que Mejoran Consistentemente

### ✅ ISFP (Introverted, Sensing, Feeling, Perceiving)

**Mejora en las 4 variantes**: +6.78% a +8.97%

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Support | 175 | Muy minoritaria |
| Baseline F1 | 0.2703 | Bajo rendimiento |
| Baseline Precision | 0.3306 | Moderada |
| Baseline Recall | 0.2286 | **MUY bajo** |

**Por qué mejora:**
- Recall extremadamente bajo (22.86%) → hay margen para mejorar
- Clase minoritaria con características distintivas
- Los sintéticos aumentan la cobertura en el espacio de embeddings

**Mejor variante para ISFP:**
- Variante A (5 prompts): +8.97%
- Variante C1 (GPT-5 terse): +8.97% (empate)

---

### ✅ ISFJ (Introverted, Sensing, Feeling, Judging)

**Mejora en las 4 variantes**: +3.33% a +14.86%

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Support | 130 | Muy minoritaria |
| Baseline F1 | 0.2655 | Bajo rendimiento |
| Baseline Precision | 0.3125 | Moderada |
| Baseline Recall | 0.2308 | **MUY bajo** |

**Por qué mejora:**
- Patrón similar a ISFP
- Clase extremadamente minoritaria (solo 130 ejemplos)
- Beneficio dramático con multi-temp ensemble (+14.86%)

**Mejor variante para ISFJ:**
- **Variante B (multi-temp)**: +14.86% 🏆
- Razón: Diversidad de temperaturas genera sintéticos más variados

---

## Clase que Empeora Consistentemente

### ❌ ESFP (Extraverted, Sensing, Feeling, Perceiving)

**Empeora en las 4 variantes**: -9.57% a -13.71%

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Support | 72 | **LA MÁS minoritaria** |
| Baseline F1 | 0.3538 | Bajo pero mejor que ISFP/ISFJ |
| Baseline Precision | 0.3966 | Similar a ISFP |
| Baseline Recall | 0.3194 | Mejor que ISFP/ISFJ |

**Por qué empeora:**
1. **Contaminación**: Sintéticos de baja calidad introducen ruido
2. **Solapamiento con otras clases**: ESFP puede ser difícil de distinguir de ENFP o ESTP
3. **Pocos ejemplos reales** (72) → LLM tiene poca información para generar sintéticos de calidad
4. **Recall ya relativamente alto** (31.94%) → sintéticos no aportan nueva información útil

**Menos malo:**
- Variante C3 (GPT-5 thorough): -9.57%
- Razón: Reasoning minimal ayuda a generar ejemplos más coherentes

---

## Patrones por Tamaño de Clase

### Clases MUY pequeñas (support < 100)

| Clase | Support | Delta Promedio | Patrón |
|-------|---------|----------------|--------|
| ESFP | 72 | **-12.26%** | ❌ Empeora (contaminación) |
| ESTJ | 96 | -1.15% | ⚠️ Neutral/ligeramente negativo |
| ESFJ | 36 | +3.00% | ✅ Mejora (pero es la MÁS pequeña) |

**Conclusión**:
- **No hay correlación directa con tamaño**
- ESFJ (36 ejemplos) mejora, pero ESFP (72) empeora
- La **calidad intrínseca** de la clase importa más que el tamaño

### Clases medianas (100-300)

| Clase | Support | Delta Promedio | Patrón |
|-------|---------|----------------|--------|
| ISFP | 175 | **+8.33%** | ✅ Mejora consistente |
| ISFJ | 130 | **+7.64%** | ✅ Mejora consistente |
| ISTJ | 249 | -0.22% | ⚠️ Neutral |

**Conclusión**:
- Las 2 clases con **mejor mejora** están en este rango
- Tamaño óptimo para aprendizaje: suficientes ejemplos para el LLM, pero suficientemente minoritarias para beneficiarse

### Clases grandes (≥ 300)

| Clase | Support | Delta Promedio | Patrón |
|-------|---------|----------------|--------|
| ESTP | 397 | +0.17% | ⚠️ Neutral (ya tiene buen F1: 0.79) |
| ENFJ | 307 | +1.61% | ✅ Mejora leve |

**Conclusión**:
- ESTP ya tiene F1=0.79 → techo de rendimiento
- ENFJ (F1=0.21 bajo) tiene más margen para mejorar

---

## Comparación GPT-5 vs GPT-4o-mini

### Performance por clase

| Clase | GPT-4o (A) | GPT-5 terse (C1) | GPT-5 thorough (C3) | Ganador |
|-------|-----------|------------------|---------------------|---------|
| ESFP | -12.72% | -13.71% | **-9.57%** | GPT-5 thorough |
| ESTP | **+0.43%** | -0.24% | +0.12% | GPT-4o |
| ISTJ | **+4.43%** | +0.49% | -0.52% | GPT-4o |
| ENFJ | **+3.04%** | -0.29% | -0.21% | GPT-4o |
| ESTJ | -2.16% | -2.46% | **+0.51%** | GPT-5 thorough |
| ISFP | **+8.97%** | +8.97% | +8.60% | GPT-4o (empate con C1) |
| ISFJ | **+8.10%** | +3.33% | +4.26% | GPT-4o |
| ESFJ | -1.56% | **+10.77%** | +5.88% | GPT-5 terse |

**Resultado**:
- GPT-4o gana en: 4/8 clases (ESTP, ISTJ, ENFJ, ISFP, ISFJ)
- GPT-5 thorough gana en: 2/8 clases (ESFP, ESTJ)
- GPT-5 terse gana en: 2/8 clases (ESFJ)

**Conclusión**:
- **GPT-4o-mini es más consistente** para clases medianas (ISFP, ISFJ, ISTJ)
- **GPT-5-mini thorough** reduce el daño en clases problemáticas (ESFP)
- **GPT-5-mini terse** sorprendentemente bueno para ESFJ (+10.77%)

---

## Hipótesis sobre Por Qué Algunas Clases Mejoran

### Teoría 1: Recall Bajo + Support Medio = Mejora

**Evidencia**:
- ISFP: recall=0.23, support=175 → mejora +8.33%
- ISFJ: recall=0.23, support=130 → mejora +7.64%

**Contra-evidencia**:
- ESFP: recall=0.32, support=72 → empeora -12.26%

**Conclusión**: Recall bajo es necesario pero no suficiente.

### Teoría 2: Distintividad de la Clase

**Clases que mejoran (ISFP, ISFJ)**:
- Características distintivas: Introvertidos + Sensing + Feeling
- Menos solapamiento con clases mayoritarias

**Clases que empeoran (ESFP)**:
- Posible confusión con ENFP (Extraverted + Feeling + Perceiving)
- O con ESTP (Extraverted + Sensing + Perceiving)
- Sintéticos ambiguos aumentan la confusión

### Teoría 3: Calidad del LLM sobre la Clase

**ESFJ** (36 ejemplos, la más pequeña):
- GPT-5 terse: +10.77%
- GPT-4o: -1.56%

→ GPT-5 parece "entender" mejor ESFJ que GPT-4o
→ No es solo cantidad de datos, sino comprensión semántica del LLM

---

## Recomendaciones

### 1. Estrategia por Clase

**Para ISFP, ISFJ (siempre mejoran)**:
- Usar **cualquier variante**, todas funcionan
- Mejor: Variante B (multi-temp) para ISFJ (+14.86%)

**Para ESFP (siempre empeora)**:
- **Evitar generación de sintéticos** para esta clase
- O usar GPT-5 thorough para minimizar daño (-9.57% vs -13.71%)

**Para clases inconsistentes (ISTJ, ENFJ, ESTJ, ESTP, ESFJ)**:
- Probar **Variante C3 (GPT-5 thorough)** primero
- Si no mejora, usar **Variante A (5 prompts GPT-4o)**

### 2. Filtrado Adaptativo

**Implementar filtro por clase**:
```python
# Clases que NO deben generar sintéticos
BLACKLIST = ['ESFP']

# Clases que se benefician de multi-temp
MULTI_TEMP_CLASSES = ['ISFJ']

# Clases que prefieren GPT-5
GPT5_CLASSES = ['ESFJ', 'ESTJ']
```

### 3. Métricas de Calidad por Clase

Monitorear:
1. **Pureza del cluster** de sintéticos por clase
2. **Similarity threshold** específico por clase
3. **Contamination rate** - % de sintéticos rechazados

---

## Próximos Experimentos

1. **Analizar ejemplos sintéticos de ESFP** para entender por qué empeoran
2. **Probar GPT-5 con reasoning=medium** para clases problemáticas
3. **Implementar filtrado selectivo**: solo generar para ISFP, ISFJ, ESFJ
4. **Aumentar similarity threshold para ESFP** (de 0.90 a 0.95)
5. **Usar class descriptions especializadas** para ESFP vs ENFP diferenciación
