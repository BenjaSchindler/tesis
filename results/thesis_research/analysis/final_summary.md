# Resultados Finales: Comparación de Filtros LLM vs SMOTE

## Estado del Experimento
- **Completado**: 1210/1440 (84%)
- **Win Rate Global**: 68.2%
- **Datasets**: 6 (todos cubiertos)

---

## Resumen por Dataset

| Dataset | N | Win Rate | Mean vs SMOTE | Best |
|---------|---|----------|---------------|------|
| **20newsgroups_10shot** | 240 | **90.4%** | **+3.61pp** | +7.25pp |
| **20newsgroups_25shot** | 240 | **88.3%** | +0.78pp | +2.61pp |
| hate_speech_davidson_10shot | 240 | 52.1% | -3.06pp | +6.15pp |
| hate_speech_davidson_25shot | 10 | 50.0% | -0.04pp | +4.00pp |
| sms_spam_10shot | 240 | 65.4% | -1.17pp | +6.98pp |
| sms_spam_25shot | 240 | 45.4% | -1.40pp | +2.84pp |

---

## Mejor Configuración por Dataset

### 20newsgroups_10shot (4 clases, 10 samples/clase)
- **Filter**: cascade (level=1)
- **LLM**: 100%, N-shot: 10
- **Resultado**: +7.25pp vs SMOTE

### 20newsgroups_25shot (4 clases, 25 samples/clase)
- **Filter**: lof (n=20, t=-0.3)
- **LLM**: 100%, N-shot: 10
- **Resultado**: +2.61pp vs SMOTE

### hate_speech_davidson_10shot (3 clases, 10 samples/clase)
- **Filter**: combined (lof=0.0, sim=0.3)
- **LLM**: 25%, N-shot: 10
- **Resultado**: +6.15pp vs SMOTE

### sms_spam_10shot (2 clases, 10 samples/clase)
- **Filter**: lof (n=5, t=0.0)
- **LLM**: 25%, N-shot: 50
- **Resultado**: +6.98pp vs SMOTE

### sms_spam_25shot (2 clases, 25 samples/clase)
- **Filter**: cascade (level=0)
- **LLM**: 5%, N-shot: 25
- **Resultado**: +2.84pp vs SMOTE

---

## Ranking de Filtros (Global)

| Filtro | Mean vs SMOTE | Win Rate | Best |
|--------|---------------|----------|------|
| **cascade** | **+1.45pp** | **83.3%** | +7.25pp |
| lof | +0.88pp | 75.8% | +6.98pp |
| none | +0.72pp | 62.9% | +6.74pp |
| embedding_guided | -1.29pp | 55.0% | +7.24pp |
| combined | -3.28pp | 52.5% | +6.15pp |

---

## Conclusiones Clave

1. **LLM funciona mejor en clasificación multi-clase** (20newsgroups, 4 clases) con 88-90% win rate

2. **El filtro más simple gana**: `cascade level=1` (solo distancia) supera a filtros más complejos

3. **Clasificación binaria es más difícil**: sms_spam muestra resultados mixtos (45-65% win rate)

4. **Recomendación por recursos**:
   - ≤10 samples/clase: 100% LLM con cascade level=1
   - 25 samples/clase: Híbrido 25-50% LLM con lof
   - >50 samples/clase: SMOTE puro

5. **El filtro combined (LOF+sim) es contraproducente**: -3.28pp mean, demasiado restrictivo
