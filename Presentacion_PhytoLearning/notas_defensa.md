# Notas de Defensa

Apuntes rápidos para repasar antes de la presentación. Organizado por tema.

---

## Filtros geométricos

### LOF (Local Outlier Factor) — `k=20`, umbral `>0`

- `k=20` = nº de vecinos para estimar densidad local. Si la clase tiene <21 muestras (10-shot), `k` se ajusta a `n-1` automáticamente.
- Umbral `>0` está sobre `decision_function` de sklearn, **no** sobre el LOF clásico. En `decision_function`: `>0` = inlier, `<0` = outlier (escala invertida vs LOF original).
- Entrenado solo con embeddings reales de la clase objetivo.
- Limitación en 10-shot: pocos vecinos → scores degeneran a uniforme.

### Cascade L1 — la mejor configuración

- L1 = solo nivel 1 = distancia euclidiana al **ancla más cercana** (ejemplo real más próximo de la clase). L4, en cambio, mide distancia al **centroide**.
- Selecciona top-N por ranking, **sin umbral binario**.
- `k=10` que aparece en el código es del nivel 3 (pureza KNN) — **no se usa en L1**.
- Por qué gana siendo el más simple:
  - Niveles 1 y 2 quedan monotónicamente relacionados con vectores L2-normalizados (distancia euclidiana ≡ coseno en la esfera unitaria), así que L2 no aporta info nueva.
  - Nivel 3 estima densidad local con pocos puntos → ruidoso en low-resource.
  - Nivel 4 (centroide) no aporta sobre L1: anclas son los reales mismos → cualquier centroide se computa a partir de ellos.
  - La media geométrica multiplica todos los scores: un score ruidoso bajo arrastra a todo el producto.

### Score compuesto `s(x) = (Π sᵢ(x))^(1/L)`

- Media geométrica de los niveles activos.
- Más estricta que la aritmética: un nivel malo hunde al candidato (no se "compensa" con otros buenos).
- Analogía: aprobar todas las materias (geométrica) vs buen promedio (aritmética).

### Embedding-Guided — el filtro que falló

**Qué hace:** en vez de premiar muestras cerca del centroide (como cascade), premia las que **llenan huecos** en el espacio de embeddings — zonas donde la clase está poco representada.

**Dos criterios opuestos:**
- **Cobertura (60%)**: distancia a los reales — *más lejos = mejor* (llena huecos vacíos).
- **Calidad (40%)**: distancia al centroide — *más cerca = mejor* (sigue siendo plausible para la clase).
- Score combinado: `0.6 × cobertura + 0.4 × calidad`.

**Selección voraz (greedy):** ordena candidatos por score, va eligiendo top-K uno por uno, salteándose los que ya están demasiado cerca de algo seleccionado (umbral de diversidad `0.1`).

**Por qué "multi-criterio":** combina dos objetivos distintos en un único score, a diferencia del cascade que es mono-criterio (solo proximidad).

**Por qué falló (−1.29 pp vs SMOTE, 55% win):**
- Cobertura y calidad **tiran en direcciones opuestas**: para llenar huecos hay que alejarse del centroide.
- En few-shot, las zonas alejadas están mal definidas → mete ruido.
- Confirma el hallazgo central: en pocos datos, **simple > complejo**, **proximidad > diversidad**.

**Defensa — para 30 segundos:**

> *"Filtro multi-criterio que combina cobertura del espacio (60%) y calidad (40%) con selección voraz. Busca muestras que llenen huecos en vez de quedarse cerca del centroide. Falló porque diversidad y proximidad tiran en direcciones opuestas: en few-shot llenar huecos saca las muestras fuera de la región segura de la clase."*

**Si te preguntan "¿por qué lo incluiste si falló?":**

> *"Lo incluí precisamente para mostrar que el principio opuesto — diversificar — no funciona. Confirma empíricamente el hallazgo central: en pocos datos, proximidad gana a diversidad."*

---

## Ponderación suave (Soft weighting)

### Pipeline

```
candidato → cascade → s(x) → min-max → s̃(x) → (·)^(1/T) → w(x) → sample_weight
```

### Ecuaciones

1. **Min-max**: `s̃(x) = (s(x) - s_min) / (s_max - s_min)` → lleva a [0,1].
2. **Temperatura**: `w(x) = max(w_min, s̃(x)^(1/T))` → peso final.

### Hiperparámetros óptimos

- `T = 0.5` → `w = s̃²` → amplifica cuadráticamente las diferencias.
- `w_min = 0` → sin piso, las peores efectivamente desaparecen.
- **Datos reales siempre peso = 1** (no se ponderan, son ground truth).

### Cuidado conceptual

- `w` **NO es un feature**. No entra a `X`, no se usa en inferencia.
- Es el argumento `sample_weight` de `.fit()` — modula cuánto pesa cada muestra en la loss durante entrenamiento.
- En inferencia el modelo solo recibe el embedding 768-d. No necesita `w`.

### `sample_weight` por clasificador

> sample_weight es nativo en LogReg, SVC, Ridge y RandomForest. El MLP de sklearn no lo soporta,
> así que implementé uno en PyTorch que multiplica la cross-entropy por el peso antes de
> retropropagar. Esto funcionó bien en los clasificadores lineales (+2pp), pero degradó
> RandomForest — probablemente porque los splits discretos no aprovechan bien los pesos
> continuos.

| Clasificador | sample_weight | Cómo pondera |
|---|---|---|
| LogReg | ✅ nativo | log-loss |
| SVC linear | ✅ nativo | costo `C·wᵢ` de violar el margen |
| Ridge | ✅ nativo | error cuadrático |
| RandomForest | ✅ nativo | impureza Gini en splits |
| MLP sklearn | ❌ no | (uso impl. propia en PyTorch) |
| kNN | — no aplica | no hay entrenamiento que ponderar |

### Resultados clave

- soft_weighted: **+2.25pp vs SMOTE** (p<0.0001, d=0.74, 83.8% win)
- binary_filter: +2.11pp (p<0.0001, d=0.69)
- **Excluyendo RandomForest**: soft weighting gana ~+2pp en todos los lineales.
- RandomForest catastrófico: -5.29pp (los splits discretos no aprovechan pesos continuos).

---

## N-shot y régimen de datos

| N-shot | Ganancia soft vs SMOTE | Win rate |
|---|---|---|
| 10-shot | **+4.52pp** | 88.6% |
| 25-shot | +1.88pp | **97.1%** |
| 50-shot | +0.34pp | n.s. |

**Patrón**: gana más en low-resource extremo, diminishing returns arriba de 50/clase.

---

## Hallazgos por nº de clases

- 6 clases (emotion): +4.55pp (d=1.24) — sweet spot.
- Binarias: 45-65% win rate — más inconsistente.
- 14+ clases: degrada.
- Scaling negativo: Spearman r=-0.260 (más clases → menores ganancias).

---

## NER (extensión, cross-task)

- Mismos filtros, otra tarea → confirma generalización.
- "Clase" = tipo de entidad dominante; embedding a nivel de oración.
- Cascade L1 gana en 4/9 datasets (más que cualquier otro).
- Combined filter: 0% aceptación en 6/9 datasets (confirma over-filtering).
- Ganancias: 10-shot +15.15pp, 25-shot +9.21pp, 50-shot +6.91pp.

---

## Preguntas anticipadas del tribunal

### "¿Por qué no usar una red neuronal en vez de filtrar?"

Probaste MLP, gana +2.16pp pero igual menos que Ridge/SVC. La diferencia entre métodos de augmentación (~2pp) es **mayor** que la diferencia entre clasificadores lineales (~0.1pp). El filtro importa más que el modelo, en este régimen.

### "¿Por qué SMOTE como baseline y no algo más moderno?"

Comparado contra 10 métodos (`exp_modern_baselines`): t5_paraphrase, contextual_bert, embedding_mixup — todos pierden contra SMOTE. SMOTE sigue siendo el benchmark relevante en low-resource.

### "¿Cómo sé que el resultado no es por azar?"

- Paired t-tests con corrección de Bonferroni.
- p<0.0001 en el experimento principal (3,675 configs).
- Cohen's d=0.74 (efecto grande).
- 5 seeds × 5 clasificadores × 7 métodos × múltiples datasets.

### "El SMOTE que usás, ¿es estándar?"

Sí. Validado en `validate_smote_implementation.py`: la implementación dummy-binary es matemáticamente equivalente al SMOTE estándar binary-mask (max diff 0.0000pp en 27 comparaciones).

### "¿Por qué LOF descrito en la tesis no coincide con el código?"

⚠️ Pendiente de corregir en `Chapters/Metodologia.tex:63`. La tesis describe LOF clásico (score >1 = outlier), pero el código usa `decision_function` de sklearn (score >0 = inlier). Aclarar antes de la defensa.

---

## Dos hallazgos importantes (B0g)

1. **La ablation de temperatura LLM no es monótona.** Si el tribunal pregunta "¿por qué $T{=}0.8$ y no $T{=}1.0$?", la respuesta honesta es: en 20news $T{=}1.0$ daba $+7.44$, pero en sms\_spam $T{=}0.8$ daba $-1.06$. Como las ablations son ruidosas, se eligió $T{=}0.8$ por **convención general en el campo**, no por optimización per-dataset.

2. **Los hiperparámetros del filtro son robustos; los del clasificador/embedding dominan la varianza.** Esto refuerza la narrativa: el filtro funciona en general, no es resultado de overfitting a hiperparámetros.

---

## TODO / pendientes

- [ ] Corregir descripción de LOF en `Escrito_Tesis/Chapters/Metodologia.tex:63` para que coincida con la implementación (`decision_function`, no LOF clásico).
- [ ] Revisar si los slides mencionan "ancla" cuando es "centroide" (ya corregido en slides 21 y cascada).
