# Banco de preguntas — Defensa de Tesis

**Aumentación de Datos con LLMs: Filtrado Geométrico para Clasificación de Texto**
Benjamín Schindler · 1 de junio de 2026 · UAI
Guía: Dr. Gonzalo Ruz · Comité: Dr. Luis Aburto, Dr. Rolando De La Cruz, Dr. Pablo Henríquez, Dr. Andrés Peters

> Cómo usar: cada pregunta tiene **R:** (qué decir, en 1–2 frases) y **→** (a qué slide/backup apuntar y el número exacto). Los números están verificados contra `results/` (ver `experiments/generate_defense_backup_figures.py`).

---

## 0. Referencia rápida (memorizar)

| Resultado | Cifra |
|---|---|
| **Headline vs SMOTE** | **+2.25 pp**, $p<0.0001$ (Bonferroni; Wilcoxon $p\!=\!4{\times}10^{-54}$), $d=0.74$, win 82–84%, $N=3675$. Absoluto 73.49% vs 71.24% |
| Por n-shot | 10→**+4.89** ($d$1.48) · 25→+2.17 ($d$1.57) · 50→+0.76 ($d$0.80) |
| Por clasificador | SVC +2.99 · Ridge +2.69 · MLP +2.16 · LogReg +1.98 · RF +1.42 (n.s.) |
| Baselines modernos | soft +2.61 (88.9%) · binary +2.39 · **todos los modernos pierden** (T5 −0.35, BERT −0.26, mixup −0.01) |
| NER (otra tarea) | cascade L1 **+9.61 pp** ($p=0.0003$, $d=2.05$) |
| Robustez | 5 LLMs (Friedman $p=0.086$) · 4 embeddings (90.5% acuerdo) |
| Config | mpnet-base-v2 (768d) · Gemini 3 Flash $T{=}0.8$ · 3× sobremuestreo · 5 semillas · macro F1 |

### Mapa de backups (agrupados por color de título en el deck)
- 🔵 **Estadística:** B0a (protocolo), B0b (Cohen's $d$), **B0c (Wilcoxon)**, B2 (caching/semillas + re-runs RF/XGB)
- 🟢 **Filtros Geométricos:** B0d₃ (RF), B0d₄ (XGB), **B0d₅ (desglose binario vs peso, por familia)**, **B6 (geometría intrínseca)** ⭐, B0f (tabla completa de filtros)
- 🟣 **Hiperparámetros:** B0e (umbrales/filtro), B0g (LLM+peso suave), **B0i (clasificadores ML — detalle)**, B0h (embeddings/protocolo)
- 🟠 **Métricas Extra:** B5 (baselines modernos), B11 (clases que más ganan), B10 (datasets), **B9 (escalabilidad)** ⭐, B13 (NER filtrado vs sin filtrar)
- 🟦 **Experimentos Extra:** B3 (closed-loop), B4 (teacher), **B12 (curriculum/híbrido)** ⭐, **B8 (circularidad)** ⭐, B1 (validación SMOTE)

---

## ⚠️ Las 6 preguntas más probables (preparar a fondo)

1. **"¿La mejora no es solo por agregar datos del LLM? ¿Qué aporta tu filtro?"** → §B, backups **B0d₃/B0d₄**.
2. **"Filtrás y clasificás con el mismo embedding: es circular."** → §C, backup **B8**.
3. **"¿Por qué SMOTE le gana a métodos modernos como T5/BERT?"** → §G, backup B5.
4. **"El caching da varianza cero en 80% de configs: tu significancia está inflada."** → §A, backup B2.
5. **"Falla con 77+ clases y casi nada en 50-shot: ¿sirve de verdad?"** → §E, backup B9.
6. **"¿Por qué el filtro simple le gana a los multi-criterio? ¿No probaste algo mejor?"** → §I, slide 22, backups B0f y B12.

---

## A. Validez estadística

**A1. "80% de las configuraciones tienen varianza cero entre semillas por el caching. ¿No infla eso la significancia?"**
R: El caching hace deterministas a los clasificadores lineales, pero hay **dos validaciones** con varianza genuina. (1) Subset estocástico del experimento principal (MLP+RF): +1.79 pp, $p=0.0015$, Wilcoxon $p=7{\times}10^{-13}$, $d=0.52$. (2) **Re-corrimos RF y XGBoost de cero** (`classifier_tuning_v2`), donde la varianza cross-seed es **real** (solo 22% cero, vs $\approx$80% cacheado): RF +1.50 pp (Wilcoxon $p=0.012$), XGBoost +9.10 pp (Wilcoxon $p<10^{-9}$, $d=1.22$). Donde la varianza es genuina el efecto no solo sobrevive: **crece**. Las pruebas $t$ pareadas siguen válidas (mismo split por par); lo único que se subestima son los CIs.
→ **B2** (ampliada con re-runs).

**A2. "¿Por qué 5 semillas y holdout en vez de validación cruzada?"**
R: Cada config se evalúa sobre el **test completo** (no submuestreado) con 5 semillas que controlan inicialización y SMOTE; el diseño es **pareado** (mismo dataset×clasificador×semilla), que es lo que da potencia. CV multiplicaría el costo de generación del LLM sin cambiar la conclusión pareada.
→ B0h (protocolo), B0a.

**A3. "¿Por qué tantos tests? ¿Controlaste comparaciones múltiples?"**
R: Sí. 7 métodos vs SMOTE ⇒ Bonferroni ($\alpha=0.05/7$); en la descomposición causal uso Holm. Reporto siempre varias herramientas juntas ($t$ pareada, Wilcoxon, Bonferroni/Holm, Cohen's $d$, win-rate) porque responden preguntas distintas y **apuntan todas en la misma dirección**.
→ **B0a, B0b, B0c**.

**A5. "El $t$-test pareado supone normalidad de las diferencias. ¿La verificaste? ¿Por qué no un test no-paramétrico?"**
R: Corrí también el **Wilcoxon de rangos con signo** (sin supuesto de distribución) sobre el resultado principal: coincide con la $t$ y da $p$ **aún menor** — soft\_weighted $p=4{\times}10^{-54}$, binary\_filter $p=7{\times}10^{-46}$, y por n-shot todos significativos (incluso 50-shot, $p=10^{-5}$, aunque ahí el efecto es pequeño, $d=0.22$). El **win-rate** (82%) es justamente la prueba de signos asociada. La conclusión no depende del supuesto de normalidad.
→ **B0c**.

**A4. "Un $p$ chico con $N$ grande puede ser trivial. ¿Es práctica la diferencia?"**
R: Por eso reporto Cohen's $d$: global $d=0.74$ (medio-grande), SVC $d=1.02$, emotion 6-clases $d=1.24$ (casi sin solapamiento). Es real **y** práctica.
→ **B0b**.

---

## B. ¿El filtro realmente aporta, o es solo el LLM? (central)

**B1. "Descompón la ganancia: ¿cuánto es del LLM y cuánto de tu filtro?"** ⭐
R: Tres pasos aditivos. El **LLM domina** (RF +2.10, XGB +8.94), pero el score geométrico agrega **por encima del LLM sin filtrar**, con un matiz importante: en **clasificadores lineales** aporta como **selección** (filtro binario +1.10 pp vs sin filtrar, $p<0.05$); en **XGBoost (boosting)** aporta como **ponderación** (peso suave +1.59 pp, $p<0.001$). Son mecanismos opuestos pero ambos usan la misma señal.
→ Desglose consolidado por familia en **B0d₅** (binario vs suave: No-árbol +2.27→+2.46, RF +1.51→+1.42, XGB +7.52→+9.10); modos por clasificador en **B0d₃/B0d₄**; slide 22 (cascade L1 +1.10 vs sin filtrar).

**B2. "Mostrame que el filtro mejora la geometría, no solo el F1."** ⭐
R: Métricas intrínsecas (sin tocar el clasificador): el filtrado baja la distancia intra-clase 0.850→0.822 ($p<0.001$), sube la separación inter-clase 0.571→0.616 ($p<0.001$), sube silhouette 0.075→0.085 ($p=0.029$) y baja Davies-Bouldin 3.822→3.370. Cierra el lazo causal entre "filtramos por geometría" y "mejora la clasificación".
→ **B6**.

**B3. "El peso suave no es significativo en el experimento principal ($p=0.10$). ¿Es una contribución real?"**
R: Honestamente, en lineales el peso suave es **marginal** (+0.13 pp, n.s.) — ahí el valor está en la **selección**. Donde el peso suave es **decisivo** es en boosting (XGBoost +1.59 pp, $p<0.001$) y bajo hiperparámetros peaked ($\tau{=}0.5$: +4.10 pp en el sweep). Es una contribución secundaria bien delimitada, no la central. **Aclaración sobre la slide 20** ("Beneficios de la ponderación suave"): ese ΔF1 (+2.99 SVC, +2.69 Ridge, etc.) es del **método completo `soft_weighted` vs SMOTE** (datos LLM + filtro + peso), no del paso de ponderar aislado; "soft_weighted" es el **nombre del método**, no solo del paso. El **B0d₅** separa explícitamente ambas cosas: el *paso* de ponderar (suave − binario) por familia.
→ **B0d₅** (desglose por familia + por n-shot), **B0d₄, B0g**, slide 20.

**B4. "¿Y si el filtro solo conserva casi-duplicados de los datos reales?"**
R: No: el ranking top-N conserva las muestras **cercanas al cluster correcto pero diversas** (la cobertura sube, no colapsa). Si solo duplicara, equivaldría a random-oversample, que **pierde** vs SMOTE (−0.32 pp). Los t-SNE (slides 18a/18b) muestran diversidad real, no duplicación.
→ slides 18a/18b, B6 (cobertura), B5.

---

## C. Circularidad del embedding (crítica metodológica más fuerte)

**C1. "Usás all-mpnet para filtrar Y como features del clasificador. El filtro favorece artificialmente su propio espacio."** ⭐
R: Es la crítica correcta, y la rompo por **tres caminos independientes**: (1) repetimos todo con **4 embeddings distintos** (mpnet, bge-large, bge-small, e5): los 4 significativos, 90.5% de acuerdo; (2) un **teacher classifier** independiente como filtro rinde **menos** que el geométrico (+2.43 vs +2.78), o sea el geométrico no necesita el espacio compartido para ganar; (3) **NER** con pipeline de evaluación totalmente externo (SpaCy+seqeval) da +9.61 pp. Honestamente: queda un 9.5% de desacuerdo entre embeddings en 50-shot, posible efecto residual.
→ **B8**, B0h, B4, slide 24.

**C2. "¿Por qué mpnet (768d) y no un embedding más grande?"**
R: Probamos 4. mpnet-base (768d) es el mejor (+2.74, 92% win) y **supera a e5-large (1024d, +0.97)**: la dimensionalidad no determina el éxito; importa la calidad del espacio. El patrón (n-shot, etc.) se mantiene en los 4.
→ B0h, slide 23.

---

## D. Generalización y validez externa

**D1. "Solo evaluaste en inglés."**
R: Cierto, es una limitación explícita. El método es agnóstico al idioma en principio (solo depende de embeddings y un LLM multilingüe); la validación multilingüe es trabajo futuro declarado.
→ slide 26 (limitaciones).

**D2. "La adaptación a NER (clase = entidad dominante) es simplista."**
R: Es una proyección deliberadamente simple para probar transferencia **sin modificar el filtro**. Aun así da 100% win y +9.61 pp con un pipeline de evaluación independiente — suficiente para mostrar que la señal geométrica es task-agnostic. NER refinado es extensión futura.
→ slide 24, B8.

**D3. "¿Elegiste los datasets a conveniencia?"**
R: No: 7 principales elegidos por **cobertura** en tres ejes (granularidad 2→20 clases, dominio: spam/odio/noticias/emoción/ontología, estilo: corto/largo, formal/informal) + 3 de escalabilidad (6/77/150 clases) para medir hasta dónde escala. 21 configuraciones de dataset en total.
→ **B10**.

---

## E. Escalabilidad y límites

**E1. "Falla con 77 clases (banking77 −0.99) y casi nada con 150. ¿Sirve de verdad?"** ⭐
R: El método es por diseño para **bajo-recurso con pocas clases**. La ganancia decae monótonamente con el número de clases (Spearman $\rho=-0.260$, $p=0.003$): sweet spot 2–6 clases (+3 a +7 pp). El mecanismo: con muchas clases el espacio se fragmenta, todo cae "lejos" del ancla y la distancia pierde poder discriminativo; además banking77 tiene clases semánticamente solapadas. Para taxonomías grandes convienen enfoques jerárquicos.
→ **B9**, slide 21.

**E2. "En 50-shot el efecto casi desaparece (+0.76)."**
R: Es el comportamiento **esperado**: el método corrige distribuciones escasas; con 50 ejemplos/clase ya hay menos que corregir. Aún así sigue siendo positivo y el caso de uso objetivo es 10–25 ejemplos/clase, donde da +4.89/+2.17 pp.
→ slide 19.

**E3. "Random Forest no se beneficia (+1.42, n.s.). ¿Tu método está roto?"**
R: No es un bug, es estructural: **bagging** (RF) ignora `sample_weight` porque cada árbol profundo memoriza. **Boosting** (XGBoost) sí lo usa: +9.10 pp vs SMOTE. La recomendación práctica es usar lineales o XGBoost, no RF, para peso suave.
→ B0d₃, B0d₄, **B0d₅** (RF: paso de ponderar −0.09 n.s. vs XGB +1.59***).

---

## F. Contaminación / novedad de las muestras (punto blando — responder honesto)

**F1. "¿Cómo sabés que el LLM no copió el test o memorizó el train?"**
R: El conjunto de **test nunca entra al prompt** (el LLM solo ve la descripción de la tarea y a lo sumo ejemplos de *train*), así que no hay fuga de test. Copiar el train equivaldría a oversampling, que rinde peor (random-oversample −0.32). La cobertura y los t-SNE muestran diversidad. Honestamente: **no calculamos una métrica explícita de novedad** (self-BLEU/distancia al vecino real) — es una mejora de validación futura.
→ slides 18a/18b, B5, B6 (cobertura).

---

## G. Líneas base y comparación justa

**G1. "Tu SMOTE puede estar mal implementado (dummy-binary)."**
R: Lo validamos: el SMOTE dummy-binary es **idéntico** al SMOTE estándar binary-mask — 27 comparaciones, diferencia máxima 0.0000 pp.
→ **B1**.

**G2. "¿Por qué SMOTE le gana a métodos modernos (T5, BERT, mixup)?"** ⭐
R: Porque esos métodos **no generan diversidad genuina**: T5/BERT parafrasean cerca del original, mixup interpola en embedding. Todos pierden vs SMOTE (T5 −0.35, BERT −0.26, mixup −0.01, EDA −0.49, back-trans −0.50). El LLM **filtrado** es el único que aporta diversidad útil (+2.61). Refuerza que la clave es diversidad controlada, no sofisticación del método.
→ **B5**.

**G3. "¿Por qué no fine-tuning generativo (ImbLLM, SentiGEN)?"**
R: Comparamos contra un **teacher** supervisado (la idea de "entrenar un selector"): el filtro geométrico es **más general** (no se especializa en separabilidad lineal) y **más eficiente** (sin entrenamiento extra): +2.78 vs +2.43. Fine-tuning agrega costo de cómputo sin garantía de ganancia.
→ **B4**.

**G4. "¿Probaste re-promptear el LLM con feedback (closed-loop)?"**
R: Sí, y **no ayuda**: closed-loop +1.12 (52% win) vs pasada única +2.08 (80% win). Las tasas de aceptación incluso **bajan** con iteraciones (−12.3 pp); el 77% de los rechazos son DISTANCE_OUTLIER. El cuello de botella es el LLM, no el prompt.
→ **B3**.

---

## H. Practicidad

**H1. "¿Esto es usable en producción? ¿Cuánto cuesta?"**
R: Sí, es barato y simple: generar 3× candidatas con el LLM, filtrar por distancia (cascade L1), ponderar por score, entrenar un clasificador lineal. ~10–50 llamadas LLM por clase = **centavos** por dataset. Sin fine-tuning ni infraestructura especial.
→ slide 25 (receta práctica).

**H2. "¿Cuándo NO usarlo?"**
R: Con >50 ejemplos/clase (el efecto se diluye), ≥77 clases (el espacio se fragmenta), o si el clasificador es un ensamble de bagging (RF). En esos casos SMOTE o enfoques jerárquicos son mejores.
→ slide 26, B9, B0d₃.

---

## I. ¿Por qué simple > complejo? (hallazgo central)

**I1. "¿Por qué el filtro de distancia simple le gana a los multi-criterio?"** ⭐
R: Tres razones: (1) en vectores L2-normalizados, distancia euclídea y similitud coseno son **monótonamente equivalentes** — agregar similitud no aporta discriminación; (2) LOF es **inestable** con ~10 vecinos reales (régimen de pocos ejemplos); (3) el operador **AND** (combined) es demasiado restrictivo y mata la diversidad (−3.28 vs SMOTE; −6.49 vs LLM sin filtrar; 0% aceptación en 6/9 datasets de NER). La ablación de cascada lo confirma: L1=L2=L3=L4 en win-rate (91.7%), los +0.008 pp son ruido.
→ slide 22, **B0f**, B0e.

**I2. "¿No probaste estrategias más inteligentes?"** ⭐
R: Sí, varias, y **ninguna supera al filtro uniforme simple**: curriculum por etapas da solo +0.13 pp sobre aplicar todo de una; el híbrido adaptativo por dificultad de clase **empeora** cuanto más se especializa (pure_geo +2.33 con 91% win vs mejor híbrido +0.77 con 36%). La parsimonia gana.
→ **B12**.

**I3. "¿Por qué top-N (ranking) y no un umbral duro?"**
R: El ranking top-N selecciona las N mejores del pool 3× sin depender de un umbral frágil. Umbral duro (combined) rechaza demasiado. Ranking +3.66 vs umbral estricto que falla.
→ B0f, B0g.

---

## Cierre sugerido (si preguntan "¿contribución en una frase?")

> "Una señal geométrica simple — distancia al cluster de la clase — basta para convertir texto sintético de un LLM en aumentación que supera a SMOTE de forma estadísticamente significativa en bajo-recurso, y la señal es robusta al LLM, al embedding y a la tarea. El hallazgo transversal es que **filtrar simple y ligero gana**: más criterios, más iteraciones o más especialización no mejoran."

---

## Notas internas (NO decir en voz alta — solo para tu conciencia)

- **NER cascade L1 = +9.61 pp.** Es lo que dicen tu tesis (Resultados.tex, Conclusiones.tex, `tab_ner_results.tex`) y el deck — usá **9.61** y defendelo así. *Pero ojo:* un recálculo fresco de `results/ner_filter_comparison/` da cascade L1 = **+9.26** (LOF relajado +9.28 sí coincide; $p=0.0003$ y $d=2.05$ corresponden a 9.26). Probablemente la tabla NER se generó de una corrida previa. Si alguien recalcula y obtiene 9.26: son un **empate virtual** cascade L1 ≈ LOF (~+9.3) y la conclusión (100% win, agnóstico a la tarea) no cambia. **Decisión pendiente con el guía:** dejar 9.61 (consistencia con la tesis) o regenerar tabla+texto a 9.26.
- **`classifier_tuning_v2` es parcial:** se omitió `dbpedia14_50shot` para ahorrar cómputo (1100 de 1320 corridas). No afecta la descomposición causal ni la significancia.
- **Curriculum e híbrido (B12)** vienen de experimentos separados con su propio SMOTE; los $\Delta$ son comparables en pp pero no son la misma corrida que el experimento principal.
- **B0d₃ y B0d₄ usan ambos el clasificador FIJO** de `tuning_v2` (corregido 2026-05-30: antes B0d₃ mostraba `random_forest_cv`, la única variante donde soft +1.47 > binary +1.22; el RF **fijo** muestra lo correcto: **binary +1.89 > soft +1.50**, paso suave −0.39 n.s. → bagging NO aprovecha el peso). Ahora B0d₃ (RF: binario gana) y B0d₄ (XGB: suave gana) son un contraste limpio y paralelo. La descomposición causal limpia (LLM/filtro/peso) usa estos clasificadores fijos; se quitó como slide aparte (B7) por la ambigüedad con la slide 20, pero el desglose vive ahora en **B0d₅**.
- **RF aparece en dos experimentos** (esperable, ambos válidos): slide 20 + headline Q&A usan RF del **exp. principal** (soft +1.42); B0d₃/B0d₄/B2 usan RF de **tuning_v2 fijo** (soft +1.50, varianza real). B0d₅ usa el principal (+1.51/+1.42) para que RF sea comparable con No-árbol. Todos coinciden en lo cualitativo: **binario ≥ suave en RF**.
- **B0d₅ (nuevo) consolida el desglose binario-vs-peso por familia** y resuelve el hueco "solo verbal": usa el **exp. principal** para No-árbol (LR/SVC/Ridge/MLP) y RF, y `tuning_v2` **fijo** para XGB (XGB no existe en el principal). Cifras verificadas (paired vs SMOTE, N=525 global): binary +2.11/78%/d0.69, soft +2.25/82%/d0.74, paso +0.13 (p=0.10 promediado por semilla / p=0.003 per-config, d trivial). Paso suave por familia: No-árbol +0.19, RF −0.09, XGB +1.59*** (d=0.92). Paso por n-shot (los 5): 10→+0.36, 25→+0.15, 50→−0.11. **OJO de consistencia:** RF aparece como +1.42 (soft, en el principal/B0d₅) y como +1.50/+1.47 en B2/B0d₃ (re-runs `tuning_v2`); son corridas distintas, mismo signo y conclusión.
- **B0i (nuevo)** detalla los hiperparámetros de los 5 clasificadores (defaults) + grillas de tuning de árboles (RF/XGB fijo y CV en train, sin fuga). Complementa el resumen de B0h.
