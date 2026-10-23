# Charla PhytoLearning — versión divulgación

**Título:** Geometría y LLMs para clases con pocos ejemplos
**Autor:** Benjamín Schindler — Laboratorio Dr. Gonzalo Ruz, UAI
**Ocasión:** Reunión Mensual PhytoLearning — viernes 31 de julio de 2026, 14:30 hrs (Zoom)

Adaptación de la presentación de defensa ([`../Presentacion_Defensa`](../Presentacion_Defensa))
para **público general no especializado**. La versión de defensa se mantiene intacta
en esa carpeta; esta es una copia independiente.

## Compilación

```bash
bash compile.sh         # compila main.pdf (xelatex × 2)
bash compile.sh clean   # limpia archivos auxiliares
```

Requisitos: XeLaTeX, paquete `beamertheme-metropolis` y la fuente Cantarell
(estándar en distribuciones GNOME / instalable con `apt install fonts-cantarell`).

## Qué cambia respecto a la versión de defensa

1. **Portada nueva** — contexto PhytoLearning, sin comité ni grados a optar.
2. **Slide nueva** «La pieza clave: convertir cada texto en un punto de un mapa»
   (`03_estado_arte.tex`) — explica embeddings antes de usarlos, con el diagrama de
   comidas del Google ML Crash Course. Es la slide fundacional de esta versión.
3. **Sin fórmulas** — SMOTE, macro F1, cascada y ponderación suave se explican con
   intuición y analogías. Los visuales (TikZ) se conservan.
4. **Jerga traducida** en todas las slides: `recall` → «% de casos que detecta»,
   `win-rate` → «% de victorias», `p < 0.0001 Bonferroni` → «real, no azar»,
   LLM → «IA», embeddings → «el mapa», oversampling → «crear más de lo raro».
5. **Slides desactivadas** con `\iffalse ... \fi` (no borradas, reactivables):
   - `03_estado_arte.tex`: «Familia 2 — aumentación textual ligera»,
     «Familia 3.5 — fine-tuning», «Tres deficiencias en los métodos actuales».
   - `06_resultados.tex`: «Hallazgo contra-intuitivo: simple $>$ complejo»,
     «Generalización a NER».
6. **«Hipótesis y objetivos» recortada** — quedan solo H1 y el objetivo general.
7. **Backup (`08_backup.tex`) intacto** — para esta charla simplemente no se muestra.

## Guion

[`notas_divulgacion.md`](notas_divulgacion.md) trae, por slide, la analogía sugerida
para decir en voz alta, más un banco de preguntas probables de público general.

## Estructura

```
Presentacion_PhytoLearning/
├── main.tex                  # Documento maestro
├── compile.sh                # Script de compilación XeLaTeX
├── notas_divulgacion.md      # Guion con analogías por slide (usar este)
├── notas_defensa.md          # Heredados de la copia: guion y banco de
├── preguntas_defensa.md      #   preguntas de la defensa (nivel técnico)
├── style/
│   ├── uai-colors.sty        # Paleta UAI sobre Metropolis
│   └── beamer-blocks.sty     # Bloques tcolorbox custom
├── secciones/
│   ├── 00_portada.tex        # Portada PhytoLearning + hoja de ruta
│   ├── 01_introduccion.tex
│   ├── 02_problema.tex
│   ├── 03_estado_arte.tex    # Incluye la slide del «mapa»
│   ├── 04_hipotesis.tex
│   ├── 05_metodologia.tex
│   ├── 06_resultados.tex
│   ├── 07_conclusiones.tex
│   └── 08_backup.tex         # Q&A (no se proyecta)
├── figuras/                  # symlink a ../Escrito_Tesis/Figures/
├── img/                      # imágenes propias de esta charla (ver Créditos)
└── logos/                    # logo FIC (UAI)
```

## Créditos de imágenes

`img/embeddings_3D_tangyuan.png` — Google, *Machine Learning Crash Course*,
[«Embedding space»](https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Se usa **sin modificaciones**, con las etiquetas originales en inglés. La atribución
aparece impresa en la slide, como exige la licencia.

## Mensajes anclaje

1. **El mapa:** cada texto es un punto; cerca = parecido. Todo depende de esto.
2. **La idea:** pedirle a la IA más de lo necesario y quedarse solo con lo bien ubicado.
3. **El resultado:** de un 71 a un 73,5 en promedio, sobre 8.000+ experimentos.
4. **Lo contra-intuitivo:** el filtro más simple le ganó a todos los sofisticados.
5. **La generalización:** el mismo filtro, sin tocarlo, funcionó en otra tarea (NER).

## Convenciones cromáticas

- **Verde** = nuestro método. **Azul** = SMOTE (referencia).
- **Gris** = otros métodos. **Naranja** = cifras destacadas o advertencias.

## Notas de mantenimiento

- Las figuras se referencian vía symlink relativo a `../Escrito_Tesis/Figures/`.
  Si mueves esta carpeta fuera de `filters/`, el symlink se rompe.
- Las cifras coinciden con las de la versión de defensa; no se re-calculó nada.
