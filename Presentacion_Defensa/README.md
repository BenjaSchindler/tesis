# Presentación de Defensa de Tesis

**Tesis:** Aumentación de Datos con LLMs: Filtrado Geométrico para Clasificación de Texto
**Autor:** Benjamín Schindler — UAI — Abril 2026
**Profesor guía:** Gonzalo Ruz

## Compilación

```bash
bash compile.sh         # compila main.pdf (xelatex × 2)
bash compile.sh clean   # limpia archivos auxiliares
```

Requisitos: XeLaTeX, paquete `beamertheme-metropolis` y la fuente Cantarell
(estándar en distribuciones GNOME / instalable con `apt install fonts-cantarell`).

## Estructura

```
Presentacion_Defensa/
├── main.tex                  # Documento maestro
├── compile.sh                # Script de compilación XeLaTeX
├── style/
│   ├── uai-colors.sty        # Paleta UAI sobre Metropolis
│   └── beamer-blocks.sty     # Bloques tcolorbox custom
├── secciones/                # Una por bloque narrativo
│   ├── 00_portada.tex        # Slides 1–2
│   ├── 01_introduccion.tex   # Slides 3–5
│   ├── 02_problema.tex       # Slides 6–7
│   ├── 03_estado_arte.tex    # Slides 8–10
│   ├── 04_hipotesis.tex      # Slide 11
│   ├── 05_metodologia.tex    # Slides 12–16
│   ├── 06_resultados.tex     # Slides 17–24 (corazón visual)
│   ├── 07_conclusiones.tex   # Slides 25–27
│   └── 08_backup.tex         # Slides B1–B5 (Q&A)
├── figuras/                  # symlink a ../Escrito_Tesis/Figures/
└── logos/                    # logo FIC (UAI)
```

## Outline de la presentación (~25 min)

| # | Slide | Bloque | Tiempo |
|---|---|---|---|
| 1–2 | Portada + roadmap | Apertura | 1.5 min |
| 3–5 | Hook, escasez, trade-off SMOTE/LLM | Motivación | 3 min |
| 6–7 | 3 deficiencias + pregunta | Problema | 2 min |
| 8–10 | Timeline, insights teóricos, Venn | SOTA | 3 min |
| 11 | H1 + H1a–d + objetivos | Hipótesis | 1 min |
| 12–16 | Pipeline, 5 filtros, cascada, soft-weight, protocolo | Metodología | 5 min |
| 17–24 | Headline, t-SNE, $n$-shot, clasificadores, simple>complejo, robustez, NER | **Resultados** | 8 min |
| 25–27 | Contribuciones, utilidad, cierre | Discusión | 2.5 min |
| B1–B5 | SMOTE, semillas, closed-loop, teacher, modernos | Backup Q&A | — |

**Total: 27 slides + 5 backup ≈ 25 min.**

## Mensajes anclaje

1. **Slide 3 (hook):** "60 muestras, +2.25 pp con un truco geométrico simple."
2. **Slide 17 (headline):** $+2{.}25$ pp vs.\ SMOTE, $p<0.0001$, $d=0.74$.
3. **Slide 22 (contra-intuitivo):** simple > complejo (Combined cae −3.28 pp).
4. **Slide 24 (NER):** mismo filtro, otra tarea, +9.61 pp.
5. **Slide 26 (utilidad):** receta concreta + costo en centavos.

## Convenciones cromáticas

- **Verde** = nuestro método (soft-weighted, binary-filter).
- **Azul** = SMOTE (referencia).
- **Gris** = otros baselines.
- **Naranja** = cifras headline o advertencias contra-intuitivas.

## Notas de mantenimiento

- Las figuras se referencian vía symlink a `../Escrito_Tesis/Figures/`. No
  duplicar; si la tesis cambia una figura, la presentación la actualiza
  automáticamente.
- Toda cifra citada coincide exactamente con
  [`Chapters/Resultados.tex`](../Escrito_Tesis/Chapters/Resultados.tex)
  y con la auto-memoria del proyecto.
- Para ver la presentación con notas de orador, agregar
  `\setbeameroption{show notes on second screen=right}` al preámbulo de
  `main.tex` y completar `\note{...}` por slide.
