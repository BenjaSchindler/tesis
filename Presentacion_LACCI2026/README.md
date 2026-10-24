# LACCI 2026 talk (15 min, English)

**Paper:** Geometric Filtering of LLM-Generated Samples for Few-Shot Text Classification
**Authors:** Benjamín Schindler, Gonzalo A. Ruz (UAI)
**Venue:** IEEE LACCI 2026, Universidad del Pacífico, Lima, Nov 3–6, 2026
(presentation upload deadline: Oct 25, 2026)

A condensed English version of [`../Presentacion_Defensa`](../Presentacion_Defensa), which is
left untouched. Every number follows the camera-ready paper
[`../Escrito_Tesis_IEEE/paper.tex`](../Escrito_Tesis_IEEE/paper.tex), not the thesis defense
(which used older figures, e.g. +2.25 pp instead of +2.61 pp).

## Two versions

| Deck | Slides | When to use |
|---|---|---|
| `main.pdf` (full) | 15 + 3 backup | Expert audience; every slide carries its statistics (d, CIs, win rates). |
| `main_simple.pdf` (simple) | 14 + 9 backup | Default for the 15-min talk: one idea per slide, assertion titles, one chart per slide. Its backup appends the full deck's result slides and backups, so the details are one click away in Q&A. |

Both use the same numbers and styles (`preamble.tex`, `style/`); `speaker_notes.md` follows
the full deck's slide order.

## Build

```bash
bash compile.sh
```

Builds both decks (`bash compile.sh main_simple` builds one). Uses [Tectonic](https://tectonic-typesetting.github.io) when it is on `PATH` (it fetches the
needed packages on the first run), otherwise XeLaTeX. Cantarell is used if installed;
otherwise the Metropolis default font is used.

## Structure

```
Presentacion_LACCI2026/
├── main.tex                  # full deck
├── main_simple.tex           # simple deck
├── preamble.tex              # shared theme, packages, title info
├── compile.sh
├── speaker_notes.md          # script, timing per slide, likely questions
├── sections_simple/
│   ├── talk.tex                  # 14 slides of the simple deck
│   └── backup.tex                # reuses sections/03_results + 05_backup
├── sections/
│   ├── 01_title_motivation.tex   # slides 1, 3–4
│   ├── problem_intro.tex         # slide 2, shared by both decks
│   ├── 02_method.tex             # slides 5–8
│   ├── 03_results.tex            # slides 9–13
│   ├── 04_closing.tex            # slides 14–15
│   └── 05_backup.tex             # 3 backup slides (Q&A only)
├── figures/
│   ├── fig_tsne_before_after_emotion_en.png   # English relabel of the thesis t-SNE figure
│   ├── fig_tsne_emotion_{before,after}.png    # its two panels, cropped by crop_tsne.py
│   └── crop_tsne.py
├── style/                    # UAI colors + tcolorbox blocks (copied from the defense)
│   └── charts.sty            # shared pgfplots/TikZ chart styles
└── logos/
```

## Differences from the defense deck

- 27 + 22 backup slides → **15 + 3 backup**; no section divider pages.
- Dropped: hook/imbalance slides, state-of-the-art families, hypotheses, macro-F1 definition,
  dataset table, NER explainer, embedding-model table, references slide.
- All charts redrawn in pgfplots/TikZ in English with the paper's numbers, sharing one style
  (`style/charts.sty`): main result with 95% CIs, curriculum curve (slide 12), gain by number
  of classes (slide 14) and an ablation waterfall (backup 1). The t-SNE panel titles and legend
  are typeset in LaTeX because the PNG's own are unreadable when projected. The thesis PNGs are in
  Spanish, and `fig_delta_by_nshot.png` does not match the paper (it shows ≈4.5/1.9/0.3 pp
  instead of 4.89/2.17/0.76 pp).
- The "simple > complex" slide no longer cites the −3.28 pp combined-filter figure, which
  does not appear in the paper or the thesis text; it uses the paper's three mechanisms and
  the NER acceptance-rate evidence instead.
