# Phase G Validation - LaTeX Tables for Thesis

Copy-paste ready LaTeX tables for thesis document.

---

## Table 1: Top 10 Configurations

```latex
\begin{table}[htbp]
\centering
\caption{Top 10 Configurations in Phase G Validation}
\label{tab:phaseG_top10}
\begin{tabular}{clcccc}
\toprule
\textbf{Rank} & \textbf{Configuration} & \textbf{Category} & \textbf{$\Delta$ F1 (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
1 & W5\_many\_shot\_10 & Prompting & +5.98 & 0.00005 & \checkmark \\
2 & W6\_temp\_high & Temperature & +5.57 & 0.0002 & \checkmark \\
3 & W5\_few\_shot\_3 & Prompting & +5.34 & 0.0005 & \checkmark \\
4 & V4\_ultra & Volume & +5.22 & 0.00001 & \checkmark \\
5 & W7\_yolo & No Filter & +5.05 & 0.0003 & \checkmark \\
6 & ENS\_WaveChampions & Ensemble & +4.40 & 0.0042 & \checkmark \\
7 & W3\_permissive\_filter & Filtering & +4.35 & 0.0001 & \checkmark \\
8 & ENS\_Top3\_G5 & Ensemble & +4.33 & 0.0002 & \checkmark \\
9 & CMB3\_skip & Component & +4.32 & 0.0006 & \checkmark \\
10 & W6\_temp\_low & Temperature & +3.89 & 0.0031 & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2: Wave Comparison

```latex
\begin{table}[htbp]
\centering
\caption{Summary of Experimental Waves in Phase G Validation}
\label{tab:phaseG_waves}
\begin{tabular}{clccc}
\toprule
\textbf{Wave} & \textbf{Focus} & \textbf{Configs} & \textbf{Best $\Delta$ (\%)} & \textbf{Avg $\Delta$ (\%)} \\
\midrule
Wave 1 & Quality gates & 3 & +3.48 & +2.80 \\
Wave 2 & Volume oversampling & 2 & +3.55 & +3.51 \\
Wave 3 & Dedup \& filtering & 2 & +4.35 & +4.12 \\
Wave 4 & Targeted generation & 1 & +1.46 & +1.46 \\
\textbf{Wave 5} & \textbf{Few/many-shot} & \textbf{3} & \textbf{+5.98} & \textbf{+4.38} \\
Wave 6 & Temperature diversity & 3 & +5.57 & +4.37 \\
Wave 7 & YOLO (no filter) & 2 & +5.05 & +3.46 \\
Wave 8 & GPT-4o reasoning & 2 & 0.00 & 0.00 \\
Wave 9 & Contrastive learning & 2 & +3.84 & +2.67 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3: Multi-Classifier Comparison

```latex
\begin{table}[htbp]
\centering
\caption{Multi-Classifier Evaluation on RARE\_massive\_oversample Synthetic Data}
\label{tab:multiclassifier}
\begin{tabular}{lccccc}
\toprule
\textbf{Classifier} & \textbf{Baseline} & \textbf{Augmented} & \textbf{$\Delta$ (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
\textbf{MLP\_512\_256\_128} & \textbf{0.2075} & \textbf{0.2333} & \textbf{+12.41} & \textbf{0.0077} & \checkmark \\
LogisticRegression & 0.2272 & 0.2308 & +1.61 & 0.3001 & \\
MLP\_256\_128 & 0.2273 & 0.2306 & +1.47 & 0.6224 & \\
LightGBM & 0.1677 & 0.1667 & $-0.63$ & 0.6672 & \\
XGBoost & 0.1788 & 0.1745 & $-2.41$ & 0.1792 & \\
\bottomrule
\multicolumn{6}{l}{\small \textit{Note: Tree-based models fail with 768D embeddings.}} \\
\end{tabular}
\end{table}
```

---

## Table 4: Rare Class Performance (Multi-Classifier)

```latex
\begin{table}[htbp]
\centering
\caption{Rare Class Improvements Across Different Classifiers}
\label{tab:rare_class_classifiers}
\begin{tabular}{lcccc}
\toprule
\textbf{Classifier} & \textbf{ESFJ $\Delta$} & \textbf{ESFP $\Delta$} & \textbf{ESTJ $\Delta$} & \textbf{Overall $\Delta$ (\%)} \\
\midrule
\textbf{MLP\_512\_256\_128} & \textbf{+0.1242} & \textbf{0.0000} & \textbf{+0.0179} & \textbf{+12.41} \\
MLP\_256\_128 & +0.1123 & 0.0000 & 0.0000 & +1.47 \\
LogisticRegression & +0.0266 & +0.0068 & +0.0024 & +1.61 \\
XGBoost & $-0.0267$ & 0.0000 & 0.0000 & $-2.41$ \\
LightGBM & $-0.0415$ & 0.0000 & 0.0000 & $-0.63$ \\
\bottomrule
\multicolumn{5}{l}{\small \textit{Synthetic samples: ESFJ=780, ESFP=992, ESTJ=811 (including originals).}} \\
\end{tabular}
\end{table}
```

---

## Table 5: Rare Class Experiments (Exp 13)

```latex
\begin{table}[htbp]
\centering
\caption{Rare Class Focused Configurations (Experiment 13)}
\label{tab:rare_class_exp13}
\begin{tabular}{lcccccc}
\toprule
\textbf{Config} & \textbf{Overall} & \textbf{p-value} & \textbf{Sig.} & \textbf{ESFJ} & \textbf{ESFP} & \textbf{ESTJ} \\
 & \textbf{$\Delta$ (\%)} & & & \textbf{$\Delta$} & \textbf{$\Delta$} & \textbf{$\Delta$} \\
\midrule
RARE\_massive & \textbf{+2.07} & \textbf{0.0453} & \checkmark & \textbf{+0.0802} & 0.0000 & 0.0000 \\
RARE\_high\_temp & +0.46 & 0.5865 & & +0.0415 & 0.0000 & 0.0000 \\
RARE\_yolo\_ext & +0.51 & 0.3665 & & +0.0148 & 0.0000 & 0.0000 \\
RARE\_contrast & $-0.24$ & 0.5566 & & 0.0000 & 0.0000 & 0.0000 \\
RARE\_few\_shot & $-0.26$ & 0.5279 & & 0.0000 & 0.0000 & 0.0000 \\
\bottomrule
\multicolumn{7}{l}{\small \textit{RARE\_massive: 20x multiplier, 738 ESFJ + 944 ESFP + 772 ESTJ synthetics.}} \\
\end{tabular}
\end{table}
```

---

## Table 6: Component Validation

```latex
\begin{table}[htbp]
\centering
\caption{Component Validation Results}
\label{tab:component_validation}
\begin{tabular}{llccc}
\toprule
\textbf{Config} & \textbf{Component Tested} & \textbf{$\Delta$ (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
V4\_ultra & High budget (tier: 30,20,15) & +5.22 & 0.00001 & \checkmark \\
CMB3\_skip & Skip clustering & +4.32 & 0.0006 & \checkmark \\
G5\_K25\_medium & K=25 neighbors & +3.21 & 0.0193 & \checkmark \\
CF1\_conf\_band & Confidence band filter & +3.01 & 0.0038 & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 7: Ensemble Configurations

```latex
\begin{table}[htbp]
\centering
\caption{Ensemble Configuration Results}
\label{tab:ensembles}
\begin{tabular}{llccc}
\toprule
\textbf{Ensemble} & \textbf{Configs Combined} & \textbf{$\Delta$ (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
ENS\_WaveChampions & W5, W6, W7 best & +4.40 & 0.0042 & \checkmark \\
ENS\_Top3\_G5 & Top 3 Phase G & +4.33 & 0.0002 & \checkmark \\
ENS\_ProblemClass & All rare class & +3.69 & 0.0033 & \checkmark \\
ENS\_TopG5\_Ext & Top 5 Waves 1-7 & +3.58 & 0.0136 & \checkmark \\
ENS\_SUPER\_G5\_F7 & Phase G+F best & +2.75 & 0.0441 & \checkmark \\
\bottomrule
\multicolumn{5}{l}{\small \textit{Note: Ensembles don't surpass best individual config (W5: +5.98\%).}} \\
\end{tabular}
\end{table}
```

---

## Table 8: Prompting Strategy Comparison (Wave 5)

```latex
\begin{table}[htbp]
\centering
\caption{Few-Shot vs Many-Shot Prompting Comparison (Wave 5)}
\label{tab:prompting_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Config} & \textbf{Examples} & \textbf{$\Delta$ (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
\textbf{W5\_many\_shot\_10} & \textbf{10} & \textbf{+5.98} & \textbf{0.00005} & \checkmark \\
W5\_few\_shot\_3 & 3 & +5.34 & 0.0005 & \checkmark \\
W5\_zero\_shot & 0 & +1.82 & 0.1247 & \\
\bottomrule
\multicolumn{5}{l}{\small \textit{Baseline F1: 0.2045 (LogisticRegression on MPNet embeddings).}} \\
\end{tabular}
\end{table}
```

---

## Table 9: Temperature Experiments (Wave 6)

```latex
\begin{table}[htbp]
\centering
\caption{LLM Temperature Impact on Synthetic Data Quality (Wave 6)}
\label{tab:temperature_exp}
\begin{tabular}{lcccc}
\toprule
\textbf{Config} & \textbf{Temperature} & \textbf{$\Delta$ (\%)} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
\textbf{W6\_temp\_high} & \textbf{1.2} & \textbf{+5.57} & \textbf{0.0002} & \checkmark \\
W6\_temp\_low & 0.3 & +3.89 & 0.0031 & \checkmark \\
W6\_temp\_extreme & 1.5 & +3.66 & 0.0058 & \checkmark \\
\bottomrule
\multicolumn{5}{l}{\small \textit{Optimal temperature: 1.2 (balance between diversity and coherence).}} \\
\end{tabular}
\end{table}
```

---

## Table 10: Phase F vs Phase G Comparison

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of Phase F and Phase G Best Configurations}
\label{tab:phase_f_vs_g}
\begin{tabular}{lcccc}
\toprule
\textbf{Phase} & \textbf{Best Config} & \textbf{$\Delta$ F1 (\%)} & \textbf{p-value} & \textbf{Key Innovation} \\
\midrule
Phase F & PF\_optimal & +2.07 & 0.0234 & Component optimization \\
\textbf{Phase G} & \textbf{W5\_many\_shot\_10} & \textbf{+5.98} & \textbf{0.00005} & \textbf{Many-shot prompting} \\
\midrule
\multicolumn{5}{l}{\textbf{Improvement:}} \\
\multicolumn{2}{l}{Absolute F1} & \multicolumn{3}{l}{0.2087 → 0.2167 (+0.0080)} \\
\multicolumn{2}{l}{Relative improvement} & \multicolumn{3}{l}{+2.07\% → +5.98\% (2.9× better)} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 11: Statistical Summary

```latex
\begin{table}[htbp]
\centering
\caption{Statistical Summary of Phase G Validation}
\label{tab:statistical_summary}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Notes} \\
\midrule
Total configurations tested & 38 & Across 9 waves + special exps \\
Statistically significant & 30/38 (78.9\%) & $p < 0.05$ \\
Best overall improvement & +5.98\% & W5\_many\_shot\_10 \\
Strongest p-value & 0.00001 & V4\_ultra \\
Baseline F1 (macro) & 0.2045 & LogReg on MPNet embeddings \\
Best F1 achieved & 0.2167 & +0.0122 absolute \\
Configs with $\Delta > +5\%$ & 5 & 13.2\% of total \\
Configs with $\Delta > +4\%$ & 11 & 28.9\% of total \\
Negative results & 5 & 13.2\% (mostly tree models) \\
\midrule
\multicolumn{3}{l}{\textbf{Rare Class Results:}} \\
ESFJ improved (MLP\_512) & +12.42\% & With 780 total samples \\
ESFP improved & 0.00\% & Unsolved across all configs \\
ESTJ improved (MLP\_512) & +1.79\% & With 811 total samples \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 12: Problem Classes Summary

```latex
\begin{table}[htbp]
\centering
\caption{Problem Classes Summary and Solutions}
\label{tab:problem_classes_summary}
\begin{tabular}{lccccc}
\toprule
\textbf{Class} & \textbf{Samples} & \textbf{Standard} & \textbf{Massive} & \textbf{MLP\_512} & \textbf{Status} \\
 & & \textbf{Config} & \textbf{Oversample} & \textbf{Classifier} & \\
\midrule
ESFJ & 42 & 0.00\% & +8.02\% & \textbf{+12.42\%} & \textbf{Solved} \\
ESFP & 48 & 0.00\% & 0.00\% & 0.00\% & \textbf{Unsolved} \\
ESTJ & 39 & 0.00\% & 0.00\% & +1.79\% & Partial \\
\bottomrule
\multicolumn{6}{l}{\small \textit{Standard: Wave 1-9 best configs; Massive: 20× oversampling; MLP\_512: Neural network.}} \\
\end{tabular}
\end{table}
```

---

## Recommended LaTeX Packages

Add to thesis preamble:

```latex
\usepackage{booktabs}      % For professional tables
\usepackage{amssymb}       % For \checkmark
\usepackage{multirow}      % For multi-row cells (if needed)
\usepackage{array}         % For advanced table formatting
\usepackage{graphicx}      % For including plots
```

---

## Including Plots in LaTeX

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{phase_g_validation/plots/top10_configs.png}
\caption{Top 10 Configurations in Phase G Validation}
\label{fig:phaseG_top10}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{phase_g_validation/plots/multiclassifier_comparison.png}
\caption{Multi-Classifier Comparison for Rare Class Improvement}
\label{fig:multiclassifier}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{phase_g_validation/plots/rare_class_heatmap.png}
\caption{Rare Class Improvements Across Configurations}
\label{fig:rare_class_heatmap}
\end{figure}
```

---

**All tables are copy-paste ready for LaTeX compilation.**
