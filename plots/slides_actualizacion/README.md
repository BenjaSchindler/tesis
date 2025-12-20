# Contenido para Actualizar Presentación PPTX

## Archivos Generados

| Archivo | Slide | Acción |
|---------|-------|--------|
| `slide_10_config_experimental.md` | Slide 10 | ACTUALIZAR |
| `slide_11_resultados_logreg.md` | Slide 11 | ACTUALIZAR |
| `slide_12a_ensembles.md` | Nueva Slide 12a | INSERTAR |
| `slide_12b_multiclasificador.md` | Nueva Slide 12b | INSERTAR |
| `slide_12c_clases_problematicas.md` | Nueva Slide 12c | INSERTAR |
| `slide_13_hallazgos_clave.md` | Slide 13 (antes 12) | ACTUALIZAR |

## Orden Final de Slides

```
1.  Título
2.  Contexto
3.  Problema
4.  Enfoques Previos
5.  SOTA Métodos
6.  Gap Metodológico
7.  Hipótesis y Objetivos
8.  Metodología Parte 1
9.  Metodología Parte 2
10. Configuración Experimental [ACTUALIZAR]
11. Resultados LogReg [ACTUALIZAR]
12a. Análisis de Ensembles [NUEVA]
12b. Validación Multi-Clasificador [NUEVA]
12c. Clases Problemáticas [NUEVA]
13. Hallazgos Clave [ACTUALIZAR - era slide 12]
14. Conclusiones [era slide 13]
```

## Figuras Disponibles para Insertar

```
Escrito_Tesis/Figures/fig_multiclassifier_comparison.pdf  → Slide 12b
Escrito_Tesis/Figures/fig_per_class_improvement.pdf       → Slide 12c
Escrito_Tesis/Figures/fig_rare_class_heatmap.pdf          → Slide 12c
Escrito_Tesis/Figures/fig_wave_comparison.pdf             → Slide 11
```

## Narrativa Principal

1. **Slide 11**: Resultados con LogReg (+1.67pp mejor config)
2. **Slide 12a**: Ensembles NO mejoran - redundancia diluye
3. **Slide 12b**: MLP Profundo es **12× mejor** que LogReg (+4.17pp)
4. **Slide 12c**: ESFJ resuelto, ESFP limitación conocida
5. **Slide 13**: Hallazgos actualizados con nuevos descubrimientos
