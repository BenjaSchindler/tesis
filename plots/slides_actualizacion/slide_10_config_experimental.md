# Slide 10: Configuración Experimental (ACTUALIZAR)

## Título
**Configuración Experimental**

---

## Cambios a Realizar

### Texto Original (aproximado)
> "273 configuraciones probadas"

### Texto Actualizado
> "**500+ configuraciones validadas** con múltiples clasificadores"

---

## Contenido Actualizado

### Dataset
- **MBTI Personality**: 8,675 publicaciones
- **16 clases** de personalidad (4 dimensiones × 2 valores)
- **Ratio de desbalance**: 47:1 (INFP vs ESTJ)

### Embeddings
- **Modelo**: all-mpnet-base-v2
- **Dimensiones**: 768
- **Ventaja**: +14.7 pp sobre TF-IDF

### Validación
- **Método**: K-Fold CV estratificado (5×3 = 15 folds)
- **Métrica primaria**: Macro F1-Score
- **Prueba estadística**: t-test pareada

### Espacio de Configuraciones

| Categoría | Parámetros Validados |
|-----------|---------------------|
| Agrupamiento | K_max ∈ {1, 2, 5, 10, 15, 18, 24} |
| Anclas | 4 estrategias (medoide, diversidad, calidad, ensemble) |
| Vecinos contexto | K ∈ {1, 5, 10, 15, 25, 50, 100, 200} |
| Temperatura LLM | τ ∈ {0.2, 0.3, 0.5, 0.7, 0.9, 1.2} |
| Filtros | 5 configuraciones de cascada |
| Prompting | n-shot ∈ {0, 3, 10, 20, 60, 70, 100, 200} |
| **Clasificadores** | **5 tipos (LogReg, MLP×2, XGBoost, LightGBM)** |

### Total de Experimentos
- **Configuraciones individuales**: 273+
- **Ensembles**: 45+
- **Validación PyTorch**: 50+
- **Multi-clasificador**: 25+
- **Total**: **500+ configuraciones**

---

## Notas del Presentador
- Enfatizar la exhaustividad de la validación
- Mencionar que se probaron múltiples clasificadores (novedad)
- La validación rigurosa (15 folds) da robustez estadística
