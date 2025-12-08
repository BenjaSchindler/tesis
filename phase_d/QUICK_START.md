# Phase D: Quick Start Guide

## Objetivo

Probar las 3 mejoras de Phase D (Contrastive Prompting, Focal Loss, Two-Stage Training) de forma **individual y combinada** para identificar cuál mejora más el macro F1, especialmente en clases LOW y MID.

## Estructura del Experimento

```
4 Variantes × 3 Seeds = 12 Experimentos
├─ Variant A: Contrastive Prompting only        (gpt-4o-mini)
├─ Variant B: Focal Loss only                   (gpt-4o-mini)
├─ Variant C: Two-Stage Training only           (gpt-4o-mini)
└─ Variant D: ALL Phase D features + GPT-5-mini (gpt-5-mini-2025-08-07)
```

**Seeds:** 42, 100, 123 (ejecutados secuencialmente en cada VM)

## Prerequisitos

1. **GCP configurado:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

3. **Archivos listos:**
   - ✓ `core/runner_phase2.py` (con Phase D integrado)
   - ✓ `core/mbti_confusers.py`
   - ✓ `core/focal_loss_training.py`
   - ✓ `core/two_stage_training.py`
   - ✓ `MBTI_500.csv`

## Uso

### 1. Lanzar Experimentos

```bash
cd phase_d
./launch_phased_4variants.sh
```

**Lo que hace:**
1. Crea 4 VMs con GPU (NVIDIA T4) en `us-central1-a`
2. Sube runner y módulos Phase D a cada VM
3. Instala dependencias Python
4. Lanza batch scripts que ejecutan 3 seeds secuencialmente
5. Auto-shutdown cuando todos los seeds terminan

**Tiempo estimado:** ~2.5 horas
**Costo estimado:** ~$13 USD

---

### 2. Monitorear Progreso (Opcional)

En una terminal nueva:

```bash
cd phase_d
./monitor_phased.sh
```

**Muestra:**
- Estado de cada VM (RUNNING / COMPLETED / STOPPED)
- Progreso de seeds (1/3, 2/3, 3/3)
- Últimas 5 líneas del log actual

Presiona **Ctrl+C** para salir.

**Tip:** Actualiza cada 60 segundos automáticamente.

---

### 3. Recolectar y Analizar Resultados

Después de ~2.5 horas:

```bash
cd phase_d
./collect_phased_results.sh
```

**Lo que hace:**
1. Descarga 12 archivos de métricas JSON (4 variantes × 3 seeds)
2. Descarga logs y CSVs sintéticos
3. Analiza resultados con Python:
   - Tabla de resultados por seed
   - Estadísticas agregadas (mean ± std)
   - Ranking de variantes
   - Identificación del ganador
4. Pregunta si eliminar VMs para ahorrar costos

**Output esperado:**

```
┌───────────────────────┬────────────┬──────────┬────────────┐
│ Variant               │ Mean Δ F1  │ Std Dev  │ Synthetics │
├───────────────────────┼────────────┼──────────┼────────────┤
│ A (Contrastive)       │ +1.05%     │ ±0.12%   │ 847 ± 23   │
│ B (Focal Loss)        │ +0.98%     │ ±0.18%   │ 892 ± 41   │
│ C (Two-Stage)         │ +0.89%     │ ±0.09%   │ 814 ± 15   │
│ D (Full Stack)        │ +1.23%     │ ±0.14%   │ 763 ± 31   │
└───────────────────────┴────────────┴──────────┴────────────┘

WINNER: D (Full Stack) (+1.230%)
```

---

## Troubleshooting

### VMs no se crean

```bash
# Verificar quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Verificar que billing está habilitado
gcloud beta billing accounts list
```

### Experimentos no arrancan

```bash
# SSH a la VM
gcloud compute ssh vm-phased-a --zone=us-central1-a

# Verificar proceso
ps aux | grep python3

# Ver log
tail -50 phased_a_seed42.log
```

### Archivos no se descargan

```bash
# Verificar que VM existe
gcloud compute instances list --filter='name~phased'

# Si VM está TERMINATED, revisar logs de por qué falló
gcloud compute instances get-serial-port-output vm-phased-a --zone=us-central1-a
```

### Costs demasiado altos

```bash
# Detener todas las VMs inmediatamente
gcloud compute instances stop vm-phased-{a,b,c,d} --zone=us-central1-a

# O eliminarlas
gcloud compute instances delete vm-phased-{a,b,c,d} --zone=us-central1-a
```

---

## Resultados Almacenados

```
phase_d/results/phased_variants/
├── phased_a_seed42_metrics.json
├── phased_a_seed42_synthetic.csv
├── phased_a_seed42.log
├── phased_a_seed100_metrics.json
├── ... (36 archivos total: 12 metrics + 12 synthetics + 12 logs)
```

---

## Siguiente Paso

Si **Variant D gana:**
- Correr con 5 seeds para validación estadística
- Analizar per-class F1 para confirmar mejoras en LOW/MID tiers
- Deploy a producción

Si **otra variante individual gana:**
- Considerar combinarla con las otras (puede haber sinergia)
- Analizar por qué D no mejoró (debugging de logs)

---

## Comandos Útiles

```bash
# Ver estado de VMs
gcloud compute instances list --filter='name~phased'

# SSH a VM específica
gcloud compute ssh vm-phased-a --zone=us-central1-a

# Ver logs en tiempo real
gcloud compute ssh vm-phased-a --zone=us-central1-a \
  --command='tail -f phased_a_seed42.log'

# Descargar archivo específico
gcloud compute scp vm-phased-a:~/phased_a_seed42_metrics.json ./ \
  --zone=us-central1-a

# Eliminar todas las VMs
gcloud compute instances delete vm-phased-{a,b,c,d} \
  --zone=us-central1-a --quiet
```

---

## Configuración Base (Phase A)

Todas las variantes usan esta configuración base:

```bash
--embedding-batch-size 256              # GPU optimizado
--llm-model gpt-4o-mini                # Base (Variant D usa gpt-5-mini)
--max-clusters 3
--prompts-per-cluster 3
--anchor-quality-threshold 0.30        # Phase A value
--f1-budget-thresholds 0.45 0.20       # Phase A values
--f1-budget-multipliers 0.0 0.5 1.0    # Phase A values
--synthetic-weight-mode flat           # Phase A mode
```

**Diferencias con Phase C:**
- ❌ Sin `--enable-adaptive-filters`
- ❌ Sin `purity-gate-threshold`
- ✅ Configuración limpia de Phase A

---

## Costo Breakdown

| Item | Cantidad | Costo Unitario | Total |
|------|----------|----------------|-------|
| n1-standard-4 VMs | 4 × 2.5 hrs | $0.20/hr | $2.00 |
| NVIDIA T4 GPUs | 4 × 2.5 hrs | $0.35/hr | $3.50 |
| OpenAI gpt-4o-mini | 9 runs | $0.50/run | $4.50 |
| OpenAI gpt-5-mini | 3 runs | $1.00/run | $3.00 |
| **TOTAL** | | | **~$13.00** |

*Nota: Si las VMs no se auto-apagan, el costo es ~$2.20/hora extra.*

---

## Support

Para problemas o preguntas:
- Ver logs en `phase_d/results/phased_variants/*.log`
- Revisar plan detallado: `.claude/plans/precious-moseying-reef.md`
- Verificar módulos: `python3 -c "from core.mbti_confusers import get_confusers"`
