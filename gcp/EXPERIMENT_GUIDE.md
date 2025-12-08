# GCP Experiment Launch Guide

Esta guía explica cómo usar el sistema de lanzamiento de experimentos en GCP de forma robusta y eficiente.

## Índice

1. [Quick Start](#quick-start)
2. [Template de Lanzamiento](#template-de-lanzamiento)
3. [Problemas Comunes y Soluciones](#problemas-comunes-y-soluciones)
4. [Best Practices](#best-practices)
5. [Ejemplos de Configuraciones](#ejemplos-de-configuraciones)

---

## Quick Start

### Prerequisitos

```bash
# 1. Autenticar con GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Configurar OpenAI API key
export OPENAI_API_KEY='sk-...'

# 3. Verificar SSH config limpio
# Asegurarse de que ~/.ssh/config no tenga entradas con placeholders (<PUERTO>, etc.)
```

### Lanzar Experimento Existente

```bash
cd phase_d
./launch_phased_4variants.sh
```

### Crear Nuevo Experimento

```bash
# 1. Copiar template
cp gcp/launch_experiment_template.sh my_experiment/launch_my_exp.sh

# 2. Editar configuración
nano my_experiment/launch_my_exp.sh

# 3. Lanzar
cd my_experiment
./launch_my_exp.sh
```

---

## Template de Lanzamiento

El template `gcp/launch_experiment_template.sh` es un script genérico que maneja todo el ciclo de vida de un experimento en GCP.

### Secciones del Template

#### 1. Configuración del Experimento

```bash
# Nombre del experimento
EXPERIMENT_NAME="my-experiment"

# Seeds a ejecutar en cada VM
SEEDS=(42 100 123)

# Número de VMs
NUM_VMS=4
```

#### 2. Archivos a Subir

```bash
FILES_TO_UPLOAD=(
    "core/runner_phase2.py"
    "core/module1.py"
    "core/module2.py"
    "MBTI_500.csv"
)
```

**Importante:** Las rutas son **relativas al directorio raíz del proyecto**, no al directorio donde está el script.

#### 3. Dependencias Python

```bash
PYTHON_DEPS="numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv"
```

Se instalan con `sudo pip3 install` (global, no venv) en las Deep Learning VMs.

#### 4. Argumentos Base

```bash
BASE_ARGS="--data-path MBTI_500.csv \
    --test-size 0.2 \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 256 \
    --llm-model gpt-4o-mini"
```

Argumentos compartidos por todas las VMs.

#### 5. Configuración por VM

```bash
# Argumentos específicos por VM
declare -A VM_ARGS
VM_ARGS[1]="--use-feature-a"
VM_ARGS[2]="--use-feature-b"
VM_ARGS[3]="--use-feature-c"
VM_ARGS[4]="--use-feature-a --use-feature-b --use-feature-c"

# Etiquetas descriptivas
declare -A VM_LABELS
VM_LABELS[1]="Feature A Only"
VM_LABELS[2]="Feature B Only"
VM_LABELS[3]="Feature C Only"
VM_LABELS[4]="All Features"

# Prefijos de archivos de salida
declare -A VM_PREFIXES
VM_PREFIXES[1]="exp_a"
VM_PREFIXES[2]="exp_b"
VM_PREFIXES[3]="exp_c"
VM_PREFIXES[4]="exp_d"
```

Esto permite configurar cada VM con diferentes parámetros.

---

## Problemas Comunes y Soluciones

### 1. Error: `/usr/bin/scp: stat local "core/runner_phase2.py": No such file or directory`

**Causa:** El script está intentando subir archivos con rutas relativas desde un directorio incorrecto.

**Solución:** El template usa `$PROJECT_ROOT` para construir rutas absolutas:

```bash
# Detección automática del project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Subida con ruta absoluta
gcloud compute scp "$PROJECT_ROOT/$file" "$vm":~/ --zone="$GCP_ZONE"
```

### 2. Error: `/home/user/.ssh/config line X: Bad port '<PUERTO>'`

**Causa:** Archivo `~/.ssh/config` tiene entradas con placeholders no reemplazados.

**Solución:**

```bash
# Opción 1: Comentar la entrada problemática
nano ~/.ssh/config
# Añadir # al inicio de las líneas problemáticas

# Opción 2: Eliminar el archivo SSH config (si no lo necesitas)
mv ~/.ssh/config ~/.ssh/config.bak
```

### 3. Batch Scripts No Se Ejecutan

**Causa:** El script no tiene permisos de ejecución o la API key no se pasó correctamente.

**Solución en el template:**

```bash
# Asegurar permisos de ejecución
chmod +x "/tmp/run_${vm_prefix}_seeds.sh"

# Pasar API key en el heredoc (no como variable de shell)
cat > "/tmp/run_${vm_prefix}_seeds.sh" << EOFBATCH
export OPENAI_API_KEY='${OPENAI_API_KEY}'  # Se expande aquí
EOFBATCH
```

### 4. VMs No Se Apagan Automáticamente

**Verificar:**

```bash
# En la VM, verificar que el proceso de shutdown esté corriendo
gcloud compute ssh vm-name --zone=us-central1-a --command="ps aux | grep shutdown"
```

**El template usa:**

```bash
# Shutdown condicional
if [ $FAILED_SEEDS -eq ${#SEEDS[@]} ]; then
    sudo shutdown -h now  # Inmediato si todo falló
else
    sudo shutdown -h +2   # 2 min delay si hubo éxito
fi
```

### 5. Imports de Módulos Fallan en la VM

**Causa:** Módulos no se subieron o están en subdirectorio incorrecto.

**Solución:**

```bash
# Verificar en la VM
gcloud compute ssh vm-name --zone=us-central1-a --command="ls -la ~/"

# Deben estar en ~/
# runner_phase2.py
# module1.py
# module2.py
# MBTI_500.csv
```

Si los módulos están en subdirectorios (ej: `core/`), el runner debe importarlos así:

```python
# En runner_phase2.py
try:
    from module1 import func
except ImportError:
    # El módulo está en el mismo directorio en la VM
    import sys
    sys.path.insert(0, '.')
    from module1 import func
```

---

## Best Practices

### 1. Nombres de Archivos Consistentes

```bash
# Usar prefijos claros
VM_PREFIXES[1]="exp_featureA"
VM_PREFIXES[2]="exp_featureB"

# Resultará en:
# exp_featureA_seed42_metrics.json
# exp_featureA_seed42_synthetic.csv
# exp_featureA_seed42.log
```

### 2. Verificar Estado Antes de Lanzar

```bash
# Verificar VMs existentes
gcloud compute instances list --filter='name~exp-'

# Si existen, eliminarlas
gcloud compute instances delete vm-exp-1 vm-exp-2 --zone=us-central1-a
```

### 3. Monitoreo Durante Ejecución

```bash
# Verificar que los procesos estén corriendo
for vm in vm-exp-{1..4}; do
    echo "=== $vm ==="
    gcloud compute ssh "$vm" --zone=us-central1-a \
      --command="ps aux | grep python3 | grep -v grep | head -2"
done

# Ver logs en tiempo real
gcloud compute ssh vm-exp-1 --zone=us-central1-a \
  --command="tail -f exp_a_seed42.log"
```

### 4. Manejo de Costos

```bash
# Detener VMs manualmente si algo sale mal
gcloud compute instances stop vm-exp-{1..4} --zone=us-central1-a

# Verificar facturación actual
gcloud compute instances list --format="table(name,zone,status,machineType)"

# Eliminar VMs cuando terminen
gcloud compute instances delete vm-exp-{1..4} --zone=us-central1-a --quiet
```

### 5. Backup de Resultados

```bash
# Descargar resultados antes de eliminar VMs
mkdir -p results/my_experiment
for vm in vm-exp-{1..4}; do
    variant=$(echo $vm | grep -oP 'exp-\K\d+')
    gcloud compute scp "$vm":~/exp_*_metrics.json results/my_experiment/ \
      --zone=us-central1-a 2>/dev/null
done
```

---

## Ejemplos de Configuraciones

### Ejemplo 1: Probar Diferentes Modelos LLM

```bash
EXPERIMENT_NAME="llm-comparison"
NUM_VMS=4
SEEDS=(42 100 123)

BASE_ARGS="--data-path MBTI_500.csv --device cuda"

# Diferentes modelos por VM
VM_ARGS[1]="--llm-model gpt-4o-mini"
VM_ARGS[2]="--llm-model gpt-4o"
VM_ARGS[3]="--llm-model gpt-5-mini-2025-08-07"
VM_ARGS[4]="--llm-model claude-3-5-sonnet"

VM_LABELS[1]="GPT-4o-mini"
VM_LABELS[2]="GPT-4o"
VM_LABELS[3]="GPT-5-mini"
VM_LABELS[4]="Claude 3.5 Sonnet"

VM_PREFIXES[1]="llm_gpt4omini"
VM_PREFIXES[2]="llm_gpt4o"
VM_PREFIXES[3]="llm_gpt5mini"
VM_PREFIXES[4]="llm_claude35"
```

### Ejemplo 2: Diferentes Configuraciones de SMOTE

```bash
EXPERIMENT_NAME="smote-config"
NUM_VMS=5
SEEDS=(42)  # Un solo seed para comparar rápido

BASE_ARGS="--data-path MBTI_500.csv --llm-model gpt-4o-mini"

# Diferentes parámetros de SMOTE
VM_ARGS[1]="--max-clusters 2 --prompts-per-cluster 3"
VM_ARGS[2]="--max-clusters 3 --prompts-per-cluster 5"
VM_ARGS[3]="--max-clusters 5 --prompts-per-cluster 3"
VM_ARGS[4]="--max-clusters 3 --prompts-per-cluster 3 --temperature 1.5"
VM_ARGS[5]="--max-clusters 3 --prompts-per-cluster 3 --use-ensemble-selection"

VM_LABELS[1]="2 clusters, 3 prompts"
VM_LABELS[2]="3 clusters, 5 prompts"
VM_LABELS[3]="5 clusters, 3 prompts"
VM_LABELS[4]="Higher temperature"
VM_LABELS[5]="With ensemble"
```

### Ejemplo 3: Combinaciones de Features (Grid Search)

```bash
EXPERIMENT_NAME="feature-grid"
NUM_VMS=8  # 2^3 combinaciones
SEEDS=(42 100)

BASE_ARGS="--data-path MBTI_500.csv --llm-model gpt-4o-mini"

# Grid: feature_a × feature_b × feature_c
VM_ARGS[1]=""  # Baseline
VM_ARGS[2]="--use-feature-a"
VM_ARGS[3]="--use-feature-b"
VM_ARGS[4]="--use-feature-c"
VM_ARGS[5]="--use-feature-a --use-feature-b"
VM_ARGS[6]="--use-feature-a --use-feature-c"
VM_ARGS[7]="--use-feature-b --use-feature-c"
VM_ARGS[8]="--use-feature-a --use-feature-b --use-feature-c"

VM_LABELS[1]="Baseline"
VM_LABELS[2]="A only"
VM_LABELS[3]="B only"
VM_LABELS[4]="C only"
VM_LABELS[5]="A+B"
VM_LABELS[6]="A+C"
VM_LABELS[7]="B+C"
VM_LABELS[8]="A+B+C (Full)"
```

---

## Estructura de Archivos Recomendada

```
project_root/
├── core/
│   ├── runner_phase2.py          # Runner principal
│   ├── module1.py                # Módulos auxiliares
│   └── module2.py
├── gcp/
│   ├── gcp_toolkit.sh            # Funciones reutilizables
│   ├── launch_experiment_template.sh  # Template genérico
│   └── EXPERIMENT_GUIDE.md       # Esta guía
├── my_experiment/
│   ├── launch_my_exp.sh          # Script adaptado del template
│   ├── collect_results.sh        # Script de recolección
│   ├── monitor.sh                # Script de monitoreo (opcional)
│   └── results/                  # Resultados descargados
│       └── exp_variant_seedX_metrics.json
└── MBTI_500.csv                  # Dataset
```

---

## Troubleshooting Checklist

Antes de lanzar un experimento, verificar:

- [ ] `$OPENAI_API_KEY` está configurada
- [ ] `~/.ssh/config` no tiene placeholders inválidos
- [ ] Autenticado con GCP (`gcloud auth list`)
- [ ] Proyecto GCP configurado (`gcloud config get-value project`)
- [ ] No hay VMs existentes con el mismo nombre
- [ ] Archivos en `FILES_TO_UPLOAD` existen en el project root
- [ ] `BASE_ARGS` y `VM_ARGS` son válidos para el runner
- [ ] Seeds están definidos correctamente

Durante la ejecución:

- [ ] VMs se crearon exitosamente
- [ ] Archivos se subieron sin errores
- [ ] Dependencias se instalaron (verificar con `nvidia-smi`)
- [ ] Batch scripts se están ejecutando (`ps aux | grep python3`)
- [ ] Logs se están generando (`tail -f exp_*_seed*.log`)

Después de completar:

- [ ] Todos los archivos de métricas existen
- [ ] VMs se apagaron automáticamente
- [ ] Resultados fueron descargados
- [ ] VMs fueron eliminadas (o verificar costo continuo)

---

## Contacto y Soporte

Si encuentras problemas no cubiertos en esta guía:

1. Verificar logs en la VM: `gcloud compute ssh vm-name --command="tail -100 run.log"`
2. Revisar serial port output: `gcloud compute instances get-serial-port-output vm-name`
3. Consultar documentación de GCP: https://cloud.google.com/compute/docs

---

## Changelog

- **2025-11-25**: Guía inicial basada en lanzamiento exitoso de Phase D
  - Template genérico de lanzamiento
  - Documentación de problemas comunes
  - Ejemplos de configuraciones
