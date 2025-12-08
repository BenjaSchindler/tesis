#!/bin/bash

# UAI HPC - Helper Script
# Script de ayuda para conectarse y transferir archivos al cluster HPC de la UAI

# Colores para mejor visualización
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           UAI HPC - Helper Script                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo

# Verificar si la configuración SSH está completa
if grep -q "<IP_PUBLICA>" ~/.ssh/config; then
    echo -e "${RED}⚠️  ADVERTENCIA: La configuración SSH aún no está completa${NC}"
    echo -e "${YELLOW}Edita el archivo ~/.ssh/config y reemplaza:${NC}"
    echo "  - <IP_PUBLICA> con la IP del servidor"
    echo "  - <PUERTO> con el puerto SSH"
    echo "  - <TU_USUARIO> con tu nombre de usuario"
    echo
    echo -e "${YELLOW}Puedes editarlo con:${NC} nano ~/.ssh/config"
    echo
fi

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}1. CONEXIÓN AL CLUSTER${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Conexión básica:${NC}"
echo "  ssh uai-hpc"
echo
echo -e "${YELLOW}Conexión con comando directo:${NC}"
echo "  ssh uai-hpc 'comando_a_ejecutar'"
echo

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}2. TRANSFERENCIA DE ARCHIVOS${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Subir un archivo:${NC}"
echo "  scp archivo.txt uai-hpc:~/"
echo "  scp archivo.txt uai-hpc:~/ruta/destino/"
echo
echo -e "${YELLOW}Subir una carpeta completa:${NC}"
echo "  scp -r carpeta/ uai-hpc:~/"
echo
echo -e "${YELLOW}Descargar un archivo:${NC}"
echo "  scp uai-hpc:~/archivo.txt ."
echo
echo -e "${YELLOW}Descargar una carpeta:${NC}"
echo "  scp -r uai-hpc:~/carpeta/ ."
echo

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}3. RSYNC (Para archivos grandes o sincronización)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Subir archivos con rsync:${NC}"
echo "  rsync -avzP carpeta/ uai-hpc:~/carpeta/"
echo
echo -e "${YELLOW}Descargar archivos con rsync:${NC}"
echo "  rsync -avzP uai-hpc:~/carpeta/ ./carpeta_local/"
echo
echo -e "${YELLOW}Sincronizar (solo cambios):${NC}"
echo "  rsync -avzP --delete carpeta/ uai-hpc:~/carpeta/"
echo

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}4. EJEMPLOS ESPECÍFICOS PARA SMOTE-LLM${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Subir proyecto completo:${NC}"
echo "  rsync -avzP --exclude='*.pyc' --exclude='__pycache__' \\"
echo "        --exclude='.git' --exclude='results/' \\"
echo "        /home/benja/Desktop/Tesis/SMOTE-LLM/ uai-hpc:~/SMOTE-LLM/"
echo
echo -e "${YELLOW}Subir solo archivos de Phase C:${NC}"
echo "  rsync -avzP phase_c/ uai-hpc:~/SMOTE-LLM/phase_c/"
echo
echo -e "${YELLOW}Descargar resultados:${NC}"
echo "  rsync -avzP uai-hpc:~/SMOTE-LLM/results/ ./results_hpc/"
echo
echo -e "${YELLOW}Ejecutar script remoto:${NC}"
echo "  ssh uai-hpc 'cd SMOTE-LLM && bash launch_script.sh'"
echo

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}5. VERIFICACIÓN DE CONEXIÓN${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Ver tu llave pública (para enviar a administradores):${NC}"
echo "  cat ~/.ssh/id_ed25519_uai_hpc.pub"
echo
echo -e "${YELLOW}Verificar configuración SSH:${NC}"
echo "  cat ~/.ssh/config | grep -A 7 'Host uai-hpc'"
echo
echo -e "${YELLOW}Probar conexión:${NC}"
echo "  ssh uai-hpc 'echo Conexion exitosa!'"
echo

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}6. TIPS Y BUENAS PRÁCTICAS${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo
echo "  • Usa rsync en lugar de scp para archivos grandes"
echo "  • Añade -P a rsync para ver progreso y poder reanudar"
echo "  • Usa --dry-run con rsync para ver qué se transferirá"
echo "  • Mantén backups de tu llave privada en lugar seguro"
echo "  • Usa screen o tmux para sesiones persistentes en el HPC"
echo

echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo
