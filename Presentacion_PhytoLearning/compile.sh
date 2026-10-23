#!/usr/bin/env bash
# Compila la presentación de defensa con XeLaTeX (necesario para fontspec/Cantarell).
# Uso: bash compile.sh [clean]

set -e
cd "$(dirname "$0")"

if [[ "$1" == "clean" ]]; then
  rm -f main.aux main.log main.nav main.out main.snm main.toc main.vrb main.synctex.gz
  echo "Archivos auxiliares eliminados."
  exit 0
fi

# Dos pasadas para resolver referencias y \tableofcontents
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

echo ""
echo "==================================================="
echo "Compilación exitosa: $(pwd)/main.pdf"
pdfinfo main.pdf 2>/dev/null | grep -E "(Pages|File size)" || true
echo "==================================================="
