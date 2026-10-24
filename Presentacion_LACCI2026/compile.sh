#!/usr/bin/env bash
# Builds main.pdf (full deck) and main_simple.pdf (simple deck).
# Uses Tectonic when available (Windows setup), otherwise XeLaTeX.
# Usage: bash compile.sh            # both decks
#        bash compile.sh main       # only one of them (main | main_simple)
set -e
cd "$(dirname "$0")"

decks=("${@:-main main_simple}")
for deck in ${decks[@]}; do
  if command -v tectonic >/dev/null 2>&1; then
    tectonic -X compile "$deck.tex"
  else
    xelatex -interaction=nonstopmode -halt-on-error "$deck.tex"
    xelatex -interaction=nonstopmode -halt-on-error "$deck.tex"
  fi
  echo "Built: $(pwd)/$deck.pdf"
done
