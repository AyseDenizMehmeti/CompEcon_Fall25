#!/usr/bin/env bash
# run.sh -- run PS8 analysis and compile writeup
set -euo pipefail

# Move to script directory
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "Working directory: $HERE"

##############################
# 1) Run Python analysis
##############################
echo ">>> Running ps8_Ayse.py ..."
python -u ps8_Ayse.py --full


##############################
# 2) Compile LaTeX
##############################
TEXFILE="ProblemSet8_Ayse.tex"  
OUTPDF="${TEXFILE%.tex}.pdf"

if [ ! -f "$TEXFILE" ]; then
  echo "ERROR: $TEXFILE not found!"
  exit 2
fi

echo ">>> Compiling LaTeX -> PDF ..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"

echo ">>> Done."
echo "PDF generated: $OUTPDF"
