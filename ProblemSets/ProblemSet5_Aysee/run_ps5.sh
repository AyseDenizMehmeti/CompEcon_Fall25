#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/Scripts/python" ]; then
  PYEXEC=".venv/Scripts/python"
elif [ -x ".venv/bin/python" ]; then
  PYEXEC=".venv/bin/python"
else
  echo ".venv not found or no python inside .venv. Create the venv and install requirements first."
  exit 1
fi

echo "Using python: $PYEXEC"

echo "1) Running unit tests..."
$PYEXEC -m pytest -q

echo "2) Running analysis script..."
$PYEXEC ps5_models.py

echo "3) Compiling LaTeX (twice for references)..."
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode ProblemSet5_Ayse.tex > /dev/null
  pdflatex -interaction=nonstopmode ProblemSet5_Ayse.tex > /dev/null
  echo "✅ ProblemSet5_Ayse.pdf created"
else
  echo "⚠️ pdflatex not found — skipping LaTeX compile"
fi
