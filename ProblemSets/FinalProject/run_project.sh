#!/bin/bash

# ============================================================
# run_project.sh 
# ============================================================

echo "=========================================="
echo "OLD NAVY PRICING ANALYSIS - FINAL PROJECT"
echo "=========================================="
echo "Author: Ayse Deniz"
echo "Course: ECON 833"
echo "=========================================="
echo ""

# -----------------------------------------------------------------
# 0. SETUP - Force fail on errors and show debug info
# -----------------------------------------------------------------
set -e  # Exit on any error
echo "Debug: Starting in directory: $(pwd)"
echo "Debug: Files here:"
ls -la
echo ""

# -----------------------------------------------------------------
# 1. CHECK DEPENDENCIES
# -----------------------------------------------------------------
echo "1. Checking dependencies..."
if ! command -v python &> /dev/null; then
    echo "   ✗ ERROR: Python not found!"; exit 1
else
    echo "   ✓ Python: $(python --version 2>&1)"
fi

if ! command -v pdflatex &> /dev/null; then
    echo "   ✗ ERROR: pdflatex not found!"; exit 1
else
    echo "   ✓ pdflatex: $(pdflatex --version | head -n1)"
fi

# -----------------------------------------------------------------
# 2. CHECK REQUIRED FILES EXIST
# -----------------------------------------------------------------
echo ""
echo "2. Checking required files..."

# Check ZIP or CSV
ZIP_FILE="oldnavy12052025daily.zip"
CSV_FILE="oldnavy12052025daily.csv"

if [ -f "$CSV_FILE" ]; then
    echo "   ✓ Found CSV file: $CSV_FILE"
elif [ -f "$ZIP_FILE" ]; then
    echo "   ✓ Found ZIP file: $ZIP_FILE"
    echo "   → Extracting CSV from ZIP..."
    unzip -o "$ZIP_FILE"
    
    # Verify extraction succeeded
    if [ -f "$CSV_FILE" ]; then
        echo "   ✓ Extracted CSV successfully"
    else
        echo "   ✗ ERROR: ZIP did not contain $CSV_FILE"
        exit 1
    fi
else
    echo "   ✗ ERROR: No data file found!"
    echo "   Expected: $CSV_FILE or $ZIP_FILE"
    exit 1
fi


# Check Python script
PYTHON_SCRIPT="FinalProject_Ayse.py"
if [ -f "$PYTHON_SCRIPT" ]; then
    echo "   ✓ Python script: $PYTHON_SCRIPT"
else
    echo "   ✗ ERROR: $PYTHON_SCRIPT not found!"; exit 1
fi

# Check LaTeX file
TEX_FILE=""
for tex in "FinalProject_Ayse.tex" "./FinalProject_Ayse.tex" "../FinalProject_Ayse.tex"; do
    if [ -f "$tex" ]; then
        TEX_FILE="$tex"
        break
    fi
done

if [ -z "$TEX_FILE" ]; then
    echo "   ✗ ERROR: FinalProject_Ayse.tex not found!"
    echo "   Searched in: $(pwd), $(pwd)/, ../"
    exit 1
else
    echo "   ✓ LaTeX file: $TEX_FILE (found at: $(realpath "$TEX_FILE" 2>/dev/null || echo "$TEX_FILE"))"
fi

# -----------------------------------------------------------------
# 3. CREATE DIRECTORIES
# -----------------------------------------------------------------
echo ""
echo "3. Setting up directories..."
mkdir -p Figures
echo "   ✓ Created/verified: Figures/"

# -----------------------------------------------------------------
# 4. RUN PYTHON ANALYSIS
# -----------------------------------------------------------------
echo ""
echo "4. Running Python analysis..."

# Check 
FIGURES_EXIST=1
for fig in discount_distribution_zoom.png gender_promo.png clearance_by_category.png \
           price_inventory_path_example.png median_product_lifecycle.png \
           confusion_matrix.png policy_feature_importance.png; do
    if [ ! -f "Figures/$fig" ]; then
        FIGURES_EXIST=0
        break
    fi
done

if [ $FIGURES_EXIST -eq 1 ]; then
    echo "   ✓ Figures already exist, skipping Python"
else
    echo "   Running Python to generate figures..."
    python "$PYTHON_SCRIPT"
    
    # Verify figures were created
    FIG_COUNT=$(ls Figures/*.png 2>/dev/null | wc -l)
    if [ $FIG_COUNT -ge 7 ]; then
        echo "   ✓ Created $FIG_COUNT figures"
    else
        echo "   ⚠ Only $FIG_COUNT figures created (expected 7)"
    fi
fi

# -----------------------------------------------------------------
# 5. COMPILE LATEX DOCUMENT
# -----------------------------------------------------------------
echo ""
echo "5. Compiling LaTeX document..."

# Get base name for output
BASE_NAME="${TEX_FILE%.*}"
PDF_FILE="${BASE_NAME}.pdf"

echo "   First compilation..."
pdflatex -interaction=nonstopmode -shell-escape "$TEX_FILE" > latex1.log 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ First compilation successful"
else
    echo "   ⚠ First compilation had issues"
    echo "   Check latex1.log"
fi

echo "   Second compilation (for references)..."
pdflatex -interaction=nonstopmode -shell-escape "$TEX_FILE" > latex2.log 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Second compilation successful"
else
    echo "   ⚠ Second compilation had issues"
fi

# -----------------------------------------------------------------
# 6. FINAL CHECK
# -----------------------------------------------------------------
echo ""
echo "6. Final check..."

if [ -f "$PDF_FILE" ]; then
    PDF_SIZE=$(du -h "$PDF_FILE" 2>/dev/null | cut -f1 || echo "unknown")
    echo "   ✓ PDF created: $PDF_FILE ($PDF_SIZE)"
    
    # Try to get page count
    if command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo "$PDF_FILE" 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
        echo "   Pages: $PAGES"
    fi
else
    echo "   ✗ ERROR: PDF was not created!"
    echo "   Check latex1.log and latex2.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "SUCCESS! Project completed."
echo "=========================================="