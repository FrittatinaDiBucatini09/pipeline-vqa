#!/bin/bash
# run_all_prep.sh
# Execute all data preparation scripts in the correct order.
# Forwards any arguments (e.g., --limit 100) to the python scripts.

set -e

ARGS="$@"
echo "🚀 Starting Data Prep Pipeline"
echo "   Arguments: $ARGS"
echo "========================================================"

# 1. MIMIC-Ext (Required for Stratified Sampling)
echo ""
echo "[1/4] Processing MIMIC-Ext..."
python3 prepare_mimic_ext.py $ARGS

# 2. GEMeX (Independent)
echo ""
echo "[2/4] Processing GEMeX..."
python3 prepare_gemex.py $ARGS

# 3. GEMeX-VQA (Independent)
echo ""
echo "[3/4] Processing GEMeX-VQA..."
python3 prepare_gemex_vqa.py $ARGS

# 4. Stratified Sampling (Depends on MIMIC-Ext output)
echo ""
echo "[4/4] Generating Stratified Samples..."
python3 create_stratified_samples.py $ARGS

echo ""
echo "========================================================"
echo "✅ All Data Preparation tasks completed successfully."
