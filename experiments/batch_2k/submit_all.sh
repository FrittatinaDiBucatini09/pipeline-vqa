#!/bin/bash
# Submit all 5 experiments with SLURM job dependencies.
# Each experiment waits for the previous one to finish (any exit code).
# This ensures sequential execution on the single GPU.
#
# Generated: 20260225_034316
# Dataset:   mimic_ext_stratified_2000_samples.csv

set -e
BATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[01] Submitting: BBox + MedGemma 1.5 4B..."
JOB_01=$(sbatch --parsable "$BATCH_DIR/exp_01_bbox_medgemma15/meta_job.sh")
echo "    Job ID: $JOB_01"

echo "[02] Submitting: BBox + MedGemma 4B (after job $JOB_01)..."
JOB_02=$(sbatch --parsable --dependency=afterany:$JOB_01 "$BATCH_DIR/exp_02_bbox_medgemma4b/meta_job.sh")
echo "    Job ID: $JOB_02"

echo "[03] Submitting: BBox + OctoMed 7B (after job $JOB_02)..."
JOB_03=$(sbatch --parsable --dependency=afterany:$JOB_02 "$BATCH_DIR/exp_03_bbox_octomed/meta_job.sh")
echo "    Job ID: $JOB_03"

echo "[04] Submitting: Segmentation + MedGemma 1.5 4B (after job $JOB_03)..."
JOB_04=$(sbatch --parsable --dependency=afterany:$JOB_03 "$BATCH_DIR/exp_04_seg_medgemma15/meta_job.sh")
echo "    Job ID: $JOB_04"

echo "[05] Submitting: Attention Map + OctoMed 7B (after job $JOB_04)..."
JOB_05=$(sbatch --parsable --dependency=afterany:$JOB_04 "$BATCH_DIR/exp_05_attn_octomed/meta_job.sh")
echo "    Job ID: $JOB_05"

echo ""
echo "============================================================"
echo "All 5 experiments submitted with sequential dependencies."
echo "Monitor with: squeue -u $USER"
echo "============================================================"
