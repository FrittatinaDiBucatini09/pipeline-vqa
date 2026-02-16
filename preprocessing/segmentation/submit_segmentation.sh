#!/bin/bash
# ==============================================================================
# SCRIPT: submit_segmentation.sh
# PURPOSE: Slurm Entry Point for the Segmentation Pipeline
# CLUSTER: UniboNLP (Slurm Managed)
# ==============================================================================
# DESCRIPTION:
#   This script is the "Trigger" in the 3-script protocol. It requests resources
#   from Slurm and then launches the Docker wrapper.
#
# PREREQUISITES:
#   1. Build the image first on the Master Node (Faretra):
#      $ ./scripts/build_image.sh
#   2. Ensure 'scripts/run_docker.sh' is executable:
#      $ chmod +x scripts/run_docker.sh
#
# USAGE:
#   $ sbatch submit_segmentation.sh
#
# MONITORING:
#   - Check status: squeue
#   - Check logs:   tail -f slurm_seg_<job_id>.out
#   - Cancel job:   scancel <job_id>
# ==============================================================================

# --- SLURM DIRECTIVES ---------------------------------------------------------

# Job Name: Appears in 'squeue' output
#SBATCH --job-name=gemex_seg

# Output Logs: Standard Output and Standard Error
# %j is replaced by the Job ID automatically
#SBATCH --output=slurm_seg_%j.out
#SBATCH --error=slurm_seg_%j.err

# Node Count: Run on a single node (required for non-MPI jobs)
#SBATCH -N 1

# GPU Request: Request 1x NVIDIA RTX 3090 (24GB VRAM)

#SBATCH --gpus=nvidia_geforce_rtx_3090:1

# Time Limit: Kill job after 4 hours to allow queue rotation
#SBATCH --time=4:00:00

# Node Constraint (Data Locality):
# CRITICAL: We force execution on 'faretra' (Node 40) specifically.
# REASON: The cluster does NOT have a distributed file system. The MIMIC-CXR
#         dataset and our code/results exist locally on Node 40. If Slurm
#         assigns this job to Node 153 (deeplearn2), it will crash because
#         the files are missing there.
#SBATCH -w faretra

# --- PIPELINE CONFIGURATION ---------------------------------------------------

# Execution Mode:
# Options: '1' (Localization Only), '2' (Segmentation Only), 'all' (End-to-End)
# CAUTION: If running '2' separately, ensure 'predictions.jsonl' exists
#          in results/step1_bboxes/ from a previous run on this SAME node.
TARGET_MODE="all"

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi

# Config Files (Container Paths):
# These paths must exist INSIDE the container (mapped to /workspace/configs).
CONF_STEP1="/workspace/configs/step1/gemex/exp_01_vqa.conf"
CONF_STEP2="/workspace/configs/step2/sam_exp03_medsam3.conf"

# --- EXECUTION HANDOFF --------------------------------------------------------

echo "----------------------------------------------------------------"
echo "Job Submitted. ID: $SLURM_JOB_ID"
echo "Node Allocated:    $SLURMD_NODENAME"
echo "Target Mode:       $TARGET_MODE"
echo "----------------------------------------------------------------"

# Launch the Docker Wrapper
# We use 'bash' explicitly to ensure the script runs even if execute bit is missing.
# The wrapper handles the complex volume mounting and GPU visibility flags.
# Added LIMIT argument (can be set via env var, e.g., sbatch --export=ALL,LIMIT=10 ...)
LIMIT="${LIMIT:-}"
bash scripts/run_docker.sh "$TARGET_MODE" "$CONF_STEP1" "$CONF_STEP2" "$LIMIT"