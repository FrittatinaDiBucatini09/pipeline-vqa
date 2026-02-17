"""
SLURM sbatch template generation for the Pipeline Orchestrator.

Provides:
  - generate_judge_inline(): Docker command block for the VQA Judge stage
    (used inside the meta-job script).
  - generate_meta_job_sbatch(): Complete sbatch script that runs all selected
    pipeline stages sequentially in a single SLURM allocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def generate_judge_inline(config_file: str, vqa_dir: str) -> str:
    """Generate the inline bash block for running the VQA Judge inside a meta-job.

    This is NOT a standalone sbatch script — it's a snippet meant to be embedded
    in the meta-job. The surrounding meta-job already has GPU allocation, env vars,
    and error handling.

    Args:
        config_file: Config file path relative to vqa/ directory
                     (e.g. "configs/judge/hard_coded_judge.conf").
        vqa_dir: Absolute path to the vqa/ directory.

    Returns:
        Bash snippet as a string.
    """
    config_arg = f'"{config_file}"' if config_file else '""'

    return f"""cd "{vqa_dir}" || exit 1
PHYS_DIR=$(pwd)

# Load HF_TOKEN from .env if not already set
if [ -z "$HF_TOKEN" ] && [ -f "$PHYS_DIR/.env" ]; then
    echo "  Loading secrets from .env..."
    set -a
    source "$PHYS_DIR/.env"
    set +a
fi

IMAGE_NAME="med_vqa_project:3090"

# Verify Docker image exists
if [[ -z "$(docker images -q "$IMAGE_NAME" 2>/dev/null)" ]]; then
    echo "[ERROR] Docker image '$IMAGE_NAME' not found. Build it first."
    exit 1
fi

echo "  Launching VQA Judge container..."
echo "  Config: {config_file or '(default)'}"

docker run --rm \\
    --name "metajob_${{SLURM_JOB_ID:-local}}_judge" \\
    --gpus "device=${{CUDA_VISIBLE_DEVICES:-0}}" \\
    --shm-size=16g \\
    -v "$PHYS_DIR":/workspace \\
    -v /llms:/llms \\
    -e HF_HOME=/llms \\
    -e HF_TOKEN="${{HF_TOKEN:-}}" \\
    "$IMAGE_NAME" \\
    /bin/bash /workspace/scripts/run_judge.sh {config_arg}"""


# Maps each preprocessing stage key to the sub-path (relative to script_dir)
# where vqa_manifest.csv and the generated images live.
_PREPROCESSING_OUTPUT_PATHS: Dict[str, str] = {
    "bbox_preproc": "results",
    "attn_map": "results",
    "segmentation": "results/step2_masks",
}

# Set of stage keys that are considered preprocessing stages.
_PREPROCESSING_STAGE_KEYS = set(_PREPROCESSING_OUTPUT_PATHS.keys())


def _generate_routing_to_preprocessing_bridge(script_dir: str) -> str:
    """Generate bash snippet that bridges routing output to preprocessing input.

    Exports ROUTED_DATASET_OVERRIDE so subsequent preprocessing stages
    consume the expanded queries JSONL produced by medclip_routing.

    Args:
        script_dir: Absolute path to the medclip_routing script directory.
    """
    output_dir = f"{script_dir}/results"

    return f"""
# ==============================================================================
# BRIDGE: Routing (medclip_routing) -> Preprocessing
# ==============================================================================
# The routing middleware generates expanded_queries.jsonl with enriched queries.
# This bridge exports ROUTED_DATASET_OVERRIDE so downstream preprocessing
# stages consume the expanded dataset.

ROUTING_OUTPUT_DIR="{output_dir}"

echo "[BRIDGE] Stage: medclip_routing -> preprocessing"
echo "[BRIDGE] Checking for expanded queries at $ROUTING_OUTPUT_DIR/expanded_queries.jsonl"

if [ ! -f "$ROUTING_OUTPUT_DIR/expanded_queries.jsonl" ]; then
    echo "[ERROR] expanded_queries.jsonl not found at $ROUTING_OUTPUT_DIR"
    echo "[ERROR] The routing stage may have failed to generate the output file."
    exit 1
fi

ROUTING_ROWS=$(wc -l < "$ROUTING_OUTPUT_DIR/expanded_queries.jsonl")
echo "[BRIDGE] Expanded queries found: $ROUTING_ROWS rows"

export ROUTED_DATASET_OVERRIDE="$ROUTING_OUTPUT_DIR/expanded_queries.jsonl"
echo "[BRIDGE] ROUTED_DATASET_OVERRIDE=$ROUTED_DATASET_OVERRIDE"
"""


def _generate_preprocessing_to_vqa_bridge(
    stage_key: str, script_dir: str
) -> str:
    """Generate bash snippet that bridges any preprocessing output to VQA input.

    Sets DATA_FILE_OVERRIDE and VQA_IMAGE_PATH so the VQA stage reads
    the preprocessed images and VQA manifest CSV.

    Args:
        stage_key: The preprocessing stage key (e.g., "bbox_preproc").
        script_dir: Absolute path to the preprocessing script directory.
    """
    output_subpath = _PREPROCESSING_OUTPUT_PATHS.get(stage_key, "results")
    output_dir = f"{script_dir}/{output_subpath}"

    return f"""
# ==============================================================================
# BRIDGE: Preprocessing ({stage_key}) -> VQA Generation
# ==============================================================================
# The preprocessing stage generates:
#   1. Modified/annotated images
#   2. vqa_manifest.csv mapping new image paths to original questions
# This bridge wires those artifacts into the VQA generation stage.

PREPROC_OUTPUT_DIR="{output_dir}"

echo "[BRIDGE] Stage: {stage_key}"
echo "[BRIDGE] Checking for VQA manifest at $PREPROC_OUTPUT_DIR/vqa_manifest.csv"

if [ ! -f "$PREPROC_OUTPUT_DIR/vqa_manifest.csv" ]; then
    echo "[ERROR] VQA manifest not found at $PREPROC_OUTPUT_DIR/vqa_manifest.csv"
    echo "[ERROR] The preprocessing stage may have failed to generate the bridge file."
    exit 1
fi

MANIFEST_ROWS=$(wc -l < "$PREPROC_OUTPUT_DIR/vqa_manifest.csv")
echo "[BRIDGE] VQA manifest found: $((MANIFEST_ROWS - 1)) image-question pairs"

export DATA_FILE_OVERRIDE="$PREPROC_OUTPUT_DIR/vqa_manifest.csv"
export VQA_IMAGE_PATH="$PREPROC_OUTPUT_DIR"

echo "[BRIDGE] DATA_FILE_OVERRIDE=$DATA_FILE_OVERRIDE"
echo "[BRIDGE] VQA_IMAGE_PATH=$VQA_IMAGE_PATH"
"""


def generate_meta_job_sbatch(
    stage_commands: List[Dict],
    run_dir: str,
    dataset_override: Optional[str] = None,
    stage_keys: Optional[List[str]] = None,
    job_name: str = "vqa_pipeline",
    time_limit: str = "15:00:00",
) -> str:
    """Generate a complete sbatch script that runs all pipeline stages sequentially.

    Args:
        stage_commands: List of dicts, each with:
            - name (str): Human-readable stage name
            - command (str): Bash command(s) to execute for this stage
        run_dir: Absolute path to the run directory (for SLURM output files).
        dataset_override: Optional CSV filename to override across all stages.
        stage_keys: Optional list of stage keys (e.g., ["bbox_preproc", "vqa_gen"])
                    used to detect inter-stage bridges.
        job_name: SLURM job name.
        time_limit: SLURM wall-clock limit (HH:MM:SS).

    Returns:
        Full sbatch script content as a string.
    """
    # --- Header ---
    lines = [
        "#!/bin/bash",
        "# ==============================================================================",
        "# AUTO-GENERATED META-JOB: Medical VQA Pipeline",
        "# Generated by the Pipeline Orchestrator",
        "# ==============================================================================",
        "",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={run_dir}/slurm_metajob_%j.out",
        f"#SBATCH --error={run_dir}/slurm_metajob_%j.err",
        "#SBATCH -N 1",
        "#SBATCH --gpus=nvidia_geforce_rtx_3090:1",
        "#SBATCH -w faretra",
        f"#SBATCH --time={time_limit}",
        "# Send SIGTERM 60s before SIGKILL on timeout, so trap can clean up",
        "#SBATCH --signal=B:TERM@60",
        "",
        "# Fail-fast: exit immediately if any command fails",
        "set -e",
        "",
    ]

    # --- Global environment ---
    lines.extend([
        "# ==============================================================================",
        "# GLOBAL ENVIRONMENT",
        "# ==============================================================================",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f'export ORCH_OUTPUT_DIR="{run_dir}"',
    ])

    if dataset_override:
        lines.append(f'export DATA_FILE_OVERRIDE="{dataset_override}"')

    lines.append("")

    # --- Trap for cleanup/logging ---
    lines.extend([
        "# ==============================================================================",
        "# ERROR HANDLING & REPORTING",
        "# ==============================================================================",
        'CURRENT_STEP="initializing"',
        "PIPELINE_START=$(date '+%Y-%m-%d %H:%M:%S')",
        "",
        "# Snapshot running containers before pipeline starts",
        'PRE_CONTAINERS=$(docker ps -q 2>/dev/null | sort)',
        "",
        "cleanup() {",
        "    EXIT_CODE=$?",
        "    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')",
        "",
        "    # Kill any Docker containers started during this meta-job",
        '    POST_CONTAINERS=$(docker ps -q 2>/dev/null | sort)',
        '    NEW_CONTAINERS=$(comm -13 <(echo "$PRE_CONTAINERS") <(echo "$POST_CONTAINERS") 2>/dev/null)',
        '    if [ -n "$NEW_CONTAINERS" ]; then',
        '        echo "[$TIMESTAMP] Stopping orphan Docker containers..."',
        '        docker kill $NEW_CONTAINERS 2>/dev/null || true',
        '        sleep 2',
        "    fi",
        "",
        "    echo ''",
        "    echo '============================================================'",
        "    if [ $EXIT_CODE -ne 0 ]; then",
        '        echo "[$TIMESTAMP] PIPELINE FAILED at step: $CURRENT_STEP"',
        '        echo "Exit code: $EXIT_CODE"',
        '        echo "Started:   $PIPELINE_START"',
        '        echo "Failed:    $TIMESTAMP"',
        "    else",
        '        echo "[$TIMESTAMP] PIPELINE COMPLETED SUCCESSFULLY"',
        '        echo "Started:  $PIPELINE_START"',
        '        echo "Finished: $TIMESTAMP"',
        "    fi",
        "    echo '============================================================'",
        "}",
        "trap cleanup EXIT",
        "trap 'exit 143' TERM  # Ensure SIGTERM (from SLURM timeout) triggers EXIT trap",
        "",
        'echo "============================================================"',
        'echo "Medical VQA Pipeline - Meta-Job"',
        f'echo "Run directory: {run_dir}"',
        f'echo "Stages: {len(stage_commands)}"',
    ])

    if dataset_override:
        lines.append(f'echo "Dataset override: {dataset_override}"')

    lines.extend([
        'echo "============================================================"',
        "",
    ])

    # --- Stage commands ---
    keys = stage_keys or []

    for i, stage_cmd in enumerate(stage_commands, 1):
        name = stage_cmd["name"]
        command = stage_cmd["command"]

        lines.extend([
            "# ==============================================================================",
            f"# STEP {i}: {name}",
            "# ==============================================================================",
            f'CURRENT_STEP="{i} - {name}"',
            f'echo "[$(date \'+%Y-%m-%d %H:%M:%S\')] START: $CURRENT_STEP"',
            "",
            command,
            "",
            f'echo "[$(date \'+%Y-%m-%d %H:%M:%S\')] DONE:  $CURRENT_STEP"',
            "",
        ])

        # Inject bridge blocks for inter-stage data flow
        idx = i - 1  # 0-based index into keys

        # Bridge: medclip_routing -> any preprocessing stage
        if (idx < len(keys) - 1
                and keys[idx] == "medclip_routing"
                and keys[idx + 1] in _PREPROCESSING_STAGE_KEYS):
            script_dir = stage_cmd.get("script_dir", "")
            if script_dir:
                lines.append(
                    _generate_routing_to_preprocessing_bridge(script_dir)
                )

        # Bridge: preprocessing -> VQA generation
        if (idx < len(keys) - 1
                and keys[idx] in _PREPROCESSING_STAGE_KEYS
                and keys[idx + 1] == "vqa_gen"):
            script_dir = stage_cmd.get("script_dir", "")
            if script_dir:
                lines.append(
                    _generate_preprocessing_to_vqa_bridge(keys[idx], script_dir)
                )

    return "\n".join(lines)
