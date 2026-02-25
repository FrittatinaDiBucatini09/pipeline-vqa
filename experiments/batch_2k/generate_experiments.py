#!/usr/bin/env python3
"""
Batch Experiment Generator — 2K Dataset
=========================================
Generates 5 meta-job scripts for running experiments with different
preprocessing + VQA model combinations on the 2k stratified dataset.

Experiments:
    1. BBox      + MedGemma 1.5 4B + Judge
    2. BBox      + MedGemma 4B     + Judge   (cache hit — skips BBox)
    3. BBox      + OctoMed 7B      + Judge   (cache hit — skips BBox)
    4. Segmentation + MedGemma 1.5 4B + Judge
    5. Attn Map  + OctoMed 7B      + Judge

Usage:
    python3 experiments/batch_2k/generate_experiments.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure orchestrator module is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "orchestrator"))

from slurm_templates import generate_meta_job_sbatch, generate_judge_inline

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASET = "mimic_ext_stratified_2000_samples.csv"

# Absolute paths to stage script directories
BBOX_DIR = str(PROJECT_ROOT / "preprocessing/bounding_box")
ATTN_DIR = str(PROJECT_ROOT / "preprocessing/attention_map")
SEG_DIR = str(PROJECT_ROOT / "preprocessing/segmentation")
VQA_DIR = str(PROJECT_ROOT / "vqa")

# Preprocessing configs (relative to their script_dir)
BBOX_CONFIG = "configs/gemex/exp_01_vqa.conf"
ATTN_CONFIG = "configs/gemex/exp_01_vqa.conf"
SEG_STEP1_CONFIG = "configs/step1/gemex/exp_01_vqa.conf"
SEG_STEP2_CONFIG = "configs/step2/sam_exp01.conf"

# VQA configs (relative to vqa/ directory)
VQA_MEDGEMMA_15 = "configs/generation/medgemma_1_5.conf"
VQA_MEDGEMMA_4B = "configs/generation/medgemma_4b.conf"
VQA_OCTOMED_7B = "configs/generation/octomed_7b.conf"

# Judge config (relative to vqa/ directory)
JUDGE_CONFIG = "configs/judge/hard_coded_judge.conf"

# Time limits per experiment type
TIME_BBOX_VQA = "10:00:00"    # BBox is fast, VQA dominates
TIME_SEG_VQA = "12:00:00"     # Segmentation takes longer
TIME_ATTN_VQA = "10:00:00"    # Attention map similar to bbox


# ==============================================================================
# STAGE COMMAND BUILDERS
# ==============================================================================

def _bbox_command() -> dict:
    return {
        "name": "Preprocessing: Bounding Box",
        "command": f'cd "{BBOX_DIR}"\nbash submit_bbox_preprocessing.sh {BBOX_CONFIG}',
        "script_dir": BBOX_DIR,
        "config_file": BBOX_CONFIG,
    }


def _attn_command() -> dict:
    return {
        "name": "Preprocessing: Attention Map",
        "command": f'cd "{ATTN_DIR}"\nbash submit_heatmap_gen.sh {ATTN_CONFIG}',
        "script_dir": ATTN_DIR,
        "config_file": ATTN_CONFIG,
    }


def _seg_command() -> dict:
    return {
        "name": "Preprocessing: Segmentation",
        "command": (
            f'export TARGET_MODE="all"\n'
            f'cd "{SEG_DIR}"\n'
            f'bash submit_segmentation.sh {SEG_STEP1_CONFIG} {SEG_STEP2_CONFIG}'
        ),
        "script_dir": SEG_DIR,
        "config_file": f"{SEG_STEP1_CONFIG} {SEG_STEP2_CONFIG}",
    }


def _vqa_command(vqa_config: str) -> dict:
    return {
        "name": "VQA Generation",
        "command": f'cd "{VQA_DIR}"\nbash submit_generation.sh {vqa_config}',
        "script_dir": VQA_DIR,
        "config_file": vqa_config,
    }


def _judge_command() -> dict:
    return {
        "name": "VQA Evaluation (Judge)",
        "command": generate_judge_inline(
            config_file=JUDGE_CONFIG,
            vqa_dir=VQA_DIR,
        ),
        "script_dir": VQA_DIR,
        "config_file": JUDGE_CONFIG,
    }


# ==============================================================================
# EXPERIMENT DEFINITIONS
# ==============================================================================

EXPERIMENTS = [
    {
        "id": "01",
        "name": "bbox_medgemma15",
        "label": "BBox + MedGemma 1.5 4B",
        "stages": [_bbox_command, lambda: _vqa_command(VQA_MEDGEMMA_15), _judge_command],
        "keys": ["bbox_preproc", "vqa_gen", "vqa_judge"],
        "time": TIME_BBOX_VQA,
        "job_name": "exp01_bbox_mg15",
    },
    {
        "id": "02",
        "name": "bbox_medgemma4b",
        "label": "BBox + MedGemma 4B",
        "stages": [_bbox_command, lambda: _vqa_command(VQA_MEDGEMMA_4B), _judge_command],
        "keys": ["bbox_preproc", "vqa_gen", "vqa_judge"],
        "time": TIME_BBOX_VQA,
        "job_name": "exp02_bbox_mg4b",
    },
    {
        "id": "03",
        "name": "bbox_octomed",
        "label": "BBox + OctoMed 7B",
        "stages": [_bbox_command, lambda: _vqa_command(VQA_OCTOMED_7B), _judge_command],
        "keys": ["bbox_preproc", "vqa_gen", "vqa_judge"],
        "time": TIME_BBOX_VQA,
        "job_name": "exp03_bbox_octo",
    },
    {
        "id": "04",
        "name": "seg_medgemma15",
        "label": "Segmentation + MedGemma 1.5 4B",
        "stages": [_seg_command, lambda: _vqa_command(VQA_MEDGEMMA_15), _judge_command],
        "keys": ["segmentation", "vqa_gen", "vqa_judge"],
        "time": TIME_SEG_VQA,
        "job_name": "exp04_seg_mg15",
    },
    {
        "id": "05",
        "name": "attn_octomed",
        "label": "Attention Map + OctoMed 7B",
        "stages": [_attn_command, lambda: _vqa_command(VQA_OCTOMED_7B), _judge_command],
        "keys": ["attn_map", "vqa_gen", "vqa_judge"],
        "time": TIME_ATTN_VQA,
        "job_name": "exp05_attn_octo",
    },
]


# ==============================================================================
# GENERATION
# ==============================================================================

def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = PROJECT_ROOT / "experiments" / "batch_2k"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BATCH EXPERIMENT GENERATOR — 2K Dataset")
    print("=" * 70)
    print(f"Dataset:    {DATASET}")
    print(f"Batch dir:  {batch_dir.relative_to(PROJECT_ROOT)}")
    print(f"Timestamp:  {timestamp}")
    print("=" * 70)

    # Verify dataset exists in all required locations
    missing = []
    for check_dir in [BBOX_DIR, ATTN_DIR, SEG_DIR, VQA_DIR]:
        ds_path = Path(check_dir) / DATASET
        if not ds_path.exists():
            missing.append(str(ds_path))
    if missing:
        print("\n[ERROR] Dataset not found in:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun: python3 data_prep/create_stratified_samples.py --limit 2000")
        return 1

    # Verify all config files exist
    config_checks = [
        (BBOX_DIR, BBOX_CONFIG),
        (ATTN_DIR, ATTN_CONFIG),
        (SEG_DIR, SEG_STEP1_CONFIG),
        (SEG_DIR, SEG_STEP2_CONFIG),
        (VQA_DIR, VQA_MEDGEMMA_15),
        (VQA_DIR, VQA_MEDGEMMA_4B),
        (VQA_DIR, VQA_OCTOMED_7B),
        (VQA_DIR, JUDGE_CONFIG),
    ]
    for base, config in config_checks:
        full = Path(base) / config
        if not full.exists():
            print(f"[ERROR] Config not found: {full}")
            return 1
    print("[OK] All configs verified.\n")

    generated_scripts = []

    for exp in EXPERIMENTS:
        exp_dir = batch_dir / f"exp_{exp['id']}_{exp['name']}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build stage commands (call the lambdas/functions)
        stage_commands = [s() for s in exp["stages"]]

        # Use orchestrator_runs convention for run_dir so cache_utils paths resolve
        run_dir = str(PROJECT_ROOT / "orchestrator_runs" / f"batch_2k_{exp['name']}_{timestamp}")

        # Create run_dir so SLURM can write output/error files there
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        script = generate_meta_job_sbatch(
            stage_commands=stage_commands,
            run_dir=run_dir,
            dataset_override=DATASET,
            stage_keys=exp["keys"],
            job_name=exp["job_name"],
            time_limit=exp["time"],
        )

        # Write meta-job script
        meta_path = exp_dir / "meta_job.sh"
        meta_path.write_text(script)
        meta_path.chmod(0o755)

        # Write experiment info
        info_path = exp_dir / "experiment_info.txt"
        info_lines = [
            f"Experiment: {exp['label']}",
            f"Dataset:    {DATASET}",
            f"Stages:     {' -> '.join(exp['keys'])}",
            f"Time limit: {exp['time']}",
            f"Run dir:    {run_dir}",
            f"Generated:  {timestamp}",
        ]
        info_path.write_text("\n".join(info_lines) + "\n")

        generated_scripts.append((exp, meta_path))
        print(f"  [{exp['id']}] {exp['label']:<35} -> {meta_path.relative_to(PROJECT_ROOT)}")

    # Generate submission helper script
    submit_path = batch_dir / "submit_all.sh"
    submit_lines = [
        "#!/bin/bash",
        "# Submit all 5 experiments with SLURM job dependencies.",
        "# Each experiment waits for the previous one to finish (any exit code).",
        "# This ensures sequential execution on the single GPU.",
        "#",
        f"# Generated: {timestamp}",
        f"# Dataset:   {DATASET}",
        "",
        "set -e",
        f'BATCH_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"',
        "",
    ]

    for i, (exp, meta_path) in enumerate(generated_scripts):
        exp_dir_name = meta_path.parent.name
        if i == 0:
            submit_lines.extend([
                f'echo "[{exp["id"]}] Submitting: {exp["label"]}..."',
                f'JOB_{exp["id"]}=$(sbatch --parsable "$BATCH_DIR/{exp_dir_name}/meta_job.sh")',
                f'echo "    Job ID: $JOB_{exp["id"]}"',
                "",
            ])
        else:
            prev_id = generated_scripts[i - 1][0]["id"]
            submit_lines.extend([
                f'echo "[{exp["id"]}] Submitting: {exp["label"]} (after job $JOB_{prev_id})..."',
                f'JOB_{exp["id"]}=$(sbatch --parsable --dependency=afterany:$JOB_{prev_id} "$BATCH_DIR/{exp_dir_name}/meta_job.sh")',
                f'echo "    Job ID: $JOB_{exp["id"]}"',
                "",
            ])

    submit_lines.extend([
        'echo ""',
        'echo "============================================================"',
        'echo "All 5 experiments submitted with sequential dependencies."',
        'echo "Monitor with: squeue -u $USER"',
        'echo "============================================================"',
    ])

    submit_path.write_text("\n".join(submit_lines) + "\n")
    submit_path.chmod(0o755)

    print(f"\n  Submission script: {submit_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("CACHING BEHAVIOR SUMMARY")
    print("=" * 70)
    print("  Exp 01: BBox RUNS         (first run, no cache)")
    print("  Exp 02: BBox CACHE-HIT    (same dataset + config -> skip)")
    print("  Exp 03: BBox CACHE-HIT    (same dataset + config -> skip)")
    print("  Exp 04: Segmentation RUNS (different preprocessing type)")
    print("  Exp 05: Attn Map RUNS     (different preprocessing type)")
    print("=" * 70)

    print("\nTo submit all experiments:")
    print(f"  bash {submit_path.relative_to(PROJECT_ROOT)}")
    print("\nOr submit individually:")
    for exp, meta_path in generated_scripts:
        print(f"  sbatch {meta_path.relative_to(PROJECT_ROOT)}  # {exp['label']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
