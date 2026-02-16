#!/usr/bin/env python3
"""
Medical VQA Pipeline Orchestrator
==================================
Interactive CLI for chaining pipeline stages and submitting them to SLURM
as a single meta-job with continuous GPU ownership.

Usage:
    python orchestrator/orchestrator.py [--dry-run]

Or via the wrapper:
    ./run_orchestrator.sh [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "orchestrator_runs"

SBATCH_JOB_RE = re.compile(r"Submitted batch job (\d+)")

console = Console()


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """Describes one stage of the Medical VQA pipeline."""

    name: str
    key: str
    script_path: str  # Relative to PROJECT_ROOT
    config_dir: Optional[str] = None  # Relative to PROJECT_ROOT
    needs_gpu: bool = True
    has_submit_script: bool = True
    description: str = ""
    # For stages using env vars instead of config file arg (segmentation)
    uses_env_vars: bool = False
    # For stages requiring positional args (bbox eval)
    positional_arg_names: list = field(default_factory=list)


STAGE_REGISTRY: list[PipelineStage] = [
    PipelineStage(
        name="Preprocessing: Bounding Box",
        key="bbox_preproc",
        script_path="preprocessing/bounding_box/submit_bbox_preprocessing.sh",
        config_dir="preprocessing/bounding_box/configs",
        description="GradCAM-based bounding box generation",
    ),
    PipelineStage(
        name="Preprocessing: Attention Map",
        key="attn_map",
        script_path="preprocessing/attention_map/submit_heatmap_gen.sh",
        config_dir="preprocessing/attention_map/configs",
        description="Heatmap/attention map generation",
    ),
    PipelineStage(
        name="Preprocessing: Segmentation",
        key="segmentation",
        script_path="preprocessing/segmentation/submit_segmentation.sh",
        config_dir="preprocessing/segmentation/configs",
        uses_env_vars=True,
        description="MedSAM segmentation pipeline",
    ),
    PipelineStage(
        name="Bounding Box Evaluation",
        key="bbox_eval",
        script_path="preprocessing/bounding_box/submit_evaluation.sh",
        config_dir=None,
        needs_gpu=False,
        positional_arg_names=["Predictions directory", "Output metrics directory"],
        description="IoU metrics evaluation (CPU-only)",
    ),
    PipelineStage(
        name="VQA Generation",
        key="vqa_gen",
        script_path="vqa/submit_generation.sh",
        config_dir="vqa/configs/generation",
        description="LLM-based VQA inference",
    ),
    PipelineStage(
        name="VQA Evaluation (Judge)",
        key="vqa_judge",
        script_path="vqa/scripts/run_judge.sh",
        config_dir="vqa/configs/judge",
        has_submit_script=False,
        description="LLM Judge evaluation",
    ),
]


# ---------------------------------------------------------------------------
# Config Discovery
# ---------------------------------------------------------------------------


def discover_configs(config_dir: str) -> list[str]:
    """Recursively find .conf files under a config directory.

    Returns paths relative to the config_dir itself, sorted alphabetically.
    Example: ["gemex/exp_01_vqa.conf", "mimic_ext/exp_01_vqa_mimic.conf"]
    """
    abs_dir = PROJECT_ROOT / config_dir
    if not abs_dir.is_dir():
        return []
    configs = sorted(abs_dir.rglob("*.conf"))
    return [str(c.relative_to(abs_dir)) for c in configs]


# ---------------------------------------------------------------------------
# Stage Configuration (Interactive Prompts)
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Holds the user-selected configuration for a single stage."""

    config_file: Optional[str] = None  # Relative to script's working dir
    env_overrides: Optional[dict] = None
    positional_args: Optional[list] = None


def configure_stage(stage: PipelineStage) -> StageConfig:
    """Prompt the user for stage-specific configuration."""
    cfg = StageConfig()

    # --- Config file selection ---
    if stage.config_dir and not stage.uses_env_vars:
        configs = discover_configs(stage.config_dir)
        if configs:
            choices = ["(default - no config argument)"] + configs
            answer = inquirer.prompt(
                [
                    inquirer.List(
                        "config",
                        message=f"Select config for {stage.name}",
                        choices=choices,
                    )
                ]
            )
            if answer and answer["config"] != "(default - no config argument)":
                # Build path relative to the script's working directory.
                # For submit scripts, cwd = script's parent dir
                #   e.g. bbox script in preprocessing/bounding_box/
                #         → configs/gemex/exp_01.conf
                # For generated wrappers (has_submit_script=False), the
                # generated sbatch cd's to the module root (e.g. vqa/),
                # not the script's immediate parent (vqa/scripts/).
                if stage.has_submit_script:
                    working_dir = Path(stage.script_path).parent
                else:
                    working_dir = Path(stage.script_path).parent.parent
                config_dir_relative = str(
                    Path(stage.config_dir).relative_to(working_dir)
                )
                cfg.config_file = f"{config_dir_relative}/{answer['config']}"

    # --- Segmentation: env var prompts ---
    if stage.uses_env_vars:
        cfg.env_overrides = _configure_segmentation()

    # --- Bbox eval: positional arg prompts ---
    if stage.positional_arg_names:
        cfg.positional_args = _configure_positional_args(stage)

    return cfg


def _configure_segmentation() -> dict:
    """Prompt for segmentation-specific environment variables."""
    answer = inquirer.prompt(
        [
            inquirer.List(
                "mode",
                message="Segmentation mode (TARGET_MODE)",
                choices=[
                    ("all - Full pipeline (localization + segmentation)", "all"),
                    ("1 - Localization only", "1"),
                    ("2 - Segmentation only", "2"),
                ],
                default="all",
            ),
            inquirer.Text(
                "limit",
                message="Sample limit (leave empty for all)",
                default="",
            ),
        ]
    )
    if not answer:
        return {"TARGET_MODE": "all"}

    env = {"TARGET_MODE": answer["mode"]}
    if answer["limit"].strip():
        env["LIMIT"] = answer["limit"].strip()
    return env


def _configure_positional_args(stage: PipelineStage) -> list:
    """Prompt for positional arguments (e.g. bbox evaluation dirs)."""
    questions = [
        inquirer.Text(f"arg_{i}", message=name)
        for i, name in enumerate(stage.positional_arg_names)
    ]
    answer = inquirer.prompt(questions)
    if not answer:
        return []
    return [answer[f"arg_{i}"] for i in range(len(stage.positional_arg_names))]


# ---------------------------------------------------------------------------
# Meta-Job Command Building
# ---------------------------------------------------------------------------


def build_stage_command(stage: PipelineStage, cfg: StageConfig) -> dict:
    """Build the bash command block for a stage inside the meta-job script.

    Returns a dict with:
        - name: Human-readable stage name
        - command: Bash commands to execute (may be multi-line)
    """
    if stage.has_submit_script:
        # For stages with submit scripts: cd to script dir and run via bash
        abs_script_dir = str(PROJECT_ROOT / Path(stage.script_path).parent)
        script_name = Path(stage.script_path).name

        # Build the command with args
        args_str = ""
        if cfg.config_file:
            args_str += f" {cfg.config_file}"
        if cfg.positional_args:
            args_str += " " + " ".join(cfg.positional_args)

        # Build env var exports for this stage (e.g., segmentation TARGET_MODE)
        env_lines = ""
        if cfg.env_overrides:
            exports = [f'export {k}="{v}"' for k, v in cfg.env_overrides.items()]
            env_lines = "\n".join(exports) + "\n"

        command = f'{env_lines}cd "{abs_script_dir}"\nbash {script_name}{args_str}'

    else:
        # Judge stage: inline Docker command
        from slurm_templates import generate_judge_inline

        vqa_dir = str(PROJECT_ROOT / Path(stage.script_path).parent.parent)
        abs_script_dir = vqa_dir
        command = generate_judge_inline(
            config_file=cfg.config_file or "",
            vqa_dir=vqa_dir,
        )

    return {
        "name": stage.name,
        "command": command,
        "script_dir": abs_script_dir,
    }


# ---------------------------------------------------------------------------
# Run Isolation & Reporting
# ---------------------------------------------------------------------------


def create_run_directory(selected_stages: list[PipelineStage]) -> Path:
    """Create the run directory with step subdirectories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, stage in enumerate(selected_stages, 1):
        step_dir = run_dir / f"step_{i:02d}_{stage.key}"
        step_dir.mkdir()

    return run_dir


def write_report(
    run_dir: Path,
    stages: list[PipelineStage],
    stage_configs: list[StageConfig],
    job_id: Optional[str] = None,
    dataset_override: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Write report.txt summarizing the pipeline run."""
    report_path = run_dir / "report.txt"
    lines = [
        "=" * 60,
        "Pipeline Orchestrator Run Report",
        "=" * 60,
        f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run Dir   : {run_dir}",
        f"Job ID    : {job_id or 'DRY_RUN'}",
        f"Dry Run   : {dry_run}",
    ]

    if dataset_override:
        lines.append(f"Dataset   : {dataset_override}")

    lines.extend([
        "",
        "-" * 60,
        f"{'Step':<6} {'Stage':<30} {'Config':<30}",
        "-" * 60,
    ])

    for i, (stage, cfg) in enumerate(zip(stages, stage_configs), 1):
        config_str = cfg.config_file or "(default)"
        if cfg.env_overrides:
            config_str = ", ".join(f"{k}={v}" for k, v in cfg.env_overrides.items())
        if cfg.positional_args:
            config_str = " | ".join(cfg.positional_args)

        lines.append(f"{i:<6} {stage.name:<30} {config_str:<30}")

    lines.extend(["", "=" * 60])
    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------


def display_summary(
    stages: list[PipelineStage],
    configs: list[StageConfig],
) -> None:
    """Print a Rich table summarizing what will be submitted."""
    table = Table(title="Pipeline Summary (Single Meta-Job)", show_lines=True)
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Config / Args", style="green")

    for i, (stage, cfg) in enumerate(zip(stages, configs), 1):
        config_str = cfg.config_file or "(default)"
        if cfg.env_overrides:
            config_str = ", ".join(f"{k}={v}" for k, v in cfg.env_overrides.items())
        if cfg.positional_args:
            config_str = " | ".join(cfg.positional_args)

        table.add_row(str(i), stage.name, config_str)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def select_stages() -> list[PipelineStage]:
    """Present a checkbox menu for stage selection."""
    choices = [(f"{s.name} - {s.description}", s) for s in STAGE_REGISTRY]
    answer = inquirer.prompt(
        [
            inquirer.Checkbox(
                "stages",
                message="Select pipeline stages (SPACE to toggle, ENTER to confirm)",
                choices=choices,
            )
        ]
    )
    if not answer or not answer["stages"]:
        return []

    # Preserve registry order
    order = {s.key: i for i, s in enumerate(STAGE_REGISTRY)}
    selected = sorted(answer["stages"], key=lambda s: order[s.key])
    return selected


def discover_datasets() -> list[str]:
    """Find .csv files in preprocessing/bounding_box to use as datasets."""
    bbox_dir = PROJECT_ROOT / "preprocessing/bounding_box"
    if not bbox_dir.is_dir():
        return []
    return sorted([f.name for f in bbox_dir.glob("*.csv")])


def select_dataset() -> Optional[str]:
    """Prompt user to select a global dataset file."""
    datasets = discover_datasets()
    if not datasets:
        console.print("[yellow]No .csv datasets found in preprocessing/bounding_box.[/yellow]")
        return None

    choices = ["(Use Script Defaults)"] + datasets
    answer = inquirer.prompt(
        [
            inquirer.List(
                "dataset",
                message="Select Dataset for this run",
                choices=choices,
            )
        ]
    )
    if not answer or answer["dataset"] == "(Use Script Defaults)":
        return None
    return answer["dataset"]


def run_pipeline(
    stages: list[PipelineStage],
    stage_configs: list[StageConfig],
    dataset_override: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Generate a meta-job script and submit it as a single SLURM job."""
    from slurm_templates import generate_meta_job_sbatch

    run_dir = create_run_directory(stages)

    console.print(
        Panel(
            f"[bold]Run directory:[/bold] {run_dir.relative_to(PROJECT_ROOT)}",
            style="blue",
        )
    )

    # Build stage command list and write traceability files
    stage_commands = []
    for i, (stage, cfg) in enumerate(zip(stages, stage_configs), 1):
        step_dir = run_dir / f"step_{i:02d}_{stage.key}"

        # Traceability files
        if cfg.config_file:
            (step_dir / "config_used.txt").write_text(cfg.config_file + "\n")
        if cfg.env_overrides:
            env_lines = [f"{k}={v}" for k, v in cfg.env_overrides.items()]
            (step_dir / "env_overrides.txt").write_text("\n".join(env_lines) + "\n")
        if dataset_override:
            (step_dir / "dataset_override.txt").write_text(dataset_override + "\n")

        cmd_info = build_stage_command(stage, cfg)
        stage_commands.append(cmd_info)

    # Generate the meta-job script
    meta_script = generate_meta_job_sbatch(
        stage_commands=stage_commands,
        run_dir=str(run_dir.resolve()),
        dataset_override=dataset_override,
        stage_keys=[s.key for s in stages],
    )

    # Write meta-job to run directory
    meta_path = run_dir / "meta_job.sh"
    meta_path.write_text(meta_script)
    meta_path.chmod(0o755)

    console.print(
        f"[green]Meta-job script:[/green] {meta_path.relative_to(PROJECT_ROOT)}"
    )

    if dry_run:
        console.print("\n[yellow]DRY RUN — Generated script:[/yellow]\n")
        console.print(meta_script)
        write_report(run_dir, stages, stage_configs,
                      dataset_override=dataset_override, dry_run=True)
        console.print(
            f"\n[green]Report written:[/green] "
            f"{run_dir.relative_to(PROJECT_ROOT)}/report.txt"
        )
        return

    # Submit the single meta-job
    result = subprocess.run(
        ["sbatch", str(meta_path)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]sbatch failed:[/red] {result.stderr.strip()}")
        write_report(run_dir, stages, stage_configs,
                      dataset_override=dataset_override, dry_run=False)
        return

    match = SBATCH_JOB_RE.search(result.stdout)
    if not match:
        console.print(f"[red]Could not parse job ID from:[/red] {result.stdout}")
        write_report(run_dir, stages, stage_configs,
                      dataset_override=dataset_override, dry_run=False)
        return

    job_id = match.group(1)
    console.print(f"\n[bold green]Submitted meta-job:[/bold green] Job ID {job_id}")
    console.print(f"  Stages: {len(stages)}")
    console.print(f"  SLURM output: {run_dir.relative_to(PROJECT_ROOT)}/slurm_metajob_{job_id}.out")
    console.print(f"  SLURM errors: {run_dir.relative_to(PROJECT_ROOT)}/slurm_metajob_{job_id}.err")

    write_report(run_dir, stages, stage_configs, job_id=job_id,
                  dataset_override=dataset_override, dry_run=False)
    console.print(
        f"[green]Report written:[/green] "
        f"{run_dir.relative_to(PROJECT_ROOT)}/report.txt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Medical VQA Pipeline Orchestrator - Interactive SLURM Pipeline Manager"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate meta-job script without submitting to SLURM",
    )
    args = parser.parse_args()

    # Banner
    console.print()
    console.print(
        Panel.fit(
            "[bold]Medical VQA Pipeline Orchestrator[/bold]\n"
            "[dim]Single Meta-Job SLURM Pipeline Manager[/dim]",
            border_style="bright_blue",
        )
    )
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE - No jobs will be submitted[/yellow]\n")

    # Step 1: Select Dataset (Global)
    dataset_override = select_dataset()
    if dataset_override:
        console.print(f"[bold green]Dataset selected:[/bold green] {dataset_override}\n")
    else:
        console.print("[dim]Using script-defined default datasets[/dim]\n")

    # Step 2: Select stages
    stages = select_stages()
    if not stages:
        console.print("[yellow]No stages selected. Exiting.[/yellow]")
        return
    console.print(f"\n[bold]{len(stages)} stage(s) selected.[/bold]\n")

    # Step 3: Configure each stage
    stage_configs: list[StageConfig] = []
    for stage in stages:
        console.print(f"[bold cyan]--- Configuring: {stage.name} ---[/bold cyan]")
        stage_configs.append(configure_stage(stage))

    # Step 4: Summary and confirmation
    display_summary(stages, stage_configs)

    if dataset_override:
        console.print(f"Global Dataset Override: [bold]{dataset_override}[/bold]")
        console.print()

    console.print("[dim]All stages will run sequentially in a single SLURM job (15h time limit).[/dim]")
    console.print()

    confirm = inquirer.prompt(
        [inquirer.Confirm("proceed", message="Proceed with submission?", default=True)]
    )
    if not confirm or not confirm["proceed"]:
        console.print("[yellow]Aborted by user.[/yellow]")
        return

    # Step 5: Generate meta-job and submit
    run_pipeline(stages, stage_configs, dataset_override=dataset_override, dry_run=args.dry_run)
    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    # Ensure imports work when running from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
