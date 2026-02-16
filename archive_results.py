#!/usr/bin/env python3
"""
Medical VQA Result Archival Script
=================================
Moves results and logs from distributed source directories to the centralized
orchestrator run directory after job completion.

Usage:
    python archive_results.py [RUN_ID_OR_PATH]

If no run ID is provided, defaults to the latest run in orchestrator_runs/.
"""

import argparse
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "orchestrator_runs"

# Mapping from step key (in folder name) to source directory
# Step folder format: step_XX_KEY
# Source directories are relative to PROJECT_ROOT
STAGE_MAPPINGS = {
    "vqa_gen": {
        "source_results": "vqa/results",
        "source_logs": "vqa/logs",  # If exists
        "target_subdir": "results"
    },
    "vqa_judge": {
        "source_results": "vqa/results/judge_results",
        "target_subdir": "judge_results"
    },
    "bbox_preproc": {
        "source_results": "preprocessing/bounding_box/results",
        "target_subdir": "results"
    },
    "bbox_eval": {
        "source_results": "preprocessing/bounding_box/results", # Eval often outputs here too, or specific dir?
        # Check if eval outputs to a different dir. Typically it might be "metrics" or similar.
        # For now assume results.
        "target_subdir": "results"
    },
    "segmentation": {
        "source_results": "preprocessing/segmentation/results",
        "target_subdir": "results"
    },
    "attn_map": {
        "source_results": "preprocessing/attention_map/results",
        "target_subdir": "results"
    }
}

def get_latest_run_dir() -> Optional[Path]:
    """Find the latest run directory in orchestrator_runs/."""
    if not RUNS_DIR.exists():
        return None
    
    # Filter for run_YYYYMMDD_HHMMSS format
    run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) descending
    return sorted(run_dirs, key=lambda x: x.name, reverse=True)[0]

def move_files(source_dir: Path, target_dir: Path, description: str):
    """Move all files from source to target directory. Falls back to copy if move fails."""
    if not source_dir.exists():
        print(f"  [yellow]Skipping {description}: Source not found ({source_dir})[/yellow]")
        return

    # Create target if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    errors = 0
    
    for item in source_dir.iterdir():
        if item.name == ".gitkeep": continue
        
        dest = target_dir / item.name
        if dest.exists():
            # Handle collision? Timestamp it?
            timestamp = datetime.now().strftime("%H%M%S")
            dest = target_dir / f"{item.stem}_{timestamp}{item.suffix}"
            
        try:
            # Try standard move first
            shutil.move(str(item), str(dest))
            count += 1
        except PermissionError:
            # Fallback: Copy via shutil (read usually ok) then Docker delete source
            try:
                # 1. Copy to destination
                if item.is_dir():
                    shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(item), str(dest))
                
                # 2. Delete source using Docker (Rootless)
                import subprocess
                parent_dir = item.parent.resolve()
                target_name = item.name
                
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{parent_dir}:/clean_target",
                    "-w", "/clean_target",
                    "alpine", "rm", "-rf", target_name
                ]
                
                # Run silence stdout/stderr unless error
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    count += 1
                else:
                     print(f"  [red]Docker cleanup failed for {item.name}: {result.stderr.strip()}[/red]")
                     errors += 1
                     count += 1 # Count as success for move (data saved), but warn
            except Exception as e_copy:
                 print(f"  [red]Error copying/cleaning {item.name}: {e_copy}[/red]")
                 errors += 1
        except Exception as e:
            print(f"  [red]Error moving {item.name}: {e}[/red]")
            errors += 1

    if count > 0:
        msg = f"  [green]Processed {count} items from {description}[/green]"
        if errors > 0:
            msg += f" [yellow]({errors} cleanup warnings)[/yellow]"
        print(msg)
        
        # Attempt to remove the now-empty source directory
        # Safety check: NEVER remove PROJECT_ROOT
        if errors == 0 and source_dir.resolve() != PROJECT_ROOT.resolve():
            try:
                # First try standard rmdir (or rmtree to catch .gitkeep etc)
                shutil.rmtree(str(source_dir))
                print(f"  [green]Removed directory: {source_dir.name}[/green]")
            except PermissionError:
                # Docker fallback for cleaning the directory itself
                try:
                    import subprocess
                    parent_dir = source_dir.parent.resolve()
                    target_name = source_dir.name
                    
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{parent_dir}:/clean_target",
                        "-w", "/clean_target",
                        "alpine", "rm", "-rf", target_name
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"  [green]Removed directory (via Docker): {source_dir.name}[/green]")
                except Exception as e_rm:
                    print(f"  [yellow]Could not remove directory {source_dir.name}: {e_rm}[/yellow]")
            except Exception as e_rm:
                 print(f"  [yellow]Could not remove directory {source_dir.name}: {e_rm}[/yellow]")
    else:
        print(f"  [dim]No files found in {source_dir}[/dim]")
        
        if source_dir.resolve() != PROJECT_ROOT.resolve():
             try:
                shutil.rmtree(str(source_dir))
                print(f"  [green]Removed empty directory: {source_dir.name}[/green]")
             except PermissionError:
                # Docker fallback
                try:
                    import subprocess
                    parent_dir = source_dir.parent.resolve()
                    target_name = source_dir.name
                    cmd = ["docker", "run", "--rm", "-v", f"{parent_dir}:/clean_target", "-w", "/clean_target", "alpine", "rm", "-rf", target_name]
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"  [green]Removed empty directory (via Docker): {source_dir.name}[/green]")
                except:
                    pass
             except:
                pass


def process_step(step_dir: Path):
    """Process a single step directory within the run."""
    # Extract key from step_XX_KEY
    # format: step_01_bbox_preproc
    parts = step_dir.name.split('_')
    if len(parts) < 3:
        return
    
    # Reassemble key (in case key has underscores)
    key = "_".join(parts[2:])
    
    if key not in STAGE_MAPPINGS:
        print(f"[dim]Skipping unknown step type: {key}[/dim]")
        return

    print(f"\nProcessing Step: [bold]{step_dir.name}[/bold] ({key})")
    mapping = STAGE_MAPPINGS[key]
    
    # Move Results
    source_results = PROJECT_ROOT / mapping["source_results"]
    target_results = step_dir / mapping.get("target_subdir", "results")
    
    print(f"  Source: {source_results.relative_to(PROJECT_ROOT) if source_results.exists() else source_results}")
    print(f"  Target: {target_results.relative_to(PROJECT_ROOT)}")
    
    move_files(source_results, target_results, "Results")
    
    # Move Logs (if applicable)
    if "source_logs" in mapping:
        source_logs = PROJECT_ROOT / mapping["source_logs"]
        target_logs = step_dir / "logs"
        move_files(source_logs, target_logs, "Logs")

def cleanup_slurm_logs(run_dir: Path):
    """Move stray SLURM logs from project root to run dir."""
    print("\nProcessing SLURM Logs...")
    logs_dir = run_dir / "slurm_logs"
    
    # Find slurm-*.out and slurm-*.err in PROJECT_ROOT
    slurm_files = list(PROJECT_ROOT.glob("slurm-*.out")) + list(PROJECT_ROOT.glob("slurm-*.err"))
    
    if not slurm_files:
        print("  [dim]No stray SLURM logs found in project root.[/dim]")
        return
        
    logs_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for log_file in slurm_files:
        dest = logs_dir / log_file.name
        try:
             shutil.move(str(log_file), str(dest))
             count += 1
        except Exception as e:
             print(f"  [red]Error moving log {log_file.name}: {e}[/red]")
             
    print(f"  [green]Moved {count} SLURM log files to {logs_dir.relative_to(PROJECT_ROOT)}[/green]")


def simple_rich_print(text):
    """
    A simple parser to replace rich-style tags with ANSI colors
    if the rich library is not installed. Supports combined tags like [bold cyan].
    """
    import re
    
    if not isinstance(text, str):
        # Convert non-string to string for printing
        text = str(text)

    # ANSI codes
    # Reset
    RESET = "\033[0m"
    
    # Styles / Colors
    STYLES = {
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def replace_tag(match):
        content = match.group(1)
        # Handle closing tags [/tag]
        if content.startswith("/"):
            return RESET
        
        # Handle opening tags [bold cyan]
        parts = content.split()
        codes = []
        valid = False
        for part in parts:
            if part in STYLES:
                codes.append(STYLES[part])
                valid = True
        
        if valid:
            return "".join(codes)
        return match.group(0) # Return original if no valid styles found

    # Regex to match anything in brackets.
    # Use non-greedy match for content inside brackets.
    formatted_text = re.sub(r'\[(.*?)\]', replace_tag, text)

    # Use sys.stdout.write to avoid recursion with the overridden print
    sys.stdout.write(formatted_text + RESET + "\n")

def main():
    # Simple rich print setup if installed, else use fallback
    global print
    try:
        from rich import print as rich_print
        print = rich_print
    except ImportError:
        print = simple_rich_print
        
        # Mock Panel if missing
        global Panel
        def Panel(text, style=None):
            # Strip tags for the border length calculation or just print simple
            # Simple box
            lines = text.split('\n')
            max_len = 0
            clean_lines = []
            
            # Very basic stripper for length calc
            for line in lines:
                clean = line
                for tag in ["bold", "dim", "red", "green", "yellow", "blue", "cyan"]:
                    clean = clean.replace(f"[{tag}]", "").replace(f"[/{tag}]", "")
                max_len = max(max_len, len(clean))
                clean_lines.append(clean)
            
            border = "─" * (max_len + 4)
            print(f"╭{border}╮")
            for line in lines:
                print(f"│  {line}  │") 
            print(f"╰{border}╯")
            return ""

    try:
        from rich.panel import Panel
    except ImportError:
        def Panel(text, style=None):
            # Return a simple string representation
            div = "─" * 40
            return f"{div}\n{text}\n{div}"

    parser = argparse.ArgumentParser(description="Cleanup results and logs into run directory.")
    parser.add_argument("run_id", nargs="?", help="Run ID (e.g. run_2026...) or path. Defaults to latest.")
    parser.add_argument("--all", action="store_true", help="Clean up ALL run directories found in orchestrator_runs/")
    args = parser.parse_args()

    # Determine Run Directory
    run_dirs = []
    
    if args.all:
        if not RUNS_DIR.exists():
            print(f"[red]Error: Runs directory not found: {RUNS_DIR}[/red]")
            sys.exit(1)
        # Find all run directories
        run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda x: x.name)
        if not run_dirs:
            print("[yellow]No run directories found in orchestrator_runs/.[/yellow]")
            return
    elif args.run_id:
        if "/" in args.run_id:
            run_dirs = [Path(args.run_id).resolve()]
        else:
            run_dirs = [RUNS_DIR / args.run_id]
    else:
        latest = get_latest_run_dir()
        if latest:
            run_dirs = [latest]

    if not run_dirs:
        print("[red]Error: No run directory specified or found. Use --all to clean all runs.[/red]")
        sys.exit(1)

    print(Panel(f"[bold]🗂️ 🧼 Starting Archival 🧼 🗂️[/bold]\n🗂️ 🪣 Targeting {len(run_dirs)} run(s)🪣  🗂️", style="blue"))

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[red]Error: Run directory not found: {run_dir}[/red]")
            continue
            
        print(f"\n[bold cyan]=== Archiving: {run_dir.name} ===[/bold cyan]")

        # 1. Iterate over step directories
        steps = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])
        
        if not steps:
            print("  [yellow]No step directories found in run folder.[/yellow]")
        
        for step_dir in steps:
            process_step(step_dir)

        is_latest = (run_dir == run_dirs[-1])
        if is_latest:
            cleanup_slurm_logs(run_dir)
        else:
             print("  [dim]Skipping SLURM log cleanup (only done for latest run to avoid ambiguity)[/dim]")
    
    print("\n[bold green]Archival and Cleanup Complete.[/bold green]")

if __name__ == "__main__":
    # Add project root to sys path for imports if needed
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
