#!/usr/bin/env python3
"""
Preprocessing Cache State Utility
==================================
CLI tool for managing preprocessing cache state files.
Called from generated meta-job bash scripts to check/write/validate
cache state and avoid redundant preprocessing runs.

Subcommands:
    check    - Compare current fingerprint against cached state (exit 0=hit, 1=miss)
    write    - Write cache state after successful stage completion
    validate - Verify integrity of cached results (manifest, row counts)

Usage from bash:
    python3 orchestrator/cache_utils.py check  --results-dir /path/to/results --dataset foo.csv --config-file /path/to.conf [--ner-enabled]
    python3 orchestrator/cache_utils.py write  --results-dir /path/to/results --dataset foo.csv --config-file /path/to.conf [--ner-enabled]
    python3 orchestrator/cache_utils.py validate --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

STATE_FILENAME = ".preproc_cache_state.json"


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _compute_config_hash(config_path: str) -> str:
    """Compute MD5 hash of config file(s) contents.

    Supports multiple space-separated paths (e.g. segmentation with step1 + step2
    configs). Returns "no_config" if no config file is provided or none exist.
    """
    if not config_path:
        return "no_config"

    # Handle multiple space-separated config paths
    paths = config_path.split()
    hasher = hashlib.md5()
    found_any = False

    for p_str in paths:
        p = Path(p_str)
        if p.is_file():
            hasher.update(p.read_bytes())
            found_any = True

    return hasher.hexdigest() if found_any else "no_config"


def _count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file (excludes header)."""
    if not csv_path.is_file():
        return 0
    with open(csv_path, "r") as f:
        lines = sum(1 for line in f if line.strip())
    return max(0, lines - 1)  # subtract header


def _count_jsonl_rows(jsonl_path: Path) -> int:
    """Count rows in a JSONL file."""
    if not jsonl_path.is_file():
        return 0
    with open(jsonl_path, "r") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def _read_state(results_dir: Path) -> dict | None:
    """Read cached state file. Returns None if missing or corrupt."""
    state_path = results_dir / STATE_FILENAME
    if not state_path.is_file():
        return None
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(results_dir: Path, state: dict) -> None:
    """Write state file to results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = results_dir / STATE_FILENAME
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[CACHE] State written to {state_path}")


def _remove_state(results_dir: Path) -> None:
    """Remove stale state file."""
    state_path = results_dir / STATE_FILENAME
    if state_path.is_file():
        state_path.unlink()


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Check if cached results match current fingerprint.

    Returns:
        0 if cache is valid (caller should SKIP the stage)
        1 if cache is invalid (caller should RUN the stage)
    """
    results_dir = Path(args.results_dir)
    current_hash = _compute_config_hash(args.config_file)
    current_dataset = args.dataset or "default"
    current_ner = args.ner_enabled

    # 1. Check state file exists
    state = _read_state(results_dir)
    if state is None:
        print("[CACHE] No cache state found. Stage must run.")
        return 1

    # 2. Check dataset match
    cached_dataset = state.get("dataset", "")
    if cached_dataset != current_dataset:
        print(
            f"[CACHE] Dataset changed (was: {cached_dataset}, now: {current_dataset}). "
            f"Stage must re-run."
        )
        return 1

    # 3. Check config hash match
    cached_hash = state.get("config_hash", "")
    if cached_hash != current_hash:
        cached_file = state.get("config_file", "unknown")
        print(
            f"[CACHE] Config changed (cached: {cached_file} [{cached_hash[:8]}...], "
            f"current: {args.config_file or 'default'} [{current_hash[:8]}...]). "
            f"Stage must re-run."
        )
        return 1

    # 4. Check NER mismatch
    cached_ner = state.get("ner_enabled", False)
    if cached_ner != current_ner:
        was = "enabled" if cached_ner else "disabled"
        now = "enabled" if current_ner else "disabled"
        print(
            f"[CACHE] NER configuration changed (was: {was}, now: {now}). "
            f"Invalidating cache..."
        )
        _remove_state(results_dir)
        return 1

    # 5. Check manifest integrity
    manifest_path = results_dir / "vqa_manifest.csv"
    if not manifest_path.is_file():
        print("[CACHE] vqa_manifest.csv missing. Stage must re-run.")
        _remove_state(results_dir)
        return 1

    manifest_rows = _count_csv_rows(manifest_path)
    if manifest_rows == 0:
        print("[CACHE] vqa_manifest.csv is empty. Stage must re-run.")
        _remove_state(results_dir)
        return 1

    cached_rows = state.get("output_row_count", 0)
    if cached_rows > 0 and manifest_rows != cached_rows:
        print(
            f"[CACHE] Manifest row count mismatch (cached: {cached_rows}, "
            f"actual: {manifest_rows}). Stage must re-run."
        )
        _remove_state(results_dir)
        return 1

    # All checks passed
    print(
        f"[CACHE] Cache valid — dataset={current_dataset}, "
        f"config={current_hash[:8]}..., ner={'on' if current_ner else 'off'}, "
        f"rows={manifest_rows}. Skipping stage."
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: write
# ---------------------------------------------------------------------------


def cmd_write(args: argparse.Namespace) -> int:
    """Write cache state after successful stage completion."""
    results_dir = Path(args.results_dir)

    manifest_path = results_dir / "vqa_manifest.csv"
    manifest_rows = _count_csv_rows(manifest_path)

    state = {
        "dataset": args.dataset or "default",
        "config_hash": _compute_config_hash(args.config_file),
        "config_file": args.config_file or "",
        "ner_enabled": args.ner_enabled,
        "output_row_count": manifest_rows,
        "manifest_exists": manifest_path.is_file(),
        "timestamp": datetime.now().isoformat(),
    }

    _write_state(results_dir, state)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate integrity of cached results."""
    results_dir = Path(args.results_dir)
    issues: list[str] = []

    # 1. State file
    state = _read_state(results_dir)
    if state is None:
        issues.append("No .preproc_cache_state.json found")
    else:
        # 2. Manifest
        manifest_path = results_dir / "vqa_manifest.csv"
        if not manifest_path.is_file():
            issues.append("vqa_manifest.csv missing")
        else:
            manifest_rows = _count_csv_rows(manifest_path)
            if manifest_rows == 0:
                issues.append("vqa_manifest.csv is empty (0 data rows)")

            cached_rows = state.get("output_row_count", 0)
            if cached_rows > 0 and manifest_rows != cached_rows:
                issues.append(
                    f"Row count mismatch: state says {cached_rows}, "
                    f"manifest has {manifest_rows}"
                )

        # 3. Predictions (optional, for bbox/step1)
        predictions_path = results_dir / "predictions.jsonl"
        if predictions_path.is_file():
            pred_rows = _count_jsonl_rows(predictions_path)
            if pred_rows == 0:
                issues.append("predictions.jsonl exists but is empty")

        # 4. Timestamp sanity
        ts_str = state.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts > datetime.now():
                    issues.append(f"Timestamp is in the future: {ts_str}")
            except ValueError:
                issues.append(f"Invalid timestamp format: {ts_str}")

    if issues:
        print(f"[VALIDATE] FAIL — {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"[VALIDATE] OK — Cache state is consistent.")
    if state:
        print(f"  Dataset:    {state.get('dataset', 'N/A')}")
        print(f"  Config:     {state.get('config_file', 'N/A')}")
        print(f"  NER:        {'enabled' if state.get('ner_enabled') else 'disabled'}")
        print(f"  Rows:       {state.get('output_row_count', 'N/A')}")
        print(f"  Timestamp:  {state.get('timestamp', 'N/A')}")
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing cache state management utility"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- check ---
    p_check = subparsers.add_parser(
        "check", help="Check if cached results match current fingerprint"
    )
    p_check.add_argument(
        "--results-dir", required=True, help="Path to the stage results directory"
    )
    p_check.add_argument(
        "--dataset", default="default", help="Dataset filename (DATA_FILE_OVERRIDE)"
    )
    p_check.add_argument(
        "--config-file", default="", help="Path to .conf file for hash computation"
    )
    p_check.add_argument(
        "--ner-enabled", action="store_true", help="NER/Routing is active upstream"
    )

    # --- write ---
    p_write = subparsers.add_parser(
        "write", help="Write cache state after successful completion"
    )
    p_write.add_argument(
        "--results-dir", required=True, help="Path to the stage results directory"
    )
    p_write.add_argument(
        "--dataset", default="default", help="Dataset filename (DATA_FILE_OVERRIDE)"
    )
    p_write.add_argument(
        "--config-file", default="", help="Path to .conf file for hash computation"
    )
    p_write.add_argument(
        "--ner-enabled", action="store_true", help="NER/Routing is active upstream"
    )

    # --- validate ---
    p_validate = subparsers.add_parser(
        "validate", help="Validate integrity of cached results"
    )
    p_validate.add_argument(
        "--results-dir", required=True, help="Path to the stage results directory"
    )

    args = parser.parse_args()

    handlers = {
        "check": cmd_check,
        "write": cmd_write,
        "validate": cmd_validate,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
