#!/usr/bin/env python3
"""
Backfill the empty 'answer' column in existing vqa_manifest.csv files
by joining against the source stratified CSV.

Usage:
    # Fix ALL known manifests (hardcoded paths):
    python backfill_manifest_answers.py

    # Dry-run (preview only):
    python backfill_manifest_answers.py --dry-run

    # Fix a single manifest with explicit source:
    python backfill_manifest_answers.py \
        --manifest path/to/vqa_manifest.csv \
        --source   path/to/mimic_ext_stratified_6000_samples.csv

Join strategy:  manifest.question == source.question
                (questions are unique enough for a 1:1 match)
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

THESIS_ROOT = Path(__file__).resolve().parent

# Default source CSV (all preprocessing dirs have a copy)
DEFAULT_SOURCE = THESIS_ROOT / "vqa" / "mimic_ext_stratified_6000_samples.csv"

# All known manifests to fix
KNOWN_MANIFESTS: List[Path] = [
    THESIS_ROOT / "orchestrator_runs/ATTENTION_4K/step_01_attn_map/results/vqa_manifest.csv",
    THESIS_ROOT / "orchestrator_runs/SEGMENTATION_4K/step_01_segmentation/results/step1_bboxes/vqa_manifest.csv",
    THESIS_ROOT / "experiments/step1_preprocessing/attention/run_2000_SAMPLES_1/step_01_attn_map/results/vqa_manifest.csv",
    THESIS_ROOT / "experiments/step1_preprocessing/bounding_box/run_2000_SAMPLES_1/step_01_bbox_preproc/results/vqa_manifest.csv",
    THESIS_ROOT / "preprocessing/attention_map/results/vqa_manifest.csv",
]


def _read_csv_fast(path: Path) -> List[Dict[str, str]]:
    """Read an entire CSV into memory in a single I/O call (NFS-friendly)."""
    raw = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    return list(reader), reader.fieldnames


def _build_answer_lookup(source_csv: Path) -> Dict[str, str]:
    """Build a question -> answer_text lookup from the source CSV."""
    rows, _ = _read_csv_fast(source_csv)
    lookup: Dict[str, str] = {}
    for row in rows:
        q = row.get("question", "").strip()
        a = row.get("answer_text", "").strip()
        if q and a and a != "[]":
            lookup[q] = a
    return lookup


def backfill_manifest(
    manifest_path: Path,
    lookup: Dict[str, str],
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Backfill 'answer' column in a manifest.

    Returns (total_rows, filled_rows, already_had_answer).
    """
    rows, fieldnames = _read_csv_fast(manifest_path)

    total = len(rows)
    filled = 0
    already = 0

    for row in rows:
        current = row.get("answer", "").strip()
        if current:
            already += 1
            continue

        q = row.get("question", "").strip()
        answer = lookup.get(q)
        if answer:
            row["answer"] = answer
            filled += 1

    if not dry_run and filled > 0:
        # Create backup
        backup = manifest_path.with_suffix(".csv.bak")
        manifest_path.rename(backup)

        # Write back
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        manifest_path.write_text(out.getvalue(), encoding="utf-8")

    return total, filled, already


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill empty 'answer' column in vqa_manifest.csv files"
    )
    parser.add_argument("--manifest", type=Path, help="Single manifest to fix")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Source CSV with answer_text column")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    manifests = [args.manifest] if args.manifest else KNOWN_MANIFESTS
    source = args.source

    if not source.exists():
        print(f"[ERROR] Source CSV not found: {source}")
        sys.exit(1)

    # Filter to existing manifests
    existing = [(m, m.exists()) for m in manifests]
    for m, ok in existing:
        if not ok:
            try:
                rel = m.relative_to(THESIS_ROOT)
            except ValueError:
                rel = m
            print(f"[SKIP]  {rel}  (file not found)")

    manifests = [m for m, ok in existing if ok]
    if not manifests:
        print("[WARN] No manifest files found.")
        sys.exit(0)

    print(f"\nSource: {source.relative_to(THESIS_ROOT)}")
    print(f"Loading answer lookup...", end=" ", flush=True)
    lookup = _build_answer_lookup(source)
    print(f"{len(lookup)} answers loaded.\n")

    if args.dry_run:
        print("── DRY RUN (no files will be modified) ──\n")

    grand_total = 0
    for manifest_path in manifests:
        try:
            rel = manifest_path.relative_to(THESIS_ROOT)
        except ValueError:
            rel = manifest_path

        total, filled, already = backfill_manifest(
            manifest_path, lookup, dry_run=args.dry_run
        )

        status = "✓" if filled > 0 else "·"
        print(f"  {status} {rel}")
        print(f"    rows={total}  filled={filled}  already={already}  "
              f"still_empty={total - filled - already}")

        if not args.dry_run and filled > 0:
            print(f"    backup → {rel}.bak")

        grand_total += filled

    verb = "would be filled" if args.dry_run else "filled"
    print(f"\nDone. {grand_total} answers {verb} across {len(manifests)} manifest(s).")


if __name__ == "__main__":
    main()
