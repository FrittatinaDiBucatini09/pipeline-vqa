#!/usr/bin/env python3
"""
Clean Generated Datasets Utility
================================
Description:
    Reads the 'generated_datasets_registry.json' file created by the data preparation scripts
    and deletes all the files listed therein. This ensures a clean slate without manually
    hunting down CSV files across the project.

Usage:
    python3 clean_generated_datasets.py
"""

import json
import os
import sys
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent / "generated_datasets_registry.json"

def main():
    print("=" * 60)
    print("🧹 CLEANING GENERATED DATASETS")
    print("=" * 60)

    if not REGISTRY_FILE.exists():
        print(f"[INFO] No registry file found at {REGISTRY_FILE}")
        print("[INFO] Nothing to clean.")
        return

    try:
        with open(REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        print("[ERROR] Registry file is corrupted.")
        return

    if not registry:
        print("[INFO] Registry is empty. Nothing to clean.")
        return

    print(f"[INFO] Found {len(registry)} entries in registry.")
    
    deleted_count = 0
    failed_count = 0
    remaining_files = []

    for file_path_str in registry:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"  [MISSING] {file_path}")
            continue

        try:
            os.remove(file_path)
            print(f"  [DELETED] {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  [FAILED]  {file_path}: {e}")
            failed_count += 1
            remaining_files.append(file_path_str)

    # Update registry
    if failed_count == 0:
        # All clean, remove registry or empty it
        try:
            os.remove(REGISTRY_FILE)
            print(f"\n[SUCCESS] Registry file deleted.")
        except Exception as e:
            print(f"\n[WARNING] Could not delete registry file: {e}")
            # Try to empty it at least
            with open(REGISTRY_FILE, 'w') as f:
                json.dump([], f)
    else:
        # Update with remaining files
        print(f"\n[WARNING] {failed_count} files could not be deleted. Updating registry...")
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(remaining_files, f, indent=4)

    print("-" * 60)
    print(f"Summary: {deleted_count} deleted, {failed_count} failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
