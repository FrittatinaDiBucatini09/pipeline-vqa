#!/usr/bin/env python3
"""
GEMeX to MIMIC-CXR Mapping Utility (Question-Centric Refactoring)
=================================================================
Description:
    Aligns metadata from the GEMeX-VQA dataset with local MIMIC-CXR images.
    
    KEY CHANGES v2:
    - Question-Centric: Iterates over questions, not just images.
    - Random Sampling: Selects a random subset of questions per image to ensure
      diversity (Open, Closed, Multi-choice) and avoid bias towards the last entry.
    - Path Validation: Ensures only questions with existing local images are kept.

Usage:
    python3 prepare_gemex.py
"""

import os
import random
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import sys
import tempfile

# Add current directory to path to import utils
sys.path.append(str(Path(__file__).parent))
import utils

# --- CONFIGURATION DEFAULTS ---
DEFAULT_MIMIC_ROOT_DIR = "/datasets/MIMIC-CXR/files"
BASE_FILENAME = "gemex_mimic_mapped"
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm', '.webp'}

def parse_args():
    parser = argparse.ArgumentParser(description='GEMeX to MIMIC-CXR Mapping Utility')
    parser.add_argument('--mimic_root_dir', type=str, default=DEFAULT_MIMIC_ROOT_DIR,
                       help='Root directory of MIMIC-CXR files')
    parser.add_argument('--max_questions_per_image', type=int, default=6,
                       help='Max questions per image (None for all)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit the total number of samples (default: None/All)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    MIMIC_ROOT_DIR = Path(args.mimic_root_dir)
    MAX_QUESTIONS_PER_IMAGE = args.max_questions_per_image
    LIMIT = args.limit
    RANDOM_SEED = args.seed

    # Generate output filename
    if LIMIT:
        OUTPUT_FILENAME = f"{BASE_FILENAME}_{LIMIT}_samples.csv"
        print(f"[INFO] Limit set to {LIMIT} samples. Output: {OUTPUT_FILENAME}")
    else:
        OUTPUT_FILENAME = f"{BASE_FILENAME}.csv"
        print(f"[INFO] No limit set. Output: {OUTPUT_FILENAME}")

    OUTPUT_CSV = Path(tempfile.gettempdir()) / OUTPUT_FILENAME

    print("[INFO] Setting random seed for reproducibility...")
    random.seed(RANDOM_SEED)

    print("[INFO] Loading FULL GEMeX dataset from Hugging Face...")
    try:
        # Load the complete dataset
        ds = load_dataset("BoKelvin/GEMeX-ThinkVG", split="train")
        df_gemex = ds.to_pandas()
        
        # Normalize column names
        if 'image_id' in df_gemex.columns and 'image_path' not in df_gemex.columns:
            df_gemex['image_path'] = df_gemex['image_id']
            
        print(f"[INFO] Raw GEMeX Entries: {len(df_gemex)}")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load dataset: {e}")
        return

    # --- 1. INDEXING LOCAL FILES ---
    if not MIMIC_ROOT_DIR.exists():
        print(f"[CRITICAL] Directory not found: {MIMIC_ROOT_DIR}")
        return

    print(f"[INFO] Indexing local files in {MIMIC_ROOT_DIR}...")
    local_files_map = {}
    
    # We use stem (filename without extension) as key for robust matching
    for f in tqdm(MIMIC_ROOT_DIR.rglob('*'), desc="Indexing Filesystem"):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            local_files_map[f.stem] = str(f)

    print(f"[INFO] Indexed {len(local_files_map)} local images.")
    if len(local_files_map) == 0:
        print("[CRITICAL] No images found locally.")
        return

    # --- 2. PRE-FILTERING & MAPPING ---
    print("-> filtering dataset for existing images...")
    
    # Create a temporary column 'stem_id' for mapping
    # Handling potential different path formats in HF dataset
    df_gemex['stem_id'] = df_gemex['image_path'].apply(lambda p: Path(str(p)).stem)
    
    # Filter: Keep only rows where the image exists locally
    # Vectorized operation is much faster than iterating rows
    df_gemex = df_gemex[df_gemex['stem_id'].isin(local_files_map.keys())].copy()
    
    print(f"[INFO] Entries with valid local images: {len(df_gemex)}")

    # --- 3. RANDOM SAMPLING PER IMAGE (Group & Sample) ---
    final_rows = []
    
    if MAX_QUESTIONS_PER_IMAGE:
        print(f"-> Applying Random Sampling (Max {MAX_QUESTIONS_PER_IMAGE} Qs/Image)...")
        
        # Group by image to handle the selection logic
        grouped = df_gemex.groupby('stem_id')
        
        for stem_id, group in tqdm(grouped, desc="Sampling Groups"):
            if len(group) > MAX_QUESTIONS_PER_IMAGE:
                # Randomly sample N rows
                sampled_group = group.sample(n=MAX_QUESTIONS_PER_IMAGE, random_state=RANDOM_SEED)
            else:
                # Keep all if less than limit
                sampled_group = group
            
            # Map the local full path efficiently
            local_path = local_files_map[stem_id]
            
            # Convert to dict records and add the resolved path
            records = sampled_group.to_dict('records')
            for r in records:
                r['resolved_local_path'] = local_path
                final_rows.append(r)
    else:
        print("-> Processing all valid entries (No Limit)...")
        # Just map the paths
        for idx, row in tqdm(df_gemex.iterrows(), total=len(df_gemex)):
            stem_id = row['stem_id']
            row_dict = row.to_dict()
            row_dict['resolved_local_path'] = local_files_map[stem_id]
            final_rows.append(row_dict)

    # --- 4. GLOBAL SAMPLING (if limit set) ---
    if LIMIT:
        if LIMIT < len(final_rows):
            print(f"[INFO] Applying global limit: sampling {LIMIT} rows from {len(final_rows)} total...")
            final_rows = random.sample(final_rows, LIMIT)
        else:
             print(f"[INFO] Global limit {LIMIT} >= Total rows {len(final_rows)}. Keeping all.")


    # --- 5. EXPORT ---
    print(f"-> Constructing Final DataFrame with {len(final_rows)} rows...")
    df_final = pd.DataFrame(final_rows)
    
    if not df_final.empty:
        # Clean up columns: ensure 'image_path' points to the REAL local file
        # The original 'image_path' from HF is usually just the ID or relative path
        df_final['original_hf_path'] = df_final['image_path']
        df_final['image_path'] = df_final['resolved_local_path']
        
        # Drop temp columns
        cols_to_drop = ['stem_id', 'resolved_local_path']
        df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True)
    else:
        print("[WARNING] Final DataFrame is empty.")

    df_final.to_csv(OUTPUT_CSV, index=False)

    print("-" * 40)
    print(f"[SUCCESS] Dataset saved to {OUTPUT_CSV}")
    if not df_final.empty:
        print(f"[STATS]   Unique Images: {df_final['image_path'].nunique()}")
        print(f"[STATS]   Total QA Pairs: {len(df_final)}")
    print("-" * 40)

    # --- AUTOMATIC DISTRIBUTION ---
    utils.distribute_file(OUTPUT_CSV)

if __name__ == "__main__":
    main()