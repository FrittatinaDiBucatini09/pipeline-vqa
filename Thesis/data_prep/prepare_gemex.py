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
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
MIMIC_ROOT_DIR = Path("/datasets/MIMIC-CXR/files")  
OUTPUT_CSV = "../gemex_mimic_mapped.csv"
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm', '.webp'}

# MAX QUESTIONS PER IMAGE:
# Set to an integer (e.g., 3) to keep a random subset of questions per image.
# Set to None to keep ALL 11 questions per image (Warning: Dataset size will explode).
MAX_QUESTIONS_PER_IMAGE = 6

# Random Seed for Reproducibility
RANDOM_SEED = 42

def main():
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

    # --- 4. EXPORT ---
    print(f"-> Constructing Final DataFrame with {len(final_rows)} rows...")
    df_final = pd.DataFrame(final_rows)
    
    # Clean up columns: ensure 'image_path' points to the REAL local file
    # The original 'image_path' from HF is usually just the ID or relative path
    df_final['original_hf_path'] = df_final['image_path']
    df_final['image_path'] = df_final['resolved_local_path']
    
    # Drop temp columns
    cols_to_drop = ['stem_id', 'resolved_local_path']
    df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True)

    df_final.to_csv(OUTPUT_CSV, index=False)

    print("-" * 40)
    print(f"[SUCCESS] Dataset saved to {OUTPUT_CSV}")
    print(f"[STATS]   Unique Images: {df_final['image_path'].nunique()}")
    print(f"[STATS]   Total QA Pairs: {len(df_final)}")
    print("-" * 40)

    # --- AUTOMATIC DISTRIBUTION TO MODULE DIRECTORIES ---
    import shutil

    # Define target directories for this CSV
    target_dirs = [
        "../preprocessing/bounding_box",
        "../preprocessing/attention_map",
        "../preprocessing/segmentation"
    ]

    output_filename = Path(OUTPUT_CSV).name

    print("\n" + "=" * 40)
    print("  AUTOMATIC CSV DISTRIBUTION")
    print("=" * 40)

    for target_dir in target_dirs:
        target_path = Path(target_dir) / output_filename
        try:
            # Create directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy CSV
            shutil.copy2(OUTPUT_CSV, target_path)
            print(f"  ✓ Copied to: {target_path}")
        except Exception as e:
            print(f"  ✗ Failed to copy to {target_path}: {e}")

    print("=" * 40)

if __name__ == "__main__":
    main()