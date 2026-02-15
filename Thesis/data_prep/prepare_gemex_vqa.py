#!/usr/bin/env python3
"""
GEMeX to MIMIC-CXR Mapping Utility (Census Edition)
===================================================
Description:
    Aligns metadata from the GEMeX-VQA dataset with local MIMIC-CXR images.
    
    UPDATES:
    - Integrity Fix: Preserves 'choices' and ensures 'question' is not corrupted by CSV formatting.
    - Diagnostic Census: Explicitly counts p10-p19 coverage.
    - Direct HTTP: Uses robust download method.

Usage:
    python3 prepare_gemex_vqa.py
"""

import os
import json
import random
import requests
import pandas as pd
import csv
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import list_repo_files

# --- CONFIGURATION ---
REPO_ID = "BoKelvin/GEMeX-VQA"
MIMIC_ROOT_DIR = Path("/datasets/MIMIC-CXR/files")  
OUTPUT_CSV = "../gemex_VQA_mimic_mapped.csv"
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm', '.webp'}

# MAX QUESTIONS PER IMAGE:
MAX_QUESTIONS_PER_IMAGE = None # Set to an integer (e.g., 5) to limit questions per image, or None for no limit.
RANDOM_SEED = 42 # For reproducibility

def download_file_direct(repo_id, filename, local_dir):
    """Bypasses LFS pointers by forcing raw HTTP download."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    local_path = os.path.join(local_dir, filename)
    
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        if size_mb > 1.0: # Assume >1MB is a real file
            print(f"   [SKIP] File {filename} seems valid locally ({size_mb:.2f} MB).")
            return local_path
    
    print(f"   [HTTP] Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f, tqdm(
            desc=filename, total=int(r.headers.get('content-length', 0)),
            unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    return local_path

def main():
    print("[INFO] Setting random seed for reproducibility...")
    random.seed(RANDOM_SEED)

    # --- 1. INDEXING LOCAL FILES ---
    if not MIMIC_ROOT_DIR.exists():
        print(f"[CRITICAL] Directory not found: {MIMIC_ROOT_DIR}")
        return

    print(f"[INFO] Indexing local files in {MIMIC_ROOT_DIR}...")
    local_files_map = {}
    folder_census = {} 
    
    for f in tqdm(MIMIC_ROOT_DIR.rglob('*'), desc="Indexing Filesystem"):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            local_files_map[f.stem] = str(f)
            
            # DIAGNOSTIC: Check pXX folder
            for part in f.parts:
                if len(part) == 3 and part.startswith('p1') and part[1:].isdigit():
                    folder_census[part] = folder_census.get(part, 0) + 1
                    break 

    print(f"[INFO] Indexed {len(local_files_map)} local images.")
    
    # --- PRINT CENSUS REPORT ---
    print("\n" + "="*40)
    print("      LOCAL DATASET CENSUS REPORT")
    print("="*40)
    if not folder_census:
        print("[WARNING] Could not detect p10-p19 structure. Check paths.")
    else:
        keys = sorted(folder_census.keys())
        for k in keys:
            print(f"  Folder {k}/ : {folder_census[k]} images")
    print("="*40 + "\n")

    if len(local_files_map) == 0:
        print("[CRITICAL] No images found locally.")
        return

    # --- 2. FILE LISTING ---
    print(f"[INFO] Fetching file list from {REPO_ID}...")
    try:
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
    except Exception as e:
        print(f"[CRITICAL] Failed to list repo files: {e}")
        return

    download_dir = Path("gemex_cache")
    download_dir.mkdir(exist_ok=True)
    matched_rows = []

    # --- 3. DOWNLOAD & MATCH ---
    for file_name in jsonl_files:
        print(f"\n-> Processing {file_name}...")
        try:
            local_path = download_file_direct(REPO_ID, file_name, str(download_dir))
            
            print("   [READ] Scanning rows...")
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Parsing JSON", unit=" lines"):
                    if not line.strip(): continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError: continue
                    
                    # --- INTEGRITY FIX: MANTENERE 'CHOICES' ---
                    # if 'choices' in sample: of sample['choices'] 
                    # ------------------------------------------

                    raw_path = sample.get('image_path') or sample.get('image_id')
                    if not raw_path: continue
                    
                    stem_id = Path(str(raw_path)).stem
                    
                    if stem_id in local_files_map:
                        sample['image_path'] = local_files_map[stem_id] 
                        sample['stem_id'] = stem_id 
                        sample['original_hf_path'] = raw_path
                        matched_rows.append(sample)
                        
        except Exception as e:
            print(f"   [ERROR] Failed to process {file_name}: {e}")

    print(f"\n[INFO] Total matches found: {len(matched_rows)}")
    
    if len(matched_rows) == 0:
        print("[WARNING] No matches found.")
        return

    # --- 4. EXPORT ---
    print("-> Creating DataFrame and applying sampling...")
    df_gemex = pd.DataFrame(matched_rows)
    final_rows = []
    
    if MAX_QUESTIONS_PER_IMAGE:
        print(f"-> Sampling max {MAX_QUESTIONS_PER_IMAGE} Qs/img...")
        grouped = df_gemex.groupby('stem_id')
        for stem_id, group in tqdm(grouped, desc="Sampling"):
            if len(group) > MAX_QUESTIONS_PER_IMAGE:
                sampled = group.sample(n=MAX_QUESTIONS_PER_IMAGE, random_state=RANDOM_SEED)
            else:
                sampled = group
            final_rows.extend(sampled.to_dict('records'))
    else:
        final_rows = df_gemex.to_dict('records')

    df_final = pd.DataFrame(final_rows)
    if 'stem_id' in df_final.columns:
        df_final.drop(columns=['stem_id'], inplace=True)
    
    # --- INTEGRITY FIX: SALVATAGGIO CSV ROBUSTO ---
    df_final.to_csv(
        OUTPUT_CSV,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\'
    )

    print("-" * 40)
    print(f"[SUCCESS] Dataset saved to {OUTPUT_CSV}")
    print(f"[STATS]   Unique Images: {df_final['image_path'].nunique()}")
    print(f"[STATS]   Total QA Pairs: {len(df_final)}")
    print("-" * 40)

    # --- AUTOMATIC DISTRIBUTION TO MODULE DIRECTORIES ---
    import shutil

    # Define target directories for this CSV
    target_dirs = [
        "../vqa",
        "../preprocessing/bounding_box"
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