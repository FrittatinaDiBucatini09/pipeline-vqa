#!/usr/bin/env python3
import json
import csv
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import tempfile

# Add current directory to path to import utils
sys.path.append(str(Path(__file__).parent))
import utils

# --- CONFIGURATION DEFAULTS ---
DEFAULT_MIMIC_EXT_DATASET_DIR = "/datasets/MIMIC-Ext-MIMIC-CXR-VQA/dataset"
DEFAULT_MIMIC_CXR_ROOT = "/datasets/MIMIC-CXR"
BASE_FILENAME = "mimic_ext_mapped"

def parse_args():
    parser = argparse.ArgumentParser(description='MIMIC-Ext-MIMIC-CXR-VQA Preparation Utility')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_MIMIC_EXT_DATASET_DIR,
                       help='Root directory of MIMIC-Ext dataset')
    parser.add_argument('--mimic_cxr_root', type=str, default=DEFAULT_MIMIC_CXR_ROOT,
                       help='Root directory of MIMIC-CXR images (for path validation)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit the number of samples (default: None/All)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()
    MIMIC_EXT_DATASET_DIR = Path(args.dataset_dir)
    LIMIT = args.limit
    SEED = args.seed

    # Generate output filename based on limit
    if LIMIT:
        OUTPUT_FILENAME = f"{BASE_FILENAME}_{LIMIT}_samples.csv"
        print(f"[INFO] Limit set to {LIMIT} samples. Output: {OUTPUT_FILENAME}")
    else:
        OUTPUT_FILENAME = f"{BASE_FILENAME}.csv"
        print(f"[INFO] No limit set. Output: {OUTPUT_FILENAME}")
        
    OUTPUT_CSV = Path(tempfile.gettempdir()) / OUTPUT_FILENAME

    target_file = MIMIC_EXT_DATASET_DIR / "train.json"
    if not target_file.exists():
        print(f"[ERROR] File not found: {target_file}")
        return

    print(f"[INFO] Loading Text-Only VQA dataset from {target_file}...")
    with open(target_file, 'r') as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} total samples.")

    MIMIC_CXR_ROOT = Path(args.mimic_cxr_root)

    # Shuffle the full dataset so we draw uniformly when applying LIMIT
    random.seed(SEED)
    random.shuffle(data)

    print(f"[INFO] Converting samples (validating image paths against {MIMIC_CXR_ROOT})...")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'question', 'visual_locations',
            'visual_regions', 'answer_text', 'grade'
        ])
        writer.writeheader()

        saved_count = 0
        skipped_count = 0

        for item in tqdm(data, desc="Processing"):
            # Stop when we've collected enough valid samples
            if LIMIT and saved_count >= LIMIT:
                break

            # 1. Path Images
            raw_path = item.get('image_path')
            if not raw_path:
                continue

            if not raw_path.startswith("files/"):
                final_path = f"files/{raw_path}"
            else:
                final_path = raw_path

            # 2. Validate image exists on disk
            full_image_path = MIMIC_CXR_ROOT / final_path
            if not full_image_path.exists():
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  [SKIP] Image not found: {full_image_path}")
                continue

            # 3. Question
            question = item.get('question')

            # 4. Answer
            ans_list = item.get('answer', [])
            answer_text = ans_list[0] if isinstance(ans_list, list) and ans_list else str(ans_list)

            # 5. Writing CSV
            writer.writerow({
                'image_path': final_path,
                'question': question,
                'visual_locations': "[]",
                'visual_regions': "[]",
                'answer_text': answer_text,
                'grade': 'Unknown'
            })

            saved_count += 1

    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} samples with missing images.")
    print(f"[SUCCESS] Saved {saved_count} validated samples to {OUTPUT_CSV}")
    print("[WARNING] This dataset has NO GROUND TRUTH BOXES.")
    print("          Run pipeline with MODE='inference' only.")

    # --- AUTOMATIC DISTRIBUTION ---
    # The user wants copies in: ../vqa, ../preprocessing/bounding_box, etc.
    # We use our utils for this.
    
    # Define targets (relative to this script's location in data_prep/)
    # TARGET_DIRS are: "../vqa", "../preprocessing/..."
    
    utils.distribute_file(OUTPUT_CSV)

if __name__ == "__main__":
    main()