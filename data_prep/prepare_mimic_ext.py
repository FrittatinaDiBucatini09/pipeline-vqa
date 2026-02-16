import json
import csv
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION DEFAULTS ---
DEFAULT_MIMIC_EXT_DATASET_DIR = "/datasets/MIMIC-Ext-MIMIC-CXR-VQA/dataset"
DEFAULT_OUTPUT_CSV = "../mimic_ext_mapped.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='MIMIC-Ext-MIMIC-CXR-VQA Preparation Utility')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_MIMIC_EXT_DATASET_DIR,
                       help='Root directory of MIMIC-Ext dataset')
    parser.add_argument('--output_csv', type=str, default=DEFAULT_OUTPUT_CSV,
                       help='Output CSV file path')
    parser.add_argument('--max_samples', type=int, default=50000,
                       help='Maximum number of samples to process')
    return parser.parse_args()

def main():
    args = parse_args()
    MIMIC_EXT_DATASET_DIR = Path(args.dataset_dir)
    OUTPUT_CSV = args.output_csv
    MAX_SAMPLES = args.max_samples

    target_file = MIMIC_EXT_DATASET_DIR / "train.json"
    if not target_file.exists():
        print(f"[ERROR] File not found: {target_file}")
        return

    print(f"[INFO] Loading Text-Only VQA dataset from {target_file}...")
    with open(target_file, 'r') as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} samples. Starting conversion...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'question', 'visual_locations', 
            'visual_regions', 'answer_text', 'grade'
        ])
        writer.writeheader()
        
        saved_count = 0
        
        for item in tqdm(data):
            
            # 1. Path Images
            raw_path = item.get('image_path')
            if not raw_path: continue
            
            if not raw_path.startswith("files/"):
                final_path = f"files/{raw_path}"
            else:
                final_path = raw_path

            # 2. Question
            question = item.get('question')
            
            # 3. Answer
            ans_list = item.get('answer', [])
            answer_text = ans_list[0] if isinstance(ans_list, list) and ans_list else str(ans_list)
            
            # 4. Writing CSV
            writer.writerow({
                'image_path': final_path,
                'question': question,
                'visual_locations': "[]", 
                'visual_regions': "[]",   
                'answer_text': answer_text,
                'grade': 'Unknown'
            })
            
            saved_count += 1
            if MAX_SAMPLES and saved_count >= MAX_SAMPLES: break

    print(f"[SUCCESS] Mapped {saved_count} samples to {OUTPUT_CSV}")
    print("[WARNING] This dataset has NO GROUND TRUTH BOXES.")
    print("          Run pipeline with MODE='inference' only.")

    # --- AUTOMATIC DISTRIBUTION TO MODULE DIRECTORIES ---
    import shutil

    # Define target directories for this CSV
    target_dirs = [
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