"""
MedCLIP Agentic Routing: Query-Aware Preprocessing for Medical VQA.

Pipeline:
1. Load all models once (SciSpacy on CPU, Gemma-2-2B-it + BiomedCLIP on GPU)
2. For each image-query pair:
   a. Evaluate query quality via SciSpacy entity extraction
   b. If query is too brief/vague, expand it via Gemma-2-2B-it
   c. Generate bounding boxes using BiomedCLIP + GradCAM
3. Output predictions.jsonl + vqa_manifest.csv
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# --- Local Imports ---
CURRENT_SCRIPT_PATH = Path(__file__).resolve().parent
if str(CURRENT_SCRIPT_PATH) not in sys.path:
    sys.path.append(str(CURRENT_SCRIPT_PATH))

try:
    import open_clip

    from utils import (
        CAMWrapper,
        cam_to_bboxes,
        evaluate_query_quality,
        expand_query,
        generate_cam_bbox,
        load_biomed_clip,
        load_gemma,
        load_scispacy,
        reshape_transform,
    )
except ImportError as e:
    sys.exit(f"\n[CRITICAL ERROR] Local dependencies missing.\nDetail: {e}\n")


# ==============================================================================
# 1. GLOBAL CONSTANTS
# ==============================================================================
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ==============================================================================
# 3. VQA MANIFEST GENERATION
# ==============================================================================

def generate_vqa_manifest(
    output_root: Path,
    question_col: str = "question",
    answer_col: str = "answer",
) -> None:
    """
    Read predictions.jsonl and generate a VQA-ready CSV manifest.

    Each JSONL record contains the full original CSV row data plus the
    bounding box predictions. This function extracts the columns needed
    by the VQA generation stage.
    """
    jsonl_path = output_root / "predictions.jsonl"
    manifest_path = output_root / "vqa_manifest.csv"

    if not jsonl_path.exists():
        print(f"[WARNING] predictions.jsonl not found at {jsonl_path}. Skipping manifest.")
        return

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_path = entry.get("image_path", "")
            question = entry.get(question_col, "")
            answer = entry.get(answer_col, "")

            if not image_path:
                continue

            records.append({
                "image_path": image_path,
                "question": question,
                "answer": answer,
            })

    if not records:
        print("[WARNING] No valid records in predictions.jsonl. VQA manifest is empty.")
        return

    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print(f"[INFO] VQA manifest generated: {manifest_path} ({len(df)} rows)")


# ==============================================================================
# 4. MODEL INITIALIZATION
# ==============================================================================

def initialize_models(args) -> Dict[str, Any]:
    """
    Load all three models once at startup.

    Memory budget (RTX 3090, 24GB):
    - SciSpacy en_core_sci_sm: CPU only (0 VRAM)
    - Gemma-2-2B-it float16: ~5-6 GB VRAM
    - BiomedCLIP ViT-B/16: ~1 GB VRAM
    - GradCAM overhead: ~1-2 GB VRAM
    - Total: ~8-9 GB VRAM (well within 24GB)
    """
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    # 1. SciSpacy (CPU)
    nlp = load_scispacy()

    # 2. Gemma-2-2B-it (GPU, float16)
    gemma_model, gemma_tokenizer = load_gemma()

    # 3. BiomedCLIP (GPU)
    clip_model, clip_preprocess, clip_tokenizer = load_biomed_clip(args.model_name)

    # 4. Initialize GradCAM wrapper
    # Target the last transformer block for ViT
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "trunk"):
        # OpenCLIP BiomedCLIP structure
        target_layers = [clip_model.visual.trunk.blocks[-1].norm1]
    elif hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer"):
        target_layers = [clip_model.visual.transformer.resblocks[-1].ln_1]
    else:
        sys.exit("[CRITICAL] Cannot determine BiomedCLIP visual backbone structure.")

    cam_wrapper = CAMWrapper(
        model=clip_model,
        target_layers=target_layers,
        tokenizer=clip_tokenizer,
        cam_version=args.cam_version,
        preprocess=clip_preprocess,
        cam_trans=reshape_transform,
    )

    vram_allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"\n[INFO] Total VRAM allocated: {vram_allocated:.2f} GB")
    print("=" * 60 + "\n")

    return {
        "nlp": nlp,
        "gemma_model": gemma_model,
        "gemma_tokenizer": gemma_tokenizer,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "clip_tokenizer": clip_tokenizer,
        "cam_wrapper": cam_wrapper,
    }


# ==============================================================================
# 5. CORE PROCESSING LOOP
# ==============================================================================

def process_dataset(args, models: Dict[str, Any]) -> None:
    """Process the dataset row-by-row with the agentic routing logic."""
    # Unpack models
    nlp = models["nlp"]
    gemma_model = models["gemma_model"]
    gemma_tokenizer = models["gemma_tokenizer"]
    cam_wrapper = models["cam_wrapper"]
    clip_preprocess = models["clip_preprocess"]

    # Load dataset
    df = load_dataframe(args.metadata_file)
    print(f"[INFO] Loaded dataset: {args.metadata_file} ({len(df)} rows)")

    if args.stop_after:
        df = df.head(args.stop_after)
        print(f"[INFO] Limited to first {args.stop_after} rows (debug mode)")

    # Prepare output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "predictions.jsonl"

    # Counters
    total = len(df)
    processed = 0
    expanded_count = 0
    failed_count = 0
    errors = []
    start_time = time.time()

    # Initialize WandB if enabled
    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project="medclip-routing",
                config=vars(args),
                mode=args.wandb_mode,
            )
        except Exception as e:
            print(f"[WARNING] WandB init failed: {e}. Continuing without logging.")

    print(f"\n[INFO] Starting processing: {total} rows")
    print(f"[INFO] Routing thresholds: entities >= {args.entity_threshold}, words >= {args.word_threshold}")
    print(f"[INFO] Output directory: {output_root}")

    with open(jsonl_path, "w") as jsonl_file:
        for idx, row in tqdm(df.iterrows(), total=total, desc="Routing"):
            try:
                # --- Extract row data ---
                image_path = str(row.get(args.path_col, ""))
                query = str(row.get(args.text_col, ""))
                answer = str(row.get("answer", ""))

                if not image_path or not query:
                    continue

                # Resolve image path
                if not os.path.isabs(image_path):
                    full_image_path = os.path.join(args.input_dir, image_path)
                else:
                    full_image_path = image_path

                if not os.path.exists(full_image_path):
                    errors.append(f"Image not found: {full_image_path}")
                    failed_count += 1
                    continue

                # --- Load image ---
                img = Image.open(full_image_path).convert("RGB")
                image_shape = (img.height, img.width)

                # --- Step 1: Query Quality Evaluation ---
                is_detailed, entities = evaluate_query_quality(
                    nlp, query,
                    entity_threshold=args.entity_threshold,
                    word_threshold=args.word_threshold,
                )

                # --- Step 2: Conditional Query Expansion ---
                final_query = query
                was_expanded = False
                if not is_detailed:
                    final_query = expand_query(
                        gemma_model,
                        gemma_tokenizer,
                        query,
                        max_new_tokens=args.gemma_max_new_tokens,
                    )
                    was_expanded = True
                    expanded_count += 1

                # --- Step 3: GradCAM Bounding Box Generation ---
                bboxes = generate_cam_bbox(
                    cam_wrapper=cam_wrapper,
                    preprocess=clip_preprocess,
                    image=img,
                    query=final_query,
                    threshold=args.cam_threshold,
                    multi_box=args.multi_box,
                    min_box_area_ratio=args.min_box_area,
                )

                # --- Write JSONL record ---
                record = {
                    "image_path": full_image_path,
                    "original_query": query,
                    "final_query": final_query,
                    "was_expanded": was_expanded,
                    "entities_detected": entities,
                    "bboxes": bboxes,
                    "question": query,
                    "answer": answer,
                }

                # Preserve additional columns from original data
                for col in ["type", "reason", "visual_regions", "visual_locations",
                            "ori_report", "q_id", "row_id", "original_hf_path", "choices"]:
                    if col in row.index:
                        val = row[col]
                        # Convert numpy types to Python native for JSON
                        if isinstance(val, (np.integer,)):
                            val = int(val)
                        elif isinstance(val, (np.floating,)):
                            val = float(val)
                        elif pd.isna(val):
                            val = ""
                        record[col] = val

                jsonl_file.write(json.dumps(record) + "\n")
                processed += 1

                # WandB logging (periodic)
                if wandb_run and processed % 100 == 0:
                    wandb_run.log({
                        "processed": processed,
                        "expanded": expanded_count,
                        "failed": failed_count,
                        "expansion_rate": expanded_count / max(processed, 1),
                    })

            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                failed_count += 1
                continue

    # --- Summary ---
    elapsed = time.time() - start_time
    throughput = processed / max(elapsed, 0.01)

    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total rows    : {total}")
    print(f"Processed     : {processed}")
    print(f"Failed        : {failed_count}")
    print(f"Expanded      : {expanded_count} ({100 * expanded_count / max(processed, 1):.1f}%)")
    print(f"Throughput    : {throughput:.2f} rows/s")
    print(f"Elapsed       : {elapsed:.1f}s")

    if errors:
        error_log = output_root / "errors.log"
        with open(error_log, "w") as f:
            f.write("\n".join(errors))
        print(f"[WARNING] {len(errors)} errors logged to {error_log}")

    if wandb_run:
        wandb_run.log({
            "final_processed": processed,
            "final_expanded": expanded_count,
            "final_failed": failed_count,
            "expansion_rate": expanded_count / max(processed, 1),
            "throughput": throughput,
        })
        wandb_run.finish()

    # Generate VQA manifest for downstream stages
    generate_vqa_manifest(
        output_root,
        question_col=args.text_col,
        answer_col="answer",
    )


# ==============================================================================
# 6. CLI ARGUMENT PARSER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedCLIP Agentic Routing: Query-Aware Preprocessing for Medical VQA"
    )

    # --- Path Arguments ---
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing medical images.")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for output predictions and manifest.")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to CSV/Parquet metadata file.")

    # --- CSV Column Mapping ---
    parser.add_argument("--path_col", type=str, default="image_path",
                        help="Column name for image paths in metadata.")
    parser.add_argument("--text_col", type=str, default="question",
                        help="Column name for text queries in metadata.")

    # --- Execution Control ---
    parser.add_argument("--stop_after", type=int, default=None,
                        help="Limit processing to first N rows (debug mode).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (reserved for future batched processing).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers.")

    # --- BiomedCLIP + GradCAM ---
    parser.add_argument("--model_name", type=str,
                        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        help="BiomedCLIP model identifier for OpenCLIP.")
    parser.add_argument("--cam_version", type=str, default="gScoreCAM",
                        help="GradCAM variant to use.")
    parser.add_argument("--cam_threshold", type=float, default=0.5,
                        help="Activation threshold for CAM binarization.")
    parser.add_argument("--multi_box", action="store_true", default=True,
                        help="Enable multi-box detection.")
    parser.add_argument("--no_multi_box", action="store_false", dest="multi_box",
                        help="Disable multi-box detection (single largest box only).")
    parser.add_argument("--min_box_area", type=float, default=0.005,
                        help="Minimum box area as fraction of image area.")

    # --- Routing Thresholds ---
    parser.add_argument("--entity_threshold", type=int, default=2,
                        help="Minimum clinical entities for a query to be 'detailed'.")
    parser.add_argument("--word_threshold", type=int, default=5,
                        help="Minimum word count for a query to be 'detailed'.")

    # --- Gemma Query Expansion ---
    parser.add_argument("--gemma_max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for Gemma query expansion.")

    # --- Experiment Tracking ---
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"],
                        help="WandB logging mode.")

    args = parser.parse_args()
    print("\n[CONFIG] Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Initialize models
    models = initialize_models(args)

    # Process dataset
    process_dataset(args, models)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    print("\n[INFO] Initializing MedCLIP Agentic Routing Pipeline...")
    main()
