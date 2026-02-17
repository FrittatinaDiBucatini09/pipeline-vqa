"""
MedCLIP Agentic Routing: NLP Query Expansion Middleware for Medical VQA.

Pipeline:
1. Load models once (SciSpacy on CPU, Gemma-2-2B-it on GPU)
2. For each row in the dataset:
   a. Evaluate query quality via SciSpacy entity extraction
   b. If query is too brief/vague, expand it via Gemma-2-2B-it
3. Output expanded_queries.jsonl with enriched question column
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- Local Imports ---
CURRENT_SCRIPT_PATH = Path(__file__).resolve().parent
if str(CURRENT_SCRIPT_PATH) not in sys.path:
    sys.path.append(str(CURRENT_SCRIPT_PATH))

try:
    from utils import (
        evaluate_query_quality,
        expand_query,
        load_gemma,
        load_scispacy,
    )
except ImportError as e:
    sys.exit(f"\n[CRITICAL ERROR] Local dependencies missing.\nDetail: {e}\n")


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV, JSONL, or Parquet file into a DataFrame."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


# ==============================================================================
# 2. MODEL INITIALIZATION
# ==============================================================================

def initialize_models() -> Dict[str, Any]:
    """
    Load NLP models once at startup.

    Memory budget (RTX 3090, 24GB):
    - SciSpacy en_core_sci_sm: CPU only (0 VRAM)
    - Gemma-2-2B-it float16: ~5-6 GB VRAM
    - Total: ~5-6 GB VRAM
    """
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    # 1. SciSpacy (CPU)
    nlp = load_scispacy()

    # 2. Gemma-2-2B-it (GPU, float16)
    gemma_model, gemma_tokenizer = load_gemma()

    vram_allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"\n[INFO] Total VRAM allocated: {vram_allocated:.2f} GB")
    print("=" * 60 + "\n")

    return {
        "nlp": nlp,
        "gemma_model": gemma_model,
        "gemma_tokenizer": gemma_tokenizer,
    }


# ==============================================================================
# 3. CORE PROCESSING LOOP
# ==============================================================================

def process_dataset(args, models: Dict[str, Any]) -> None:
    """Process the dataset row-by-row with query evaluation and expansion."""
    # Unpack models
    nlp = models["nlp"]
    gemma_model = models["gemma_model"]
    gemma_tokenizer = models["gemma_tokenizer"]

    # Load dataset
    df = load_dataframe(args.metadata_file)
    print(f"[INFO] Loaded dataset: {args.metadata_file} ({len(df)} rows)")

    if args.stop_after:
        df = df.head(args.stop_after)
        print(f"[INFO] Limited to first {args.stop_after} rows (debug mode)")

    # Prepare output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "expanded_queries.jsonl"

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
                query = str(row.get(args.text_col, ""))

                if not query:
                    continue

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

                # --- Build output record with all original columns ---
                record = {}
                for col in row.index:
                    val = row[col]
                    if isinstance(val, (np.integer,)):
                        val = int(val)
                    elif isinstance(val, (np.floating,)):
                        val = float(val)
                    elif pd.isna(val):
                        val = ""
                    record[col] = val

                # Overwrite question column with expanded text
                record["original_question"] = query
                record[args.text_col] = final_query
                record["was_expanded"] = was_expanded
                record["entities_detected"] = entities

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

    print(f"\n[INFO] Output written to: {jsonl_path}")


# ==============================================================================
# 4. CLI ARGUMENT PARSER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedCLIP Agentic Routing: NLP Query Expansion Middleware for Medical VQA"
    )

    # --- Path Arguments ---
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for output expanded queries.")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to CSV/JSONL/Parquet metadata file.")

    # --- CSV Column Mapping ---
    parser.add_argument("--path_col", type=str, default="image_path",
                        help="Column name for image paths in metadata (passed through).")
    parser.add_argument("--text_col", type=str, default="question",
                        help="Column name for text queries in metadata.")

    # --- Execution Control ---
    parser.add_argument("--stop_after", type=int, default=None,
                        help="Limit processing to first N rows (debug mode).")

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
    models = initialize_models()

    # Process dataset
    process_dataset(args, models)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    print("\n[INFO] Initializing MedCLIP Agentic Routing Pipeline...")
    main()
