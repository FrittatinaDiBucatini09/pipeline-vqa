#!/usr/bin/env python3
"""
evaluate_bbox.py — IoU Metrics for Bounding Box Predictions

Standalone evaluation script for the GEMeX WSOL Pipeline.
Compares predicted bounding boxes against ground-truth `visual_locations`.

IMPORTANT: This script is SELF-CONTAINED (No-Cross-Import Rule).
           It does NOT import anything from `generation/`.

Usage:
    python evaluate_bbox.py \
        --predictions_file ../generation/results/predictions.jsonl \
        --input_dir /dataset/mimic-cxr-jpg/2.0.0 \
        --output_dir ./results
"""

import argparse
import ast
import json
import re
import sys
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("[CRITICAL] Pillow not installed. Run: pip install Pillow")


# ==============================================================================
# 1. DUPLICATED UTILITIES (from generation/scripts/bbox_preprocessing.py)
#    Reason: No-Cross-Import Rule — evaluation must be fully self-contained.
# ==============================================================================

def parse_gold_coordinates(raw_value) -> List[List[float]]:
    """
    Parses strings containing nested lists like "[[x1,y1,x2,y2], [x1,y1,x2,y2]]"
    Robust against CSV formatting artifacts and various data types.
    """
    # Handle empty/NaN cases
    if raw_value is None or raw_value == "" or str(raw_value).lower() == "nan":
        return []

    # Case 1: Already loaded as list (e.g., from Parquet or JSON)
    if isinstance(raw_value, list):
        # Sub-case: flat list [x,y,x,y] -> wrap in [[x,y,x,y]]
        if len(raw_value) == 4 and all(isinstance(n, (int, float)) for n in raw_value):
            return [raw_value]
        return raw_value  # Already properly formatted

    # Case 2: String representation - try parsing
    try:
        cleaned = str(raw_value).strip()

        # Attempt 1: AST parsing (handles Python-style lists)
        parsed = ast.literal_eval(cleaned)

        if isinstance(parsed, list):
            if not parsed:
                return []
            if isinstance(parsed[0], list):
                return parsed
            if len(parsed) == 4:
                return [parsed]

    except (ValueError, SyntaxError):
        # Attempt 2: Regex fallback for messy formats
        try:
            matches = re.findall(
                r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,'
                r'\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
                cleaned
            )
            if matches:
                return [[float(x) for x in m] for m in matches]
        except Exception:
            pass

    return []


# ==============================================================================
# 2. IoU COMPUTATION
# ==============================================================================

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Both boxes must be in the same coordinate space: [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def scale_gold_to_pixel(gold_box: List[float], img_w: int, img_h: int,
                        ref_dim: float = 224.0) -> List[float]:
    """
    Scales a gold bounding box from 224-reference space to pixel space.
    GEMeX gold annotations are encoded relative to a 224x224 grid.
    """
    scale_x = img_w / ref_dim
    scale_y = img_h / ref_dim
    return [
        gold_box[0] * scale_x,
        gold_box[1] * scale_y,
        gold_box[2] * scale_x,
        gold_box[3] * scale_y,
    ]


def best_match_iou(gold_boxes: List[List[float]],
                   pred_boxes: List[List[float]]) -> Tuple[float, List[float]]:
    """
    For each gold box, finds the predicted box with the highest IoU (greedy matching).
    
    Returns:
        - mean_iou: Average of the best IoU for each gold box.
        - per_gold_ious: List of best IoU values, one per gold box.
    """
    if not gold_boxes:
        return 0.0, []

    if not pred_boxes:
        return 0.0, [0.0] * len(gold_boxes)

    per_gold_ious = []
    for g_box in gold_boxes:
        best_iou = 0.0
        for p_box in pred_boxes:
            iou = compute_iou(g_box, p_box)
            if iou > best_iou:
                best_iou = iou
        per_gold_ious.append(best_iou)

    mean_iou = float(np.mean(per_gold_ious)) if per_gold_ious else 0.0
    return mean_iou, per_gold_ious


# ==============================================================================
# 3. IMAGE DIMENSION RESOLUTION
# ==============================================================================

def get_image_dimensions(image_path_field: str, input_dir: Path) -> Optional[Tuple[int, int]]:
    """
    Resolves the image file and returns (width, height).
    
    The image_path in predictions.jsonl is a relative path like:
      files/p10/p10268877/.../image_idx1.jpg
    
    Strategy:
      1. Look for the ORIGINAL image (strip _idxN suffix) under input_dir.
      2. If not found, try the path as-is.
    """
    rel_path = Path(image_path_field)
    
    # Strip the _idxN suffix added by the generation pipeline
    # e.g., "image_idx1.jpg" -> "image.jpg"
    stem = rel_path.stem
    suffix = rel_path.suffix
    clean_stem = re.sub(r'_idx\d+$', '', stem)
    original_filename = f"{clean_stem}{suffix}"
    original_rel_path = rel_path.parent / original_filename
    
    # Try original path first (most common case)
    candidates = [
        input_dir / original_rel_path,
        input_dir / rel_path,
        Path(image_path_field),  # Absolute path fallback
    ]
    
    for candidate in candidates:
        if candidate.exists():
            try:
                with Image.open(candidate) as img:
                    return img.size  # (width, height)
            except Exception:
                continue
    
    return None


# ==============================================================================
# 4. MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate(args):
    """Main evaluation loop."""
    predictions_path = Path(args.predictions_file)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        sys.exit(f"[ERROR] Predictions file not found: {predictions_path}")

    # --- Load Predictions ---
    records = []
    with open(predictions_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed line {line_num}: {e}")

    print(f"[INFO] Loaded {len(records)} predictions from: {predictions_path}")

    # --- Process Each Record ---
    IoU_THRESHOLDS = [0.1, 0.25, 0.5, 0.75]
    
    all_ious = []           # Flat list of all per-gold-box IoUs
    sample_results = []     # Per-record results for CSV
    skipped = 0
    dim_cache = {}          # Cache image dimensions by original path

    # Per-type aggregation
    type_ious = {}  # { "location": [ious], "abnormality": [ious] }

    for idx, record in enumerate(records):
        # 1. Parse gold boxes from visual_locations
        raw_gold = record.get("visual_locations", "[]")
        gold_boxes_224 = parse_gold_coordinates(raw_gold)

        # 2. Parse predicted boxes (already native JSON lists)
        pred_boxes = record.get("predicted_boxes", [])
        if isinstance(pred_boxes, str):
            pred_boxes = parse_gold_coordinates(pred_boxes)
        
        # Normalize format: handle both [[x1,y1,x2,y2]] and [[[x1,y1,x2,y2], "label"]]
        # Some strategies (e.g., S_regions_composite) output nested lists with labels
        normalized_pred_boxes = []
        for box in pred_boxes:
            if isinstance(box, list) and len(box) > 0:
                # Check if this is nested: [[coords], "label"]
                if isinstance(box[0], list) and len(box[0]) == 4:
                    # Extract coordinates from nested structure
                    normalized_pred_boxes.append(box[0])
                elif len(box) == 4 and all(isinstance(c, (int, float)) for c in box):
                    # Already flat coordinates
                    normalized_pred_boxes.append(box)
                else:
                    # Unexpected format, try to use as-is
                    normalized_pred_boxes.append(box)
        pred_boxes = normalized_pred_boxes


        # 3. Get image dimensions for coordinate scaling
        image_path_field = record.get("image_path", "")
        
        # Use cache to avoid re-opening images from the same original
        cache_key = re.sub(r'_idx\d+', '', image_path_field)
        if cache_key in dim_cache:
            dims = dim_cache[cache_key]
        else:
            dims = get_image_dimensions(image_path_field, input_dir)
            dim_cache[cache_key] = dims

        if dims is None:
            if gold_boxes_224:
                print(f"[WARNING] Image not found for record {idx}: {image_path_field}. Skipping.")
                skipped += 1
            continue

        img_w, img_h = dims

        # 4. Scale gold boxes from 224-space to pixel space
        gold_boxes_pixel = [
            scale_gold_to_pixel(g, img_w, img_h, ref_dim=args.ref_dim)
            for g in gold_boxes_224
        ]

        # 5. Compute IoU
        mean_iou, per_gold = best_match_iou(gold_boxes_pixel, pred_boxes)

        # 6. Aggregate
        all_ious.extend(per_gold)

        q_type = record.get("type", "unknown")
        if q_type not in type_ious:
            type_ious[q_type] = []
        type_ious[q_type].extend(per_gold)

        # 7. Per-sample record
        question = record.get("question", "")
        sample_results.append({
            "index": idx,
            "question": question[:80],
            "type": q_type,
            "num_gold": len(gold_boxes_pixel),
            "num_pred": len(pred_boxes),
            "mean_iou": round(mean_iou, 4),
            "per_gold_ious": str([round(v, 4) for v in per_gold]),
            "gold_boxes_pixel": str([[int(c) for c in b] for b in gold_boxes_pixel]),
            "predicted_boxes": str(pred_boxes),
        })

    # --- Compute Summary Statistics ---
    all_ious_np = np.array(all_ious) if all_ious else np.array([0.0])

    summary = {
        "total_records": len(records),
        "evaluated_records": len(sample_results),
        "skipped_records": skipped,
        "total_gold_boxes": len(all_ious),
        "mean_iou": float(np.mean(all_ious_np)),
        "median_iou": float(np.median(all_ious_np)),
        "std_iou": float(np.std(all_ious_np)),
    }

    # Accuracy @ thresholds
    for t in IoU_THRESHOLDS:
        hits = int(np.sum(all_ious_np >= t))
        acc = hits / len(all_ious_np) if len(all_ious_np) > 0 else 0.0
        summary[f"acc@{t}"] = acc
        summary[f"hits@{t}"] = hits

    # --- Per-Type Breakdown ---
    type_summary = {}
    for t_name, t_ious in type_ious.items():
        t_np = np.array(t_ious) if t_ious else np.array([0.0])
        type_summary[t_name] = {
            "count": len(t_ious),
            "mean_iou": float(np.mean(t_np)),
            "acc@0.5": float(np.sum(t_np >= 0.5) / len(t_np)) if len(t_np) > 0 else 0.0,
        }

    # --- Write Report ---
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 56 + "\n")
        f.write("     GEMeX BBOX EVALUATION REPORT (IoU Metrics)\n")
        f.write("=" * 56 + "\n\n")

        f.write(f"  Predictions File : {predictions_path}\n")
        f.write(f"  Input Dir        : {input_dir}\n")
        f.write(f"  Reference Dim    : {args.ref_dim}\n\n")

        f.write("-" * 56 + "\n")
        f.write("  SUMMARY\n")
        f.write("-" * 56 + "\n")
        f.write(f"  Total Records    : {summary['total_records']}\n")
        f.write(f"  Evaluated        : {summary['evaluated_records']}\n")
        f.write(f"  Skipped (no img) : {summary['skipped_records']}\n")
        f.write(f"  Total Gold Boxes : {summary['total_gold_boxes']}\n\n")

        f.write("-" * 56 + "\n")
        f.write("  IoU METRICS\n")
        f.write("-" * 56 + "\n")
        f.write(f"  Mean IoU         : {summary['mean_iou']:.4f}\n")
        f.write(f"  Median IoU       : {summary['median_iou']:.4f}\n")
        f.write(f"  Std IoU          : {summary['std_iou']:.4f}\n\n")

        f.write("-" * 56 + "\n")
        f.write("  ACCURACY @ IoU THRESHOLDS\n")
        f.write("-" * 56 + "\n")
        for t in IoU_THRESHOLDS:
            f.write(f"  Acc @ {t:<5}      : {summary[f'acc@{t}']:.4f}  "
                    f"({summary[f'hits@{t}']}/{summary['total_gold_boxes']})\n")

        f.write("\n")
        f.write("-" * 56 + "\n")
        f.write("  PER-TYPE BREAKDOWN\n")
        f.write("-" * 56 + "\n")
        for t_name, t_data in type_summary.items():
            f.write(f"  [{t_name}]  N={t_data['count']}  "
                    f"Mean IoU={t_data['mean_iou']:.4f}  "
                    f"Acc@0.5={t_data['acc@0.5']:.4f}\n")

        f.write("\n" + "=" * 56 + "\n")

    print(f"\n[SUCCESS] Report saved to: {report_path}")

    # --- Write Per-Sample CSV ---
    csv_path = output_dir / "per_sample_results.csv"
    if sample_results:
        fieldnames = sample_results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_results)
        print(f"[SUCCESS] Per-sample CSV saved to: {csv_path}")

    # --- Print Summary to Console ---
    print(f"\n{'=' * 40}")
    print(f"  Mean IoU  : {summary['mean_iou']:.4f}")
    print(f"  Median    : {summary['median_iou']:.4f}")
    for t in IoU_THRESHOLDS:
        print(f"  Acc@{t:<5} : {summary[f'acc@{t}']:.4f}")
    print(f"{'=' * 40}")


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WSOL Bounding Box Predictions (IoU Metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--predictions_file', type=str, required=True,
        help="Path to predictions.jsonl (output of bbox_preprocessing.py)"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help="Root directory for source images (e.g., /dataset/mimic-cxr-jpg/2.0.0). "
             "Used to resolve image paths and get dimensions for coordinate scaling."
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help="Directory for output report and CSV. Default: ./results"
    )
    parser.add_argument(
        '--ref_dim', type=float, default=224.0,
        help="Reference dimension for gold bounding box coordinates. Default: 224.0"
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
