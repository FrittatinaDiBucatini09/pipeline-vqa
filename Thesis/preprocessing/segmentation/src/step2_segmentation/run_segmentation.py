import argparse
import sys
import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Try to import from local utils (module mode) or direct (script mode)
try:
    from .segmentation_utils import (
        load_model, preprocess_image, save_mask,
        generate_visual_prompt_image, apply_mask_overlay,
        build_text_prompt, xyxy_to_cxcywh_norm,
        ALL_MODEL_KEYS, TEXT_PROMPT_MODES
    )
except ImportError:
    try:
        from segmentation_utils import (
            load_model, preprocess_image, save_mask,
            generate_visual_prompt_image, apply_mask_overlay,
            build_text_prompt, xyxy_to_cxcywh_norm,
            ALL_MODEL_KEYS, TEXT_PROMPT_MODES
        )
    except ImportError as e:
        sys.exit(f"[CRITICAL] Could not import segmentation_utils: {e}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}

def parse_box(box_data: Any) -> List[int]:
    """Parses box data from various formats (list, string) to [x1, y1, x2, y2]."""
    if isinstance(box_data, list):
        # Handle nested list [[x1, y1, x2, y2]]
        if len(box_data) > 0 and isinstance(box_data[0], list):
            return [int(x) for x in box_data[0]]
        return [int(x) for x in box_data]
    return []


# ==============================================================================
# SAM1 / SAM2 INFERENCE (Bounding Box Prompts)
# ==============================================================================

def run_bbox_inference(predictor, img_input, prompt_box):
    """Run inference with SAM1/SAM2 using bounding box prompt."""
    predictor.set_image(img_input)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt_box[None, :],  # (1, 4)
        multimask_output=False,
    )
    return masks[0]  # (H, W)


# ==============================================================================
# SAM3 / MEDSAM3 INFERENCE (Text + Optional BBox Prompts)
# ==============================================================================

def run_sam3_text_inference(predictor, image_path, text_prompt):
    """Run MedSAM3 inference with text prompt only."""
    results = predictor.predict(image_path, [text_prompt])
    return _extract_sam3_best_mask(results, image_path)


def run_sam3_bbox_inference(predictor, image_path, box, img_width, img_height):
    """
    Run SAM3 inference with bounding box prompt via Sam3Processor API.
    Uses add_geometric_prompt with normalized [cx, cy, w, h] coordinates.
    """
    from PIL import Image as PILImage
    
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        raise ImportError("sam3 library not found for bbox inference.")
    
    # Get the underlying SAM3 model from the LoRA inference wrapper
    sam3_model = predictor.model
    processor = Sam3Processor(sam3_model)
    
    # Load image
    pil_image = PILImage.open(image_path).convert("RGB")
    inference_state = processor.set_image(pil_image)
    
    # Convert box to SAM3 format: [cx, cy, w, h] normalized [0,1]
    norm_box = xyxy_to_cxcywh_norm(box, img_width, img_height)
    
    # Add geometric (box) prompt
    output = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box,
        label=True,  # positive prompt
    )
    
    masks = output.get("masks")
    if masks is not None and len(masks) > 0:
        # Take the best mask (highest scoring)
        scores = output.get("scores", [1.0])
        if hasattr(masks, 'cpu'):
            mask = masks[0].cpu().numpy()
        elif isinstance(masks, np.ndarray):
            mask = masks[0]
        else:
            mask = np.array(masks[0])
        
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        return (mask > 0.5).astype(np.uint8)
    
    return None


def run_sam3_combined_inference(predictor, image_path, text_prompt, box, img_width, img_height):
    """
    Run SAM3 inference with both text and bbox prompts.
    Falls back to text-only if combined prompting fails.
    """
    from PIL import Image as PILImage
    
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        # Fallback to text-only via LoRA inference
        return run_sam3_text_inference(predictor, image_path, text_prompt)
    
    try:
        sam3_model = predictor.model
        processor = Sam3Processor(sam3_model)
        
        pil_image = PILImage.open(image_path).convert("RGB")
        inference_state = processor.set_image(pil_image)
        
        # Set text prompt first
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )
        
        # Then add geometric prompt to refine
        norm_box = xyxy_to_cxcywh_norm(box, img_width, img_height)
        output = processor.add_geometric_prompt(
            state=inference_state,
            box=norm_box,
            label=True,
        )
        
        masks = output.get("masks")
        if masks is not None and len(masks) > 0:
            if hasattr(masks, 'cpu'):
                mask = masks[0].cpu().numpy()
            elif isinstance(masks, np.ndarray):
                mask = masks[0]
            else:
                mask = np.array(masks[0])
            
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            return (mask > 0.5).astype(np.uint8)
    except Exception as e:
        print(f"[WARN] Combined text+bbox failed ({e}), falling back to text-only")
    
    return run_sam3_text_inference(predictor, image_path, text_prompt)


def _extract_sam3_best_mask(results, image_path):
    """
    Extract the best mask from SAM3LoRAInference results.
    Returns a binary mask (H, W) or None if no detections.
    """
    best_mask = None
    best_score = -1.0
    
    for key in results:
        if key == '_image':
            continue
        result = results[key]
        if result['num_detections'] > 0 and result['masks'] is not None:
            # Find the detection with the highest score
            max_idx = np.argmax(result['scores'])
            score = result['scores'][max_idx]
            if score > best_score:
                best_score = score
                mask = result['masks'][max_idx]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                best_mask = mask.astype(np.uint8)
    
    if best_mask is None:
        # Create empty mask with image dimensions
        from PIL import Image as PILImage
        try:
            img = PILImage.open(image_path)
            w, h = img.size
            best_mask = np.zeros((h, w), dtype=np.uint8)
        except Exception:
            best_mask = np.zeros((512, 512), dtype=np.uint8)
    
    return best_mask


# ==============================================================================
# MAIN INFERENCE LOOP
# ==============================================================================

def run_inference(args):
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Model (auto-detects SAM1 vs SAM2 vs SAM3)
    print(f"Loading model: {args.model_checkpoint}")
    predictor, model_family = load_model(
        args.model_checkpoint,
        device=device,
        model_cfg=args.model_cfg,
        sam3_cfg=args.sam3_cfg,
    )
    print(f"Model family: {model_family.upper()}")
    
    is_sam3 = (model_family == "sam3")
    
    if is_sam3:
        print(f"Text prompt mode: {args.text_prompt_mode}")
        print(f"Use bbox with SAM3: {args.sam3_use_bbox}")
    
    # Output Directories
    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "masks")
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    # 2. Load Data (JSONL)
    print(f"Reading predictions from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        
    data = [json.loads(line) for line in lines]
    
    # 3. Filtering/Limiting
    if args.limit:
        print(f"Limiting to first {args.limit} samples.")
        data = data[:args.limit]

    print(f"Processing {len(data)} samples in Mode: {args.scenario}")

    # Counters
    processed = 0
    skipped = 0

    # 4. Processing Loop
    for item in tqdm(data):
        image_path = item.get('image_path') or item.get('path')
        # Handle relative paths
        if args.input_root and not os.path.isabs(image_path):
            image_path = os.path.join(args.input_root, image_path)
            
        if not os.path.exists(image_path):
            print(f"[WARN] Image not found: {image_path}")
            skipped += 1
            continue

        # Parse bounding box (used by SAM1/SAM2, optional for SAM3)
        box = parse_box(
            item.get('box') or item.get('bbox') or 
            item.get('visual_locations') or item.get('predicted_boxes')
        )
        if not box:
            boxes = item.get('boxes') or item.get('valid_boxes') or item.get('predicted_boxes')
            if boxes and isinstance(boxes, list) and len(boxes) > 0:
                if isinstance(boxes[0], list):
                    box = [int(x) for x in boxes[0]]
                else:
                    box = [int(x) for x in boxes]

        # For SAM1/SAM2: bbox is required
        if not is_sam3 and (not box or len(box) < 4):
            skipped += 1
            continue

        # =====================================================================
        # SAM3 / MedSAM3 PATH (Text-Prompted Segmentation)
        # =====================================================================
        if is_sam3:
            # Build text prompt from JSONL fields
            text_prompt = build_text_prompt(item, mode=args.text_prompt_mode)
            
            # Decide inference strategy
            if args.sam3_use_bbox and box and len(box) >= 4:
                # Load image to get dimensions
                try:
                    image = preprocess_image(image_path)
                    img_h, img_w = image.shape[:2]
                except Exception as e:
                    print(f"[ERR] {e}")
                    skipped += 1
                    continue
                
                # Combined text + bbox inference
                best_mask = run_sam3_combined_inference(
                    predictor, image_path, text_prompt, box, img_w, img_h
                )
            else:
                # Text-only inference
                best_mask = run_sam3_text_inference(predictor, image_path, text_prompt)
                # Load image for overlay (needed below)
                try:
                    image = preprocess_image(image_path)
                except Exception as e:
                    print(f"[ERR] {e}")
                    skipped += 1
                    continue

        # =====================================================================
        # SAM1 / SAM2 PATH (Bounding Box Segmentation)
        # =====================================================================
        else:
            try:
                image = preprocess_image(image_path)
            except Exception as e:
                print(f"[ERR] {e}")
                skipped += 1
                continue

            img_input = image.copy()
            prompt_box = np.array(box)

            # Scenario handling
            if args.scenario == 'B':
                img_input = generate_visual_prompt_image(img_input, box, color=(255, 0, 0))
            elif args.scenario == 'C':
                img_input = generate_visual_prompt_image(img_input, box, color=(255, 0, 0))
                prompt_box = prompt_box + 50 
                
            best_mask = run_bbox_inference(predictor, img_input, prompt_box)

        # =====================================================================
        # SAVE RESULTS (shared by all model families)
        # =====================================================================
        if best_mask is None:
            skipped += 1
            continue
            
        filename = Path(image_path).stem
        q_idx = item.get('question_idx', item.get('idx', item.get('q_id', '')))
        out_name = f"{filename}_q{q_idx}" if q_idx else filename
            
        mask_path = os.path.join(mask_dir, f"{out_name}_mask.png")
        save_mask(best_mask, mask_path)
        
        if args.save_overlays:
            overlay = apply_mask_overlay(image, best_mask)
            if box and len(box) >= 4:
                prompt_box_draw = np.array(box)
                cv2.rectangle(
                    overlay,
                    (int(prompt_box_draw[0]), int(prompt_box_draw[1])),
                    (int(prompt_box_draw[2]), int(prompt_box_draw[3])),
                    (0, 255, 255), 2
                )
            cv2.imwrite(
                os.path.join(overlay_dir, f"{out_name}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
        
        processed += 1

    print(f"Done. Processed: {processed} | Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Medical Image Segmentation (SAM1 / MedSAM2 / MedSAM3)")
    
    # Input / Output
    parser.add_argument('--input_file', type=str, required=True, help="Path to predictions.jsonl from Step 1")
    parser.add_argument('--input_root', type=str, default="", help="Root directory to prepend to relative image paths")
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Model Selection
    parser.add_argument('--model_checkpoint', type=str, default="medsam",
                        help=f"Model key or path. Keys: {sorted(ALL_MODEL_KEYS)}")
    parser.add_argument('--model_cfg', type=str, default="configs/sam2.1_hiera_t512.yaml",
                        help="SAM2 config YAML (only for MedSAM2 models)")
    parser.add_argument('--sam3_cfg', type=str, default="configs/full_lora_config.yaml",
                        help="MedSAM3/SAM3 config YAML (only for MedSAM3 models)")
    
    # Experiment
    parser.add_argument('--scenario', type=str, choices=['A', 'B', 'C'], default='A',
                        help="A=Clean, B=VisualPrompt, C=Adversarial (SAM1/SAM2 only)")
    parser.add_argument('--limit', type=int, help="Process only N samples")
    parser.add_argument('--save_overlays', action='store_true', help="Save visualization overlays")
    
    # MedSAM3 Text Prompt Options
    parser.add_argument('--text_prompt_mode', type=str, default="question",
                        choices=sorted(TEXT_PROMPT_MODES),
                        help="Text prompt mode for MedSAM3: "
                             "'regions' = visual_regions only, "
                             "'regions_question' = visual_regions + question, "
                             "'question' = question only (default)")
    parser.add_argument('--sam3_use_bbox', action='store_true',
                        help="Also pass bounding box as geometric prompt to SAM3 (in addition to text)")
    
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
