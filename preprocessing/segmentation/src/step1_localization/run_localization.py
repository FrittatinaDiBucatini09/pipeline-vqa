import argparse
import sys
import ast
import re
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import multiprocessing
import os
import time
from transformers import AutoTokenizer
import wandb

# ==============================================================================
# 1. DEPENDENCY MANAGEMENT & DYNAMIC IMPORTS
# ==============================================================================
try:
    # Try relative import (when running as module)
    from .utils import CAMWrapper, load_clip
    import open_clip
except ImportError:
    # Fallback for standalone execution (if needed, though module run is preferred)
    try:
        from utils import CAMWrapper, load_clip
        import open_clip
    except ImportError as e:
        sys.exit(f"\n[CRITICAL ERROR] Local dependencies missing.\nDetail: {e}\n")

try:
    import pydensecrf.densecrf as dcrf
except ImportError:
    sys.exit("\n[CRITICAL ERROR] 'pydensecrf' not found.\nRun: pip install git+https://github.com/lucasb-eyer/pydensecrf.git\n")

# ==============================================================================
# 1.1 GLOBAL THREADING CONTROL (Prevents OpenMP deadlocks with ProcessPoolExecutor)
# ==============================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ==============================================================================
# 2. GLOBAL CONSTANTS & CONFIGURATION
# ==============================================================================
COLOR_GOLD = (0, 255, 0)      # Green for Ground Truth
COLOR_INFERENCE = (0, 0, 255) # Red for Model Predictions
LINE_THICKNESS = 2
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
CLIP_VERSIONS = ["RN50x16", "RN50x4", "RN50", "RN101", "ViT-B/32", "ViT-B/16", "hila"]
CAM_VERSIONS = ['GradCAM', 'ScoreCAM', 'GracCAM++', 'AblationCAM', 'XGradCAM', 'gScoreCAM']

# ==============================================================================
# 3. WORKER FUNCTIONS (MULTIPROCESSING)
# ==============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))

def identity_transform(x):
    """
    Converts PIL Image to Tensor.
    Required because custom_collate uses torch.stack(), which only accepts tensors.
    """
    # Converts PIL -> Numpy array -> Tensor (C, H, W) normalized 0-1
    # Assumes 'x' is a PIL RGB image
    return torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0

def sanitize_for_json(obj):
    """
    Recursively cleans an object (dict, list, value) to make it
    JSON serializable. Replaces NaN, Infinity and corrupted floats with None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        # Handles NaN, Inf, -Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        # Converts numpy types (e.g., int64) to native Python types
        return sanitize_for_json(obj.item())
    elif pd.isna(obj): # Handles pd.NA, pd.NaT
        return None
    return obj

def _run_crf_task(task_args: Tuple) -> List[List[int]]:
    """
    CPU-Bound Worker: Runs Dense Conditional Random Fields (CRF) refinement.
    Updated for Multi-Box support and Adaptive Padding.
    """
    image_np, cam_map, params = task_args
    proc_size = 512
    
    # Extract Multi-Box params safely
    multi_mode = params.get('multi_label', False)
    min_area_ratio = params.get('min_box_area_ratio', 0.01)
    
    # 1. Standard Preprocessing (Resize)
    h_orig, w_orig = image_np.shape[:2]
    img_area = h_orig * w_orig 

    img_small = cv2.resize(image_np, (proc_size, proc_size), interpolation=cv2.INTER_LINEAR)
    cam_small = cv2.resize(cam_map, (proc_size, proc_size), interpolation=cv2.INTER_LINEAR)
    
    # --- RANGE NORMALIZATION ---
    if cam_small.max() <= 1.0 + 1e-7: 
        cam_small = cam_small * 255.0
    
    # --- OPTIONAL HARD CUT LOGIC ---
    cutoff = params.get('crf_cutoff')
    if cutoff is not None and cutoff > 0.0:
        thresh_val = cutoff * 255.0
        cam_small[cam_small < thresh_val] = 0
    
    # 2. Probability Normalization
    anno_norm = cam_small / 255.0
    anno_norm = np.clip(anno_norm, 0.00001, 0.99999)

    # 3. CRF Unary Potentials
    tau = params['crf_tau']
    n_energy = -np.log((1.0 - anno_norm + 1e-8)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + 1e-8) / (tau * _sigmoid(anno_norm))

    d = dcrf.DenseCRF2D(proc_size, proc_size, params['crf_M'])
    unary = np.zeros((params['crf_M'], proc_size * proc_size), dtype='float32')
    unary[0, :] = n_energy.flatten()
    unary[1, :] = p_energy.flatten()
    d.setUnaryEnergy(unary)

    # 4. Pairwise Potentials
    d.addPairwiseGaussian(sxy=params['crf_gaussian_sxy'], compat=params['crf_gaussian_compat'])
    img_contiguous = np.ascontiguousarray(img_small)
    d.addPairwiseBilateral(sxy=params['crf_bilateral_sxy'], srgb=params['crf_bilateral_srgb'], 
                           rgbim=img_contiguous, compat=params['crf_bilateral_compat'])

    # 5. Inference
    q_values = d.inference(1)
    res_map_small = np.argmax(q_values, axis=0).reshape((proc_size, proc_size))
    
    # 6. Post-processing & Multi-Box Logic
    # Resize mask to ORIGINAL dimensions immediately
    res_map_orig = cv2.resize(res_map_small.astype('float32'), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    final_mask = (res_map_orig * 255).astype('uint8')
    
    _, binary = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract padding parameters (with safe defaults)
    p_max = params.get('pad_max', 0.25)
    p_min = params.get('pad_min', 0.02)

    if not contours:
        return []
    
    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_boxes = []

    if not multi_mode:
        # SINGLE MODE
        x, y, w, h = cv2.boundingRect(contours[0])
        # FIX: Define raw_box and apply padding
        raw_box = [x, y, x + w, y + h]
        padded_box = _apply_adaptive_padding(raw_box, w_orig, h_orig, p_max, p_min)
        valid_boxes.append(padded_box)
    else:
        # MULTI MODE
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate Area
            box_area = w * h

            # Filter Noise
            if box_area >= (img_area * min_area_ratio):
                # FIX: Define raw_box before using it
                raw_box = [x, y, x + w, y + h]
                padded_box = _apply_adaptive_padding(raw_box, w_orig, h_orig, p_max, p_min)
                valid_boxes.append(padded_box)

    return valid_boxes

# ==============================================================================
# 4. DATA LOADING LAYER
# ==============================================================================
class GEMeXDataset(Dataset):
    # Modificata per accettare una lista di dizionari {'path': ..., 'prompt': ...}
    def __init__(self, samples: List[Dict[str, Any]], input_root: Path, 
                 preprocess_fn: Any):
        self.samples = samples
        self.input_root = input_root
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Retrieve the specific path+prompt pair for this row directly"""
        item = self.samples[idx]
        fpath = item['path']
        prompt = item['prompt']

        try:
            rel_path = str(fpath.relative_to(self.input_root))
        except ValueError:
            rel_path = fpath.name

        img_cv = cv2.imread(str(fpath))
        if img_cv is None: return None

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.preprocess(img_pil)

        return {
            'tensor': tensor,
            'img_cv': img_cv,
            'path': str(fpath),
            'rel_path': rel_path,
            'prompt': item['prompt'],
            'visual_regions': item.get('visual_regions', []),
            'gold_boxes': item.get('gold_boxes', []),
            'raw_metadata': item['raw_row_data']
        }

def custom_collate(batch):
    """Filters out failed loads (None) before stacking tensors."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    tensors = torch.stack([b['tensor'] for b in batch])
    return {
        'tensor': tensors,
        'img_cv': [b['img_cv'] for b in batch],
        'path': [b['path'] for b in batch],
        'rel_path': [b['rel_path'] for b in batch],
        'prompt': [b['prompt'] for b in batch],
        'visual_regions': [b['visual_regions'] for b in batch],
        'gold_boxes': [b['gold_boxes'] for b in batch],
        'raw_metadata': [b['raw_metadata'] for b in batch]
    }

def parse_gold_coordinates(raw_value):
    """
    Parses strings containing nested lists like "[[x1,y1,x2,y2], [x1,y1,x2,y2]]"
    Robust against CSV formatting artifacts and various data types.
    """
    # Handle empty/NaN cases
    if pd.isna(raw_value) or raw_value == "" or str(raw_value).lower() == "nan":
        return []

    # Case 1: Already loaded as list (e.g., from Parquet)
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
            # Empty list
            if not parsed:
                return []
            # List of lists [[...]]
            if isinstance(parsed[0], list):
                return parsed
            # Flat list [x,y,x,y], wrap it
            if len(parsed) == 4:
                return [parsed]
            
    except (ValueError, SyntaxError):
        # Attempt 2: Regex fallback for messy formats
        # Search for [number, number, number, number] patterns
        try:
            matches = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', cleaned)
            if matches:
                return [[float(x) for x in m] for m in matches]
        except Exception:
            pass

    # No valid format found
    return []

def parse_visual_regions_list(raw_value: Any) -> List[str]:
    """
    Parses the visual_regions column which can be a Python list or a string
    representing a list (e.g., "['bilateral lung', 'heart']").
    """
    if pd.isna(raw_value) or raw_value == "" or str(raw_value).lower() == "nan":
        return []

    # If already a list (e.g., loaded from Parquet or JSON)
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if v]

    # If it's a string, try to evaluate it as Python literal
    try:
        cleaned = str(raw_value).strip()
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if v]
    except (ValueError, SyntaxError):
        pass

    # Fallback: If parsing fails, assume it's a single unformatted string
    # or try to manually clean it from square brackets
    cleaned_fallback = str(raw_value).replace('[', '').replace(']', '').replace("'", "").replace('"', "")
    return [x.strip() for x in cleaned_fallback.split(',') if x.strip()]

# ==============================================================================
# 5. MODEL ARCHITECTURE & WRAPPERS
# ==============================================================================
class BiomedCLIPWrapper(nn.Module):
    """
    Wraps OpenCLIP models to expose a standard interface for the CAM library.
    Handles embedding normalization and dot-product similarity.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.text_embeddings = None

    def set_text(self, text_tokens: torch.Tensor):
        """Pre-computes text embeddings. Handles both Single (1, D) and Batch (N, D)."""
        with torch.no_grad():
            emb = self.model.encode_text(text_tokens)
            self.text_embeddings = emb / emb.norm(dim=-1, keepdim=True)

    def forward(self, images: torch.Tensor, texts: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.text_embeddings is None: raise ValueError("Text input missing.")
        
        image_embeddings = self.model.encode_image(images)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()

        # Compute cosine similarity logits
        logits = logit_scale * image_embeddings @ self.text_embeddings.t()
        
        # Output shape: (Batch_Size_Images, Batch_Size_Texts)
        # In Composite Mode: (N, N) where diagonal elements (i,i) are the pairs we want.
        return logits

class BatchCAMWrapper(CAMWrapper):
    """Enables batch processing for Class Activation Maps."""
    def __call__(self, text: str, input_tensor: torch.Tensor, targets=None) -> np.ndarray:
        if input_tensor.dim() < 4: input_tensor = input_tensor.unsqueeze(0)
        if not input_tensor.is_cuda: input_tensor = input_tensor.cuda()
        return self.cam(input_tensor=input_tensor, targets=targets)

class SafeBiomedTokenizer:
    """
    Fixes a specific compatibility issue with 'hf-hub:' prefixes in newer 
    Transformers versions when loading from OpenCLIP.
    """
    def __init__(self, model_name):
        clean_name = model_name.replace("hf-hub:", "")
        print(f"[Tokenizer] Loading from clean name: {clean_name}")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(clean_name)
        
    def __call__(self, text):
        if isinstance(text, str): text = [text]
        out = self.hf_tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        return out["input_ids"]

# ==============================================================================
# 6. PIPELINE ORCHESTRATION
# ==============================================================================
class BiomedCLIPInferencePipeline:
    def __init__(self, args: argparse.Namespace, device: str = 'cuda'):
        self.device = device
        self.args = args
        self.cam_version = args.cam_version
        
        # Load Model & Transforms
        _, _, _, self.cam_transform, _ = load_clip(args.clip_version)
        biomed_core, self.preprocess = open_clip.create_model_from_pretrained(args.model_name)
        self.tokenizer = SafeBiomedTokenizer(args.model_name)
        
        biomed_core.to(self.device).eval()
        self.clip_model = BiomedCLIPWrapper(biomed_core)

        # Target Layer Selection for CAM (Last Norm layer is standard for ViT)
        try:
            target_layer = self.clip_model.model.visual.trunk.blocks[11].norm2
        except AttributeError:
            target_layer = self.clip_model.model.visual.trunk.blocks[-1]

        self.cam_wrapper = BatchCAMWrapper(
            model=self.clip_model, preprocess=None, target_layers=[target_layer],
            tokenizer=self.tokenizer, cam_version=self.cam_version, topk=args.topk,
            is_transformer=True, cam_trans=self.cam_transform
        )

    def process_batch(self, batch_tensors: torch.Tensor, prompts: List[str]):
        """Runs the Forward Pass and CAM generation using vectorized operations."""
        unique_prompts = list(set(prompts))
        
        # Use Automatic Mixed Precision (AMP)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            
            # CASE A: All prompts are identical (Standard Batching)
            if len(unique_prompts) == 1:
                text_tokens = self.tokenizer(unique_prompts[0]).to(self.device)
                self.clip_model.set_text(text_tokens)
                # Target=None implies "Maximize the highest score" (index 0 since 1 prompt = 1 output)
                return self.cam_wrapper(text=None, input_tensor=batch_tensors.to(self.device), targets=None)
            
            # CASE B: Mixed Prompts (Composite Mode / Batched List)
            else:
                batch_tensors = batch_tensors.to(self.device)
                batch_size = len(prompts)
                
                # 1. Tokenize ALL prompts at once -> (N, Token_Len)
                all_tokens = self.tokenizer(prompts).to(self.device)
                
                # 2. Update Model State with N embeddings -> text_embeddings is (N, Dim)
                self.clip_model.set_text(all_tokens)
                
                # 3. Create specific targets for Grad-CAM
                # We want Image[i] to be explained by Prompt[i].
                # Since output is (N, N), the score for (Image i, Prompt i) is at index 'i'.
                # We tell Grad-CAM: "For image i in the batch, explain class index i".
                targets = [ClassifierOutputTarget(i) for i in range(batch_size)]
                
                # 4. Single Batched Call
                # Warning: CAMWrapper.__call__ expects 'text' arg, but we already set it manually.
                # Passing text=None ensures we use the batched embeddings set in step 2.
                return self.cam_wrapper(text=None, input_tensor=batch_tensors, targets=targets)

# ==============================================================================
# 7. UTILITIES & ASYNC I/O
# ==============================================================================
def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)

def save_result_async(save_path, img, boxes, mode, prompt=None, draw_label=False, inference_color=(0, 0, 255)):
    """
    Handles both standard mode (list of boxes) and composite mode (list of box+label tuples).
    """
    try:
        color = COLOR_GOLD if mode == 'gold' else inference_color
        
        if boxes:
            for item in boxes:
                # --- POLYMORPHIC HANDLING ---
                # Case 1: Composite Mode -> item is ([x1,y1,x2,y2], "Label")
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
                    box, specific_label = item
                    current_color = inference_color  # Or generate random color here if you want different colors
                # Case 2: Standard Mode -> item is [x1,y1,x2,y2]
                else:
                    box = item
                    specific_label = prompt
                    current_color = color

                # Drawing
                if len(box) >= 4:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), current_color, LINE_THICKNESS)
                    
                    if draw_label and specific_label:
                        # Text drawing logic (same as before, but uses specific_label)
                        label_text = specific_label[:30]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                        
                        # Text background for readability
                        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), current_color, -1)
                        cv2.putText(img, label_text, (x1, y1 - 2), font, font_scale, (255, 255, 255), thickness)

        cv2.imwrite(str(save_path), img)
    except Exception as e:
        print(f"[ERROR Async Save] {save_path}: {e}")

def _get_safe_body_bounds(body_mask, img_w, img_h):
    """
    Computes the bounding rectangle that encloses the active body region.
    If the mask is empty or None, returns the full image dimensions.
    
    Args:
        body_mask (np.ndarray): Binary mask of body region (float32 0.0-1.0)
        img_w (int): Original image width
        img_h (int): Original image height
    
    Returns:
        List[int]: Bounding coordinates [min_x, min_y, max_x, max_y]
    
    Notes:
        - Handles disconnected anatomical regions (e.g., separated lungs)
        - Safe fallback to full image bounds on mask failure
    """
    if body_mask is None:
        return [0, 0, img_w, img_h]
    
    # Convert float mask to uint8 for OpenCV contour processing
    mask_uint8 = (body_mask * 255).astype('uint8')
    
    # Extract external contours of white regions (body area)
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # No detectable body region: fallback to full image
        return [0, 0, img_w, img_h]
    
    # Compute unified bounding box across ALL contours
    # (accounts for disconnected anatomical structures like separate lungs)
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return [x, y, x + w, y + h]

def _generate_body_mask(img_bgr):
    """
    Generates a mask that isolates the body by REMOVING WHITE ARTIFACTS.
    Strategy: High-Intensity Clipping.
    Everything > 240 (digital white) is considered text/artifact and removed.
    """
    if img_bgr is None: 
        return None
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- STAGE A: Body Detection (Anatomy) ---
    # Standard Otsu Thresholding to separate body from black background
    _, body_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology to fill holes in the body (e.g., dark lungs) and exclude external noise
    kernel_body = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_body, iterations=4)
    body_mask = cv2.dilate(body_mask, kernel_body, iterations=2)  # Expand for safety on skin edges

    # --- STAGE B: White Artifact Detection (Text/Markers) ---
    # Identify pixels with very high intensity (digital text is often 255 or >240)
    # WARNING: If bones are being clipped, raise this value (e.g., from 235 to 245).
    # For MIMIC-CXR, text is usually very distinct.
    GLARE_THRESHOLD = 245 
    _, glare_mask = cv2.threshold(gray, GLARE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Artifact mask dilation:
    # Needed to cover the anti-aliased edges of letters that might be < 235
    kernel_glare = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    glare_mask = cv2.dilate(glare_mask, kernel_glare, iterations=4)  # High iterations to cover entire words

    # --- STAGE C: Fusion ---
    # Final Mask = (Body) - (White Artifacts)
    # Use bitwise_not to invert the glare_mask (becomes black where there is text)
    final_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(glare_mask))
    
    # Normalize to 0-1 float
    return final_mask.astype(np.float32) / 255.0

def _is_anatomical_box(img_np, box, min_mean_intensity=15):
    """Rejects boxes that contain mostly black pixels (background/artifacts)."""
    x1, y1, x2, y2 = map(int, box)
    h, w = img_np.shape[:2]
    
    # Clip coordinates safely
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1: return False

    crop = img_np[y1:y2, x1:x2]
    
    # Check mean intensity (Bone/Tissue is usually > 20-30 in grayscale)
    if np.mean(crop) < min_mean_intensity:
        return False
    return True

def _get_bbox_from_raw_cam(cam_map, orig_size, img_ref=None, threshold=0.5, 
                          multi_mode=False, min_area_ratio=0.01, check_anatomy=False,
                          pad_max=0.25, pad_min=0.02, safe_bounds=None):
    """
    Extracts bounding boxes with optional anatomical validation.
    """
    h_orig, w_orig = orig_size
    img_area = h_orig * w_orig
    
    # Normalize and Threshold
    cam_min, cam_max = cam_map.min(), cam_map.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam_map - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = cam_map
        
    binary = (cam_norm > threshold).astype('uint8') * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: 
        return []
    
    # Sort by Area (Largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_boxes = []

    # If check_anatomy is active and we are NOT in multi-mode,
    # we check the first 3 candidates to find the best valid one.
    candidates = contours if multi_mode else contours[:3]

    for cnt in candidates:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Scale to original image size
        scaled_box = _scale_box(x, y, w, h, cam_map.shape, orig_size, pad_max, pad_min, safe_bounds)
        
        # 1. Filter by Area (Noise)
        box_area = (scaled_box[2] - scaled_box[0]) * (scaled_box[3] - scaled_box[1])
        if box_area < (img_area * min_area_ratio):
            continue

        # 2. Filter by Anatomy (Content Check)
        if check_anatomy and img_ref is not None:
            if not _is_anatomical_box(img_ref, scaled_box):
                # If rejected, the loop continues to the next candidate (the "second" box)
                continue
        
        valid_boxes.append(scaled_box)
        
        # If we're not in multi-mode, we just need to find one valid box
        if not multi_mode:
            break
            
    # Fallback: If the anatomical filter is too aggressive and removed everything,
    # revert to original behavior (take the largest even if "ugly")
    # This avoids returning [] if there's still a strong signal.
    if not valid_boxes and contours and not multi_mode and check_anatomy:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        valid_boxes.append(_scale_box(x, y, w, h, cam_map.shape, orig_size, pad_max, pad_min))

    return valid_boxes

def _scale_box(x, y, w, h, cam_shape, orig_size, pad_max, pad_min, safe_bounds=None):
    """Updated signature to accept dynamic padding params."""
    h_cam, w_cam = cam_shape[:2]
    h_orig, w_orig = orig_size
    
    scale_x = w_orig / w_cam
    scale_y = h_orig / h_cam
    
    final_x1 = int(x * scale_x)
    final_y1 = int(y * scale_y)
    final_x2 = int((x + w) * scale_x)
    final_y2 = int((y + h) * scale_y)

    return _apply_adaptive_padding(
        [final_x1, final_y1, final_x2, final_y2], 
        w_orig, h_orig, 
        max_pad=pad_max, min_pad=pad_min,
        safe_bounds=safe_bounds
    )

def _apply_adaptive_padding(box, img_w, img_h, max_pad, min_pad, safe_bounds=None):
    """
    Applies dynamic padding based on the relative size of the bounding box.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    img_area = img_w * img_h
    if img_area <= 0: 
        return box
    
    # Calculate relative area
    box_relative_area = (w * h) / img_area
    
    # Fixed thresholds for interpolation (below 5% = max, above 40% = min)
    lower_bound = 0.05
    upper_bound = 0.40
    
    if box_relative_area <= lower_bound:
        pad_ratio = max_pad
    elif box_relative_area >= upper_bound:
        pad_ratio = min_pad
    else:
        # Linear interpolation
        alpha = (box_relative_area - lower_bound) / (upper_bound - lower_bound)
        pad_ratio = max_pad - alpha * (max_pad - min_pad)
    
    # Apply expansion
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    if safe_bounds:
        limit_x1, limit_y1, limit_x2, limit_y2 = safe_bounds
    else:
        limit_x1, limit_y1, limit_x2, limit_y2 = 0, 0, img_w, img_h

    nx1 = max(limit_x1, x1 - pad_w)
    ny1 = max(limit_y1, y1 - pad_h)
    nx2 = min(limit_x2, x2 + pad_w)
    ny2 = min(limit_y2, y2 + pad_h)
    
    return [nx1, ny1, nx2, ny2]

# ==============================================================================
# 8. MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP + CRF Localization (WandB Enabled)")
    # Path Arguments
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--metadata_file', type=str)
    
    # Execution Flags
    parser.add_argument('--mode', type=str, choices=['gold', 'inference'], required=True)
    parser.add_argument('--stop_after', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--skip_crf', action='store_true', help="Bypass Dense Conditional Random Field post-processing; use direct CAM thresholding.")
    parser.add_argument('--cam_threshold', type=float, default=0.5, help="Activation threshold for CAM binarization when CRF is disabled.")
    parser.add_argument('--output_format', type=str, choices=['image', 'jsonl'], default='jsonl', help="Serialization format: visual overlays ('image') or structured metadata ('jsonl').")
    parser.add_argument('--draw_labels', action='store_true', 
                        help="If enabled, overlays the prompt text on top of the bounding boxes in the output images.")
    parser.add_argument('--box_color', type=str, default="0,0,255", 
                        help="BGR Color triplet for inference boxes (e.g., '255,0,0' for Blue). Default: Red (0,0,255).")
    
    # Visual Regions
    parser.add_argument('--use_visual_regions', action='store_true',
                        help="Use the visual_regions column as prompt for MedCLIP.")
    parser.add_argument('--composite_regions', action='store_true',
                    help="VQA MODE: Generates ONE image per tuple containing ALL detected boxes for the regions list, each with its specific label.")
    parser.add_argument('--include_context_in_inference', action='store_true',
                        help="If set, appends the context prompt (Question or Fallback) to the visual regions list during inference, but filters it out from results.")
    parser.add_argument('--visual_regions_col', type=str, default='visual_regions',
                        help="Name of the column containing the list of regions (e.g., visual_regions).")
    parser.add_argument('--explode_regions', action='store_true',
                        help="If enabled, creates a separate inference task for EACH region in the list (1 region = 1 box).")
    
    # Adaptive Padding Arguments
    parser.add_argument('--pad_max', type=float, default=0.25, 
                        help="Maximum padding ratio for small boxes (e.g., 0.25 = +25%).")
    parser.add_argument('--pad_min', type=float, default=0.02, 
                        help="Minimum padding ratio for large boxes (e.g., 0.02 = +2%).")
    parser.add_argument('--enable_smart_padding', action='store_true',
                        help="If enabled, restricts bounding box expansion to the anatomical body mask, avoiding black background areas.")

    # Artifact Removal Flags
    parser.add_argument('--enable_body_mask', action='store_true',
                        help="Applies Otsu thresholding to mask CAM activations outside the patient body.")
    parser.add_argument('--enable_anatomical_check', action='store_true',
                        help="Performs a heuristic check on the output box content to reject black/empty regions.")

    # Model Hyperparameters
    parser.add_argument('--model_name', type=str, default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')
    parser.add_argument('--cam_version', type=str, default='gScoreCAM')
    parser.add_argument('--topk', type=int, default=300)
    
    # CRF Hyperparameters
    parser.add_argument('--crf_cutoff', type=float, default=None, 
                        help="Threshold (0.0-1.0) for pre-cleaning the CAM before CRF. Removes background noise.")
    parser.add_argument('--crf_M', type=int, default=2)
    parser.add_argument('--crf_tau', type=float, default=1.05)
    parser.add_argument('--crf_gaussian_sxy', type=int, default=25)
    parser.add_argument('--crf_gaussian_compat', type=int, default=3)
    parser.add_argument('--crf_bilateral_sxy', type=int, default=25)
    parser.add_argument('--crf_bilateral_srgb', type=int, default=5)
    parser.add_argument('--crf_bilateral_compat', type=int, default=5)
    
    # Data Mapping Config
    parser.add_argument('--path_col', type=str, default='image_path')
    parser.add_argument('--box_col', type=str, default='visual_locations')
    parser.add_argument('--text_col', type=str, default='question')
    parser.add_argument('--prompt', type=str, default="medical abnormality")
    parser.add_argument('--use_dynamic_prompts', action='store_true')

    # Multiple Box Handling
    parser.add_argument('--multi_label', action='store_true', 
                        help="If enabled, it returns all bounding boxes above the threshold instead of just the largest one.")
    parser.add_argument('--min_box_area_ratio', type=float, default=0.01, 
                        help="Minimum area of the contour relative to the image (0.01 = 1%) to be considered valid in multi-mode.")

    args = parser.parse_args()
    
    # --- WandB Initialization ---
    if "WANDB_API_KEY" not in os.environ:
        print("[WARNING] WANDB_API_KEY not found. Runs will be offline.")
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project="GEMeX-Preprocessing",
        name=f"{args.mode}-{args.cam_version}",
        config=vars(args),
        dir=args.output_dir
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # --- Metadata Loading ---
    print(f"[INFO] Loading Metadata from: {args.metadata_file}")
    df = load_dataframe(args.metadata_file) if args.metadata_file else pd.DataFrame()
    prompt_lookup = {}
    gold_lookup = {}
    model = None

    # --- Mode: Inference (Load Models) ---
    if args.mode == 'inference':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing Model on {device}...")
        model = BiomedCLIPInferencePipeline(args, device=device)

    # --- Mode: Gold Standard (Load Ground Truth) ---
    if args.mode == 'gold' and not df.empty:
        print("[INFO] Parsing Gold Standard Bounding Boxes (GEMeX VQA Format)...")
        for _, row in df.iterrows():
            # Normalize the path
            path_val = str(row[args.path_col]).strip()
            
            # Robust coordinate parsing
            raw_box = row.get(args.box_col, "[]")
            parsed_boxes = parse_gold_coordinates(raw_box)
            
            # Lookup Management:
            # We save both with the pure filename (e.g., "image.jpg") 
            # and with the full relative path if present
            key_stem = Path(path_val).name
            
            # Map: Filename -> List of Boxes [[x,y,x,y], ...]
            if parsed_boxes:
                gold_lookup[key_stem] = parsed_boxes
                gold_lookup[path_val] = parsed_boxes
            else:
                gold_lookup[key_stem] = []
                gold_lookup[path_val] = []

    # --- File & Prompt Discovery Strategy ---
    dataset_samples = []
    if not df.empty and args.path_col in df.columns:
        print("[INFO] Using Metadata CSV for Query-Image pairs...")

        for _, row in df.iterrows():
            rel_path = str(row[args.path_col]).strip()
            
            # Handle absolute vs. relative path resolution
            if "/datasets/" in rel_path:
                full_path = Path(rel_path)
            else:
                full_path = input_root / rel_path

            # Validate file existence before processing
            if full_path.exists():
                prompts_to_process = []
                
                # ------------------------------------------------------------
                # 1. CONTEXT PROMPT DETERMINATION (Dynamic/Static Logic)
                # ------------------------------------------------------------
                # Always compute context prompt (serves as fallback AND optional context injection)
                context_prompt = args.prompt  # Default: use global fallback prompt
                if args.use_dynamic_prompts:
                    row_text = str(row.get(args.text_col, "")).strip()
                    # Use row-specific text if available and valid
                    if row_text and row_text.lower() != 'nan':
                        context_prompt = row_text
                
                # ------------------------------------------------------------
                # 2. VISUAL REGIONS PROCESSING PIPELINE
                # ------------------------------------------------------------
                if args.use_visual_regions:
                    raw_regions = row.get(args.visual_regions_col, "[]")
                    regions_list = parse_visual_regions_list(raw_regions)
                    
                    if regions_list:
                        if args.composite_regions:
                            # HYBRID MODE: Combine visual regions with optional context
                            final_inference_list = regions_list.copy()
                            
                            # Inject context prompt into inference list if enabled
                            if args.include_context_in_inference:
                                final_inference_list.append(context_prompt)
                            
                            # Pass enriched list for processing
                            prompts_to_process = [final_inference_list]
                            
                        elif args.explode_regions:
                            # INDIVIDUAL MODE: Process each visual region separately
                            prompts_to_process = regions_list
                        else:
                            # AGGREGATED MODE: Merge all visual regions into single prompt
                            prompts_to_process = [", ".join(regions_list)]
                    else:
                        # FALLBACK: Empty regions list → use context prompt
                        prompts_to_process = [context_prompt]

                # ------------------------------------------------------------
                # 3. STANDARD MODE FALLBACK
                # ------------------------------------------------------------
                # Used if visual regions are disabled or processing yielded no prompts
                if not prompts_to_process:
                    prompts_to_process = [context_prompt]

                # ------------------------------------------------------------
                # 4. METADATA SERIALIZATION & ANNOTATION PARSING
                # ------------------------------------------------------------
                vis_regions = [] 
                try:
                    raw_regions_str = str(row.get('visual_regions', '[]'))
                    parsed_regions = ast.literal_eval(raw_regions_str)
                    if isinstance(parsed_regions, list):
                        vis_regions = parsed_regions
                except (ValueError, SyntaxError):
                    vis_regions = []

                # Parse ground-truth bounding box annotations
                raw_locs = row.get(args.box_col, "[]")
                current_gold_boxes = parse_gold_coordinates(raw_locs)

                # ------------------------------------------------------------
                # 5. SAMPLE CONSTRUCTION FOR INFERENCE QUEUE
                # ------------------------------------------------------------
                for p_payload in prompts_to_process:
                    dataset_samples.append({
                        'path': full_path,                       # Absolute image path
                        'prompt': p_payload,                     # May contain: [R1, R2, Context]
                        'visual_regions': vis_regions,           # Original [R1, R2] from CSV (no context)
                        'gold_boxes': current_gold_boxes,        # Ground-truth bounding boxes
                        'raw_row_data': row.to_dict()            # Complete CSV row for traceability
                    })

        print(f"[INFO] Total Image-Question pairs loaded: {len(dataset_samples)}")

    else:
        print("[INFO] Scanning input directory (Fallback Mode)...")
        # Fallback: Use the same default prompt for all images found on disk
        raw_files = [f for f in input_root.rglob('*') if f.suffix.lower() in VALID_EXTENSIONS]
        
        for f in raw_files:
            dataset_samples.append({
                'path': f,
                'prompt': args.prompt
            })

    # Debugging limit
    if args.stop_after: 
        print(f"[INFO] STOP_AFTER active: processing first {args.stop_after} pairs.")
        dataset_samples = dataset_samples[:args.stop_after]
    
    # --- DataLoader Initialization ---
    dataset = GEMeXDataset(
        samples=dataset_samples, 
        input_root=input_root,
        preprocess_fn=model.preprocess if model else identity_transform
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)

    # --- Runtime Initialization ---
    processed = 0
    failed_count = 0
    errors = []
    
    # Thread Pool for I/O (Disk Writes)
    io_executor = ThreadPoolExecutor(max_workers=4)
    
    # Process Pool for CPU Tasks (CRF)
    # Detects SLURM allocation to prevent oversubscription
    cpu_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    crf_executor = ProcessPoolExecutor(max_workers=min(cpu_cores, 8))

    print(f"\n[INFO] Starting Loop (Batch: {args.batch_size}). Logging to WandB.")
    start_time = time.time()

    # Parse User Color (BGR format for OpenCV)
    try:
        # Converts string "0,0,255" -> tuple (0, 0, 255)
        user_inference_color = tuple(map(int, args.box_color.split(',')))
    except Exception:
        print("[WARNING] Invalid color format. Using default Red (0,0,255).")
        user_inference_color = (0, 0, 255)
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch is None: continue

            current_batch_size = len(batch['path']) 
            tensors, orig_imgs, paths, rel_paths, prompts, raw_metadatas = (
                batch['tensor'], batch['img_cv'], batch['path'], 
                batch['rel_path'], batch['prompt'], batch['raw_metadata']
            )
            
            try:
                batch_boxes = []  # Contains results for saving

                # === COMPOSITE ===
                if args.mode == 'inference':
                    if args.composite_regions and isinstance(prompts[0], list):
                        # In this mode, we process image by image within the batch
                        # to handle the complexity of having N different regions for each image.

                        for i in range(current_batch_size):
                            # Extract current image tensor and corresponding inference prompts
                            current_img_tensor = tensors[i]
                            current_inference_prompts = prompts[i]  # Format: ['Heart', 'Lung', 'Is there Cardiomegaly?']
                            
                            # ------------------------------------------------------------
                            # REGION MAPPING & SANITY CHECKS
                            # ------------------------------------------------------------
                            # Retrieve ground-truth visual regions for proper slicing logic
                            # Note: batch['visual_regions'] contains original annotations without context prompts
                            real_visual_regions = batch['visual_regions'][i]
                            
                            # Safety fallback: if region list is empty, infer count from prompts
                            num_real_regions = len(real_visual_regions)
                            if num_real_regions == 0:
                                num_real_regions = len(current_inference_prompts)
                                # Exclude context prompt from count if hybrid mode is active
                                if args.include_context_in_inference and num_real_regions > 1:
                                    num_real_regions -= 1

                            composite_results = []  # Stores (box, region_label) tuples
                            
                            # Skip processing if no prompts available
                            if not current_inference_prompts:
                                batch_boxes.append([])
                                continue

                            # ------------------------------------------------------------
                            # BATCH INFERENCE EXECUTION
                            # ------------------------------------------------------------
                            # Model processes ALL prompts: visual regions + optional context
                            # Create batch dimension: repeat image for each prompt
                            mini_batch_tensors = current_img_tensor.unsqueeze(0).repeat(
                                len(current_inference_prompts), 1, 1, 1
                            )
                            
                            # Forward pass: generate CAMs for all prompts
                            cams = model.process_batch(mini_batch_tensors, current_inference_prompts)

                            # Precompute body mask for optimization (applies to all regions)
                            cached_body_mask = None
                            if args.enable_body_mask:
                                cached_body_mask = _generate_body_mask(orig_imgs[i])

                            # ------------------------------------------------------------
                            # REGION-SPECIFIC POST-PROCESSING
                            # ------------------------------------------------------------
                            # Iterate only over actual visual regions (exclude context prompt if present)
                            for r_idx in range(num_real_regions):
                                # Boundary check for CAM array
                                if r_idx >= len(cams):
                                    break
                                
                                cam = cams[r_idx]
                                region_label = current_inference_prompts[r_idx]  # Anatomical region name
                                
                                # Convert to numpy if tensor
                                if isinstance(cam, torch.Tensor): 
                                    cam = cam.detach().cpu().numpy()

                                # Get original image dimensions for scaling
                                h, w = orig_imgs[i].shape[:2]

                                # --------------------------------------------------------
                                # BODY MASK APPLICATION
                                # --------------------------------------------------------
                                if cached_body_mask is not None:
                                    h_cam, w_cam = cam.shape
                                    # Resize mask to CAM resolution (preserve binary nature)
                                    mask_resized = cv2.resize(
                                        cached_body_mask, (w_cam, h_cam), 
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                    cam = cam * mask_resized

                                # --------------------------------------------------------
                                # BOUNDING BOX EXTRACTION (SINGLE-BOX ENFORCEMENT)
                                # --------------------------------------------------------
                                boxes = _get_bbox_from_raw_cam(
                                    cam, 
                                    orig_size=(h, w),
                                    img_ref=orig_imgs[i],
                                    threshold=args.cam_threshold, 
                                    multi_mode=False,  # ENFORCE SINGLE BOX PER REGION
                                    min_area_ratio=args.min_box_area_ratio,
                                    check_anatomy=args.enable_anatomical_check,
                                    pad_max=args.pad_max,
                                    pad_min=args.pad_min
                                )

                                # Append results with region label for composite output
                                for b in boxes:
                                    composite_results.append((b, region_label))
                            
                            # Store all detected regions for current sample
                            batch_boxes.append(composite_results)

                    else:
                        # --- STANDARD / EXPLODED MODE ---
                        # We enter here if we are NOT in Composite Mode.
                        # Process the batch normally.
                        cams = model.process_batch(tensors, prompts)
    
                        # Prepare lists for results
                        batch_boxes = []  # For Fast Mode
                        crf_tasks = []    # For CRF Mode
                        valid_indices = []

                        # --- COMMON PRE-PROCESSING (LAYER 1: BODY MASK) ---
                        # Iterate through generated CAMs to apply filter BEFORE CRF or BBox extraction
                        processed_cams = []
    
                        for i, cam in enumerate(cams):
                            if cam is None: 
                                processed_cams.append(None)
                                continue
                                
                            # Ensure it's numpy
                            if isinstance(cam, torch.Tensor): 
                                cam = cam.detach().cpu().numpy()
        
                            # === LAYER 1: BODY MASKING (Now Global) ===
                            safe_body_bounds = None

                            if args.enable_body_mask:
                                # Calculate mask on the original image
                                body_mask = _generate_body_mask(orig_imgs[i])
                                if body_mask is not None:
                                    h_orig, w_orig = orig_imgs[i].shape[:2]

                                    if args.enable_smart_padding:
                                        safe_body_bounds = _get_safe_body_bounds(body_mask, w_orig, h_orig)

                                    h_cam, w_cam = cam.shape
                                    # Resize mask to match CAM (usually 14x14 or 224x224)
                                    mask_resized = cv2.resize(body_mask, (w_cam, h_cam), interpolation=cv2.INTER_NEAREST)
                                    # Apply mask: zero activations outside the body
                                    cam = cam * mask_resized
                            # ===========================================
                            
                            processed_cams.append(cam)

                        # --- BRANCHING: FAST MODE vs CRF ---
                        if args.skip_crf:
                            # === FAST MODE (GPU/CPU Lightweight) ===
                            for i, cam in enumerate(processed_cams):
                                if cam is None:
                                    batch_boxes.append([])
                                    continue

                                orig_h, orig_w = orig_imgs[i].shape[:2]
            
                                # === LAYER 2: ANATOMICAL CHECK (Now Global) ===
                                box = _get_bbox_from_raw_cam(
                                    cam, 
                                    (orig_h, orig_w), 
                                    img_ref=orig_imgs[i],  # Needed to check box content
                                    threshold=args.cam_threshold, 
                                    multi_mode=args.multi_label,
                                    min_area_ratio=args.min_box_area_ratio,
                                    # Parameters passed from global flags
                                    check_anatomy=args.enable_anatomical_check,
                                    pad_max=args.pad_max,
                                    pad_min=args.pad_min,
                                    safe_bounds=safe_body_bounds
                                )
                                batch_boxes.append(box)
                        else:
                            # === CRF MODE (Precision Refinement) ===
                            # Note: We pass 'cam' which has ALREADY been masked by Body Masking above!
                            for i, cam in enumerate(processed_cams):
                                if cam is None: 
                                    continue
                                
                                # Prepare parallel task
                                task_args = (orig_imgs[i], cam, vars(args))
                                crf_tasks.append(task_args)
                                valid_indices.append(i)
        
                            # Parallel execution
                            if crf_tasks:
                                results = list(crf_executor.map(_run_crf_task, crf_tasks))
                                
                                # Realign results
                                batch_boxes = [[] for _ in range(len(paths))]
                                for idx, res in zip(valid_indices, results):
                                    batch_boxes[idx] = res
                            else:
                                batch_boxes = [[] for _ in range(len(paths))]

                        # 3. WandB Logging (Periodic Sampling)
                        if batch_idx % 50 == 0 and batch_boxes[0]:
                            debug_img = orig_imgs[0].copy()
                            for b in batch_boxes[0]:
                                x1, y1, x2, y2 = map(int, b[:4])
                                cv2.rectangle(debug_img, (x1, y1), (x2, y2), COLOR_INFERENCE, 2)
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                            wandb.log({
                                "example_prediction": wandb.Image(debug_img, caption=f"Prompt: {prompts[0]}")
                            })

                elif args.mode == 'gold':
                    REF_DIM = 224.0  # Reference dimension (GEMeX Dataset)

                    # We retrieve the specific boxes from the batch, not from global lookup!
                    batch_gold_boxes = batch['gold_boxes']

                    for i, raw_boxes in enumerate(batch_gold_boxes):
                        # raw_boxes is the list of boxes specific to THIS question/CSV row
                        
                        # Retrieve actual dimensions of current image
                        real_h, real_w = orig_imgs[i].shape[:2]
                        
                        scaled_boxes = []
                        for box in raw_boxes:
                            if len(box) == 4:
                                x1, y1, x2, y2 = box
                                
                                # --- SCALING LOGIC ---
                                scale_x = real_w / REF_DIM
                                scale_y = real_h / REF_DIM
                                
                                nx1 = int(x1 * scale_x)
                                ny1 = int(y1 * scale_y)
                                nx2 = int(x2 * scale_x)
                                ny2 = int(y2 * scale_y)
                                
                                scaled_boxes.append([nx1, ny1, nx2, ny2])
                        
                        # Add scaled boxes to the final list for drawing
                        batch_boxes.append(scaled_boxes)

                # 4. Async Output Saving with Unique Filenames
                jsonl_path = output_root / "predictions.jsonl"

                # Base index to calculate unique suffixes in this batch
                global_idx_base = batch_idx * dataloader.batch_size

                with open(jsonl_path, 'a') as f_json:
                    for i, p in enumerate(paths):
                        # Retrieve current data
                        original_rel_path = Path(rel_paths[i])
                        current_prompt = prompts[i]
                        current_boxes = batch_boxes[i]
                        
                        # Default path for JSONL (points to the original)
                        # Will be overwritten if we are in 'image' mode
                        final_json_image_path = str(original_rel_path)

                        # --- IMAGE MODE LOGIC ---
                        if args.output_format == 'image':
                            # 1. Generate unique filename
                            # Example: 02aa804e_idx105.jpg
                            stem = original_rel_path.stem
                            suffix = original_rel_path.suffix
                            unique_idx = processed + 1
                            new_filename = f"{stem}_idx{unique_idx}{suffix}"
                            
                            # 2. Determine the relative output path
                            # Maintains the original folder structure (e.g., files/p10/...)
                            relative_save_path = original_rel_path.parent / new_filename
                            full_save_path = output_root / relative_save_path
                            
                            # 3. Update the path for JSONL
                            # DO NOT overwrite image_path with the debug image path.
                            # Downstream steps (Step 2) need the ORIGINAL raw image path.
                            # final_json_image_path = str(relative_save_path) <- DISABLED
                            
                            # We can log the visual path separately if needed
                            visual_debug_path = str(relative_save_path)
                            
                            # 4. Create directories if they don't exist
                            if not full_save_path.parent.exists(): 
                                full_save_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 5. Launch asynchronous save (Draw Boxes + Write to Disk)
                            io_executor.submit(
                                save_result_async, 
                                full_save_path, 
                                orig_imgs[i].copy(), 
                                current_boxes,
                                args.mode,
                                prompt=current_prompt,
                                draw_label=args.draw_labels,
                                inference_color=user_inference_color
                            )

                        # --- JSONL WRITING (COMMON TO BOTH MODES) ---
                        raw_data = raw_metadatas[i] if 'raw_metadata' in batch else {}

                        record = raw_data.copy()
                        record.update({
                            "image_path": final_json_image_path,
                            "predicted_boxes": current_boxes,
                            "prompt_used": current_prompt,
                            "mode": args.mode,
                            # Aggiungi timestamp o versione se utile
                            "processed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        clean_record = sanitize_for_json(record)

                        # Safe JSON Serialization
                        try:
                            f_json.write(json.dumps(clean_record) + "\n")
                        except (TypeError, ValueError) as e:
                            print(f"[ERROR JSON] Impossibile serializzare l'immagine {p}: {e}")
                            safe_fallback = {
                                "image_path": str(final_json_image_path),
                                "error": "Serialization Failed (Invalid Data)",
                                "predicted_boxes": current_boxes
                            }
                            f_json.write(json.dumps(safe_fallback) + "\n")
                            wandb.log({"serialization_errors": 1})
                        
                        # Increment global counter
                        processed += 1

                    # Force disk write at the end of each batch (Safety for HPC)
                    f_json.flush()

                wandb.log({"processed_images": processed})

            except Exception as e:
                failed_count += current_batch_size
                err_msg = f"Batch Error: {e}"
                print(err_msg)
                errors.append(err_msg)
                wandb.log({"errors": 1})

    finally:
        # ==================================================================
        # DETERMINISTIC CLEANUP (Guaranteed execution on success or failure)
        # ==================================================================
        print("\n[CLEANUP] Starting resource teardown...")

        # 1. Shutdown Thread Pool (I/O writes) — wait for pending saves
        try:
            io_executor.shutdown(wait=True, cancel_futures=False)
            print("[CLEANUP] I/O executor shut down.")
        except Exception as e:
            print(f"[WARNING] I/O executor shutdown error: {e}")

        # 2. Shutdown Process Pool (CRF) — cancel any queued futures
        try:
            crf_executor.shutdown(wait=True, cancel_futures=True)
            print("[CLEANUP] CRF executor shut down.")
        except Exception as e:
            print(f"[WARNING] CRF executor shutdown error: {e}")

        # 3. Release DataLoader workers explicitly
        try:
            del dataloader
            del dataset
            print("[CLEANUP] DataLoader workers released.")
        except NameError:
            pass

        # 4. Release GPU memory
        try:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            print("[CLEANUP] GPU memory released.")
        except Exception as e:
            print(f"[WARNING] GPU cleanup error: {e}")

        # 5. Finalize WandB telemetry
        try:
            wandb.finish()
            print("[CLEANUP] WandB finalized.")
        except Exception as e:
            print(f"[WARNING] WandB finish error: {e}")

        # 6. Write final report (inside finally to guarantee output)
        try:
            total_time = time.time() - start_time
            throughput = processed / total_time if total_time > 0 else 0

            report_path = output_root / "report.txt"
            with open(report_path, "w") as f:
                f.write("========================================\n")
                f.write(f"       GEMeX PREPROCESSING REPORT       \n")
                f.write("========================================\n\n")
                f.write(f"Date:            {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode:            {args.mode}\n")
                f.write(f"CAM Version:     {args.cam_version}\n")
                f.write(f"CRF Enabled:     {'No' if args.skip_crf else 'Yes'}\n")
                f.write(f"Output Format:   {args.output_format}\n")
                f.write(f"Batch Size:      {args.batch_size}\n")
                f.write(f"Throughput:      {throughput:.2f} img/sec\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Success:   {processed}\n")
                f.write(f"Total Failed:    {failed_count}\n")
                f.write(f"Total Errors:    {len(errors)} (Batch Failures)\n")
                f.write("-" * 40 + "\n")

                if errors:
                    f.write("\n--- ERROR LOG ---\n")
                    f.write("\n".join(errors))

            print(f"\n[SUCCESS] Processing complete.")
            print(f"[REPORT]  Saved to: {report_path}")
            print(f"[STATS]   Success: {processed} | Failed: {failed_count} | Speed: {throughput:.2f} img/s")
        except Exception as e:
            print(f"[WARNING] Report generation error: {e}")

if __name__ == "__main__":
    # 'spawn' is required for CUDA compatibility in multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    print("\n[INFO] Initializing Pipeline...")
    main()

    # ==================================================================
    # FORCED EXIT STRATEGY
    # Ensures the Python interpreter terminates even if third-party
    # threads (OpenMP, WandB telemetry, OpenCL) are deadlocked.
    # sys.exit(0) attempts clean shutdown; os._exit(0) bypasses
    # interpreter cleanup as a last resort after a timeout.
    # ==================================================================
    print("[EXIT] Requesting interpreter shutdown...")
    try:
        sys.exit(0)
    except SystemExit:
        pass
    finally:
        # If we reach here, clean shutdown was blocked. Force terminate.
        print("[EXIT] Forcing process termination (os._exit).")
        os._exit(0)