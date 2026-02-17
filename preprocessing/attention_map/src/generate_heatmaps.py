"""
Attention Map Heatmap Generation Pipeline
==========================================
Adapted from bbox_preprocessing.py (Bounding Box pipeline).

Generates dense attention heatmap overlay images using BiomedCLIP + gScoreCAM.
Instead of extracting discrete bounding boxes, this pipeline:
  1. Generates CAM activations using the same model/CAM logic.
  2. Normalizes the CAM to [0, 1] per image.
  3. Applies a colormap (JET or TURBO) to produce a colored heatmap.
  4. Alpha-blends the heatmap onto the original X-Ray image.
  5. Saves the result as a .png file.

Logic Retained:   BiomedCLIP model loading, tokenizer, data loading, body masking, WandB.
Logic Removed:    CRF, contours, adaptive padding, JSONL coordinates.
Logic Added:      Normalization, colormap application, alpha blending, PNG output.
"""

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
from concurrent.futures import ThreadPoolExecutor
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import time
from transformers import AutoTokenizer
import wandb

# ==============================================================================
# 1. DEPENDENCY MANAGEMENT & DYNAMIC IMPORTS
# ==============================================================================
CURRENT_SCRIPT_PATH = Path(__file__).resolve().parent
if str(CURRENT_SCRIPT_PATH) not in sys.path:
    sys.path.append(str(CURRENT_SCRIPT_PATH))

try:
    from utils import CAMWrapper, load_clip
    import open_clip
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
except ImportError as e:
    print(f"\n[WARNING] Local dependencies or pydensecrf missing: {e}\n")
    if "pydensecrf" in str(e):
        print("CRF will be disabled unless pydensecrf is installed.")

# ==============================================================================
# 1.1 GLOBAL THREADING CONTROL (Prevents OpenMP deadlocks with ThreadPoolExecutor)
# ==============================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ==============================================================================
# 2. GLOBAL CONSTANTS & CONFIGURATION
# ==============================================================================
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}

# Colormap mapping
COLORMAP_DICT = {
    'jet': cv2.COLORMAP_JET,
    'turbo': cv2.COLORMAP_TURBO,
    'inferno': cv2.COLORMAP_INFERNO,
    'hot': cv2.COLORMAP_HOT,
}

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def identity_transform(x):
    """
    Converts PIL Image to Tensor.
    Required because custom_collate uses torch.stack(), which only accepts tensors.
    """
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
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return sanitize_for_json(obj.item())
    elif pd.isna(obj):
        return None
    return obj


# ==============================================================================
# 4. DATA LOADING LAYER
# ==============================================================================
class GEMeXDataset(Dataset):
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
        'raw_metadata': [b['raw_metadata'] for b in batch]
    }


def parse_visual_regions_list(raw_value: Any) -> List[str]:
    """
    Parses the visual_regions column which can be a Python list or a string
    representing a list (e.g., "['bilateral lung', 'heart']").
    """
    if pd.isna(raw_value) or raw_value == "" or str(raw_value).lower() == "nan":
        return []

    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if v]

    try:
        cleaned = str(raw_value).strip()
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if v]
    except (ValueError, SyntaxError):
        pass

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
        logits = logit_scale * image_embeddings @ self.text_embeddings.t()
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
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            
            # CASE A: All prompts are identical (Standard Batching)
            if len(unique_prompts) == 1:
                text_tokens = self.tokenizer(unique_prompts[0]).to(self.device)
                self.clip_model.set_text(text_tokens)
                return self.cam_wrapper(text=None, input_tensor=batch_tensors.to(self.device), targets=None)
            
            # CASE B: Mixed Prompts (Composite Mode)
            else:
                batch_tensors = batch_tensors.to(self.device)
                batch_size = len(prompts)
                
                all_tokens = self.tokenizer(prompts).to(self.device)
                self.clip_model.set_text(all_tokens)
                
                targets = [ClassifierOutputTarget(i) for i in range(batch_size)]
                return self.cam_wrapper(text=None, input_tensor=batch_tensors, targets=targets)


# ==============================================================================
# 7. HEATMAP VISUALIZATION (NEW)
# ==============================================================================

def _generate_body_mask(img_bgr):
    """
    Generates a mask that isolates the body by REMOVING WHITE ARTIFACTS.
    Strategy: High-Intensity Clipping.
    Everything > 240 (digital white) is considered text/artifact and removed.
    """
    if img_bgr is None: 
        return None
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Body Detection (Anatomy) — Standard Otsu Thresholding
    _, body_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_body = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_body, iterations=4)
    body_mask = cv2.dilate(body_mask, kernel_body, iterations=2)

    # White Artifact Detection (Text/Markers)
    GLARE_THRESHOLD = 245 
    _, glare_mask = cv2.threshold(gray, GLARE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    kernel_glare = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    glare_mask = cv2.dilate(glare_mask, kernel_glare, iterations=4)

    # Fusion: Body - White Artifacts
    final_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(glare_mask))
    
    return final_mask.astype(np.float32) / 255.0


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _run_crf_task(image_np, cam_map, params):
    """
    Runs Dense CRF refinement on the CAM map.
    Returns: Refined Probability Map (Float32, 0.0-1.0)
    """
    # 1. Resize to processing size (speed optimization)
    proc_size = 512
    h_orig, w_orig = image_np.shape[:2]
    
    img_small = cv2.resize(image_np, (proc_size, proc_size), interpolation=cv2.INTER_LINEAR)
    cam_small = cv2.resize(cam_map, (proc_size, proc_size), interpolation=cv2.INTER_LINEAR)
    
    # Range Normalization (Ensure 0-255 for CRF energy calculation)
    if cam_small.max() <= 1.0 + 1e-7: 
        cam_small = cam_small * 255.0
        
    # Optional Hard Cutoff
    cutoff = params.get('crf_cutoff', 0.0)
    if cutoff > 0.0:
        cam_small[cam_small < (cutoff * 255.0)] = 0

    # 2. Probability Setup
    # We treat the CAM as the "Foreground" probability.
    # Background = 1.0 - Foreground
    fg_prob = cam_small / 255.0
    fg_prob = np.clip(fg_prob, 1e-5, 1.0 - 1e-5)
    bg_prob = 1.0 - fg_prob
    
    # Stack: [Background, Foreground] -> Shape (2, H, W)
    prob_stack = np.stack([bg_prob, fg_prob], axis=0)
    
    # 3. Unary Potentials
    # U = -log(P)
    unary = -np.log(prob_stack)
    unary = unary.reshape((2, -1)) # Flatten to (2, N)
    
    # 4. CRF Setup
    d = dcrf.DenseCRF2D(proc_size, proc_size, 2)
    d.setUnaryEnergy(unary.astype(np.float32))
    
    # 5. Pairwise Potentials
    # Smoothness (Gaussian)
    d.addPairwiseGaussian(
        sxy=params['crf_gaussian_sxy'], 
        compat=params['crf_gaussian_compat']
    )
    
    # Appearance (Bilateral)
    img_contiguous = np.ascontiguousarray(img_small)
    d.addPairwiseBilateral(
        sxy=params['crf_bilateral_sxy'], 
        srgb=params['crf_bilateral_srgb'], 
        rgbim=img_contiguous, 
        compat=params['crf_bilateral_compat']
    )
    
    # 6. Inference
    # d.inference(N) returns Q-values (unnormalized logs? No, properly normalized probs usually)
    # But PyDenseCRF inference returns the marginal probabilities.
    Q = d.inference(params['crf_M'])
    Q = np.array(Q).reshape((2, proc_size, proc_size))
    
    # Extract Foreground refined probability
    res_fg = Q[1, :, :]
    
    # Resize back to ORIGINAL dimensions
    res_orig = cv2.resize(res_fg, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    return res_orig


def generate_and_save_heatmap(
    cam_map: np.ndarray, 
    original_bgr: np.ndarray, 
    save_path: Path, 
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    body_mask: Optional[np.ndarray] = None,
    save_raw_cam: bool = False,
    prompt: str = ""
):
    """
    Core heatmap generation function.
    
    1. Apply optional body mask to suppress background artifacts.
    2. Normalize CAM to [0, 1].
    3. Resize to original image dimensions.
    4. Apply colormap (JET/TURBO) -> colored heatmap.
    5. Alpha-blend heatmap onto original X-Ray.
    6. Optionally annotate prompt text.
    7. Save as PNG.
    
    Args:
        cam_map:       Raw CAM output (float, any range).
        original_bgr:  Original image in BGR format.
        save_path:     Output file path.
        alpha:         Blending factor (0=original only, 1=heatmap only).
        colormap:      OpenCV colormap constant.
        body_mask:     Optional float32 mask [0.0-1.0] to suppress background.
        save_raw_cam:  If True, saves raw grayscale CAM alongside the overlay.
        prompt:        Optional prompt text to annotate on the image.
    """
    h_orig, w_orig = original_bgr.shape[:2]
    
    # Ensure numpy
    if isinstance(cam_map, torch.Tensor):
        cam_map = cam_map.detach().cpu().numpy()
    
    # 1. Apply body mask (suppress background activations)
    if body_mask is not None:
        h_cam, w_cam = cam_map.shape[:2]
        mask_resized = cv2.resize(body_mask, (w_cam, h_cam), interpolation=cv2.INTER_NEAREST)
        cam_map = cam_map * mask_resized
    
    # 2. Normalize to [0, 1]
    cam_min, cam_max = cam_map.min(), cam_map.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam_map - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_map)
    
    # 3. Resize to original image dimensions
    cam_resized = cv2.resize(cam_norm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # 4. Apply colormap
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    
    # 5. Alpha-blend
    # Ensure original is 3-channel (X-Rays may be grayscale loaded as BGR)
    if len(original_bgr.shape) == 2:
        original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(heatmap_bgr, alpha, original_bgr, 1 - alpha, 0)
    
    # 6. Optional: Annotate prompt text
    if prompt:
        label_text = str(prompt)[:60]  # Truncate long prompts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Background rectangle for readability
        cv2.rectangle(overlay, (5, 5), (10 + tw, 10 + th + 4), (0, 0, 0), -1)
        cv2.putText(overlay, label_text, (8, 8 + th), font, font_scale, (255, 255, 255), thickness)
    
    # 7. Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), overlay)
    
    # Optional: Save raw grayscale CAM
    if save_raw_cam:
        raw_path = save_path.parent / f"{save_path.stem}_raw_cam.png"
        cv2.imwrite(str(raw_path), cam_uint8)


# ==============================================================================
# 8. UTILITIES & ASYNC I/O
# ==============================================================================
def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def generate_vqa_manifest(manifest_records: List[Dict], output_root: Path) -> None:
    """
    Writes a VQA-ready CSV manifest from the collected records.

    Each record maps a generated heatmap image to its original question/answer,
    enabling the downstream VQA generation stage to consume preprocessed images.

    Args:
        manifest_records: List of dicts with keys: image_path, question, answer.
        output_root: Path to the preprocessing output directory.
    """
    manifest_path = output_root / "vqa_manifest.csv"

    if not manifest_records:
        print("[WARNING] No records collected for VQA manifest. Skipping.")
        return

    df = pd.DataFrame(manifest_records)
    df.to_csv(manifest_path, index=False)
    print(f"[INFO] VQA manifest generated: {manifest_path} ({len(df)} rows)")


def _save_heatmap_async(cam_map, original_bgr, save_path, alpha, colormap, 
                        body_mask, save_raw_cam, prompt):
    """Thread-safe wrapper for async heatmap saving."""
    try:
        generate_and_save_heatmap(
            cam_map=cam_map,
            original_bgr=original_bgr,
            save_path=save_path,
            alpha=alpha,
            colormap=colormap,
            body_mask=body_mask,
            save_raw_cam=save_raw_cam,
            prompt=prompt
        )
    except Exception as e:
        print(f"[ERROR Async Save] {save_path}: {e}")


# ==============================================================================
# 9. MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP + CAM → Attention Heatmap Visualization")
    
    # --- Path Arguments ---
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Root directory containing input images.")
    parser.add_argument('--output_dir', type=str, default='./results',
                        help="Directory for output heatmap images.")
    parser.add_argument('--metadata_file', type=str,
                        help="Path to CSV/Parquet metadata file.")
    
    # --- Execution Flags ---
    parser.add_argument('--stop_after', type=int, default=None,
                        help="Limit processing to the first N image-query pairs.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--draw_labels', action='store_true',
                        help="If enabled, overlays the prompt text on the heatmap image.")
    
    # --- Heatmap-Specific Arguments (NEW) ---
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Alpha blending factor: 0.0 = original only, 1.0 = heatmap only. Default: 0.5")
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['jet', 'turbo', 'inferno', 'hot'],
                        help="Colormap for the heatmap visualization. Default: jet")
    parser.add_argument('--save_raw_cam', action='store_true',
                        help="If enabled, additionally saves the raw grayscale CAM image.")
    
    # --- Visual Regions ---
    parser.add_argument('--use_visual_regions', action='store_true',
                        help="Use the visual_regions column as prompt for MedCLIP.")
    parser.add_argument('--composite_regions', action='store_true',
                        help="Generates ONE heatmap per tuple with ALL region activations overlaid.")
    parser.add_argument('--include_context_in_inference', action='store_true',
                        help="Appends the context prompt to the visual regions list during inference.")
    parser.add_argument('--visual_regions_col', type=str, default='visual_regions',
                        help="Name of the column containing the list of regions.")
    parser.add_argument('--explode_regions', action='store_true',
                        help="Creates a separate heatmap for EACH region in the list.")
    
    # --- Artifact Removal ---
    parser.add_argument('--enable_body_mask', action='store_true',
                        help="Applies Otsu thresholding to mask CAM activations outside the patient body.")
    
    # --- Model Hyperparameters ---
    parser.add_argument('--model_name', type=str, 
                        default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')
    parser.add_argument('--cam_version', type=str, default='gScoreCAM')
    parser.add_argument('--topk', type=int, default=300)
    
    # --- CRF Parameters (Re-integrated) ---
    parser.add_argument('--skip_crf', action='store_true', help="Skip CRF refinement (default: False if not set).")
    parser.add_argument('--crf_M', type=int, default=6)
    parser.add_argument('--crf_tau', type=float, default=0.80) 
    parser.add_argument('--crf_gaussian_sxy', type=int, default=15)
    parser.add_argument('--crf_gaussian_compat', type=int, default=2)
    parser.add_argument('--crf_bilateral_sxy', type=int, default=15)
    parser.add_argument('--crf_bilateral_srgb', type=int, default=5)
    parser.add_argument('--crf_bilateral_compat', type=int, default=3)
    parser.add_argument('--crf_cutoff', type=float, default=0.0)

    # --- Data Mapping Config ---
    parser.add_argument('--path_col', type=str, default='image_path')
    parser.add_argument('--text_col', type=str, default='question')
    parser.add_argument('--prompt', type=str, default="medical abnormality",
                        help="Default/fallback prompt for inference.")
    parser.add_argument('--use_dynamic_prompts', action='store_true',
                        help="Use per-row text column as prompts instead of the global fallback.")

    args = parser.parse_args()
    
    # --- Resolve colormap ---
    cv2_colormap = COLORMAP_DICT.get(args.colormap, cv2.COLORMAP_JET)

    # CRF Params Dictionary
    crf_params = {
        'crf_M': args.crf_M,
        'crf_tau': args.crf_tau, # Not used in simplified logic above but kept for compat
        'crf_gaussian_sxy': args.crf_gaussian_sxy,
        'crf_gaussian_compat': args.crf_gaussian_compat,
        'crf_bilateral_sxy': args.crf_bilateral_sxy,
        'crf_bilateral_srgb': args.crf_bilateral_srgb,
        'crf_bilateral_compat': args.crf_bilateral_compat,
        'crf_cutoff': args.crf_cutoff
    }
    
    # --- WandB Initialization ---
    if "WANDB_API_KEY" not in os.environ:
        print("[WARNING] WANDB_API_KEY not found. Runs will be offline.")
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project="GEMeX-AttentionMap",
        name=f"heatmap-{args.cam_version}-{args.colormap}",
        config=vars(args),
        dir=args.output_dir
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # --- Metadata Loading ---
    print(f"[INFO] Loading Metadata from: {args.metadata_file}")
    df = load_dataframe(args.metadata_file) if args.metadata_file else pd.DataFrame()

    # --- Initialize Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Initializing Model on {device}...")
    model = BiomedCLIPInferencePipeline(args, device=device)

    # --- File & Prompt Discovery Strategy ---
    dataset_samples = []
    if not df.empty and args.path_col in df.columns:
        print("[INFO] Using Metadata CSV for Query-Image pairs...")

        for _, row in df.iterrows():
            rel_path = str(row[args.path_col]).strip()
            
            if "/datasets/" in rel_path:
                full_path = Path(rel_path)
            else:
                full_path = input_root / rel_path

            if full_path.exists():
                prompts_to_process = []
                
                # Context prompt determination
                context_prompt = args.prompt
                if args.use_dynamic_prompts:
                    row_text = str(row.get(args.text_col, "")).strip()
                    if row_text and row_text.lower() != 'nan':
                        context_prompt = row_text
                
                # Visual Regions Processing
                if args.use_visual_regions:
                    raw_regions = row.get(args.visual_regions_col, "[]")
                    regions_list = parse_visual_regions_list(raw_regions)
                    
                    if regions_list:
                        if args.composite_regions:
                            final_inference_list = regions_list.copy()
                            if args.include_context_in_inference:
                                final_inference_list.append(context_prompt)
                            prompts_to_process = [final_inference_list]
                        elif args.explode_regions:
                            prompts_to_process = regions_list
                        else:
                            prompts_to_process = [", ".join(regions_list)]
                    else:
                        prompts_to_process = [context_prompt]

                if not prompts_to_process:
                    prompts_to_process = [context_prompt]

                # Metadata Serialization
                vis_regions = [] 
                try:
                    raw_regions_str = str(row.get('visual_regions', '[]'))
                    parsed_regions = ast.literal_eval(raw_regions_str)
                    if isinstance(parsed_regions, list):
                        vis_regions = parsed_regions
                except (ValueError, SyntaxError):
                    vis_regions = []

                for p_payload in prompts_to_process:
                    dataset_samples.append({
                        'path': full_path,
                        'prompt': p_payload,
                        'visual_regions': vis_regions,
                        'raw_row_data': row.to_dict()
                    })

        print(f"[INFO] Total Image-Question pairs loaded: {len(dataset_samples)}")

    else:
        print("[INFO] Scanning input directory (Fallback Mode)...")
        raw_files = [f for f in input_root.rglob('*') if f.suffix.lower() in VALID_EXTENSIONS]
        
        for f in raw_files:
            dataset_samples.append({
                'path': f,
                'prompt': args.prompt,
                'visual_regions': [],
                'raw_row_data': {}
            })

    # Debugging limit
    if args.stop_after: 
        print(f"[INFO] STOP_AFTER active: processing first {args.stop_after} pairs.")
        dataset_samples = dataset_samples[:args.stop_after]
    
    # --- DataLoader Initialization ---
    dataset = GEMeXDataset(
        samples=dataset_samples, 
        input_root=input_root,
        preprocess_fn=model.preprocess
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_collate, pin_memory=True)

    # --- Runtime Initialization ---
    processed = 0
    failed_count = 0
    errors = []
    vqa_manifest_records = []  # Collects (image_path, question, answer) for VQA bridge

    # Thread Pool for I/O (Disk Writes)
    io_executor = ThreadPoolExecutor(max_workers=4)

    print(f"\n[INFO] Starting Heatmap Generation Loop (Batch: {args.batch_size}). Logging to WandB.")
    start_time = time.time()
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch is None: continue

            current_batch_size = len(batch['path']) 
            tensors, orig_imgs, paths, rel_paths, prompts = (
                batch['tensor'], batch['img_cv'], batch['path'], 
                batch['rel_path'], batch['prompt']
            )
            
            try:
                # === COMPOSITE MODE ===
                if args.composite_regions and isinstance(prompts[0], list):
                    for i in range(current_batch_size):
                        current_img_tensor = tensors[i]
                        current_inference_prompts = prompts[i]
                        
                        real_visual_regions = batch['visual_regions'][i]
                        num_real_regions = len(real_visual_regions)
                        if num_real_regions == 0:
                            num_real_regions = len(current_inference_prompts)
                            if args.include_context_in_inference and num_real_regions > 1:
                                num_real_regions -= 1

                        if not current_inference_prompts:
                            continue

                        # Batch inference: repeat image for each prompt
                        mini_batch_tensors = current_img_tensor.unsqueeze(0).repeat(
                            len(current_inference_prompts), 1, 1, 1
                        )
                        cams = model.process_batch(mini_batch_tensors, current_inference_prompts)

                        # Precompute body mask
                        cached_body_mask = None
                        if args.enable_body_mask:
                            cached_body_mask = _generate_body_mask(orig_imgs[i])

                        # Generate heatmap for EACH region
                        for r_idx in range(num_real_regions):
                            if r_idx >= len(cams):
                                break
                            
                            cam = cams[r_idx]
                            region_label = current_inference_prompts[r_idx]
                            
                            if isinstance(cam, torch.Tensor): 
                                cam = cam.detach().cpu().numpy()

                            # Apply CRF if requested
                            if not args.skip_crf:
                                try:
                                    cam = _run_crf_task(orig_imgs[i], cam, crf_params)
                                except Exception as e:
                                    print(f"[ERROR] CRF Failed: {e}")

                            # Build unique output path per region
                            original_rel_path = Path(rel_paths[i])
                            stem = original_rel_path.stem
                            region_tag = str(region_label).replace(' ', '_')[:20]
                            new_filename = f"{stem}_idx{processed + 1}_{region_tag}.png"
                            relative_save_path = original_rel_path.parent / new_filename
                            full_save_path = output_root / relative_save_path

                            io_executor.submit(
                                _save_heatmap_async,
                                cam, orig_imgs[i].copy(), full_save_path,
                                args.alpha, cv2_colormap, cached_body_mask,
                                args.save_raw_cam,
                                region_label if args.draw_labels else ""
                            )
                            processed += 1

                            # Collect record for VQA manifest
                            raw_data = batch['raw_metadata'][i]
                            vqa_manifest_records.append({
                                'image_path': str(relative_save_path),
                                'question': raw_data.get(args.text_col, ''),
                                'answer': raw_data.get('answer', ''),
                            })

                # === STANDARD / EXPLODED MODE ===
                else:
                    cams = model.process_batch(tensors, prompts)

                    for i, cam in enumerate(cams):
                        if cam is None:
                            continue
                            
                        if isinstance(cam, torch.Tensor): 
                            cam = cam.detach().cpu().numpy()

                        # Apply CRF if requested
                        if not args.skip_crf:
                            try:
                                cam = _run_crf_task(orig_imgs[i], cam, crf_params)
                            except Exception as e:
                                print(f"[ERROR] CRF Failed: {e}")

                        # Body mask
                        cached_body_mask = None
                        if args.enable_body_mask:
                            cached_body_mask = _generate_body_mask(orig_imgs[i])

                        # Build unique output path
                        original_rel_path = Path(rel_paths[i])
                        stem = original_rel_path.stem
                        unique_idx = processed + 1
                        new_filename = f"{stem}_idx{unique_idx}.png"
                        relative_save_path = original_rel_path.parent / new_filename
                        full_save_path = output_root / relative_save_path

                        io_executor.submit(
                            _save_heatmap_async,
                            cam, orig_imgs[i].copy(), full_save_path,
                            args.alpha, cv2_colormap, cached_body_mask,
                            args.save_raw_cam,
                            prompts[i] if args.draw_labels else ""
                        )
                        processed += 1

                        # Collect record for VQA manifest
                        raw_data = batch['raw_metadata'][i]
                        vqa_manifest_records.append({
                            'image_path': str(relative_save_path),
                            'question': raw_data.get(args.text_col, ''),
                            'answer': raw_data.get('answer', ''),
                        })

                # WandB Logging (Periodic Sampling)
                if batch_idx % 50 == 0:
                    # Log a sample heatmap to WandB
                    sample_cam = cams[0] if not isinstance(prompts[0], list) else None
                    if sample_cam is not None:
                        if isinstance(sample_cam, torch.Tensor):
                            sample_cam = sample_cam.detach().cpu().numpy()
                        
                        # Generate a quick preview
                        h, w = orig_imgs[0].shape[:2]
                        cam_min, cam_max = sample_cam.min(), sample_cam.max()
                        if cam_max - cam_min > 1e-8:
                            cam_preview = (sample_cam - cam_min) / (cam_max - cam_min)
                        else:
                            cam_preview = np.zeros_like(sample_cam)
                        cam_preview = cv2.resize(cam_preview, (w, h))
                        cam_uint8 = (cam_preview * 255).astype(np.uint8)
                        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2_colormap)
                        overlay = cv2.addWeighted(heatmap_bgr, args.alpha, orig_imgs[0], 1 - args.alpha, 0)
                        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        
                        wandb.log({
                            "example_heatmap": wandb.Image(overlay_rgb, caption=f"Prompt: {prompts[0]}")
                        })

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

        # 2. Release DataLoader workers explicitly
        try:
            del dataloader
            del dataset
            print("[CLEANUP] DataLoader workers released.")
        except NameError:
            pass

        # 3. Release GPU memory
        try:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            print("[CLEANUP] GPU memory released.")
        except Exception as e:
            print(f"[WARNING] GPU cleanup error: {e}")

        # 4. Finalize WandB telemetry
        try:
            wandb.finish()
            print("[CLEANUP] WandB finalized.")
        except Exception as e:
            print(f"[WARNING] WandB finish error: {e}")

        # 5. Write final report (inside finally to guarantee output)
        try:
            total_time = time.time() - start_time
            throughput = processed / total_time if total_time > 0 else 0

            report_path = output_root / "report.txt"
            with open(report_path, "w") as f:
                f.write("========================================\n")
                f.write(f"    ATTENTION MAP HEATMAP REPORT        \n")
                f.write("========================================\n\n")
                f.write(f"Date:            {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"CAM Version:     {args.cam_version}\n")
                f.write(f"Colormap:        {args.colormap}\n")
                f.write(f"Alpha:           {args.alpha}\n")
                f.write(f"Body Mask:       {'Yes' if args.enable_body_mask else 'No'}\n")
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

            print(f"\n[SUCCESS] Heatmap generation complete.")
            print(f"[REPORT]  Saved to: {report_path}")
            print(f"[STATS]   Success: {processed} | Failed: {failed_count} | Speed: {throughput:.2f} img/s")
        except Exception as e:
            print(f"[WARNING] Report generation error: {e}")

        # 6. Generate VQA-ready manifest for downstream pipeline stages
        try:
            generate_vqa_manifest(vqa_manifest_records, output_root)
        except Exception as e:
            print(f"[WARNING] VQA manifest generation error: {e}")


if __name__ == "__main__":
    print("\n[INFO] Initializing Attention Map Pipeline...")
    main()

    # Forced exit to prevent zombie processes from deadlocked third-party threads
    print("[EXIT] Requesting interpreter shutdown...")
    try:
        sys.exit(0)
    except SystemExit:
        pass
    finally:
        print("[EXIT] Forcing process termination (os._exit).")
        os._exit(0)
