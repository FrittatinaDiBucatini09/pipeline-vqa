import torch
import numpy as np
import cv2
import os
import json
import ast
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union

import urllib.request
from tqdm import tqdm
import warnings

# Suppress noisy warnings from SAM libraries and PyTorch attention kernels
warnings.filterwarnings('ignore', category=FutureWarning, module='segment_anything.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sam2.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sam3.*')
warnings.filterwarnings('ignore', message='.*Memory efficient.*')
warnings.filterwarnings('ignore', message='.*Flash [Aa]ttention.*')
warnings.filterwarnings('ignore', message='.*CuDNN attention.*')
warnings.filterwarnings('ignore', message='.*Expected query.*dtype.*')
warnings.filterwarnings('ignore', message='.*scaled_dot_product_attention.*')

# ==============================================================================
# MODEL REGISTRY
# ==============================================================================

# SAM1 / MedSAM (original) — direct URL downloads
SAM1_MODELS = {
    "medsam": "https://zenodo.org/records/10689643/files/medsam_vit_b.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

# MedSAM2 (SAM 2.1 based) — HuggingFace downloads
MEDSAM2_MODELS = {
    "medsam2_latest":     "MedSAM2_latest.pt",
    "medsam2_2411":       "MedSAM2_2411.pt",
    "medsam2_us_heart":   "MedSAM2_US_Heart.pt",
    "medsam2_mri_liver":  "MedSAM2_MRI_LiverLesion.pt",
    "medsam2_ct_lesion":  "MedSAM2_CTLesion.pt",
}

MEDSAM2_HF_REPO = "wanglab/MedSAM2"
MEDSAM2_DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"

# MedSAM3 (SAM3 + LoRA) — HuggingFace downloads
MEDSAM3_MODELS = {
    "medsam3": "best_lora_weights.pt",
}

MEDSAM3_HF_REPO = "lal-Joey/MedSAM3_v1"
MEDSAM3_DEFAULT_CFG = "configs/full_lora_config.yaml"

# Combined lookup for quick validation
ALL_MODEL_KEYS = set(SAM1_MODELS.keys()) | set(MEDSAM2_MODELS.keys()) | set(MEDSAM3_MODELS.keys())

# Valid text prompt modes for MedSAM3
TEXT_PROMPT_MODES = {"regions", "regions_question", "question"}

def is_medsam2_key(key: str) -> bool:
    """Returns True if the key refers to a MedSAM2 model."""
    return key in MEDSAM2_MODELS

def is_medsam3_key(key: str) -> bool:
    """Returns True if the key refers to a MedSAM3 model."""
    return key in MEDSAM3_MODELS

def is_known_model_key(key: str) -> bool:
    """Returns True if the key is any known model."""
    return key in ALL_MODEL_KEYS

# ==============================================================================
# TEXT PROMPT BUILDER (MedSAM3)
# ==============================================================================

def build_text_prompt(item: Dict[str, Any], mode: str = "question") -> str:
    """
    Builds a text prompt from JSONL item fields for MedSAM3.
    
    Modes:
        - "regions":          Use visual_regions only (e.g., "trachea")
        - "regions_question": Concatenate visual_regions + question  
        - "question":         Use question/prompt_used only
    
    Fallback: If mode requires visual_regions but it's missing/empty,
    falls back to "question" mode with a warning.
    """
    # Extract question text
    question = item.get("prompt_used") or item.get("question") or ""
    
    # Extract visual_regions (may be a string repr of a list)
    visual_regions_raw = item.get("visual_regions", "")
    regions_text = ""
    
    if visual_regions_raw:
        # Parse string representation of list if needed
        if isinstance(visual_regions_raw, str):
            try:
                parsed = ast.literal_eval(visual_regions_raw)
                if isinstance(parsed, list):
                    regions_text = ", ".join(str(r) for r in parsed)
                else:
                    regions_text = str(parsed)
            except (ValueError, SyntaxError):
                regions_text = visual_regions_raw.strip("[]'\"")
        elif isinstance(visual_regions_raw, list):
            regions_text = ", ".join(str(r) for r in visual_regions_raw)
    
    # Build prompt based on mode
    if mode == "regions":
        if regions_text:
            return regions_text
        else:
            if question:
                print(f"[WARN] visual_regions missing, falling back to question")
            return question or "medical finding"
    
    elif mode == "regions_question":
        if regions_text and question:
            return f"{regions_text}. {question}"
        elif regions_text:
            return regions_text
        elif question:
            print(f"[WARN] visual_regions missing, falling back to question")
            return question
        else:
            return "medical finding"
    
    else:  # "question" mode (default)
        return question or "medical finding"


# ==============================================================================
# DOWNLOAD UTILITIES
# ==============================================================================

def download_sam1_model(model_name: str, checkpoint_dir: str) -> str:
    """Downloads SAM1/MedSAM model weights via direct URL."""
    url = SAM1_MODELS.get(model_name)
    if not url:
        raise ValueError(f"Unknown SAM1 model: {model_name}. Available: {list(SAM1_MODELS.keys())}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = url.split('/')[-1]
    output_path = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(output_path):
        print(f"[INFO] Model {model_name} already exists at {output_path}")
        return output_path
        
    print(f"[INFO] Downloading {model_name} from {url}...")
    
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
            
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        
    return output_path


def download_medsam2_model(model_name: str, checkpoint_dir: str) -> str:
    """Downloads MedSAM2 model weights from HuggingFace Hub."""
    filename = MEDSAM2_MODELS.get(model_name)
    if not filename:
        raise ValueError(f"Unknown MedSAM2 model: {model_name}. Available: {list(MEDSAM2_MODELS.keys())}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_path = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(output_path):
        print(f"[INFO] Model {model_name} already exists at {output_path}")
        return output_path
    
    print(f"[INFO] Downloading {model_name} from HuggingFace ({MEDSAM2_HF_REPO})...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required for MedSAM2 downloads. pip install huggingface_hub")
    
    hf_hub_download(
        repo_id=MEDSAM2_HF_REPO,
        filename=filename,
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"[INFO] Downloaded {filename} to {output_path}")
    return output_path


def download_medsam3_model(model_name: str, checkpoint_dir: str) -> str:
    """Downloads MedSAM3 LoRA weights from HuggingFace Hub."""
    filename = MEDSAM3_MODELS.get(model_name)
    if not filename:
        raise ValueError(f"Unknown MedSAM3 model: {model_name}. Available: {list(MEDSAM3_MODELS.keys())}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_path = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(output_path):
        print(f"[INFO] Model {model_name} already exists at {output_path}")
        return output_path
    
    print(f"[INFO] Downloading {model_name} LoRA weights from HuggingFace ({MEDSAM3_HF_REPO})...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required for MedSAM3 downloads. pip install huggingface_hub")
    
    hf_hub_download(
        repo_id=MEDSAM3_HF_REPO,
        filename=filename,
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"[INFO] Downloaded {filename} to {output_path}")
    return output_path


def download_model(model_name: str, checkpoint_dir: str) -> str:
    """Unified download dispatcher for any known model key."""
    if model_name in SAM1_MODELS:
        return download_sam1_model(model_name, checkpoint_dir)
    elif model_name in MEDSAM2_MODELS:
        return download_medsam2_model(model_name, checkpoint_dir)
    elif model_name in MEDSAM3_MODELS:
        return download_medsam3_model(model_name, checkpoint_dir)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(ALL_MODEL_KEYS)}")


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_medsam_model(checkpoint_path: str, device: str = "cuda", checkpoint_dir: str = "/workspace/checkpoints"):
    """
    Loads a SAM1-based model (MedSAM / SAM ViT-B/L/H).
    Accepts a file path OR a known model key.
    Returns a SamPredictor instance.
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError("segment_anything library not found. pip install git+https://github.com/facebookresearch/segment-anything.git")

    # Resolve key → path
    if checkpoint_path in SAM1_MODELS:
        print(f"[INFO] '{checkpoint_path}' is a known SAM1 key. Checking cache...")
        checkpoint_path = download_sam1_model(checkpoint_path, checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM1 checkpoint not found at {checkpoint_path}")

    # Infer model type from filename
    model_type = "vit_b"
    if "vit_l" in checkpoint_path:
        model_type = "vit_l"
    elif "vit_h" in checkpoint_path:
        model_type = "vit_h"
        
    print(f"[INFO] Loading SAM1 model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def load_medsam2_model(checkpoint_path: str, device: str = "cuda",
                       checkpoint_dir: str = "/workspace/checkpoints",
                       model_cfg: str = MEDSAM2_DEFAULT_CFG):
    """
    Loads a MedSAM2 (SAM 2.1 based) model for 2D image segmentation.
    Accepts a file path OR a known model key.
    Returns a SAM2ImagePredictor instance.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "sam2 library not found. Install MedSAM2:\n"
            "  git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2 && pip install -e ."
        )

    # Resolve key → path
    if checkpoint_path in MEDSAM2_MODELS:
        print(f"[INFO] '{checkpoint_path}' is a known MedSAM2 key. Checking cache...")
        checkpoint_path = download_medsam2_model(checkpoint_path, checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MedSAM2 checkpoint not found at {checkpoint_path}")

    print(f"[INFO] Loading MedSAM2 model (cfg: {model_cfg})")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def load_medsam3_model(checkpoint_path: str, device: str = "cuda",
                       checkpoint_dir: str = "/workspace/checkpoints",
                       model_cfg: str = MEDSAM3_DEFAULT_CFG):
    """
    Loads a MedSAM3 (SAM3 + LoRA) model for text-prompted segmentation.
    Accepts a file path OR a known model key.
    Returns a SAM3LoRAInference instance (from MedSAM3 repo).
    
    The returned object supports:
        - predictor.predict(image_path, [text_prompts]) → dict with masks
        - The model also supports bbox via SAM3's add_geometric_prompt API
    """
    try:
        from infer_sam import SAM3LoRAInference
    except ImportError:
        raise ImportError(
            "MedSAM3 / SAM3 library not found. Install:\n"
            "  git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .\n"
            "  git clone https://github.com/Joey-S-Liu/MedSAM3.git && cd MedSAM3 && pip install -e ."
        )
    
    # Resolve key → path (LoRA weights)
    if checkpoint_path in MEDSAM3_MODELS:
        print(f"[INFO] '{checkpoint_path}' is a known MedSAM3 key. Checking cache...")
        checkpoint_path = download_medsam3_model(checkpoint_path, checkpoint_dir)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MedSAM3 LoRA weights not found at {checkpoint_path}")
    
    print(f"[INFO] Loading MedSAM3 (SAM3 + LoRA, cfg: {model_cfg})")
    # MedSAM3's code uses many relative paths internally (config, BPE vocab,
    # asset files).  Temporarily switch CWD to its repo root so they resolve.
    original_cwd = os.getcwd()
    try:
        os.chdir("/opt/MedSAM3")
        predictor = SAM3LoRAInference(
            config_path=model_cfg,
            weights_path=checkpoint_path,
            device=device,
        )
    finally:
        os.chdir(original_cwd)
    return predictor


def load_model(checkpoint_path: str, device: str = "cuda",
               checkpoint_dir: str = "/workspace/checkpoints",
               model_cfg: str = MEDSAM2_DEFAULT_CFG,
               sam3_cfg: str = MEDSAM3_DEFAULT_CFG):
    """
    Unified model loader. Automatically detects SAM1 vs MedSAM2 vs MedSAM3.
    Returns (predictor, model_family) where model_family is 'sam1', 'sam2', or 'sam3'.
    """
    # MedSAM3 key
    if checkpoint_path in MEDSAM3_MODELS:
        predictor = load_medsam3_model(checkpoint_path, device, checkpoint_dir, sam3_cfg)
        return predictor, "sam3"
    
    # MedSAM2 key
    if checkpoint_path in MEDSAM2_MODELS:
        predictor = load_medsam2_model(checkpoint_path, device, checkpoint_dir, model_cfg)
        return predictor, "sam2"
    
    # SAM1 key
    if checkpoint_path in SAM1_MODELS:
        predictor = load_medsam_model(checkpoint_path, device, checkpoint_dir)
        return predictor, "sam1"
    
    # File path inference
    lower_path = checkpoint_path.lower()
    if "medsam3" in lower_path or "sam3" in lower_path or "lora" in lower_path:
        predictor = load_medsam3_model(checkpoint_path, device, checkpoint_dir, sam3_cfg)
        return predictor, "sam3"
    elif "medsam2" in lower_path:
        predictor = load_medsam2_model(checkpoint_path, device, checkpoint_dir, model_cfg)
        return predictor, "sam2"
    else:
        predictor = load_medsam_model(checkpoint_path, device, checkpoint_dir)
        return predictor, "sam1"


# ==============================================================================
# SAM3 BBOX HELPER
# ==============================================================================

def xyxy_to_cxcywh_norm(box: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Converts [x1, y1, x2, y2] pixel coords to [cx, cy, w, h] normalized [0,1]
    as SAM3's add_geometric_prompt expects.
    """
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / img_width
    cy = ((y1 + y2) / 2.0) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return [cx, cy, w, h]


# ==============================================================================
# IMAGE PROCESSING UTILITIES
# ==============================================================================

def preprocess_image(image_path: str, target_size: int = 1024) -> np.ndarray:
    """
    Loads and preprocesses an image for MedSAM.
    MedSAM expects RGB images.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def generate_visual_prompt_image(image: np.ndarray, box: List[int], color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 5) -> np.ndarray:
    """
    Draws a box on the image for Scenario B (Visual Prompting).
    In this scenario, the prompt is injected visually into the pixel data.
    """
    img_copy = image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Overlays a binary mask on the image for visualization.
    """
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def save_mask(mask: np.ndarray, output_path: str):
    """
    Saves the mask as a PNG file (0=Background, 255=Foreground).
    """
    if mask.dtype == bool:
        mask_uint8 = (mask.astype(np.uint8) * 255)
    else:
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        
    cv2.imwrite(output_path, mask_uint8)

def rle_encoding(mask):
    """
    Run-length encoding for RLE storage (optional).
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
