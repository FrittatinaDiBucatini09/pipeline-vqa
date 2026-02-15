"""
Utility functions and wrappers for the Bounding Box Preprocessing pipeline.
Handles CAM (Class Activation Map) generation and Model/Tensor transformations.
"""

import warnings
from typing import Optional, Tuple, Any

import torch
import numpy as np

# --- 1. SAFE IMPORTS ---
try:
    from pytorch_grad_cam import (
        GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, 
        XGradCAM, EigenCAM
    )
except ImportError:
    raise ImportError(
        "The 'grad-cam' library is not installed. Please check requirements.txt."
    )


# --- 2. HELPER FUNCTIONS ---

def reshape_transform(tensor: torch.Tensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
    """
    Reshapes Vision Transformer (ViT) output tensors for compatibility with Grad-CAM.
    
    Handles two formats:
    1. (Batch, Tokens, Dim) -> Standard for OpenCLIP
    2. (Tokens, Batch, Dim) -> Legacy format
    """
    # Detect format: (Batch, Tokens, Dim) or (Tokens, Batch, Dim)
    # OpenCLIP usually outputs (Batch, Tokens, Dim), e.g., (1, 197, 768)
    
    # If the first dimension is small (batch) and the second is large (tokens),
    # assume (Batch, Tokens, Dim).
    if tensor.shape[0] < tensor.shape[1]:
        # Format: (Batch, Tokens, Dim)
        # Remove the CLS token (index 0 in the token dimension)
        patches = tensor[:, 1:, :]
        
        if height is None or width is None:
            num_patches = patches.shape[1]
            # Calculate grid side length (assuming square image)
            grid_side = int(num_patches ** 0.5)
            height = width = grid_side
            
        # Reshape to (Batch, H, W, Dim)
        result = patches.reshape(tensor.shape[0], height, width, tensor.shape[2])
        
        # Permute to (Batch, Dim, H, W) as required by grad-cam
        result = result.permute(0, 3, 1, 2)
        return result
        
    else:
        # Fallback for legacy format (Tokens, Batch, Dim)
        if height is None or width is None:
            grid_square = len(tensor) - 1
            height = width = int(grid_square**0.5)
            
        result = tensor[1:, :, :].reshape(height, width, tensor.size(2))
        result = result.permute(2, 0, 1)
        return result.unsqueeze(0)


def load_clip(clip_version: str, resize: str = 'adapt', custom: bool = False) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Simplified loader utility for CLIP models.
    Determines if reshape transformation is needed based on architecture (ViT).
    """
    try:
        import clip
    except ImportError:
        clip = None
    
    print(f"[Utils] Loading utilities for version: {clip_version}")
    
    # Assign reshape transform only for Vision Transformers
    if 'vit' in clip_version.lower():
        cam_trans = reshape_transform
    else:
        cam_trans = None
    
    # Returns placeholders for model/preprocess as they are handled elsewhere in this pipeline
    return None, None, None, cam_trans, clip


# --- 3. SAFE CAM WRAPPER ---

class CAMWrapper(object):
    """
    Wrapper class for the pytorch-grad-cam library.
    Handles compatibility issues between CLIP models and the CAM library.
    """
    
    CAM_DICT = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        # Fallbacks for unsupported or custom names
        "gscorecam": ScoreCAM,   
        "groupcam": GradCAM,     
        "hilacam": GradCAM,      
    }

    def __init__(self, model: torch.nn.Module, target_layers: list, tokenizer: Any, 
                 cam_version: str, preprocess: Any = None, target_category: Any = None, 
                 is_clip: bool = True, mute: bool = False, cam_trans: Any = None, 
                 is_transformer: bool = False, model_domain: str = "biomedclip", **kwargs):
        
        self.model = model
        # Normalize CAM version string
        raw_version = cam_version.lower() if cam_version else "gradcam"
        
        self.target_layers = target_layers
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans

        if raw_version in self.CAM_DICT:
            self.version = raw_version
        else:
            print(f"[Warning] Unknown CAM version '{cam_version}'. Defaulting to ScoreCAM.")
            self.version = 'scorecam'
        
        self._load_cam()

    def _load_cam(self):
        """Initializes the specific CAM implementation."""
        # ScoreCAM requires a larger batch size for variance calculation
        batch_size = 16 if self.version in ['scorecam', 'gscorecam'] else 1
        cam_cls = self.CAM_DICT[self.version]
        
        # Arguments for the CAM constructor
        args = {
            "model": self.model,
            "target_layers": self.target_layers,
            "reshape_transform": self.cam_trans,
        }
        
        # Try initializing with standard args; fallback to including batch_size if needed
        try:
            self.cam = cam_cls(**args)
        except TypeError:
            args["batch_size"] = batch_size
            self.cam = cam_cls(**args)

    def __call__(self, text: str, img: Any, label: Any = None, heatmap_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generates the Class Activation Map.
        
        Args:
            text (str): The prompt/query text.
            img (PIL.Image or Tensor): The input image.
            label: Optional target label.
            heatmap_size: Desired output size.
            
        Returns:
            np.ndarray: The generated heatmap (grayscale).
        """
        if self.preprocess is not None:
            img = self.preprocess(img)
            
        text_token = None
        if self.tokenizer and text:
            text_token = self.tokenizer(text).cuda()
        
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()

        # --- CRITICAL FIX ---
        # 1. Manually set text embeddings if the model supports it.
        # This decouples text encoding from the visual forward pass used by CAM.
        if hasattr(self.model, 'set_text') and text_token is not None:
            self.model.set_text(text_token)

        # 2. Pass ONLY the image to the CAM object.
        # Since text is already set in the model state via set_text(),
        # we avoid passing a tuple (image, text) which confuses the grad-cam library.
        grayscale_cam = self.cam(input_tensor=img, targets=label)

        # Return the heatmap for the first image in the batch
        return grayscale_cam[0, :]