"""
Utility functions for the MedCLIP Agentic Routing preprocessing stage.

Handles:
- Model loading (SciSpacy, Gemma-2-2B-it, BiomedCLIP)
- Query quality evaluation via SciSpacy entity extraction
- Query expansion via Gemma-2-2B-it
- GradCAM-based bounding box generation via BiomedCLIP
"""

import warnings
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

# --- 1. SAFE IMPORTS ---

try:
    from pytorch_grad_cam import (
        GradCAM,
        ScoreCAM,
        GradCAMPlusPlus,
        AblationCAM,
        XGradCAM,
        EigenCAM,
    )
except ImportError:
    raise ImportError(
        "The 'grad-cam' library is not installed. Please check requirements.txt."
    )


# ==============================================================================
# 2. VIT RESHAPE TRANSFORM (Reused from bbox pipeline)
# ==============================================================================

def reshape_transform(
    tensor: torch.Tensor,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """
    Reshapes Vision Transformer (ViT) output tensors for compatibility with Grad-CAM.

    Handles two formats:
    1. (Batch, Tokens, Dim) -> Standard for OpenCLIP
    2. (Tokens, Batch, Dim) -> Legacy format
    """
    if tensor.shape[0] < tensor.shape[1]:
        patches = tensor[:, 1:, :]
        if height is None or width is None:
            num_patches = patches.shape[1]
            grid_side = int(num_patches**0.5)
            height = width = grid_side
        result = patches.reshape(tensor.shape[0], height, width, tensor.shape[2])
        result = result.permute(0, 3, 1, 2)
        return result
    else:
        if height is None or width is None:
            grid_square = len(tensor) - 1
            height = width = int(grid_square**0.5)
        result = tensor[1:, :, :].reshape(height, width, tensor.size(2))
        result = result.permute(2, 0, 1)
        return result.unsqueeze(0)


# ==============================================================================
# 3. CAM WRAPPER (Reused from bbox pipeline)
# ==============================================================================

class CAMWrapper:
    """Wrapper for pytorch-grad-cam compatible with BiomedCLIP."""

    CAM_DICT = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "gscorecam": ScoreCAM,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: list,
        tokenizer: Any,
        cam_version: str,
        preprocess: Any = None,
        cam_trans: Any = None,
    ):
        self.model = model
        self.target_layers = target_layers
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans

        raw_version = cam_version.lower() if cam_version else "gradcam"
        if raw_version in self.CAM_DICT:
            self.version = raw_version
        else:
            print(f"[Warning] Unknown CAM version '{cam_version}'. Defaulting to ScoreCAM.")
            self.version = "scorecam"

        self._load_cam()

    def _load_cam(self):
        batch_size = 16 if self.version in ["scorecam", "gscorecam"] else 1
        cam_cls = self.CAM_DICT[self.version]
        args = {
            "model": self.model,
            "target_layers": self.target_layers,
            "reshape_transform": self.cam_trans,
        }
        try:
            self.cam = cam_cls(**args)
        except TypeError:
            args["batch_size"] = batch_size
            self.cam = cam_cls(**args)

    def __call__(
        self,
        text: str,
        img: Any,
        label: Any = None,
    ) -> np.ndarray:
        """Generate a Class Activation Map for a single image-text pair."""
        if self.preprocess is not None:
            img = self.preprocess(img)

        text_token = None
        if self.tokenizer and text:
            text_token = self.tokenizer(text).cuda()

        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()

        if hasattr(self.model, "set_text") and text_token is not None:
            self.model.set_text(text_token)

        grayscale_cam = self.cam(input_tensor=img, targets=label)
        return grayscale_cam[0, :]


# ==============================================================================
# 4. MODEL LOADERS
# ==============================================================================

def load_scispacy():
    """Load the SciSpacy en_core_sci_sm model on CPU."""
    import spacy

    nlp = spacy.load("en_core_sci_sm")
    print("[INFO] SciSpacy en_core_sci_sm loaded (CPU)")
    return nlp


def load_gemma(model_name: str = "google/gemma-2-2b-it") -> Tuple[Any, Any]:
    """
    Load Gemma-2-2B-it for query expansion.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"[INFO] Gemma model loaded: {model_name} (float16, CUDA)")
    return model, tokenizer


def load_biomed_clip(
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
) -> Tuple[Any, Any, Any]:
    """
    Load BiomedCLIP via OpenCLIP.

    Returns:
        (model, preprocess, tokenizer) tuple.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.cuda().eval()
    print(f"[INFO] BiomedCLIP loaded: {model_name} (CUDA)")
    return model, preprocess, tokenizer


# ==============================================================================
# 5. QUERY QUALITY EVALUATION
# ==============================================================================

def evaluate_query_quality(
    nlp,
    query: str,
    entity_threshold: int = 2,
    word_threshold: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Evaluate whether a text query is detailed enough for visual grounding.

    Heuristic:
    - The query must contain at least `entity_threshold` clinical entities.
    - The query must be at least `word_threshold` words long.

    Args:
        nlp: SciSpacy Language model.
        query: The text query to evaluate.
        entity_threshold: Minimum number of clinical entities required.
        word_threshold: Minimum word count required.

    Returns:
        (is_detailed, entities) tuple.
    """
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    word_count = len(query.split())

    is_detailed = len(entities) >= entity_threshold and word_count >= word_threshold
    return is_detailed, entities


# ==============================================================================
# 6. QUERY EXPANSION
# ==============================================================================

EXPANSION_PROMPT_TEMPLATE = (
    "You are a radiology assistant. A user submitted a brief medical image query "
    "that lacks clinical detail. Rewrite and expand it into a highly descriptive "
    "clinical prompt suitable for a medical vision model. Keep it under 50 words.\n\n"
    "Original query: \"{query}\"\n\n"
    "Expanded clinical prompt:"
)


def expand_query(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Expand a brief query into a detailed clinical prompt using Gemma-2-2B-it.

    Args:
        model: The loaded Gemma model.
        tokenizer: The corresponding tokenizer.
        query: The original brief query.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The expanded query string.
    """
    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    expanded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Take only the first sentence/line to avoid runaway generation
    expanded = expanded.split("\n")[0].strip()
    if not expanded:
        return query  # Fallback to original if generation is empty

    return expanded


# ==============================================================================
# 7. GRADCAM BOUNDING BOX GENERATION
# ==============================================================================

def cam_to_bboxes(
    heatmap: np.ndarray,
    image_shape: Tuple[int, int],
    threshold: float = 0.5,
    min_box_area_ratio: float = 0.005,
    multi_box: bool = True,
) -> List[List[int]]:
    """
    Convert a GradCAM heatmap to bounding box coordinates.

    Args:
        heatmap: Grayscale CAM output (H, W), values in [0, 1].
        image_shape: (height, width) of the original image.
        threshold: Activation threshold for binarization.
        min_box_area_ratio: Minimum box area as fraction of total image area.
        multi_box: If True, return all valid boxes. If False, return only the largest.

    Returns:
        List of bounding boxes as [x1, y1, x2, y2].
    """
    h, w = image_shape[:2]

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Binarize
    binary = (heatmap_resized >= threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    image_area = h * w
    min_area = image_area * min_box_area_ratio

    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        box_area = bw * bh
        if box_area >= min_area:
            boxes.append([x, y, x + bw, y + bh])

    if not boxes:
        return []

    if not multi_box:
        # Return only the largest box
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return [boxes[0]]

    return boxes


def generate_cam_bbox(
    cam_wrapper: CAMWrapper,
    preprocess: Any,
    image,
    query: str,
    threshold: float = 0.5,
    multi_box: bool = True,
    min_box_area_ratio: float = 0.005,
) -> List[List[int]]:
    """
    Generate bounding boxes for an image-query pair using GradCAM.

    Args:
        cam_wrapper: Initialized CAMWrapper instance.
        preprocess: BiomedCLIP preprocessing transform.
        image: PIL.Image in RGB.
        query: The text query (original or expanded).
        threshold: CAM activation threshold.
        multi_box: Whether to return multiple boxes.
        min_box_area_ratio: Minimum box area as fraction of image area.

    Returns:
        List of bounding boxes as [x1, y1, x2, y2].
    """
    image_np = np.array(image)
    image_shape = image_np.shape[:2]

    # Generate heatmap
    preprocessed = preprocess(image)
    heatmap = cam_wrapper(text=query, img=preprocessed)

    # Convert heatmap to bboxes
    bboxes = cam_to_bboxes(
        heatmap,
        image_shape,
        threshold=threshold,
        min_box_area_ratio=min_box_area_ratio,
        multi_box=multi_box,
    )

    return bboxes
