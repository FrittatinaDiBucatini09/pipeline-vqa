"""
Step 2: Segmentation Refinement Module
--------------------------------------
Implements MedSAM (Segment Anything Model) logic to refine 
bounding boxes into pixel-perfect masks.
"""

from .utils import MedSAMWrapper, overlay_mask

__all__ = ['MedSAMWrapper', 'overlay_mask']