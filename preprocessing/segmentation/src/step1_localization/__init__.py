"""
Step 1: Weakly-Supervised Localization Module
---------------------------------------------
Implements BiomedCLIP + gScoreCAM logic to generate bounding boxes 
from text prompts.
"""

from .utils import CAMWrapper, load_clip, reshape_transform

__all__ = ['CAMWrapper', 'load_clip', 'reshape_transform']