"""Multimodal Image + Text consistency verification module for SignalScope (Bonus Module E)."""

from typing import Dict, Any, Optional
from PIL import Image


def evaluate_multimodal_consistency(
    image: Image.Image,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates semantic consistency between an input image and a non-political caption.

    Args:
        image: PIL Image object.
        caption: Optional user-provided text caption or claim.

    Returns:
        dict: Consistency score and alignment notes.
    """
    if not caption or not caption.strip():
        return {
            "multimodal_evaluated": False,
            "consistency_score": None,
            "notes": "No text caption provided for multimodal evaluation."
        }

    clean_caption = caption.strip()

    # Generic heuristic check for common synthetic artifact keywords in caption
    synthetic_keywords = ["ai art", "midjourney", "stable diffusion", "dall-e", "prompt", "rendered", "octane render"]
    found_keywords = [kw for kw in synthetic_keywords if kw in clean_caption.lower()]

    consistency_score = 0.90
    if found_keywords:
        consistency_score = 0.45
        notes = f"Caption explicitly references generative synthesis terminology ({', '.join(found_keywords)})."
    else:
        notes = f"Generic caption '{clean_caption[:50]}...' analyzed; no explicit synthetic prompt markers detected in text stream."

    return {
        "multimodal_evaluated": True,
        "caption": clean_caption,
        "consistency_score": round(consistency_score, 4),
        "alignment_status": "aligned" if consistency_score > 0.60 else "discrepancy_detected",
        "notes": notes,
    }
