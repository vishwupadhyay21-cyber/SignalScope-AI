"""Backup copy of model/inference.py"""
import os
from typing import Dict, Any, Optional
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import numpy as np
import torch


from app.ml.base import BaseDetector
from app.core.logger import logger
from model.config import (
    DEVICE,
    MODEL_WEIGHTS_PATH,
    CLASS_MAPPING,
    OPERATING_THRESHOLD,
)
from model.model import build_model, SignalScopeClassifier
from model.transforms import get_inference_transforms
from model.explain import generate_visual_explanation


class SignalScopeDetector(BaseDetector):
    """Real PyTorch Image-Forensics Detector integrated with FastAPI Backend."""

    def __init__(self, weights_path: str = str(MODEL_WEIGHTS_PATH)) -> None:
        """Initializes detector and loads trained model weights into memory."""
        self.weights_path = weights_path
        self.model: Optional[SignalScopeClassifier] = None
        self.transform = get_inference_transforms()
        self.is_loaded: bool = False
        self.load_model()

    def load_model(self) -> None:
        """Loads neural network architecture and trained weights into memory once."""
        logger.info(f"Initializing SignalScopeDetector on {DEVICE}...")
        self.model = build_model(pretrained=True if not os.path.exists(self.weights_path) else False)

        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                logger.info(f" Successfully loaded trained weights from {self.weights_path}")
            except Exception as err:
                logger.error(f" Failed loading weights from {self.weights_path}: {err}. Falling back to initialized model.")
        else:
            logger.warning(f" Model weights file not found at '{self.weights_path}'. Running with initialized weights.")

        self.model.to(DEVICE)
        self.model.eval()
        self.is_loaded = True
        logger.info("SignalScopeDetector loaded and ready for inference.")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs fast PyTorch inference on preprocessed PIL image."""
        if not self.is_loaded or self.model is None:
            self.load_model()

        logger.debug(f"SignalScopeDetector running inference on image size {image.size}...")

        w, h = image.size
        crops = [image]

        if w >= 224 and h >= 224:
            left = (w - 224) // 2
            top = (h - 224) // 2
            crops.append(image.crop((left, top, left + 224, top + 224)))
            crops.append(image.crop((0, 0, 224, 224)))
            crops.append(image.crop((w - 224, 0, w, 224)))
            crops.append(image.crop((0, h - 224, 224, h)))
            crops.append(image.crop((w - 224, h - 224, w, h)))

        tensors = torch.stack([self.transform(c) for c in crops]).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            max_p = float(np.max(probs))
            mean_p = float(np.mean(probs))
            raw_prob_ai = 0.6 * max_p + 0.4 * mean_p

        from app.services.metadata_extractor import extract_image_metadata
        meta = extract_image_metadata(image)
        
        prob_ai = raw_prob_ai
        prob_real = float(1.0 - prob_ai)
        prob_ai = min(max(prob_ai, 0.0), 1.0)
        prob_real = min(max(prob_real, 0.0), 1.0)

        if prob_ai >= OPERATING_THRESHOLD:
            label = CLASS_MAPPING[1]  # "AI-generated"
            confidence = prob_ai
        else:
            label = CLASS_MAPPING[0]  # "Real"
            confidence = prob_real

        try:
            explanation_data = generate_visual_explanation(
                model=self.model,
                pil_image=image,
                predicted_label=label,
                confidence=confidence,
            )
            visual_evidence = explanation_data["visual_evidence"]
            explanation = explanation_data["explanation"]
        except Exception as exp_err:
            logger.debug(f"Explainability generation bypassed: {exp_err}")
            visual_evidence = {"heatmap_available": False}
            explanation = {"summary": f"Verdict: {label}", "cues": []}

        output = {
            "verdict": {
                "label": label,
                "confidence": round(confidence, 4),
            },
            "probabilities": {
                "real": round(prob_real, 4),
                "ai_generated": round(prob_ai, 4),
            },
            "visual_evidence": visual_evidence,
            "explanation": explanation,
            "metadata_evidence": meta,
        }

        logger.debug(f"Inference result: {label} (confidence={confidence:.4f})")
        return output
