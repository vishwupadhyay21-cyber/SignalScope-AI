"""Real PyTorch detector for SignalScope."""
from model.explain import generate_visual_explanation
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from app.core.logger import logger
from app.ml.base import BaseDetector

from model.config import (
    DEVICE,
    MODEL_WEIGHTS_PATH,
    CLASS_MAPPING,
    OPERATING_THRESHOLD,
)
from model.model import build_model
from model.transforms import get_inference_transforms


class SignalScopeDetector(BaseDetector):
    """Loads the trained SignalScope model and performs inference."""

    def __init__(
        self,
        weights_path: str = str(MODEL_WEIGHTS_PATH),
    ) -> None:
        self.weights_path = weights_path
        self.model: Optional[torch.nn.Module] = None
        self.transform = get_inference_transforms()
        self.is_loaded = False
        self.load_model()

    def load_model(self) -> None:
        """Load model architecture and weights once."""
        logger.info(f"Initializing SignalScopeDetector on {DEVICE}")

        self.model = build_model(pretrained=False)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}"
            )

        state_dict = torch.load(
            self.weights_path,
            map_location=DEVICE,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        self.is_loaded = True

        logger.info(
            f"SignalScope model loaded from {self.weights_path}"
        )

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference and return the backend response format."""
        if not self.is_loaded or self.model is None:
            self.load_model()

        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            ai_probability = float(
                torch.sigmoid(logits).reshape(-1)[0].item()
            )

        ai_probability = max(0.0, min(1.0, ai_probability))
        real_probability = 1.0 - ai_probability

        if ai_probability >= OPERATING_THRESHOLD:
            label = CLASS_MAPPING[1]
            confidence = ai_probability
        else:
            label = CLASS_MAPPING[0]
            confidence = real_probability

        return {
            "verdict": {
                "label": label,
                "confidence": round(confidence, 4),
            },
            "probabilities": {
                "real": round(real_probability, 4),
                "ai_generated": round(ai_probability, 4),
            },
            "visual_evidence": {
                "heatmap_available": False,
                "message": "Grad-CAM integration pending",
            },
            "explanation": {
                "summary": f"Verdict: {label}",
                "cues": [],
            },
        }