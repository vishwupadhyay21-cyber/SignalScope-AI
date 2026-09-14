"""Dummy Detector implementation for development and testing.

Simulates the behavior of a future deep learning model without requiring
heavy weights, GPU, or model training.
"""

from typing import Dict, Any
from PIL import Image

from app.core.logger import logger
from app.ml.base import BaseDetector


class DummyDetector(BaseDetector):
    """A simulated detector that returns a deterministic verdict and probability distribution."""

    def __init__(self) -> None:
        """Initializes the dummy detector and simulates loading the model."""
        self.is_loaded: bool = False
        self.load_model()

    def load_model(self) -> None:
        """Simulates loading model architecture and weights."""
        logger.info("Initializing DummyDetector [Development Mode]...")
        self.is_loaded = True
        logger.info("DummyDetector loaded successfully.")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Returns a fixed mock prediction mimicking a trained binary classifier.

        Args:
            image: Preprocessed PIL Image.

        Returns:
            dict: Structured prediction with verdict and probabilities.
        """
        logger.debug(f"DummyDetector running inference on image sized {image.size}...")

        # Fixed prediction output conforming to Phase 3 requirements
        return {
            "verdict": {
                "label": "AI-generated",
                "confidence": 0.85,
            },
            "probabilities": {
                "real": 0.15,
                "ai_generated": 0.85,
            },
        }
