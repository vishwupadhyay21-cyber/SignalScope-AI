"""Abstract Base Detector Interface.

Defines the contract that any real or dummy AI detection model must implement.
This enables swapping between dummy and trained PyTorch models seamlessly.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image


class BaseDetector(ABC):
    """Abstract base class representing an AI media detector."""

    @abstractmethod
    def load_model(self) -> None:
        """Loads model weights, network architecture, or external resources into memory."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs inference on a preprocessed PIL image and returns prediction probabilities.

        Args:
            image: A preprocessed PIL Image instance (e.g. RGB, standard dimensions).

        Returns:
            dict: Prediction dictionary containing 'verdict' and 'probabilities'.
                Example:
                {
                    "verdict": {
                        "label": "AI-generated",
                        "confidence": 0.85
                    },
                    "probabilities": {
                        "real": 0.15,
                        "ai_generated": 0.85
                    }
                }
        """
        pass
