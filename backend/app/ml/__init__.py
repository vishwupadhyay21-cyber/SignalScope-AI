"""Machine Learning detector module for SignalScope.

Provides modular interfaces for model loading, preprocessing, and inference.
"""

from app.ml.base import BaseDetector
from app.ml.dummy import DummyDetector
from app.ml.factory import get_detector
from app.ml.preprocessor import preprocess_image

__all__ = ["BaseDetector", "DummyDetector", "get_detector", "preprocess_image"]
