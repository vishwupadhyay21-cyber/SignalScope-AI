from typing import Optional

from app.ml.base import BaseDetector
from app.ml.inference import SignalScopeDetector

_detector_instance: Optional[BaseDetector] = None


def get_detector() -> BaseDetector:
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = SignalScopeDetector()

    return _detector_instance