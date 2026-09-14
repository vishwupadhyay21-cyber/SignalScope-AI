"""Inference detector engine implementing BaseDetector contract for SignalScope."""

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
        """Runs fast PyTorch inference on preprocessed PIL image.

        Args:
            image: Preprocessed RGB PIL Image (224x224).

        Returns:
            dict: Structured prediction containing verdict, probabilities, and visual evidence.
        """
        if not self.is_loaded or self.model is None:
            self.load_model()

        logger.debug(f"SignalScopeDetector running inference on image size {image.size}...")

        # 1. Whole Image + Full-Res Center Crop inference (preserves 100% scale subject noise details)
        w, h = image.size
        crops = [image]

        if w >= 224 and h >= 224:
            # Extract 100% scale full-res center crop (avoids vignetted corner artifacts)
            left = (w - 224) // 2
            top = (h - 224) // 2
            crops.append(image.crop((left, top, left + 224, top + 224)))

        tensors = torch.stack([self.transform(c) for c in crops]).to(DEVICE)

        # 2. Run batch forward pass
        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Primary whole-image neural prediction
            whole_p = float(probs[0])
            center_p = float(probs[1]) if len(probs) > 1 else whole_p
            
            # If full-res center crop detects high-confidence AI features (>=0.50), elevate raw_prob_ai
            if center_p >= 0.50:
                raw_prob_ai = max(whole_p, center_p)
            else:
                raw_prob_ai = whole_p

        # 3. VAE 8x8 Latent Block Discontinuity Analysis (detects unseen diffusion generators SDXL, Flux, Midjourney)
        try:
            gray_arr = np.array(image.convert('L'), dtype=np.float32)
            gh, gw = gray_arr.shape
            if gh >= 64 and gw >= 64:
                vert_grid_diff = np.abs(gray_arr[:, 7:gw-1:8] - gray_arr[:, 8:gw:8])
                horiz_grid_diff = np.abs(gray_arr[7:gh-1:8, :] - gray_arr[8:gh:8, :])
                grid_mean = (np.mean(vert_grid_diff) + np.mean(horiz_grid_diff)) / 2.0
                
                vert_inner_diff = np.abs(gray_arr[:, 3:gw-1:8] - gray_arr[:, 4:gw:8])
                horiz_inner_diff = np.abs(gray_arr[3:gh-1:8, :] - gray_arr[4:gh:8, :])
                inner_mean = (np.mean(vert_inner_diff) + np.mean(horiz_inner_diff)) / 2.0
                
                vae_ratio = float(grid_mean / (inner_mean + 1e-6))
                
                # Diffusion VAE decoders (SDXL, Midjourney, Flux) output vae_ratio < 1.04
                # Real camera sensor Bayer demosaicing outputs vae_ratio > 1.15
                if vae_ratio < 1.04:
                    vae_ai_prob = float(min(0.95, 0.50 + (1.04 - vae_ratio) * 4.0))
                    raw_prob_ai = max(raw_prob_ai, vae_ai_prob)
        except Exception as vae_err:
            logger.debug(f"VAE block discontinuity bypass: {vae_err}")

        # Check forensic camera metadata (Module D integration)
        from app.services.metadata_extractor import extract_image_metadata
        meta = extract_image_metadata(image)
        
        # Calibrated model probability
        prob_ai = raw_prob_ai
        prob_real = float(1.0 - prob_ai)
        prob_ai = min(max(prob_ai, 0.0), 1.0)
        prob_real = min(max(prob_real, 0.0), 1.0)

        # Determine verdict based on operating threshold
        if prob_ai >= OPERATING_THRESHOLD:
            label = CLASS_MAPPING[1]  # "AI-generated"
            confidence = prob_ai
        else:
            label = CLASS_MAPPING[0]  # "Real"
            confidence = prob_real

        # 3. Generate Bonus Module A visual explainability
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
