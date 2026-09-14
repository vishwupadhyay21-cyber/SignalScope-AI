"""Pydantic schemas for the image analysis endpoint."""

from typing import Optional
from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    """Detailed structural information about the verified image."""

    format: str = Field(..., description="Format detected from image content (e.g., JPEG, PNG, WEBP)")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class Verdict(BaseModel):
    """Classification verdict from the detector."""

    label: str = Field(..., description="Predicted class label (e.g., AI-generated or Real)", example="AI-generated")
    confidence: float = Field(..., description="Confidence score for the winning label (0.0 to 1.0)", example=0.85)


class Probabilities(BaseModel):
    """Confidence probabilities across all classes."""

    real: float = Field(..., description="Probability that the image is Real (0.0 to 1.0)", example=0.15)
    ai_generated: float = Field(..., description="Probability that the image is AI-generated (0.0 to 1.0)", example=0.85)


class ImageMetadata(BaseModel):
    """Forensic metadata extracted from image headers and EXIF."""

    available: bool = Field(..., description="Whether EXIF metadata was discovered in the image", example=True)
    image_format: Optional[str] = Field(None, description="Image container format (JPEG, PNG, WEBP)", example="JPEG")
    width: Optional[int] = Field(None, description="Image width in pixels", example=1920)
    height: Optional[int] = Field(None, description="Image height in pixels", example=1080)
    camera_make: Optional[str] = Field(None, description="Camera manufacturer if available", example="Canon")
    camera_model: Optional[str] = Field(None, description="Camera model if available", example="EOS R5")
    software: Optional[str] = Field(None, description="Software used if available", example="Adobe Photoshop")
    datetime: Optional[str] = Field(None, description="Capture timestamp if available", example="2026:09:11 12:00:00")


class AnalysisResponse(BaseModel):
    """Structured response returned by the image analysis pipeline."""

    status: str = Field(default="success", description="Status indicator", example="success")
    analysis_id: str = Field(..., description="Unique UUID identifier for this analysis run", example="a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    filename: Optional[str] = Field(None, description="Original filename of the uploaded image", example="sample.jpg")
    verdict: Verdict
    probabilities: Probabilities
    metadata: ImageMetadata
