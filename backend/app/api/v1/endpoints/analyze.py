"""Image analysis, detection, and metadata extraction endpoint."""

import uuid
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.schemas.analyze import AnalysisResponse
from app.services.image_validator import validate_and_process_image
from app.services.metadata_extractor import extract_image_metadata
from app.ml.preprocessor import preprocess_image
from app.ml.factory import get_detector
from app.core.logger import logger

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Analyze Image for AI vs Real Classification and Forensic Metadata",
    description=(
        "Accepts an image file via multipart/form-data, validates format and integrity in memory, "
        "extracts forensic EXIF metadata safely without influencing the classification verdict, "
        "preprocesses the image to standard dimensions (224x224 RGB), passes it to the detector model, "
        "and returns prediction verdict, confidence probabilities, and metadata with a unique analysis ID."
    ),
    responses={
        200: {
            "description": "Image successfully analyzed.",
            "model": AnalysisResponse,
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "analysis_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "filename": "sample.jpg",
                        "verdict": {
                            "label": "AI-generated",
                            "confidence": 0.85,
                        },
                        "probabilities": {
                            "real": 0.15,
                            "ai_generated": 0.85,
                        },
                        "metadata": {
                            "available": True,
                            "camera_make": "Canon",
                            "camera_model": "EOS R5",
                            "software": "Adobe Photoshop",
                            "datetime": "2026:09:11 12:00:00",
                        },
                    }
                }
            },
        },
        400: {
            "description": "No file provided, empty file, or corrupted/invalid image data.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "code": 400,
                        "message": "Uploaded file is corrupted or not a valid image.",
                    }
                }
            },
        },
        413: {
            "description": "File exceeds maximum configured size limit.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "code": 413,
                        "message": "File size exceeds the maximum allowed limit of 10MB.",
                    }
                }
            },
        },
        415: {
            "description": "Unsupported image format (only JPEG, PNG, WEBP allowed).",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "code": 415,
                        "message": "Unsupported image format 'GIF'. Allowed formats: JPEG, PNG, WEBP.",
                    }
                }
            },
        },
    },
)
async def analyze_image(
    file: Optional[UploadFile] = File(
        None,
        description="Image file to upload and analyze (JPEG, PNG, or WEBP format).",
    ),
):
    """Executes the complete Phase 4 image analysis pipeline:

    1. Validate image in memory (format, size, integrity).
    2. Extract available forensic EXIF metadata (treated purely as evidence, not proof).
    3. Preprocess image (RGB conversion, resizing to 224x224).
    4. Run prediction with the configured detector (DummyDetector in Dev).
    5. Generate a unique analysis tracking UUID.
    6. Return unified structured JSON response with verdict, probabilities, and metadata.
    """
    # 1. Validation (Phase 2)
    validation_result = await validate_and_process_image(file)
    validated_image = validation_result["image"]

    # 2. Metadata & EXIF Analysis (Phase 4)
    # Forensic context only; does NOT modify or influence prediction
    metadata = extract_image_metadata(validated_image)

    # 3. Preprocessing & Normalization (Phase 3 & Phase 5)
    try:
        preprocessed_img = preprocess_image(validated_image)
    except Exception as prep_err:
        logger.error(
            f"Image preprocessing failed for '{validation_result.get('filename')}': {prep_err}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image preprocessing failed. The uploaded image could not be processed for analysis.",
        )

    # 4. Model Inference (Phase 3)
    detector = get_detector()
    prediction = detector.predict(preprocessed_img)

    # 5. Generate unique tracking ID
    analysis_id = str(uuid.uuid4())

    # 6. Return structured response
    return {
        "status": "success",
        "analysis_id": analysis_id,
        "filename": validation_result["filename"],
        "verdict": prediction["verdict"],
        "probabilities": prediction["probabilities"],
        "metadata": metadata,
    }
