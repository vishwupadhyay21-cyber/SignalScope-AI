"""Image validation service.

Processes uploaded images safely in memory, verifies their integrity and formats,
and rejects invalid, oversized, or corrupted files without saving them to disk.
"""

import io
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.logger import logger


# Mapping of verified Pillow formats to standard MIME types
FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def _sanitize_filename_for_log(filename: str, max_length: int = 100) -> str:
    """Removes log-injection characters from a filename before it is written to logs.

    Strips newlines, carriage returns, and null bytes that an attacker could embed
    in a crafted filename to forge or corrupt log entries.

    Args:
        filename: Raw filename string received from the upload.
        max_length: Maximum characters to include (prevents very long log lines).

    Returns:
        A safe, printable string suitable for log output.
    """
    sanitized = filename.replace("\n", "").replace("\r", "").replace("\x00", "")
    return sanitized[:max_length]


async def validate_and_process_image(file: Optional[UploadFile]) -> dict:
    """Validates an uploaded image file completely in memory.

    Performs the following checks:
    1. Ensures a file was actually provided.
    2. Reads the upload with a hard byte cap to prevent memory exhaustion.
    3. Enforces configurable file size limits.
    4. Verifies file integrity using Pillow (detects corrupt/fake images).
    5. Validates format against allowed list (JPEG, PNG, WEBP).
    6. Extracts image dimensions safely.

    Args:
        file: The UploadFile object received from the multipart/form-data request.

    Returns:
        dict: A dictionary containing verified image metadata matching ImageAnalysisSuccessResponse.

    Raises:
        HTTPException: With appropriate HTTP status code (400, 413, 415) and error message.
    """
    # 1. Check if a file was provided
    if file is None or not file.filename or file.filename.strip() == "":
        logger.warning("Upload rejected: No file provided in the request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided. Please upload an image file.",
        )

    # Sanitize the filename once for safe use in all log statements below.
    # This prevents an attacker from injecting newlines into log entries via filename.
    safe_name = _sanitize_filename_for_log(file.filename)

    # 2. Stream-capped read — never buffer more than MAX_REQUEST_BODY_SIZE_MB into RAM.
    # Passing (cap + 1) means: if we get cap+1 bytes back, the upload is over the limit
    # and we reject it immediately without reading the rest of the payload.
    cap = settings.max_request_body_size_bytes
    contents = await file.read(cap + 1)

    if len(contents) > cap:
        logger.warning(
            f"Upload rejected: File '{safe_name}' exceeds the hard body-read cap "
            f"({settings.MAX_REQUEST_BODY_SIZE_MB}MB)."
        )
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File size exceeds the maximum allowed limit of {settings.MAX_IMAGE_SIZE_MB}MB.",
        )

    # Check for empty file
    if len(contents) == 0:
        logger.warning(f"Upload rejected: File '{safe_name}' is empty (0 bytes).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 3. Check file size against the configurable image limit
    if len(contents) > settings.max_image_size_bytes:
        logger.warning(
            f"Upload rejected: File '{safe_name}' size ({len(contents)} bytes) "
            f"exceeds limit ({settings.max_image_size_bytes} bytes / {settings.MAX_IMAGE_SIZE_MB}MB)."
        )
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File size exceeds the maximum allowed limit of {settings.MAX_IMAGE_SIZE_MB}MB.",
        )

    # 4. In-memory validation with Pillow
    # We do NOT trust the file extension or declared content-type.
    # We open the actual binary bytes to verify it is a valid image.
    try:
        # Step A: Verify file header and structure
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
            detected_format = img.format

        # Step B: Pillow's verify() invalidates the file pointer, so we re-open to inspect data & dimensions
        pil_image = Image.open(io.BytesIO(contents))
        # Force loading pixel data to ensure the image is not truncated or corrupted halfway through
        pil_image.load()
        width, height = pil_image.size
        final_format = (pil_image.format or detected_format or "").upper()

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as err:
        logger.warning(f"Upload rejected: File '{safe_name}' failed image decoding: {err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a valid image.",
        )

    # 5. Verify image format is allowed (JPEG, PNG, WEBP)
    if final_format not in settings.ALLOWED_IMAGE_FORMATS:
        logger.warning(
            f"Upload rejected: File '{safe_name}' has unsupported format '{final_format}'. "
            f"Allowed: {settings.ALLOWED_IMAGE_FORMATS}."
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format '{final_format}'. Allowed formats: {', '.join(settings.ALLOWED_IMAGE_FORMATS)}.",
        )

    # 6. Verify image dimensions do not exceed maximum safety limit
    if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
        logger.warning(
            f"Upload rejected: File '{safe_name}' dimensions ({width}x{height}) "
            f"exceed maximum safety limit of {settings.MAX_IMAGE_DIMENSION}px."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions ({width}x{height}) exceed the maximum allowed limit of {settings.MAX_IMAGE_DIMENSION}px.",
        )

    # Resolve canonical MIME type
    canonical_mime = FORMAT_TO_MIME.get(final_format, f"image/{final_format.lower()}")

    logger.info(
        f"Image verified successfully: {safe_name} [{final_format}, {width}x{height}, {len(contents)} bytes]"
    )

    return {
        "status": "success",
        "message": "Image validation successful",
        "filename": file.filename,
        "content_type": canonical_mime,
        "image_info": {
            "format": final_format,
            "width": width,
            "height": height,
        },
        "image": pil_image,
    }
