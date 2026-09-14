"""Image metadata and EXIF analysis service.

Extracts useful forensic metadata (camera make, model, software, timestamp, dimensions)
from in-memory validated images without modifying the image or saving files to disk.

IMPORTANT FORENSIC PRINCIPLE:
Metadata is treated purely as additional forensic evidence, NOT as proof of authenticity
or AI generation. Metadata must never influence or override the detector model verdict.
"""

from typing import Dict, Any, Optional
from PIL import Image, ExifTags

from app.core.logger import logger

# Tag constants for standard EXIF fields
TAG_MAKE = 0x010F        # 271 - Camera manufacturer
TAG_MODEL = 0x0110       # 272 - Camera model
TAG_SOFTWARE = 0x0131    # 305 - Software / editing tool
TAG_DATETIME = 0x0132    # 306 - Date/time modified
TAG_DATETIME_ORIG = 0x9003  # 36867 - Date/time original (in Exif IFD)


def _sanitize_string(value: Any, max_length: int = 256) -> Optional[str]:
    """Sanitizes raw EXIF tag values into safe, printable strings.

    Strips null bytes, leading/trailing whitespace, and truncates unusually long strings.
    """
    if value is None:
        return None
    try:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        # Remove null terminators and surrounding whitespace
        cleaned = text.replace("\x00", "").strip()
        if not cleaned:
            return None
        return cleaned[:max_length]
    except Exception:
        return None


def extract_image_metadata(image: Image.Image) -> Dict[str, Any]:
    """Extracts forensic metadata and EXIF properties safely from a PIL Image.

    This function is completely non-crashing: if an image has no EXIF, invalid tags,
    or corrupted metadata headers, it safely degrades to `{"available": False}`.

    Args:
        image: A validated PIL Image instance.

    Returns:
        dict: Structured metadata dictionary conforming to ImageMetadata schema.
    """
    try:
        exif = image.getexif()
        if not exif or len(exif) == 0:
            logger.debug("No EXIF metadata found in image.")
            return {"available": False}

        # 1. Extract standard IFD0 tags
        camera_make = _sanitize_string(exif.get(TAG_MAKE))
        camera_model = _sanitize_string(exif.get(TAG_MODEL))
        software = _sanitize_string(exif.get(TAG_SOFTWARE))
        datetime_str = _sanitize_string(exif.get(TAG_DATETIME))

        # 2. Check Exif sub-IFD (for DateTimeOriginal) if datetime wasn't in IFD0
        if not datetime_str:
            try:
                # 0x8769 is the Exif sub-IFD pointer
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd:
                    datetime_str = _sanitize_string(exif_ifd.get(TAG_DATETIME_ORIG))
            except Exception as ifd_err:
                logger.debug(f"Could not read Exif sub-IFD: {ifd_err}")

        # 3. Assemble structured result
        result: Dict[str, Any] = {
            "available": True,
            "image_format": getattr(image, "format", None),
            "width": image.width if hasattr(image, "width") else None,
            "height": image.height if hasattr(image, "height") else None,
        }

        # Only include forensic fields if they have meaningful values
        if camera_make:
            result["camera_make"] = camera_make
        if camera_model:
            result["camera_model"] = camera_model
        if software:
            result["software"] = software
        if datetime_str:
            result["datetime"] = datetime_str

        logger.info(
            f"Forensic metadata extracted: make='{camera_make}', model='{camera_model}', "
            f"software='{software}', datetime='{datetime_str}'"
        )
        return result

    except Exception as err:
        # Guarantee: metadata extraction failure MUST NEVER crash the API
        logger.warning(f"Metadata extraction encountered an unexpected error: {err}. Returning available=False.")
        return {"available": False}
