"""Image preprocessing and normalization utilities for machine learning inference.

Normalizes diverse image formats, color modes, transparency channels, EXIF orientations,
and aspect ratios into a uniform 224x224 RGB image representation for future model inference.
"""

from typing import Tuple
from PIL import Image, ImageOps

from app.core.logger import logger


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """Normalizes and prepares a validated PIL Image for model inference.

    Steps performed:
    1. EXIF Orientation Normalization:
       Corrects physical image rotation using embedded EXIF orientation tags
       (e.g., photos taken sideways on smartphones or cameras).
    2. Color & Transparency Normalization:
       - Transparent images (RGBA, LA, or paletted with alpha) are composited
         onto a solid white RGB background to avoid black silhouette artifacts.
       - Grayscale images (L mode) and other color spaces are converted to 3-channel RGB.
    3. Aspect-Ratio Safe Resizing:
       Scales and center-crops the image to the standard model input dimensions
       (target_size, defaulting to 224x224) using high-quality Bilinear resampling
       without stretching or squishing non-square inputs.

    Args:
        image: Original validated in-memory PIL Image.
        target_size: Target (width, height) tuple for model input. Default: (224, 224).

    Returns:
        Image.Image: Normalized 3-channel RGB PIL Image of dimensions target_size.

    Raises:
        ValueError: If preprocessing fails due to corrupted or invalid image data.
    """
    logger.debug(f"Starting image preprocessing: mode={image.mode}, size={image.size}")

    try:
        # Step 1: EXIF Orientation Normalization
        try:
            transposed = ImageOps.exif_transpose(image)
            if transposed is not None:
                image = transposed
        except Exception as exif_err:
            logger.debug(f"EXIF orientation transpose bypassed: {exif_err}")

        # Step 2: Color Space & Alpha Normalization to RGB
        has_alpha = (
            image.mode in ("RGBA", "LA")
            or (image.mode == "P" and "transparency" in getattr(image, "info", {}))
        )

        if has_alpha:
            # Composite transparent areas over a neutral white canvas
            rgba_image = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(canvas, rgba_image)
            image = composite.convert("RGB")
        elif image.mode != "RGB":
            # Grayscale (L), CMYK, or Paletted (P) without transparency
            image = image.convert("RGB")

        # Step 3: Aspect-Ratio Safe Resizing
        if image.size != target_size:
            image = ImageOps.fit(image, target_size, method=Image.Resampling.BILINEAR)

        logger.debug(f"Preprocessing completed: mode={image.mode}, size={image.size}")
        return image

    except Exception as err:
        logger.error(f"Error during image preprocessing: {err}", exc_info=True)
        raise ValueError(f"Image preprocessing failed: {err}") from err
