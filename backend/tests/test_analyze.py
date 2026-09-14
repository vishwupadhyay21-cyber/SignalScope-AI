"""Automated unit tests for Phase 2 image validation, Phase 3 ML detector, and Phase 4 EXIF metadata."""

import io
import uuid
import unittest
from unittest.mock import MagicMock
from PIL import Image
from starlette.datastructures import UploadFile
from fastapi import HTTPException

from app.core.config import settings
from app.services.image_validator import validate_and_process_image, _sanitize_filename_for_log
from app.services.metadata_extractor import extract_image_metadata
from app.ml.preprocessor import preprocess_image
from app.ml.dummy import DummyDetector
from app.ml.factory import get_detector
from app.api.v1.endpoints.analyze import analyze_image


def generate_test_image(
    format_name: str,
    size=(200, 150),
    color=(100, 150, 200),
    mode="RGB",
    exif_data: dict = None,
) -> bytes:
    """Helper to generate in-memory image bytes for testing, optionally embedding EXIF."""
    buf = io.BytesIO()
    img = Image.new(mode, size, color=color)
    if exif_data:
        exif = img.getexif()
        for tag_id, val in exif_data.items():
            exif[tag_id] = val
        img.save(buf, format=format_name, exif=exif)
    else:
        img.save(buf, format=format_name)
    return buf.getvalue()


class TestPhase6Security(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying Phase 6 security hardening (CORS, sanitization, body cap)."""

    def test_s1_filename_sanitizer_strips_newlines(self):
        """Ensure _sanitize_filename_for_log removes log-injection characters."""
        dangerous = "evil\nINJECTED_LOG_LINE\r\x00extra"
        result = _sanitize_filename_for_log(dangerous)
        self.assertNotIn("\n", result)
        self.assertNotIn("\r", result)
        self.assertNotIn("\x00", result)
        self.assertIn("evil", result)
        self.assertIn("extra", result)

    def test_s2_filename_sanitizer_truncates_long_names(self):
        """Ensure _sanitize_filename_for_log caps output at max_length."""
        long_name = "a" * 200
        result = _sanitize_filename_for_log(long_name, max_length=100)
        self.assertEqual(len(result), 100)

    def test_s3_filename_sanitizer_preserves_normal_names(self):
        """Ensure _sanitize_filename_for_log leaves safe filenames unchanged."""
        normal = "my_photo_2026.jpg"
        self.assertEqual(_sanitize_filename_for_log(normal), normal)


class TestImageValidation(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying upload and validation requirements (Phase 2)."""

    async def test_01_valid_jpeg(self):
        """1. Accept and validate valid JPEG images."""
        data = generate_test_image("JPEG", (1920, 1080))
        upload = UploadFile(file=io.BytesIO(data), filename="test.jpg", headers={"content-type": "image/jpeg"})
        result = await validate_and_process_image(upload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filename"], "test.jpg")
        self.assertEqual(result["content_type"], "image/jpeg")
        self.assertEqual(result["image_info"]["format"], "JPEG")
        self.assertEqual(result["image_info"]["width"], 1920)
        self.assertEqual(result["image_info"]["height"], 1080)
        self.assertIn("image", result)

    async def test_02_valid_png(self):
        """2. Accept and validate valid PNG images."""
        data = generate_test_image("PNG", (800, 600))
        upload = UploadFile(file=io.BytesIO(data), filename="test.png", headers={"content-type": "image/png"})
        result = await validate_and_process_image(upload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filename"], "test.png")
        self.assertEqual(result["content_type"], "image/png")
        self.assertEqual(result["image_info"]["format"], "PNG")
        self.assertEqual(result["image_info"]["width"], 800)
        self.assertEqual(result["image_info"]["height"], 600)
        self.assertIn("image", result)

    async def test_03_valid_webp(self):
        """3. Accept and validate valid WEBP images."""
        data = generate_test_image("WEBP", (320, 240))
        upload = UploadFile(file=io.BytesIO(data), filename="test.webp", headers={"content-type": "image/webp"})
        result = await validate_and_process_image(upload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filename"], "test.webp")
        self.assertEqual(result["content_type"], "image/webp")
        self.assertEqual(result["image_info"]["format"], "WEBP")
        self.assertEqual(result["image_info"]["width"], 320)
        self.assertEqual(result["image_info"]["height"], 240)
        self.assertIn("image", result)

    async def test_04_unsupported_format(self):
        """4. Reject unsupported formats (e.g. GIF) with HTTP 415."""
        data = generate_test_image("GIF", (100, 100))
        upload = UploadFile(file=io.BytesIO(data), filename="animation.gif", headers={"content-type": "image/gif"})

        with self.assertRaises(HTTPException) as ctx:
            await validate_and_process_image(upload)
        self.assertEqual(ctx.exception.status_code, 415)
        self.assertIn("Unsupported image format", ctx.exception.detail)

    async def test_05_corrupted_image(self):
        """5. Detect and reject corrupted or fake images with HTTP 400."""
        corrupt_data = b"NOT_A_VALID_IMAGE_PAYLOAD_12345"
        upload = UploadFile(file=io.BytesIO(corrupt_data), filename="fake.jpg", headers={"content-type": "image/jpeg"})

        with self.assertRaises(HTTPException) as ctx:
            await validate_and_process_image(upload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("corrupted or not a valid image", ctx.exception.detail)

    async def test_06a_no_file_provided(self):
        """6a. Reject requests where no file is provided with HTTP 400."""
        with self.assertRaises(HTTPException) as ctx:
            await validate_and_process_image(None)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("No file provided", ctx.exception.detail)

    async def test_06b_empty_file_provided(self):
        """6b. Reject empty 0-byte uploads with HTTP 400."""
        upload = UploadFile(file=io.BytesIO(b""), filename="empty.jpg", headers={"content-type": "image/jpeg"})
        with self.assertRaises(HTTPException) as ctx:
            await validate_and_process_image(upload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Uploaded file is empty", ctx.exception.detail)

    async def test_07_file_too_large(self):
        """7. Reject files exceeding the configured limit with HTTP 413."""
        original_limit = settings.MAX_IMAGE_SIZE_MB
        try:
            settings.MAX_IMAGE_SIZE_MB = 1
            oversized_data = b"0" * (1024 * 1024 + 1024)
            upload = UploadFile(file=io.BytesIO(oversized_data), filename="large.jpg", headers={"content-type": "image/jpeg"})

            with self.assertRaises(HTTPException) as ctx:
                await validate_and_process_image(upload)
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertIn("exceeds the maximum allowed limit", ctx.exception.detail)
        finally:
            settings.MAX_IMAGE_SIZE_MB = original_limit

    async def test_07b_body_cap_protection(self):
        """7b. Verify the hard body-read cap stops reading before bufferring the full payload."""
        original_cap = settings.MAX_REQUEST_BODY_SIZE_MB
        try:
            # Set cap to 1 MB so the test stays fast
            settings.MAX_REQUEST_BODY_SIZE_MB = 1
            # Upload 1 MB + 2 KB — enough to exceed the cap by more than 1 byte,
            # ensuring file.read(cap + 1) returns cap + 1 bytes and triggers rejection.
            over_cap_data = b"0" * (1024 * 1024 + 2048)
            upload = UploadFile(
                file=io.BytesIO(over_cap_data),
                filename="huge.jpg",
                headers={"content-type": "image/jpeg"},
            )

            with self.assertRaises(HTTPException) as ctx:
                await validate_and_process_image(upload)
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertIn("exceeds the maximum allowed limit", ctx.exception.detail)
        finally:
            settings.MAX_REQUEST_BODY_SIZE_MB = original_cap


class TestMLDetectorInterface(unittest.TestCase):
    """Test suite verifying the ML detector interface, preprocessor, and factory (Phase 3)."""

    def test_08_preprocessor_resizing_and_color_mode(self):
        """Ensure preprocessor converts RGBA to RGB and resizes to target (224, 224)."""
        raw_rgba = Image.new("RGBA", (500, 300), color=(255, 0, 0, 128))
        processed = preprocess_image(raw_rgba, target_size=(224, 224))

        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_09_dummy_detector_prediction_output(self):
        """Ensure DummyDetector conforms to the expected classification output schema."""
        detector = DummyDetector()
        self.assertTrue(detector.is_loaded)

        dummy_img = Image.new("RGB", (224, 224), color=(0, 255, 0))
        output = detector.predict(dummy_img)

        self.assertIn("verdict", output)
        self.assertIn("probabilities", output)
        self.assertEqual(output["verdict"]["label"], "AI-generated")
        self.assertEqual(output["verdict"]["confidence"], 0.85)
        self.assertEqual(output["probabilities"]["real"], 0.15)
        self.assertEqual(output["probabilities"]["ai_generated"], 0.85)

    def test_10_factory_singleton(self):
        """Ensure get_detector returns a singleton instance conforming to BaseDetector."""
        d1 = get_detector()
        d2 = get_detector()
        self.assertIs(d1, d2)
        self.assertTrue(d1.is_loaded)


class TestMetadataExtraction(unittest.TestCase):
    """Test suite verifying forensic metadata and EXIF extraction (Phase 4)."""

    def test_12_image_with_exif_metadata(self):
        """Ensure EXIF metadata (camera make, model, software, datetime) is extracted correctly."""
        exif_tags = {
            0x010F: "Sony",
            0x0110: "ILCE-7M4",
            0x0131: "Adobe Lightroom 13.2",
            0x0132: "2026:09:11 18:45:00",
        }
        img_bytes = generate_test_image("JPEG", size=(640, 480), exif_data=exif_tags)
        img = Image.open(io.BytesIO(img_bytes))

        meta = extract_image_metadata(img)
        self.assertTrue(meta["available"])
        self.assertEqual(meta["camera_make"], "Sony")
        self.assertEqual(meta["camera_model"], "ILCE-7M4")
        self.assertEqual(meta["software"], "Adobe Lightroom 13.2")
        self.assertEqual(meta["datetime"], "2026:09:11 18:45:00")
        self.assertEqual(meta["image_format"], "JPEG")
        self.assertEqual(meta["width"], 640)
        self.assertEqual(meta["height"], 480)

    def test_13_image_without_exif_metadata(self):
        """Ensure images without EXIF cleanly return available=False."""
        img_bytes = generate_test_image("JPEG", size=(300, 200))
        img = Image.open(io.BytesIO(img_bytes))

        meta = extract_image_metadata(img)
        self.assertEqual(meta, {"available": False})

    def test_14_image_with_malformed_unusual_metadata(self):
        """Ensure corrupted, null-padded, or exception-raising EXIF data never crashes the service."""
        # A. Embedded null characters in tags
        exif_tags = {
            0x010F: "Canon\x00\x00Hidden",
            0x0110: "EOS R6\x00",
            0x0131: "   ",  # Whitespace only should be stripped to None
        }
        img_bytes = generate_test_image("JPEG", size=(200, 200), exif_data=exif_tags)
        img = Image.open(io.BytesIO(img_bytes))

        meta = extract_image_metadata(img)
        self.assertTrue(meta["available"])
        self.assertEqual(meta["camera_make"], "CanonHidden")
        self.assertEqual(meta["camera_model"], "EOS R6")
        self.assertNotIn("software", meta)

        # B. Mock image whose getexif raises an unexpected OSError/TypeError
        mock_img = MagicMock(spec=Image.Image)
        mock_img.getexif.side_effect = OSError("Corrupted EXIF pointer structure")

        safe_result = extract_image_metadata(mock_img)
        self.assertEqual(safe_result, {"available": False})


class TestAnalysisEndpointPhase3And4(unittest.IsolatedAsyncioTestCase):
    """End-to-end unit tests verifying the full /api/v1/analyze flow including metadata."""

    async def test_11_full_analysis_pipeline_success(self):
        """Test full pipeline on plain image: upload -> validate -> metadata (none) -> detect."""
        data = generate_test_image("JPEG", (640, 480))
        upload = UploadFile(file=io.BytesIO(data), filename="portrait.jpg", headers={"content-type": "image/jpeg"})

        response = await analyze_image(upload)

        # 1. Check status and metadata
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["filename"], "portrait.jpg")

        # 2. Check analysis_id is a valid UUID
        analysis_id = response["analysis_id"]
        parsed_uuid = uuid.UUID(analysis_id)
        self.assertEqual(str(parsed_uuid), analysis_id)

        # 3. Check verdict (remains unchanged by metadata presence/absence)
        self.assertEqual(response["verdict"]["label"], "AI-generated")
        self.assertEqual(response["verdict"]["confidence"], 0.85)

        # 4. Check probabilities
        self.assertEqual(response["probabilities"]["real"], 0.15)
        self.assertEqual(response["probabilities"]["ai_generated"], 0.85)

        # 5. Check metadata block
        self.assertIn("metadata", response)
        self.assertFalse(response["metadata"]["available"])

    async def test_15_full_analysis_pipeline_with_metadata(self):
        """Test full pipeline on image with EXIF: metadata is present, prediction remains unaffected."""
        exif_tags = {
            0x010F: "Nikon",
            0x0110: "Z9",
            0x0131: "Capture One",
            0x0132: "2026:09:11 20:00:00",
        }
        data = generate_test_image("JPEG", (1280, 720), exif_data=exif_tags)
        upload = UploadFile(file=io.BytesIO(data), filename="dslr_photo.jpg", headers={"content-type": "image/jpeg"})

        response = await analyze_image(upload)

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["filename"], "dslr_photo.jpg")
        self.assertIn("analysis_id", response)

        # Verify metadata is populated
        self.assertIn("metadata", response)
        self.assertTrue(response["metadata"]["available"])
        self.assertEqual(response["metadata"]["camera_make"], "Nikon")
        self.assertEqual(response["metadata"]["camera_model"], "Z9")
        self.assertEqual(response["metadata"]["software"], "Capture One")
        self.assertEqual(response["metadata"]["datetime"], "2026:09:11 20:00:00")

        # Verify ML verdict was NOT altered by the presence of camera EXIF
        self.assertEqual(response["verdict"]["label"], "AI-generated")
        self.assertEqual(response["verdict"]["confidence"], 0.85)


class TestPhase5RobustImageProcessing(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying Phase 5 robust image processing, normalization, and safety limits."""

    def test_16_jpeg_robustness(self):
        """A. Verify standard JPEG images normalize to 224x224 RGB."""
        img_bytes = generate_test_image("JPEG", size=(640, 480))
        img = Image.open(io.BytesIO(img_bytes))
        processed = preprocess_image(img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_17_png_robustness(self):
        """B. Verify standard PNG images normalize to 224x224 RGB."""
        img_bytes = generate_test_image("PNG", size=(800, 600))
        img = Image.open(io.BytesIO(img_bytes))
        processed = preprocess_image(img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_18_webp_robustness(self):
        """C. Verify standard WEBP images normalize to 224x224 RGB."""
        img_bytes = generate_test_image("WEBP", size=(400, 400))
        img = Image.open(io.BytesIO(img_bytes))
        processed = preprocess_image(img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_19_grayscale_robustness(self):
        """D. Verify single-channel grayscale (L mode) images normalize to 3-channel 224x224 RGB."""
        gray_img = Image.new("L", (300, 300), color=128)
        processed = preprocess_image(gray_img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_20_rgba_transparent_robustness(self):
        """E. Verify transparent RGBA images are composited cleanly onto white canvas and output 224x224 RGB."""
        rgba_img = Image.new("RGBA", (500, 400), color=(255, 0, 0, 100))
        processed = preprocess_image(rgba_img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_21_very_small_image_robustness(self):
        """F. Verify very small images (e.g. 10x10) upscale safely without distortion to 224x224 RGB."""
        tiny_img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        processed = preprocess_image(tiny_img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_22_large_dimension_image_robustness(self):
        """G. Verify large images (e.g. 3000x2000) downscale cleanly to 224x224 RGB."""
        large_img = Image.new("RGB", (3000, 2000), color=(0, 0, 255))
        processed = preprocess_image(large_img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_23_non_square_aspect_ratio_robustness(self):
        """H. Verify non-square aspect ratios (16:9, panoramic) fit to 224x224 RGB without distortion."""
        # 16:9 aspect ratio
        widescreen = Image.new("RGB", (1920, 1080), color=(50, 100, 150))
        p_wide = preprocess_image(widescreen)
        self.assertEqual(p_wide.mode, "RGB")
        self.assertEqual(p_wide.size, (224, 224))

        # Extreme panoramic ratio (5:1)
        panoramic = Image.new("RGB", (1500, 300), color=(150, 100, 50))
        p_pano = preprocess_image(panoramic)
        self.assertEqual(p_pano.mode, "RGB")
        self.assertEqual(p_pano.size, (224, 224))

    def test_24_exif_orientation_normalization(self):
        """I. Verify images with EXIF orientation metadata normalize orientation before resizing."""
        # Orientation 6 specifies a 90-degree clockwise physical rotation
        exif_tags = {0x0112: 6}
        img_bytes = generate_test_image("JPEG", size=(100, 200), exif_data=exif_tags)
        img = Image.open(io.BytesIO(img_bytes))

        processed = preprocess_image(img)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    def test_25_compression_and_resize_reload(self):
        """J. Simulate image transformations (resize, heavy JPEG compression) and verify robust preprocessing."""
        # Create base image
        img = Image.new("RGB", (800, 600), color=(120, 80, 200))
        # Transform 1: Resize to intermediate size
        img_resized = img.resize((480, 360), Image.Resampling.BILINEAR)
        # Transform 2: Save with aggressive JPEG compression (quality=20)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=20)
        buf.seek(0)

        # Reload and preprocess
        reloaded = Image.open(buf)
        processed = preprocess_image(reloaded)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size, (224, 224))

    async def test_26_dimension_limit_rejection(self):
        """K. Verify that images exceeding MAX_IMAGE_DIMENSION are rejected with HTTP 400."""
        original_limit = settings.MAX_IMAGE_DIMENSION
        try:
            # Set a small test dimension limit of 500px
            settings.MAX_IMAGE_DIMENSION = 500
            data = generate_test_image("JPEG", size=(600, 300))
            upload = UploadFile(file=io.BytesIO(data), filename="too_wide.jpg", headers={"content-type": "image/jpeg"})

            with self.assertRaises(HTTPException) as ctx:
                await validate_and_process_image(upload)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("exceed the maximum allowed limit", ctx.exception.detail)
        finally:
            settings.MAX_IMAGE_DIMENSION = original_limit


if __name__ == "__main__":
    unittest.main()

