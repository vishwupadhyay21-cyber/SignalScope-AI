"""HTTP-level integration tests for all SignalScope backend endpoints.

These tests exercise every route through the full FastAPI stack —
middleware, error handlers, routing, Pydantic validation — using
Starlette's TestClient (backed by httpx2).

Covered endpoints
-----------------
GET  /                     Root welcome endpoint
GET  /api/v1/health        Health check (Phase 1 / Phase 5)
POST /api/v1/analyze       Image analysis pipeline (Phases 2-4)

Also covers
-----------
- Consistent error response structure  {"status","code","message"}
- Consistent success response structure
- HTTP status codes for all rejection paths
- analysis_id uniqueness and UUID format
- Swagger/OpenAPI gating by ENVIRONMENT  (Phase 6)
- No file / missing field / wrong method (malformed requests)
"""

import io
import uuid
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient
from PIL import Image

from app.main import app
from app.core.config import settings

# ---------------------------------------------------------------------------
# Shared test client (one instance for the whole module)
# ---------------------------------------------------------------------------
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image_bytes(
    fmt: str = "JPEG",
    size: tuple = (200, 150),
    mode: str = "RGB",
    color: tuple = (100, 150, 200),
    exif_data: dict = None,
) -> bytes:
    """Returns in-memory image bytes for the given Pillow format, optionally with EXIF."""
    buf = io.BytesIO()
    img = Image.new(mode, size, color=color)
    if exif_data:
        exif = img.getexif()
        for tag_id, val in exif_data.items():
            exif[tag_id] = val
        img.save(buf, format=fmt, exif=exif)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _upload_file(data: bytes, filename: str, content_type: str = "image/jpeg"):
    """Returns the files dict used for multipart/form-data POST."""
    return {"file": (filename, io.BytesIO(data), content_type)}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
class TestRootEndpoint(unittest.TestCase):
    """Tests for the root welcome endpoint GET /."""

    def test_r1_root_returns_200(self):
        """GET / must return HTTP 200."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_r2_root_response_fields(self):
        """GET / response must contain project, version, environment, status, docs_url, health_check_url."""
        response = client.get("/")
        body = response.json()
        for key in ("project", "version", "environment", "status", "docs_url", "health_check_url"):
            self.assertIn(key, body, f"Missing key '{key}' in root response")

    def test_r3_root_status_running(self):
        """GET / status field must equal 'running'."""
        body = client.get("/").json()
        self.assertEqual(body["status"], "running")

    def test_r4_root_version_matches_settings(self):
        """GET / version must match settings.VERSION."""
        body = client.get("/").json()
        self.assertEqual(body["version"], settings.VERSION)

    def test_r5_root_health_check_url_correct(self):
        """GET / health_check_url must point to /api/v1/health."""
        body = client.get("/").json()
        self.assertIn("/api/v1/health", body["health_check_url"])

    def test_r6_root_wrong_method_returns_405(self):
        """POST / must be rejected with HTTP 405 Method Not Allowed."""
        response = client.post("/")
        self.assertEqual(response.status_code, 405)


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------
class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /api/v1/health."""

    def test_h1_health_returns_200(self):
        """Health check must return HTTP 200."""
        response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)

    def test_h2_health_status_field(self):
        """Health response must have status='healthy'."""
        body = client.get("/api/v1/health").json()
        self.assertEqual(body["status"], "healthy")

    def test_h3_health_version_field(self):
        """Health response must include version matching settings.VERSION."""
        body = client.get("/api/v1/health").json()
        self.assertIn("version", body)
        self.assertEqual(body["version"], settings.VERSION)

    def test_h4_health_environment_field(self):
        """Health response must include environment matching settings.ENVIRONMENT."""
        body = client.get("/api/v1/health").json()
        self.assertIn("environment", body)
        self.assertEqual(body["environment"], settings.ENVIRONMENT)

    def test_h5_health_wrong_method_returns_405(self):
        """POST /api/v1/health must be rejected with HTTP 405."""
        response = client.post("/api/v1/health")
        self.assertEqual(response.status_code, 405)

    def test_h6_health_content_type_json(self):
        """Health response Content-Type must be application/json."""
        response = client.get("/api/v1/health")
        self.assertIn("application/json", response.headers.get("content-type", ""))


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — success paths
# ---------------------------------------------------------------------------
class TestAnalyzeEndpointSuccess(unittest.TestCase):
    """HTTP-level tests for successful POST /api/v1/analyze requests."""

    def test_a1_valid_jpeg_returns_200(self):
        """Valid JPEG upload must return HTTP 200."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "photo.jpg"),
        )
        self.assertEqual(response.status_code, 200)

    def test_a2_valid_png_returns_200(self):
        """Valid PNG upload must return HTTP 200."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("PNG"), "photo.png", "image/png"),
        )
        self.assertEqual(response.status_code, 200)

    def test_a3_valid_webp_returns_200(self):
        """Valid WEBP upload must return HTTP 200."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("WEBP"), "photo.webp", "image/webp"),
        )
        self.assertEqual(response.status_code, 200)

    def test_a4_success_response_structure(self):
        """Successful response must contain status, analysis_id, filename, verdict,
        probabilities, and metadata at the top level."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "check.jpg"),
        )
        body = response.json()
        for key in ("status", "analysis_id", "filename", "verdict", "probabilities", "metadata"):
            self.assertIn(key, body, f"Missing key '{key}' in success response")

    def test_a5_success_status_field(self):
        """Successful response status field must equal 'success'."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "ok.jpg"),
        )
        self.assertEqual(response.json()["status"], "success")

    def test_a6_analysis_id_is_valid_uuid4(self):
        """analysis_id in the response must be a valid UUID (version 4)."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "uuid_check.jpg"),
        )
        analysis_id = response.json()["analysis_id"]
        parsed = uuid.UUID(analysis_id)
        # UUID4 has version=4
        self.assertEqual(parsed.version, 4)
        self.assertEqual(str(parsed), analysis_id)

    def test_a7_analysis_id_unique_per_request(self):
        """Two consecutive requests must return distinct analysis_ids."""
        data = _make_image_bytes("JPEG")
        id1 = client.post("/api/v1/analyze", files=_upload_file(data, "a.jpg")).json()["analysis_id"]
        id2 = client.post("/api/v1/analyze", files=_upload_file(data, "b.jpg")).json()["analysis_id"]
        self.assertNotEqual(id1, id2)

    def test_a8_filename_preserved_in_response(self):
        """Filename from the upload must be echoed back in the response."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "my_image.jpg"),
        )
        self.assertEqual(response.json()["filename"], "my_image.jpg")

    def test_a9_verdict_structure(self):
        """Verdict block must contain label (str) and confidence (float in 0–1)."""
        body = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "verdict.jpg"),
        ).json()
        verdict = body["verdict"]
        self.assertIn("label", verdict)
        self.assertIn("confidence", verdict)
        self.assertIsInstance(verdict["label"], str)
        self.assertIsInstance(verdict["confidence"], float)
        self.assertGreaterEqual(verdict["confidence"], 0.0)
        self.assertLessEqual(verdict["confidence"], 1.0)

    def test_a10_probabilities_structure(self):
        """Probabilities block must contain real and ai_generated floats that sum to ~1.0."""
        body = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "proba.jpg"),
        ).json()
        probs = body["probabilities"]
        self.assertIn("real", probs)
        self.assertIn("ai_generated", probs)
        self.assertAlmostEqual(probs["real"] + probs["ai_generated"], 1.0, places=5)

    def test_a11_metadata_available_field_present(self):
        """Metadata block must contain the 'available' boolean field."""
        body = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "meta.jpg"),
        ).json()
        self.assertIn("available", body["metadata"])
        self.assertIsInstance(body["metadata"]["available"], bool)

    def test_a12_content_type_json(self):
        """Analyze response Content-Type must be application/json."""
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(_make_image_bytes("JPEG"), "ct.jpg"),
        )
        self.assertIn("application/json", response.headers.get("content-type", ""))

    def test_a13_image_with_exif_metadata_in_http_response(self):
        """Valid image with EXIF metadata returns full metadata block via HTTP."""
        exif_tags = {
            0x010F: "Canon",
            0x0110: "EOS R5",
            0x0131: "Adobe Photoshop",
            0x0132: "2026:09:12 10:00:00",
        }
        data = _make_image_bytes("JPEG", size=(400, 300), exif_data=exif_tags)
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(data, "exif_photo.jpg"),
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        meta = body.get("metadata", {})
        self.assertTrue(meta.get("available"))
        self.assertEqual(meta.get("camera_make"), "Canon")
        self.assertEqual(meta.get("camera_model"), "EOS R5")
        self.assertEqual(meta.get("software"), "Adobe Photoshop")
        self.assertEqual(meta.get("datetime"), "2026:09:12 10:00:00")
        self.assertEqual(meta.get("width"), 400)
        self.assertEqual(meta.get("height"), 300)
        # Verify model classification verdict was unaffected
        self.assertEqual(body["verdict"]["label"], "AI-generated")
        self.assertEqual(body["verdict"]["confidence"], 0.85)

    def test_a14_corrupt_exif_does_not_crash_endpoint(self):
        """Corrupted EXIF metadata header safely degrades to available=False over HTTP."""
        data = _make_image_bytes("JPEG", size=(100, 100))
        with patch("PIL.Image.Image.getexif", side_effect=OSError("Fatal corrupted EXIF header")):
            response = client.post(
                "/api/v1/analyze",
                files=_upload_file(data, "corrupt_exif.jpg"),
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("metadata", body)
        self.assertFalse(body["metadata"]["available"])

    def test_a15_filename_with_control_characters_handled_safely(self):
        """Upload with filename containing control/newline characters is safely processed."""
        data = _make_image_bytes("JPEG", size=(100, 100))
        response = client.post(
            "/api/v1/analyze",
            files=_upload_file(data, "my\nphoto\r.jpg"),
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("analysis_id", body)


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — error paths
# ---------------------------------------------------------------------------
class TestAnalyzeEndpointErrors(unittest.TestCase):
    """HTTP-level tests for all rejection paths of POST /api/v1/analyze."""

    def _assert_error_structure(self, body: dict, expected_code: int):
        """Helper: verify the consistent error response shape."""
        self.assertIn("status", body)
        self.assertIn("code", body)
        self.assertIn("message", body)
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["code"], expected_code)
        self.assertIsInstance(body["message"], str)
        self.assertGreater(len(body["message"]), 0)

    def test_e1_no_file_field_returns_400(self):
        """POST with no file field at all must return HTTP 400 with error structure."""
        # Sending no multipart body — our handler returns 400, not 422,
        # because file is Optional and defaults to None.
        response = client.post("/api/v1/analyze")
        self.assertEqual(response.status_code, 400)
        self._assert_error_structure(response.json(), 400)

    def test_e2_empty_file_returns_400(self):
        """POST with a 0-byte file must return HTTP 400."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self._assert_error_structure(response.json(), 400)

    def test_e3_corrupted_file_returns_400(self):
        """POST with corrupt image bytes must return HTTP 400."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("fake.jpg", io.BytesIO(b"NOT_AN_IMAGE_XYZ"), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self._assert_error_structure(response.json(), 400)

    def test_e4_unsupported_format_returns_415(self):
        """POST with a GIF file must return HTTP 415 Unsupported Media Type."""
        gif_data = _make_image_bytes("GIF")
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("animation.gif", io.BytesIO(gif_data), "image/gif")},
        )
        self.assertEqual(response.status_code, 415)
        self._assert_error_structure(response.json(), 415)

    def test_e5_oversized_file_returns_413(self):
        """POST with a file exceeding MAX_IMAGE_SIZE_MB must return HTTP 413."""
        original_limit = settings.MAX_IMAGE_SIZE_MB
        try:
            settings.MAX_IMAGE_SIZE_MB = 1
            oversized = b"X" * (1024 * 1024 + 1024)
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("big.jpg", io.BytesIO(oversized), "image/jpeg")},
            )
            self.assertEqual(response.status_code, 413)
            self._assert_error_structure(response.json(), 413)
        finally:
            settings.MAX_IMAGE_SIZE_MB = original_limit

    def test_e6_wrong_method_get_returns_405(self):
        """GET /api/v1/analyze must be rejected with HTTP 405."""
        response = client.get("/api/v1/analyze")
        self.assertEqual(response.status_code, 405)

    def test_e7_unknown_route_returns_404(self):
        """A completely unknown route must return HTTP 404 with error structure."""
        response = client.get("/api/v1/does_not_exist")
        self.assertEqual(response.status_code, 404)
        self._assert_error_structure(response.json(), 404)

    def test_e8_error_message_is_safe_string(self):
        """Error messages must not expose internal Python exception details."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("corrupt.jpg", io.BytesIO(b"bad_data"), "image/jpeg")},
        )
        message = response.json().get("message", "")
        # Should not contain Python tracebacks or internal module paths
        self.assertNotIn("Traceback", message)
        self.assertNotIn("File \"", message)
        self.assertNotIn("line ", message)

    def test_e9_dimensions_exceeding_limit_returns_400(self):
        """Image dimensions exceeding MAX_IMAGE_DIMENSION return HTTP 400."""
        orig = settings.MAX_IMAGE_DIMENSION
        try:
            settings.MAX_IMAGE_DIMENSION = 500
            data = _make_image_bytes("JPEG", size=(600, 300))
            response = client.post(
                "/api/v1/analyze",
                files=_upload_file(data, "too_wide.jpg"),
            )
            self.assertEqual(response.status_code, 400)
            self._assert_error_structure(response.json(), 400)
            self.assertIn("exceed the maximum allowed limit", response.json()["message"])
        finally:
            settings.MAX_IMAGE_DIMENSION = orig

    def test_e10_body_cap_protection_returns_413(self):
        """Upload payload exceeding MAX_REQUEST_BODY_SIZE_MB is stopped by stream cap with HTTP 413."""
        orig = settings.MAX_REQUEST_BODY_SIZE_MB
        try:
            settings.MAX_REQUEST_BODY_SIZE_MB = 1
            oversized_data = b"0" * (1024 * 1024 + 2048)
            response = client.post(
                "/api/v1/analyze",
                files=_upload_file(oversized_data, "huge.jpg"),
            )
            self.assertEqual(response.status_code, 413)
            self._assert_error_structure(response.json(), 413)
            self.assertIn("exceeds the maximum allowed limit", response.json()["message"])
        finally:
            settings.MAX_REQUEST_BODY_SIZE_MB = orig

    def test_e11_wrong_form_field_name_returns_400(self):
        """POST with mismatched multipart field name (e.g. 'image' instead of 'file') returns HTTP 400."""
        response = client.post(
            "/api/v1/analyze",
            files={"wrong_field": ("test.jpg", io.BytesIO(_make_image_bytes("JPEG")), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self._assert_error_structure(response.json(), 400)
        self.assertIn("No file provided", response.json()["message"])

    def test_e12_whitespace_filename_returns_400(self):
        """POST with whitespace-only filename returns HTTP 400."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("   ", io.BytesIO(_make_image_bytes("JPEG")), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self._assert_error_structure(response.json(), 400)
        self.assertIn("No file provided", response.json()["message"])

    def test_e13_validation_error_returns_422_structure(self):
        """Request triggering validation error (empty string filename) returns consistent 422 error JSON."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("", io.BytesIO(b"abc"), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        self._assert_error_structure(response.json(), 422)

    def test_e14_unhandled_server_error_returns_500_structure(self):
        """Unexpected internal exception returns consistent 500 error JSON without server crash."""
        data = _make_image_bytes("JPEG", size=(50, 50))
        with patch("app.api.v1.endpoints.analyze.get_detector", side_effect=RuntimeError("Unexpected ML crash")):
            response = client.post(
                "/api/v1/analyze",
                files=_upload_file(data, "crash.jpg"),
            )
        self.assertEqual(response.status_code, 500)
        self._assert_error_structure(response.json(), 500)
        self.assertEqual(response.json()["message"], "An unexpected internal server error occurred.")


# ---------------------------------------------------------------------------
# Phase 6 — Swagger/OpenAPI gating
# ---------------------------------------------------------------------------
class TestSwaggerGating(unittest.TestCase):
    """Verify that Swagger UI and OpenAPI schema visibility obeys ENVIRONMENT."""

    def test_g1_docs_visible_in_development(self):
        """GET /docs must return 200 when ENVIRONMENT != 'production'."""
        # The test environment uses the .env value (development)
        if settings.ENVIRONMENT == "production":
            self.skipTest("Running in production environment – docs are expected to be hidden.")
        response = client.get("/docs")
        self.assertEqual(response.status_code, 200)

    def test_g2_openapi_json_visible_in_development(self):
        """GET /api/v1/openapi.json must return 200 in non-production."""
        if settings.ENVIRONMENT == "production":
            self.skipTest("Running in production environment.")
        response = client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 200)

    def test_g3_openapi_schema_contains_analyze_route(self):
        """The OpenAPI schema must declare the /api/v1/analyze path."""
        if settings.ENVIRONMENT == "production":
            self.skipTest("Running in production environment.")
        schema = client.get("/api/v1/openapi.json").json()
        paths = schema.get("paths", {})
        self.assertIn("/api/v1/analyze", paths)

    def test_g4_openapi_schema_contains_health_route(self):
        """The OpenAPI schema must declare the /api/v1/health path."""
        if settings.ENVIRONMENT == "production":
            self.skipTest("Running in production environment.")
        schema = client.get("/api/v1/openapi.json").json()
        paths = schema.get("paths", {})
        self.assertIn("/api/v1/health", paths)

    def test_g5_docs_disabled_in_production(self):
        """FastAPI app initialized in production environment gates /docs and /openapi.json."""
        from fastapi import FastAPI
        with patch.object(settings, "ENVIRONMENT", "production"):
            prod_app = FastAPI(
                docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
                redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
                openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENVIRONMENT != "production" else None,
            )
            prod_client = TestClient(prod_app)
            self.assertEqual(prod_client.get("/docs").status_code, 404)
            self.assertEqual(prod_client.get("/redoc").status_code, 404)
            self.assertEqual(prod_client.get(f"{settings.API_V1_STR}/openapi.json").status_code, 404)


# ---------------------------------------------------------------------------
# Phase 6 — CORS and Security Headers
# ---------------------------------------------------------------------------
class TestCORSAndSecurityHeaders(unittest.TestCase):
    """Tests verifying Phase 6 CORS and security headers."""

    def test_c1_cors_allowed_origin_header(self):
        """Requests with an Origin header receive Access-Control-Allow-Origin."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)

    def test_c2_cors_preflight_options(self):
        """Preflight OPTIONS request returns HTTP 200 with allowed methods."""
        response = client.options(
            "/api/v1/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
