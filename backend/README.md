# SignalScope Backend (SIH 2026)
> **Problem Statement 2:** SignalScope – Telling Real From Synthetic in the Age of Generative Media.

Welcome to the backend repository of **SignalScope**! This is the production-ready FastAPI backend service that powers our AI vs. Real image classification, metadata forensics, and explainability pipeline.

---

## 🎯 Implementation Roadmap Overview

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Environment & FastAPI API Skeleton** — Modular routing (`/api/v1`), Swagger UI, health checks, config & logging. | ✅ Complete |
| **Phase 2** | **Image Upload & In-Memory Validation** — Format validation (JPEG/PNG/WEBP), Pillow integrity checks, 10MB limit, zero disk footprint. | ✅ Complete |
| **Phase 3** | **ML Model Integration Interface** — `BaseDetector` ABC, `DummyDetector` baseline, standard preprocessing (224x224 RGB), singleton factory. | ✅ Complete |
| **Phase 4** | **Image Metadata & EXIF Analysis** — Non-crashing forensic metadata extraction (camera, model, software, timestamp), forensic separation from verdict. | ✅ Complete |
| **Phase 5** | **Robust Image Processing & Normalization** — Grayscale/alpha handling, EXIF orientation transpose, aspect-ratio safe resizing, dimension limit safety (8192px). | ✅ Complete |
| **Phase 6** | **Security & Production Hardening** — Configurable CORS origins, stream body-cap protection (12MB), log injection sanitization, production Swagger gating. | ✅ Complete |
| **Phase 7** | **Complete Backend API Testing** — 79 automated tests across unit services and full HTTP routes (0 failures). | ✅ Complete |
| **Phase 8** | **Final Backend Completion & Production Readiness** — Code audit, contract finalization, smoke testing, and handoff documentation. | ✅ Complete |

---

## 🏗️ Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async ASGI)
- **Server**: [Uvicorn](https://www.uvicorn.org/) (Standard async runtime)
- **Validation & Serialization**: [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Image Processing**: [Pillow (PIL)](https://pillow.readthedocs.io/)
- **Testing**: Python `unittest` & `starlette.testclient` (backed by `httpx`)

---

## 🚀 Beginner Quickstart Guide (Local Setup)

### Step 1: Open Terminal in Project Directory
Ensure your terminal is located in the repository root:
```powershell
cd d:\SIH
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```
*(Or if using uv: `uv venv .venv --python 3.11`)*

### Step 3: Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt `cmd`):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Copy `.env.example` to `.env` if not already present:
```powershell
Copy-Item .env.example .env
```

### Step 6: Start the Backend Server
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 API Contract Reference

### 1. Root Welcome Endpoint
- **URL:** `GET /`
- **Description:** Basic service discovery and welcome information.
- **Response (200 OK):**
  ```json
  {
    "project": "SignalScope Backend",
    "version": "0.1.0",
    "environment": "development",
    "status": "running",
    "docs_url": "/docs",
    "health_check_url": "/api/v1/health"
  }
  ```

### 2. Health Check Endpoint
- **URL:** `GET /api/v1/health`
- **Description:** System health check verifying server availability and environment.
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "environment": "development"
  }
  ```

### 3. Image Analysis Endpoint
- **URL:** `POST /api/v1/analyze`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Image file binary (Required). Formats allowed: `image/jpeg`, `image/png`, `image/webp`.
- **Validation Constraints:**
  - Allowed formats: `JPEG`, `PNG`, `WEBP`
  - Max image size: `10MB` (`MAX_IMAGE_SIZE_MB`)
  - Request body read cap: `12MB` (`MAX_REQUEST_BODY_SIZE_MB`)
  - Max image dimensions: `8192px` width/height (`MAX_IMAGE_DIMENSION`)
- **Successful Response (200 OK):**
  ```json
  {
    "status": "success",
    "analysis_id": "93e8e7db-f807-4d23-9d9a-1ac92ad57972",
    "filename": "camera_shot.jpg",
    "verdict": {
      "label": "AI-generated",
      "confidence": 0.85
    },
    "probabilities": {
      "real": 0.15,
      "ai_generated": 0.85
    },
    "metadata": {
      "available": true,
      "image_format": "JPEG",
      "width": 1920,
      "height": 1080,
      "camera_make": "Canon",
      "camera_model": "EOS R5",
      "software": "Adobe Photoshop 2026",
      "datetime": "2026:09:11 21:15:00"
    }
  }
  ```
  *(Note: If the uploaded image contains no EXIF tags or corrupted metadata, `"metadata"` safely returns `{"available": false}` without altering the verdict).*

---

## ⚠️ Consistent Error Responses

All API errors return a uniform, safe JSON structure without leaking internal stack traces or paths:

```json
{
  "status": "error",
  "code": 400,
  "message": "Uploaded file is corrupted or not a valid image."
}
```

| HTTP Status | Trigger Condition | Example Message |
| :--- | :--- | :--- |
| **400 Bad Request** | Missing file, empty 0-byte file, corrupted image bytes, or dimensions > 8192px | `"Uploaded file is corrupted or not a valid image."` |
| **404 Not Found** | Unrecognized endpoint route | `"Not Found"` |
| **405 Method Not Allowed** | Incorrect HTTP method (e.g. GET on `/api/v1/analyze`) | `"Method Not Allowed"` |
| **413 Content Too Large** | Upload size > 10MB or stream cap > 12MB exceeded | `"File size exceeds the maximum allowed limit of 10MB."` |
| **415 Unsupported Media Type** | Format not in JPEG, PNG, WEBP (e.g. GIF, SVG, BMP) | `"Unsupported image format 'GIF'. Allowed formats: JPEG, PNG, WEBP."` |
| **422 Unprocessable Content** | Malformed multipart or missing required form structure | `"Field required"` |
| **500 Internal Server Error** | Unexpected internal service failure | `"An unexpected internal server error occurred."` |

---

## 🔒 Security & Hardening Features

- **Safe In-Memory Processing**: Uploaded images are read, validated, preprocessed, and analyzed entirely in RAM. No untrusted user data is written to disk.
- **Log Injection Sanitization**: All uploaded filenames are stripped of newline (`\n`), carriage return (`\r`), and null (`\x00`) characters before logging.
- **Stream Body-Cap Protection**: The upload stream stops reading if the payload exceeds `MAX_REQUEST_BODY_SIZE_MB` (12MB) to prevent RAM exhaustion attacks.
- **Dimension Safety Limit**: Images exceeding `8192px` in width or height are rejected with HTTP 400 before heavy preprocessing to block decompression bomb exploits.
- **Configurable CORS**: CORS is configured via `CORS_ORIGINS` in `.env`. Credentials are explicitly disabled to prevent credential-leakage attacks.
- **Environment Gating**: In production (`ENVIRONMENT="production"`), `/docs`, `/redoc`, and `/api/v1/openapi.json` are disabled to prevent API surface exposure.

---

## 🧪 Running Automated Tests

Run the complete backend test suite (79 tests):

```powershell
python -m unittest discover -s tests -v
```

Expected output:
```text
Ran 79 tests in ~0.7s
OK
```

- **`tests/test_analyze.py` (31 tests)**: Isolated unit tests for image validation, preprocessor, EXIF extractor, detector interface, and dimension safety limits.
- **`tests/test_http.py` (48 tests)**: End-to-end integration tests exercising Starlette `TestClient` across all HTTP routes, success paths, error paths, malformed payloads, CORS headers, and Swagger environment gating.

---

## 🤝 Teammate Integration & Handoff Guide

### For Frontend Developers
1. **API Base URL**: `http://localhost:8000` (or configured host/port).
2. **Health Monitoring**: Poll `GET /api/v1/health` to ensure the backend is alive.
3. **Image Analysis**: Send a `multipart/form-data` POST request to `POST /api/v1/analyze` with the file under key `file`.
4. **Response Handling**:
   - `verdict.label`: String (`"AI-generated"` or `"Real"`).
   - `verdict.confidence`: Float between `0.0` and `1.0`.
   - `probabilities.real` and `probabilities.ai_generated`: Floats summing to `1.0`.
   - `metadata.available`: Boolean indicating if EXIF forensic details are available.

### For ML Engineers
1. **Abstract Contract**: Located in [app/ml/base.py](file:///d:/SIH/app/ml/base.py) (`BaseDetector`). Your model class must implement `load_model()` and `predict(image: Image.Image) -> Dict[str, Any]`.
2. **Current Development Model**: [app/ml/dummy.py](file:///d:/SIH/app/ml/dummy.py) provides deterministic responses for testing without PyTorch weights.
3. **Swapping in Trained Weights**:
   - Open [app/ml/factory.py](file:///d:/SIH/app/ml/factory.py).
   - Replace the instance in `get_detector()` with your trained detector class.
   - Zero changes to the FastAPI endpoints, validation pipelines, or frontend contracts are required!

---

## 📁 Project Directory Structure

```text
d:\SIH\
├── app/
│   ├── __init__.py               # Python package initialization
│   ├── main.py                   # FastAPI application entrypoint, CORS, exception handlers
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py         # Aggregates v1 endpoints (Health, Analysis)
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py     # Health check endpoint (GET /api/v1/health)
│   │           └── analyze.py    # Complete image analysis pipeline (POST /api/v1/analyze)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic Settings (.env configuration & limits)
│   │   └── logger.py             # Centralized structured logger
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseDetector abstract base class
│   │   ├── dummy.py              # DummyDetector implementation for dev/testing
│   │   ├── factory.py            # Singleton provider for active detector model
│   │   └── preprocessor.py       # Aspect-ratio safe resize (224x224 RGB) & normalization
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── analyze.py            # Pydantic schemas (Verdict, Probabilities, ImageMetadata, AnalysisResponse)
│   └── services/
│       ├── __init__.py
│       ├── image_validator.py    # Stream-capped in-memory image validation
│       └── metadata_extractor.py # Safe forensic EXIF metadata extraction
├── tests/
│   ├── __init__.py
│   ├── test_analyze.py           # Unit tests (31 tests: services, ML preprocessor, EXIF)
│   └── test_http.py              # HTTP integration tests (48 tests: routes, errors, security)
├── .env                          # Local environment variables
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusions
├── requirements.txt              # Production runtime dependencies
└── README.md                     # Comprehensive documentation & handoff guide
```
