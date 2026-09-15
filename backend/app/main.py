"""SignalScope FastAPI Application Entrypoint."""
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import logger
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info(f"API Docs available at: http://localhost:{settings.PORT}/docs")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for SignalScope – Telling Real From Synthetic in the Age of Generative Media (SIH 2026).",
    # Swagger UI and ReDoc are disabled in production to avoid exposing the API surface.
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://signalscope-ai-1-nnr2.onrender.com",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # No credentials needed — this is a stateless, cookie-free file-upload API.
    # allow_credentials=True + allow_origins=["*"] is also forbidden by the CORS spec.
    allow_credentials=False,
    allow_methods=["GET", "POST"],  # Only methods actually served by this API
    allow_headers=["*"],
)


from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

# ---------------------------------------------------------------------------
# Global Error Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles HTTPExceptions (such as 404 Not Found) gracefully with consistent JSON."""
    logger.warning(f"HTTP {exc.status_code} error on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.status_code,
            "message": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles 422 validation errors with a consistent JSON response format."""
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc}")
    error_msg = exc.errors()[0]["msg"] if exc.errors() else "Invalid request data."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": 422,
            "message": error_msg,
        },
    )



@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches any unhandled exceptions to prevent the server from crashing silently."""
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": 500,
            "message": "An unexpected internal server error occurred.",
        },
    )


# ---------------------------------------------------------------------------
# Root Endpoint (Convenience redirect / Welcome info)
# ---------------------------------------------------------------------------
@app.get("/", summary="Root Welcome Endpoint", tags=["General"])
async def root():
    """Welcome endpoint providing project details and links to documentation."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
        "docs_url": "/docs",
        "health_check_url": f"{settings.API_V1_STR}/health",
    }


# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix=settings.API_V1_STR)
