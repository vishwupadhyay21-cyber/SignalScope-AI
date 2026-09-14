"""Health check endpoint."""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description=(
        "Returns the operational status of the API server, "
        "including the current version and environment name."
    ),
    response_model=dict,
    responses={
        200: {
            "description": "Server is running and healthy.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "0.1.0",
                        "environment": "development",
                    }
                }
            },
        }
    },
)
async def health_check():
    """Returns healthy status, API version, and active environment."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
