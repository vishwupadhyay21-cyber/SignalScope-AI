"""API v1 router aggregator."""

from fastapi import APIRouter
from app.api.v1.endpoints import health, analyze

api_router = APIRouter()

# Register health check endpoints under the "Health" tag
api_router.include_router(health.router, tags=["Health"])

# Register image analysis endpoints under the "Analysis" tag
api_router.include_router(analyze.router, tags=["Analysis"])
