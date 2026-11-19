"""Pydantic request and response schemas for Recommendation API."""

from datetime import datetime, timezone
from typing import Dict, List
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Schema representing an individual recommended item."""

    rank: int = Field(..., description="Rank position of recommendation (1 to N)")
    item_id: int = Field(..., description="Unique product ID")
    score: float = Field(..., description="Normalized prediction confidence score")


class RecommendationResponse(BaseModel):
    """Schema for /recommend/{visitor_id} endpoint."""

    visitor_id: int = Field(..., description="Requested visitor ID")
    count: int = Field(..., description="Number of recommendations returned")
    source: str = Field(..., description="Algorithm source ('hybrid', 'collaborative', or 'popularity_fallback')")
    recommendations: List[RecommendationItem] = Field(..., description="Ordered list of recommended products")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC response timestamp")


class PopularResponse(BaseModel):
    """Schema for /popular endpoint."""

    count: int = Field(..., description="Number of recommendations returned")
    source: str = Field(default="popularity_global", description="Recommendation source")
    recommendations: List[RecommendationItem] = Field(..., description="Ordered list of globally popular products")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC response timestamp")


class HealthResponse(BaseModel):
    """Schema for /health readiness and liveness probe."""

    status: str = Field(..., description="Service health status ('healthy' or 'degraded')")
    service: str = Field(..., description="Microservice name")
    version: str = Field(..., description="Deployed API version")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")
    models_loaded: Dict[str, bool] = Field(..., description="Status of individual sub-model artifacts")
    memory_usage_mb: float = Field(..., description="Process RAM consumption in Megabytes")


class RootResponse(BaseModel):
    """Schema for root / endpoint."""

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    status: str = Field(..., description="Operational status")
    docs_url: str = Field(..., description="Interactive OpenAPI documentation URL")
    endpoints: Dict[str, str] = Field(..., description="Summary of available REST endpoints")
