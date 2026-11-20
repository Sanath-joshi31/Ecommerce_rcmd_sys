"""FastAPI application entrypoint for E-Commerce Product Recommendation API."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.dashboard import get_dashboard_html
from app.eda_dashboard import get_eda_dashboard_html
from app.schemas import (
    HealthResponse,
    PopularResponse,
    RecommendationResponse,
    RootResponse,
)
from app.services import RecommendationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info("Starting up Recommendation API server...")
    service = RecommendationService.get_instance(artifacts_dir="artifacts/approved")
    app.state.service = service
    logger.info("Recommendation Service initialized. Readiness: %s", service.is_ready)
    yield
    logger.info("Shutting down Recommendation API server...")


app = FastAPI(
    title="E-Commerce Product Recommendation API",
    description="Enterprise-grade recommendation microservice powered by Implicit ALS, Content-Based TF-IDF, and Popularity Baseline.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    summary="API Metadata and Available Endpoints",
    tags=["General"],
)
async def root():
    """Retrieve service metadata, operational status, and interactive documentation links."""
    return {
        "name": "E-Commerce Product Recommendation API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "endpoints": {
            "root": "GET /",
            "health": "GET /health",
            "popular": "GET /popular?top_n=10",
            "recommend": "GET /recommend/{visitor_id}?top_n=10&filter_seen=true",
        },
    }


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Interactive Web Console",
    tags=["General"],
)
async def dashboard():
    """Interactive visual console for testing real-time recommendation inferences."""
    return HTMLResponse(content=get_dashboard_html())


@app.get(
    "/eda",
    response_class=HTMLResponse,
    summary="Interactive Exploratory Data Analysis Dashboard",
    tags=["General"],
)
async def eda_dashboard():
    """Interactive visual EDA dashboard displaying event funnel, temporal split, and matrix sparsity."""
    return HTMLResponse(content=get_eda_dashboard_html())


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and Readiness Health Probe",
    tags=["Monitoring"],
)
async def health_check():
    """Diagnostic health probe returning uptime, memory consumption, and model loading status."""
    service: RecommendationService = app.state.service
    health = service.get_health_status()
    if health["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health,
        )
    return health


@app.get(
    "/popular",
    response_model=PopularResponse,
    summary="Top-N Globally Popular Products",
    tags=["Recommendations"],
)
async def get_popular_products(
    top_n: int = Query(default=10, ge=1, le=100, description="Number of items to retrieve (1 to 100)"),
):
    """Retrieve top-N globally popular products based on weighted historical customer interactions."""
    service: RecommendationService = app.state.service
    try:
        return service.get_popular_recommendations(top_n=top_n)
    except Exception as e:
        logger.error("Error retrieving popular products: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve popular products: {str(e)}",
        )


@app.get(
    "/recommend/{visitor_id}",
    response_model=RecommendationResponse,
    summary="Top-N Personalized Product Recommendations",
    tags=["Recommendations"],
)
async def get_recommendations(
    visitor_id: int = Path(..., gt=0, description="Positive integer visitor identifier"),
    top_n: int = Query(default=10, ge=1, le=100, description="Number of recommendations (1 to 100)"),
    filter_seen: bool = Query(default=True, description="Exclude items previously viewed or purchased"),
):
    """Generate top-N personalized hybrid recommendations for a visitor.

    - Warm Visitors: Returns personalized hybrid recommendations (`source: 'hybrid'`).
    - Cold-Start Visitors: Automatically degrades to popularity fallback (`source: 'popularity_fallback'`).
    """
    service: RecommendationService = app.state.service
    try:
        return service.recommend(
            visitor_id=visitor_id,
            top_n=top_n,
            filter_seen=filter_seen,
        )
    except Exception as e:
        logger.error("Error generating recommendations for visitor %s: %s", visitor_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
