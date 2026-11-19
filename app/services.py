"""Recommendation Service loading approved artifacts and orchestrating real-time inference."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import psutil

from src.models.popularity import PopularityRecommender
from src.models.collaborative_filtering import ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.recommendation.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RecommendationService:
    """Singleton service encapsulating model artifacts, inference routines, and health telemetry."""

    _instance: Optional["RecommendationService"] = None

    def __init__(self, artifacts_dir: str | Path = "artifacts/approved"):
        self.artifacts_dir = Path(artifacts_dir)
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

        self.popularity_model: Optional[PopularityRecommender] = None
        self.als_model: Optional[ALSRecommender] = None
        self.content_model: Optional[ContentBasedRecommender] = None
        self.hybrid_model: Optional[HybridRecommender] = None

        self.models_loaded: Dict[str, bool] = {
            "popularity": False,
            "als": False,
            "content": False,
            "hybrid": False,
        }
        self.is_ready = False

    @classmethod
    def get_instance(cls, artifacts_dir: str | Path = "artifacts/approved") -> "RecommendationService":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls(artifacts_dir=artifacts_dir)
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        """Load approved model weights, index mappings, and feature caches from disk."""
        logger.info("Initializing RecommendationService from %s...", self.artifacts_dir)
        try:
            # 1. Load Popularity Baseline
            pop_path = self.artifacts_dir / "popularity_features.parquet"
            if pop_path.exists():
                self.popularity_model = PopularityRecommender()
                self.popularity_model.load_from_parquet(pop_path)
                self.models_loaded["popularity"] = True
                logger.info("Loaded Popularity Baseline.")
            else:
                logger.warning("Popularity features missing at %s", pop_path)

            # 2. Load Collaborative Filtering (ALS)
            als_path = self.artifacts_dir / "als_model.pkl"
            if als_path.exists():
                self.als_model = ALSRecommender()
                self.als_model.load(als_path)
                self.models_loaded["als"] = True
                logger.info("Loaded ALS Recommender.")
            else:
                logger.warning("ALS model missing at %s", als_path)

            # 3. Load Content-Based Filter
            content_path = self.artifacts_dir / "content_model.pkl"
            if content_path.exists():
                self.content_model = ContentBasedRecommender()
                self.content_model.load(content_path)
                self.models_loaded["content"] = True
                logger.info("Loaded Content-Based Recommender.")
            else:
                logger.warning("Content-based model missing at %s", content_path)

            # 4. Initialize Hybrid Engine
            if self.popularity_model is not None:
                self.hybrid_model = HybridRecommender(
                    popularity_model=self.popularity_model,
                    als_model=self.als_model,
                    content_model=self.content_model,
                    collaborative_weight=0.5,
                    content_weight=0.3,
                    popularity_weight=0.2,
                )
                self.models_loaded["hybrid"] = True
                self.is_ready = True
                logger.info("Hybrid Recommendation Engine initialized and ready.")
            else:
                logger.error("Failed to initialize Hybrid Engine: Popularity baseline missing.")
                self.is_ready = False

        except Exception as e:
            logger.error("Error loading model artifacts: %s", e, exc_info=True)
            self.is_ready = False

    def get_health_status(self) -> Dict[str, Any]:
        """Compute service health, uptime, and memory consumption telemetry."""
        uptime = round(time.time() - self.start_time, 2)
        mem_info = self.process.memory_info()
        mem_mb = round(mem_info.rss / (1024 * 1024), 2)

        status = "healthy" if self.is_ready and all(self.models_loaded.values()) else "degraded"
        if not self.models_loaded["popularity"]:
            status = "unhealthy"

        return {
            "status": status,
            "service": "ecommerce-recommendation-api",
            "version": "1.0.0",
            "uptime_seconds": uptime,
            "models_loaded": self.models_loaded,
            "memory_usage_mb": mem_mb,
        }

    def get_popular_recommendations(self, top_n: int = 10) -> Dict[str, Any]:
        """Retrieve top-N global popular products."""
        if not self.models_loaded["popularity"] or self.popularity_model is None:
            raise RuntimeError("Popularity baseline model is not available.")

        recs = self.popularity_model.recommend(top_n=top_n)
        formatted_recs = [
            {"rank": rank, "item_id": int(iid), "score": round(float(score), 4)}
            for rank, (iid, score) in enumerate(recs, 1)
        ]
        return {
            "count": len(formatted_recs),
            "source": "popularity_global",
            "recommendations": formatted_recs,
        }

    def recommend(
        self,
        visitor_id: int,
        top_n: int = 10,
        filter_seen: bool = True,
    ) -> Dict[str, Any]:
        """Generate top-N personalized hybrid recommendations with cold-start fallback.

        Args:
            visitor_id: External visitor ID.
            top_n: Number of recommendations.
            filter_seen: Whether to exclude already interacted products.

        Returns:
            Dictionary matching RecommendationResponse schema.
        """
        if not self.is_ready or self.hybrid_model is None:
            raise RuntimeError("Recommendation service is not initialized.")

        recs, source = self.hybrid_model.recommend(
            visitor_id=visitor_id,
            top_n=top_n,
            filter_already_interacted=filter_seen,
        )

        formatted_recs = [
            {"rank": rank, "item_id": int(iid), "score": round(float(score), 4)}
            for rank, (iid, score) in enumerate(recs, 1)
        ]

        return {
            "visitor_id": int(visitor_id),
            "count": len(formatted_recs),
            "source": source,
            "recommendations": formatted_recs,
        }
