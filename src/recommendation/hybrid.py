"""Hybrid Recommendation Engine combining ALS, Content-Based, and Popularity."""

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.models.popularity import PopularityRecommender
from src.models.collaborative_filtering import ALSRecommender
from src.models.content_based import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def min_max_normalize(scores_dict: Dict[int, float]) -> Dict[int, float]:
    """Normalize dictionary scores to [0.0, 1.0]."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    if np.isclose(max_v, min_v):
        return {k: 1.0 for k in scores_dict}
    return {k: (v - min_v) / (max_v - min_v + 1e-8) for k, v in scores_dict.items()}


class HybridRecommender:
    """Hybrid Recommendation Engine.

    Blends candidate predictions from:
    1. Collaborative Filtering (ALS)
    2. Content-Based Filtering (TF-IDF Attribute Similarity)
    3. Global Popularity Baseline
    """

    def __init__(
        self,
        popularity_model: PopularityRecommender,
        als_model: Optional[ALSRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_weight: float = 0.5,
        content_weight: float = 0.3,
        popularity_weight: float = 0.2,
    ):
        self.popularity_model = popularity_model
        self.als_model = als_model
        self.content_model = content_model

        # Ensure weights sum to 1.0
        total_w = collaborative_weight + content_weight + popularity_weight
        self.w_cf = collaborative_weight / total_w
        self.w_content = content_weight / total_w
        self.w_pop = popularity_weight / total_w

    def set_weights(self, w_cf: float, w_content: float, w_pop: float):
        """Update blending weights."""
        total = w_cf + w_content + w_pop
        assert total > 0, "Weights must sum to a positive number."
        self.w_cf = w_cf / total
        self.w_content = w_content / total
        self.w_pop = w_pop / total

    def recommend(
        self,
        visitor_id: int,
        top_n: int = 10,
        filter_already_interacted: bool = True,
        candidate_pool_size: int = 50,
    ) -> Tuple[List[Tuple[int, float]], str]:
        """Generate top-N hybrid recommendations.

        Args:
            visitor_id: Visitor identifier.
            top_n: Number of recommendations to return.
            filter_already_interacted: Whether to filter out previously seen products.
            candidate_pool_size: Number of candidates retrieved per sub-model before fusion.

        Returns:
            Tuple of (recommendations_list [(item_id, hybrid_score)], source_str).
        """
        # 1. Retrieve candidates from Collaborative Filtering
        cf_candidates: Dict[int, float] = {}
        if self.als_model is not None:
            cf_recs = self.als_model.recommend(
                visitor_id=visitor_id,
                top_n=candidate_pool_size,
                filter_already_interacted=filter_already_interacted,
            )
            cf_candidates = dict(cf_recs)

        # 2. Retrieve candidates from Content-Based Filtering
        content_candidates: Dict[int, float] = {}
        if self.content_model is not None:
            content_recs = self.content_model.recommend(
                visitor_id=visitor_id,
                top_n=candidate_pool_size,
                filter_already_interacted=filter_already_interacted,
            )
            content_candidates = dict(content_recs)

        # 3. Check for cold-start (unseen visitor with no personalized candidates)
        if not cf_candidates and not content_candidates:
            # Cold-start fallback
            pop_recs = self.popularity_model.recommend(
                visitor_id=visitor_id,
                top_n=top_n,
                filter_items=None,
            )
            return pop_recs, "popularity_fallback"

        # 4. Normalization of candidate scores
        norm_cf = min_max_normalize(cf_candidates)
        norm_content = min_max_normalize(content_candidates)

        # 5. Union candidate pool
        all_candidate_items: Set[int] = set(norm_cf.keys()).union(set(norm_content.keys()))

        # Add top popularity items to enrich candidate pool
        pop_candidates = self.popularity_model.recommend(
            visitor_id=visitor_id,
            top_n=candidate_pool_size,
            filter_items=None,
        )
        pop_dict = dict(pop_candidates)
        norm_pop = min_max_normalize(pop_dict)
        all_candidate_items = all_candidate_items.union(set(norm_pop.keys()))

        # 6. Weighted Linear Rank Fusion
        hybrid_scores: Dict[int, float] = {}
        for item in all_candidate_items:
            s_cf = norm_cf.get(item, 0.0)
            s_content = norm_content.get(item, 0.0)
            s_pop = norm_pop.get(item, 0.0)

            score = (self.w_cf * s_cf) + (self.w_content * s_content) + (self.w_pop * s_pop)
            hybrid_scores[item] = score

        # 7. Sort and truncate to Top-N
        sorted_recommendations = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        return sorted_recommendations, "hybrid"
