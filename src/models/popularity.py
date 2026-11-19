"""Popularity-based recommendation engine (Baseline & Cold-Start Fallback)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PopularityRecommender:
    """Popularity Recommender recommending highest weighted products from training data.

    Calculates:
        Popularity Score = (Views * 1) + (Carts * 3) + (Purchases * 5)
    """

    def __init__(self, name: str = "PopularityBaseline"):
        self.name = name
        self.popularity_df: Optional[pd.DataFrame] = None
        self.sorted_items: List[int] = []
        self.sorted_scores: List[float] = []
        self.item_to_score: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame, weights: Dict[str, float] = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}):
        """Train popularity model on training events."""
        logger.info("Fitting PopularityRecommender on %d training events...", len(train_df))
        work = train_df.copy()
        work["weight"] = work["event"].map(weights).fillna(1.0).astype(np.float32)
        work["is_view"] = (work["event"] == "view").astype(np.int32)
        work["is_cart"] = (work["event"] == "addtocart").astype(np.int32)
        work["is_trans"] = (work["event"] == "transaction").astype(np.int32)

        pop = work.groupby("itemid", as_index=False).agg(
            popularity_score=("weight", "sum"),
            unique_visitors=("visitorid", "nunique"),
            total_views=("is_view", "sum"),
            total_carts=("is_cart", "sum"),
            total_purchases=("is_trans", "sum"),
        )
        pop = pop.sort_values(by=["popularity_score", "unique_visitors"], ascending=[False, False]).reset_index(drop=True)
        pop["popularity_rank"] = pop.index + 1

        self.popularity_df = pop
        self.sorted_items = pop["itemid"].tolist()
        self.sorted_scores = pop["popularity_score"].tolist()
        self.item_to_score = dict(zip(self.sorted_items, self.sorted_scores))
        logger.info("PopularityRecommender fitted successfully on %d items.", len(self.sorted_items))
        return self

    def load_from_parquet(self, path: str | Path):
        """Load precomputed popularity features from Parquet file."""
        pop = pd.read_parquet(path)
        self.popularity_df = pop
        self.sorted_items = pop["itemid"].tolist()
        self.sorted_scores = pop["popularity_score"].tolist()
        self.item_to_score = dict(zip(self.sorted_items, self.sorted_scores))
        logger.info("Loaded PopularityRecommender with %d items from %s", len(self.sorted_items), path)
        return self

    def recommend(
        self,
        visitor_id: Optional[int] = None,
        top_n: int = 10,
        filter_items: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-N popular product recommendations.

        Args:
            visitor_id: Optional visitor ID (unused for global popularity, preserved for interface).
            top_n: Number of recommendations to return.
            filter_items: Optional set of item IDs to exclude (already seen/purchased).

        Returns:
            List of (item_id, popularity_score) tuples.
        """
        filter_set = filter_items or set()
        recommendations = []

        for item, score in zip(self.sorted_items, self.sorted_scores):
            if item not in filter_set:
                recommendations.append((item, float(score)))
                if len(recommendations) >= top_n:
                    break

        return recommendations
