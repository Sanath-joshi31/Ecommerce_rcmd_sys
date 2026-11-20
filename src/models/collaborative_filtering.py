"""Implicit Alternating Least Squares (ALS) Collaborative Filtering Recommender."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix

try:
    import implicit
except (ImportError, OSError) as e:
    implicit = None
    logging.getLogger(__name__).warning("implicit C-library could not be imported (%s); collaborative model in fallback mode.", e)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ALSRecommender:
    """Collaborative Filtering Recommender using Implicit Alternating Least Squares (ALS).

    Optimizes latent user and item factor representations on sparse implicit feedback.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 20,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.use_gpu = use_gpu

        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.user_item_matrix: Optional[csr_matrix] = None
        self.visitorid_to_idx: Dict[int, int] = {}
        self.idx_to_visitorid: Dict[int, int] = {}
        self.itemid_to_idx: Dict[int, int] = {}
        self.idx_to_itemid: Dict[int, int] = {}

    def fit(
        self,
        user_item_matrix: csr_matrix,
        mappings: Dict[str, Any],
    ):
        """Fit ALS model on sparse user-item interaction matrix.

        Args:
            user_item_matrix: CSR matrix with shape (num_users, num_items).
            mappings: Dict containing bidirectional user and item index dictionaries.
        """
        logger.info(
            "Initializing ALS model (factors=%d, reg=%.3f, iter=%d)...",
            self.factors,
            self.regularization,
            self.iterations,
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
        )

        self.user_item_matrix = user_item_matrix.astype(np.float32)
        self.visitorid_to_idx = mappings["visitorid_to_idx"]
        self.idx_to_visitorid = mappings["idx_to_visitorid"]
        self.itemid_to_idx = mappings["itemid_to_idx"]
        self.idx_to_itemid = mappings["idx_to_itemid"]

        logger.info("Fitting ALS on user-item matrix of shape %s...", str(user_item_matrix.shape))
        self.model.fit(self.user_item_matrix)
        logger.info("ALS model training completed successfully.")
        return self

    def recommend(
        self,
        visitor_id: int,
        top_n: int = 10,
        filter_already_interacted: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a visitor.

        Args:
            visitor_id: External visitor ID.
            top_n: Number of items to recommend.
            filter_already_interacted: Whether to suppress previously interacted items.

        Returns:
            List of (item_id, predicted_score) tuples. Empty list if visitor is unknown (cold-start).
        """
        if self.model is None or self.model.user_factors is None or self.model.item_factors is None:
            raise RuntimeError("ALS model has not been trained or loaded.")

        if visitor_id not in self.visitorid_to_idx:
            # Cold-start visitor
            return []

        user_idx = self.visitorid_to_idx[visitor_id]

        # Case 1: user_item_matrix available
        if self.user_item_matrix is not None:
            user_vector = self.user_item_matrix[user_idx]
            ids, scores = self.model.recommend(
                userid=user_idx,
                user_items=user_vector,
                N=top_n,
                filter_already_liked_items=filter_already_interacted,
            )
            recommendations = []
            for item_idx, score in zip(ids, scores):
                if item_idx in self.idx_to_itemid:
                    real_item_id = self.idx_to_itemid[item_idx]
                    recommendations.append((real_item_id, float(score)))
            return recommendations

        # Case 2: In-memory lightweight inference using factor dot product
        user_factor = self.model.user_factors[user_idx]
        scores = self.model.item_factors.dot(user_factor)

        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        recommendations = [
            (self.idx_to_itemid[idx], float(scores[idx]))
            for idx in top_indices
            if idx in self.idx_to_itemid
        ]
        return recommendations

    def save(self, filepath: str | Path):
        """Save ALS model factors and mappings to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "random_state": self.random_state,
            "user_factors": self.model.user_factors if self.model else None,
            "item_factors": self.model.item_factors if self.model else None,
            "visitorid_to_idx": self.visitorid_to_idx,
            "idx_to_visitorid": self.idx_to_visitorid,
            "itemid_to_idx": self.itemid_to_idx,
            "idx_to_itemid": self.idx_to_itemid,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved ALSRecommender artifact to: %s", path)

    def load(self, filepath: str | Path, user_item_matrix: Optional[csr_matrix] = None):
        """Load ALS model factors and mappings from disk."""
        path = Path(filepath)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.factors = data["factors"]
        self.regularization = data["regularization"]
        self.iterations = data["iterations"]
        self.random_state = data["random_state"]
        self.visitorid_to_idx = data["visitorid_to_idx"]
        self.idx_to_visitorid = data["idx_to_visitorid"]
        self.itemid_to_idx = data["itemid_to_idx"]
        self.idx_to_itemid = data["idx_to_itemid"]

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.user_factors = data["user_factors"]
        self.model.item_factors = data["item_factors"]
        if user_item_matrix is not None:
            self.user_item_matrix = user_item_matrix

        logger.info("Loaded ALSRecommender successfully from: %s", path)
        return self
