"""Attribute-based Content Recommender using TF-IDF and Sparse Cosine Similarity."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommender using tokenized item properties and TF-IDF representations.

    Leverages anonymized item properties to build sparse attribute profiles.
    User recommendations are computed by projecting user interaction history into the
    item TF-IDF vector space and calculating sparse cosine similarities.
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_tfidf: Optional[csr_matrix] = None
        self.item_ids: List[int] = []
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.user_history: Dict[int, Dict[int, float]] = {}

    def fit(
        self,
        item_properties_df: pd.DataFrame,
        user_item_df: Optional[pd.DataFrame] = None,
    ):
        """Build TF-IDF representations from item properties.

        Args:
            item_properties_df: DataFrame containing ['itemid', 'property', 'value'].
            user_item_df: Optional DataFrame with ['visitorid', 'itemid', 'total_weight']
                          to cache known user interaction history.
        """
        logger.info("Building content attribute tokens for %d property records...", len(item_properties_df))

        # Combine property and value into discrete attribute tokens: e.g. "prop_830_val_10294"
        sample_props = item_properties_df.copy()
        sample_props["token"] = (
            "p_"
            + sample_props["property"].astype(str)
            + "_v_"
            + sample_props["value"].astype(str)
        )

        # Aggregate tokens per item
        item_docs = (
            sample_props.groupby("itemid")["token"]
            .apply(lambda tokens: " ".join(tokens))
            .reset_index()
        )

        self.item_ids = item_docs["itemid"].tolist()
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}

        logger.info("Vectorizing attribute profiles for %d items using TF-IDF...", len(item_docs))
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.item_tfidf = self.vectorizer.fit_transform(item_docs["token"]).tocsr()
        logger.info(
            "Content TF-IDF Matrix built: %d items x %d features | Non-zeros: %d",
            self.item_tfidf.shape[0],
            self.item_tfidf.shape[1],
            self.item_tfidf.nnz,
        )

        if user_item_df is not None:
            self.cache_user_history(user_item_df)

        return self

    def cache_user_history(self, user_item_df: pd.DataFrame):
        """Cache user interaction history for low-latency recommendation queries."""
        # Pre-filter to only items present in the fitted TF-IDF matrix to eliminate memory exhaustion
        if self.item_to_idx is not None and len(self.item_to_idx) > 0:
            user_item_df = user_item_df[user_item_df["itemid"].isin(self.item_to_idx)]

        logger.info("Caching user history for content recommendations across %d records...", len(user_item_df))
        history: Dict[int, Dict[int, float]] = {}
        for row in user_item_df[["visitorid", "itemid", "total_weight"]].itertuples(index=False):
            vid, iid, w = row
            if vid not in history:
                history[vid] = {}
            history[vid][iid] = float(w)
        self.user_history = history
        logger.info("Cached interaction history for %d visitors.", len(history))

    def recommend(
        self,
        visitor_id: int,
        top_n: int = 10,
        filter_already_interacted: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate content-based recommendations by projecting visitor history into item space.

        Args:
            visitor_id: Visitor identifier.
            top_n: Number of recommendations to retrieve.
            filter_already_interacted: Whether to exclude previously interacted items.

        Returns:
            List of (item_id, cosine_similarity_score) tuples.
        """
        if self.item_tfidf is None:
            raise RuntimeError("Content model has not been fitted or loaded.")

        if visitor_id not in self.user_history or not self.user_history[visitor_id]:
            # Cold-start visitor
            return []

        user_items = self.user_history[visitor_id]
        known_indices = [
            (self.item_to_idx[iid], weight)
            for iid, weight in user_items.items()
            if iid in self.item_to_idx
        ]

        if not known_indices:
            return []

        # Build user profile vector as weighted average of interacted item TF-IDF vectors
        user_vector = np.zeros(self.item_tfidf.shape[1], dtype=np.float32)
        total_weight = 0.0
        seen_indices = set()

        for idx, w in known_indices:
            user_vector += self.item_tfidf[idx].toarray().flatten() * w
            total_weight += w
            seen_indices.add(idx)

        if total_weight > 0:
            user_vector /= total_weight

        # Normalize user vector for cosine similarity
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector /= norm

        # Compute cosine similarity: (num_items, features) dot (features,)
        scores = self.item_tfidf.dot(user_vector)

        if filter_already_interacted:
            for idx in seen_indices:
                scores[idx] = -1.0

        # Retrieve top-N
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        recommendations = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                recommendations.append((self.idx_to_item[idx], score))

        return recommendations

    def save(self, filepath: str | Path):
        """Serialize content-based model components."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vectorizer": self.vectorizer,
            "item_tfidf": self.item_tfidf,
            "item_ids": self.item_ids,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "user_history": self.user_history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved ContentBasedRecommender to: %s", path)

    def load(self, filepath: str | Path):
        """Load content-based model components."""
        path = Path(filepath)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.max_features = data["max_features"]
        self.min_df = data["min_df"]
        self.vectorizer = data["vectorizer"]
        self.item_tfidf = data["item_tfidf"]
        self.item_ids = data["item_ids"]
        self.item_to_idx = data["item_to_idx"]
        self.idx_to_item = data["idx_to_item"]
        self.user_history = data["user_history"]
        logger.info("Loaded ContentBasedRecommender successfully from: %s", path)
        return self
