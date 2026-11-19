"""Unit tests for individual recommendation model architectures."""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from src.models.popularity import PopularityRecommender
from src.models.collaborative_filtering import ALSRecommender
from src.models.content_based import ContentBasedRecommender


def test_popularity_recommender():
    """Verify PopularityRecommender calculation, sorting, and seen-item filtering."""
    # Synthetic training events: Item 101 has 1 purchase (w=5) and 1 view (w=1) -> 6
    # Item 102 has 2 views (w=1+1=2) -> 2
    # Item 103 has 1 cart (w=3) -> 3
    data = pd.DataFrame({
        "visitorid": [1, 1, 2, 3],
        "itemid": [101, 101, 102, 103],
        "event": ["view", "transaction", "view", "addtocart"]
    })

    pop = PopularityRecommender()
    pop.fit(data)

    recs = pop.recommend(top_n=3)
    assert len(recs) == 3
    # Ranks should be: Item 101 (6.0), Item 103 (3.0), Item 102 (2.0 - note Item 102 had 1 view in row 2)
    assert recs[0][0] == 101
    assert recs[0][1] == 6.0

    # Test filtering item 101
    recs_filtered = pop.recommend(top_n=2, filter_items={101})
    assert len(recs_filtered) == 2
    rec_item_ids = [iid for iid, _ in recs_filtered]
    assert 101 not in rec_item_ids


def test_als_recommender_synthetic():
    """Verify ALSRecommender trains and produces valid predictions on sparse matrix."""
    # Build 3 users x 3 items sparse matrix
    # User 0 likes Item 0, 1
    # User 1 likes Item 1, 2
    # User 2 likes Item 0, 2
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    data = [5.0, 3.0, 4.0, 5.0, 1.0, 5.0]
    mat = csr_matrix((data, (rows, cols)), shape=(3, 3), dtype=np.float32)

    mappings = {
        "visitorid_to_idx": {1001: 0, 1002: 1, 1003: 2},
        "idx_to_visitorid": {0: 1001, 1: 1002, 2: 1003},
        "itemid_to_idx": {501: 0, 502: 1, 503: 2},
        "idx_to_itemid": {0: 501, 1: 502, 2: 503},
    }

    als = ALSRecommender(factors=8, regularization=0.01, iterations=5, random_state=42)
    als.fit(mat, mappings)

    # Known visitor recommendation
    recs = als.recommend(visitor_id=1001, top_n=2, filter_already_interacted=False)
    assert len(recs) > 0
    assert all(isinstance(iid, int) for iid, _ in recs)

    # Unknown visitor recommendation (must return empty list)
    cold_recs = als.recommend(visitor_id=999999, top_n=5)
    assert cold_recs == []


def test_content_based_recommender():
    """Verify ContentBasedRecommender TF-IDF tokenization and user history recommendation."""
    props_df = pd.DataFrame({
        "itemid": [201, 201, 202, 203],
        "property": [10, 20, 10, 99],
        "value": ["A", "B", "A", "Z"]
    })
    user_item_df = pd.DataFrame({
        "visitorid": [1, 2],
        "itemid": [201, 203],
        "total_weight": [5.0, 3.0]
    })

    content = ContentBasedRecommender(max_features=50, min_df=1)
    content.fit(props_df, user_item_df)

    # User 1 interacted with Item 201 (has property 10=A).
    # Item 202 also has property 10=A. Item 202 should be recommended.
    recs = content.recommend(visitor_id=1, top_n=2, filter_already_interacted=True)
    assert len(recs) > 0
    assert recs[0][0] == 202
