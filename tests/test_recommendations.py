"""Unit tests for Hybrid Recommender, normalization, and cold-start fallback logic."""

import pytest
import pandas as pd
from src.models.popularity import PopularityRecommender
from src.recommendation.hybrid import HybridRecommender, min_max_normalize


def test_min_max_normalize():
    """Verify normalization handles diverse score ranges and edge cases."""
    # Standard range
    scores = {101: 10.0, 102: 5.0, 103: 0.0}
    norm = min_max_normalize(scores)
    assert pytest.approx(norm[101]) == 1.0
    assert pytest.approx(norm[102]) == 0.5
    assert pytest.approx(norm[103]) == 0.0

    # Identical values
    flat = {201: 5.0, 202: 5.0}
    norm_flat = min_max_normalize(flat)
    assert norm_flat[201] == 1.0
    assert norm_flat[202] == 1.0

    # Empty dictionary
    assert min_max_normalize({}) == {}


def test_hybrid_cold_start_fallback():
    """Verify HybridRecommender gracefully falls back to popularity when sub-models return empty."""
    pop_df = pd.DataFrame({
        "itemid": [1, 2, 3, 4, 5],
        "popularity_score": [100.0, 80.0, 60.0, 40.0, 20.0],
        "unique_visitors": [50, 40, 30, 20, 10],
        "popularity_rank": [1, 2, 3, 4, 5]
    })
    pop = PopularityRecommender()
    pop.sorted_items = pop_df["itemid"].tolist()
    pop.sorted_scores = pop_df["popularity_score"].tolist()
    pop.item_to_score = dict(zip(pop.sorted_items, pop.sorted_scores))

    # Construct hybrid with no ALS or Content sub-models
    hybrid = HybridRecommender(popularity_model=pop, als_model=None, content_model=None)

    recs, source = hybrid.recommend(visitor_id=88888, top_n=3)
    assert source == "popularity_fallback"
    assert len(recs) == 3
    assert recs[0][0] == 1
    assert recs[1][0] == 2
    assert recs[2][0] == 3


def test_hybrid_weight_updating():
    """Verify hybrid weights updating and normalization."""
    pop = PopularityRecommender()
    hybrid = HybridRecommender(popularity_model=pop, collaborative_weight=2.0, content_weight=2.0, popularity_weight=1.0)
    assert pytest.approx(hybrid.w_cf) == 0.4
    assert pytest.approx(hybrid.w_content) == 0.4
    assert pytest.approx(hybrid.w_pop) == 0.2

    hybrid.set_weights(w_cf=3.0, w_content=0.0, w_pop=1.0)
    assert pytest.approx(hybrid.w_cf) == 0.75
    assert pytest.approx(hybrid.w_content) == 0.0
    assert pytest.approx(hybrid.w_pop) == 0.25
