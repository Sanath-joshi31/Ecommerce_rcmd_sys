"""Evaluation metrics for top-K recommendation systems."""

import logging
from typing import Dict, List, Set, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def precision_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    """Calculate Precision@K: Fraction of top-K recommended items that are relevant."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in actual)
    return hits / float(k)


def recall_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    """Calculate Recall@K: Fraction of actual relevant items captured in top-K."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in actual)
    return hits / float(len(actual))


def ndcg_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # index i is rank i+1; log2(rank + 1)

    # Ideal DCG: all actual items placed at top ranks
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def average_precision_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    """Calculate Average Precision at K (AP@K) for a single user."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    catalog_items: Set[int],
    k_list: List[int] = [5, 10, 20],
) -> Dict[str, Any]:
    """Compute comprehensive ranking metrics across a population of evaluation users.

    Computes:
    - Precision@K for each K
    - Recall@K for each K
    - NDCG@K for each K
    - MAP@K for each K
    - Catalog Coverage@K for each K

    Args:
        recommendations: Dict mapping visitor_id to list of recommended item_ids.
        ground_truth: Dict mapping visitor_id to set of ground-truth interacted item_ids.
        catalog_items: Set of all unique product IDs in catalog.
        k_list: List of K thresholds (e.g. [5, 10, 20]).

    Returns:
        Dict mapping metric names to evaluated scores.
    """
    eval_users = [u for u in ground_truth.keys() if u in recommendations]
    if not eval_users:
        logger.warning("No overlapping users between recommendations and ground truth.")
        return {}

    n_users = len(eval_users)
    n_catalog = len(catalog_items) if catalog_items else 1
    results = {"evaluated_users": n_users}

    for k in k_list:
        precisions = []
        recalls = []
        ndcgs = []
        aps = []
        unique_recommended_items = set()

        for u in eval_users:
            rec = recommendations[u]
            act = ground_truth[u]
            precisions.append(precision_at_k(rec, act, k))
            recalls.append(recall_at_k(rec, act, k))
            ndcgs.append(ndcg_at_k(rec, act, k))
            aps.append(average_precision_at_k(rec, act, k))
            unique_recommended_items.update(rec[:k])

        coverage = len(unique_recommended_items) / float(n_catalog) * 100.0

        results[f"precision@{k}"] = round(float(np.mean(precisions)), 5)
        results[f"recall@{k}"] = round(float(np.mean(recalls)), 5)
        results[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 5)
        results[f"map@{k}"] = round(float(np.mean(aps)), 5)
        results[f"coverage@{k}_pct"] = round(float(coverage), 3)

    return results
