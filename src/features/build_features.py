"""Feature engineering, interaction weighting, and sparse matrix generation."""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


def calculate_interaction_weights(
    df: pd.DataFrame, weights: Dict[str, float] = DEFAULT_WEIGHTS
) -> pd.Series:
    """Calculate discrete behavioral interaction weight for each event.

    Args:
        df: DataFrame containing 'event' column.
        weights: Dictionary mapping event names to numerical confidence weights.

    Returns:
        Series of numeric interaction weights.
    """
    return df["event"].map(weights).fillna(1.0).astype(np.float32)


def aggregate_user_item_interactions(
    df: pd.DataFrame, weights: Dict[str, float] = DEFAULT_WEIGHTS
) -> pd.DataFrame:
    """Aggregate event logs into a consolidated (visitorid, itemid) interaction table.

    Computes:
    - total_weight: Sum of implicit feedback confidence.
    - event_count: Number of distinct actions.
    - view_count: Frequency of product views.
    - cart_count: Frequency of add-to-cart actions.
    - purchase_count: Frequency of purchases.
    - last_timestamp: Most recent interaction epoch ms.

    Args:
        df: Cleaned events DataFrame.
        weights: Event weighting configuration.

    Returns:
        Aggregated DataFrame indexed by (visitorid, itemid).
    """
    logger.info("Aggregating %d event interactions by user-product pairs...", len(df))
    work_df = df.copy()
    work_df["weight"] = calculate_interaction_weights(work_df, weights)
    work_df["is_view"] = (work_df["event"] == "view").astype(np.int32)
    work_df["is_cart"] = (work_df["event"] == "addtocart").astype(np.int32)
    work_df["is_trans"] = (work_df["event"] == "transaction").astype(np.int32)

    grouped = work_df.groupby(["visitorid", "itemid"], as_index=False).agg(
        total_weight=("weight", "sum"),
        event_count=("weight", "count"),
        view_count=("is_view", "sum"),
        cart_count=("is_cart", "sum"),
        purchase_count=("is_trans", "sum"),
        last_timestamp=("timestamp", "max"),
    )

    grouped["total_weight"] = grouped["total_weight"].astype(np.float32)
    logger.info("Generated %d unique user-product interaction records", len(grouped))
    return grouped


def build_popularity_features(
    train_df: pd.DataFrame, weights: Dict[str, float] = DEFAULT_WEIGHTS
) -> pd.DataFrame:
    """Compute global item popularity features strictly from training split.

    Args:
        train_df: Chronologically split training events.
        weights: Event weighting dictionary.

    Returns:
        DataFrame sorted by popularity_score descending with rank.
    """
    logger.info("Computing global popularity features on %d training records...", len(train_df))
    work_df = train_df.copy()
    work_df["weight"] = calculate_interaction_weights(work_df, weights)
    work_df["is_view"] = (work_df["event"] == "view").astype(np.int32)
    work_df["is_cart"] = (work_df["event"] == "addtocart").astype(np.int32)
    work_df["is_trans"] = (work_df["event"] == "transaction").astype(np.int32)

    pop = work_df.groupby("itemid", as_index=False).agg(
        popularity_score=("weight", "sum"),
        unique_visitors=("visitorid", "nunique"),
        total_views=("is_view", "sum"),
        total_carts=("is_cart", "sum"),
        total_purchases=("is_trans", "sum"),
    )

    pop = pop.sort_values(by=["popularity_score", "unique_visitors"], ascending=[False, False]).reset_index(drop=True)
    pop["popularity_rank"] = pop.index + 1
    logger.info("Computed popularity metrics for %d distinct training items", len(pop))
    return pop


def build_sparse_interaction_matrix(
    user_item_df: pd.DataFrame,
) -> Tuple[csr_matrix, Dict[str, Any]]:
    """Build a Compressed Sparse Row (CSR) matrix and bidirectional index mappings.

    Args:
        user_item_df: Aggregated DataFrame containing ['visitorid', 'itemid', 'total_weight'].

    Returns:
        Tuple of (csr_matrix [num_users x num_items], mappings_dict).
    """
    logger.info("Building CSR interaction matrix for %d interactions...", len(user_item_df))

    unique_visitors = np.sort(user_item_df["visitorid"].unique())
    unique_items = np.sort(user_item_df["itemid"].unique())

    visitorid_to_idx = {vid: idx for idx, vid in enumerate(unique_visitors)}
    idx_to_visitorid = {idx: vid for vid, idx in visitorid_to_idx.items()}

    itemid_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx_to_itemid = {idx: iid for iid, idx in itemid_to_idx.items()}

    rows = user_item_df["visitorid"].map(visitorid_to_idx).to_numpy(dtype=np.int32)
    cols = user_item_df["itemid"].map(itemid_to_idx).to_numpy(dtype=np.int32)
    data = user_item_df["total_weight"].to_numpy(dtype=np.float32)

    num_users = len(unique_visitors)
    num_items = len(unique_items)

    mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32)

    total_possible = num_users * num_items
    density = (mat.nnz / total_possible) * 100 if total_possible > 0 else 0.0
    sparsity = 100.0 - density

    mappings = {
        "visitorid_to_idx": visitorid_to_idx,
        "idx_to_visitorid": idx_to_visitorid,
        "itemid_to_idx": itemid_to_idx,
        "idx_to_itemid": idx_to_itemid,
        "num_users": num_users,
        "num_items": num_items,
        "non_zeros": mat.nnz,
        "sparsity_pct": round(sparsity, 4),
        "density_pct": round(density, 6),
    }

    logger.info(
        "Sparse Matrix built: %d users x %d items | Non-zeros: %d | Sparsity: %.4f%%",
        num_users,
        num_items,
        mat.nnz,
        sparsity,
    )
    return mat, mappings
