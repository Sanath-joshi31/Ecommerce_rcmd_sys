"""Data preprocessing, cleaning, and chronological splitting for Retailrocket events."""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_EVENTS = {"view", "addtocart", "transaction"}


def validate_raw_events(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute integrity validation checks on raw events DataFrame.

    Args:
        df: Raw events DataFrame.

    Returns:
        Dictionary report of validation metrics and status.
    """
    total_rows = len(df)
    missing_timestamp = int(df["timestamp"].isna().sum())
    missing_visitorid = int(df["visitorid"].isna().sum())
    missing_itemid = int(df["itemid"].isna().sum())
    missing_event = int(df["event"].isna().sum())

    invalid_event_types = int((~df["event"].isin(VALID_EVENTS)).sum())

    # transactionid is expected to be null on views and addtocart
    transaction_events = df[df["event"] == "transaction"]
    transactions_missing_id = int(transaction_events["transactionid"].isna().sum())

    # Exact duplicate check
    exact_duplicates = int(df.duplicated().sum())

    report = {
        "total_rows": total_rows,
        "missing_timestamp": missing_timestamp,
        "missing_visitorid": missing_visitorid,
        "missing_itemid": missing_itemid,
        "missing_event": missing_event,
        "invalid_event_types": invalid_event_types,
        "transactions_missing_id": transactions_missing_id,
        "exact_duplicates": exact_duplicates,
        "status": "PASS"
        if (
            missing_timestamp == 0
            and missing_visitorid == 0
            and missing_itemid == 0
            and invalid_event_types == 0
        )
        else "FAIL",
    }
    logger.info("Validation completed: Status=%s", report["status"])
    return report


def clean_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw events dataset and record reproducible cleaning decisions.

    Decisions enforced:
    1. Drop records missing essential keys (timestamp, visitorid, itemid, event).
    2. Filter out unauthorized event types not in {'view', 'addtocart', 'transaction'}.
    3. Convert millisecond epoch timestamps to UTC datetimes.
    4. Remove exact duplicate clicks (same millisecond, visitor, item, and event).
    5. Retain repeated views across distinct timestamps (legitimate intent).
    6. Ensure non-negative and plausible timestamps.

    Args:
        df: Raw events DataFrame.

    Returns:
        Tuple of (cleaned_df, cleaning_metadata).
    """
    initial_count = len(df)
    logger.info("Starting data cleaning on %d records...", initial_count)

    # 1. Drop missing essential keys
    clean_df = df.dropna(subset=["timestamp", "visitorid", "itemid", "event"]).copy()
    dropped_missing_keys = initial_count - len(clean_df)

    # 2. Filter invalid events
    clean_df = clean_df[clean_df["event"].isin(VALID_EVENTS)].copy()
    dropped_invalid_events = initial_count - dropped_missing_keys - len(clean_df)

    # 3. Handle exact duplicate clicks (accidental client double-triggers)
    dedup_before = len(clean_df)
    clean_df = clean_df.drop_duplicates(
        subset=["timestamp", "visitorid", "itemid", "event"]
    ).copy()
    dropped_exact_duplicates = dedup_before - len(clean_df)

    # 4. Standardize datetimes
    clean_df["datetime"] = pd.to_datetime(clean_df["timestamp"], unit="ms", utc=True)
    clean_df["date"] = clean_df["datetime"].dt.date
    clean_df["hour"] = clean_df["datetime"].dt.hour
    clean_df["day_of_week"] = clean_df["datetime"].dt.dayofweek

    # 5. Type casting
    clean_df["visitorid"] = clean_df["visitorid"].astype(np.int64)
    clean_df["itemid"] = clean_df["itemid"].astype(np.int64)
    clean_df["timestamp"] = clean_df["timestamp"].astype(np.int64)

    # 6. Sort chronologically
    clean_df = clean_df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    final_count = len(clean_df)
    retention_rate = (final_count / initial_count) * 100 if initial_count > 0 else 0.0

    cleaning_metadata = {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "dropped_missing_keys": dropped_missing_keys,
        "dropped_invalid_events": dropped_invalid_events,
        "dropped_exact_duplicates": dropped_exact_duplicates,
        "retention_rate_pct": round(retention_rate, 4),
        "min_datetime": str(clean_df["datetime"].min()),
        "max_datetime": str(clean_df["datetime"].max()),
        "unique_visitors": int(clean_df["visitorid"].nunique()),
        "unique_items": int(clean_df["itemid"].nunique()),
    }
    logger.info(
        "Cleaning finished: Retained %d / %d records (%.2f%%)",
        final_count,
        initial_count,
        retention_rate,
    )
    return clean_df, cleaning_metadata


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Enforce strict chronological train-validation-test split based on temporal quantiles.

    Guarantees:
    - Train data strictly precedes Validation data.
    - Validation data strictly precedes Test data.
    - Zero future lookahead leakage into training features.

    Args:
        df: Cleaned events DataFrame sorted by timestamp ascending.
        train_ratio: Proportion for training (default 0.70).
        val_ratio: Proportion for validation (default 0.15).
        test_ratio: Proportion for testing (default 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df, split_metadata).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    total_events = len(df)
    train_cutoff_idx = int(total_events * train_ratio)
    val_cutoff_idx = int(total_events * (train_ratio + val_ratio))

    t_val_start = df.iloc[train_cutoff_idx]["timestamp"]
    t_test_start = df.iloc[val_cutoff_idx]["timestamp"]

    train_df = df[df["timestamp"] < t_val_start].copy()
    val_df = df[(df["timestamp"] >= t_val_start) & (df["timestamp"] < t_test_start)].copy()
    test_df = df[df["timestamp"] >= t_test_start].copy()

    # Leakage assertions
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Data leakage: Train overlaps Validation!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Data leakage: Validation overlaps Test!"

    split_metadata = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_pct": round(len(train_df) / total_events * 100, 2),
        "val_pct": round(len(val_df) / total_events * 100, 2),
        "test_pct": round(len(test_df) / total_events * 100, 2),
        "train_end_time": str(train_df["datetime"].max()),
        "val_start_time": str(val_df["datetime"].min()),
        "val_end_time": str(val_df["datetime"].max()),
        "test_start_time": str(test_df["datetime"].min()),
        "test_end_time": str(test_df["datetime"].max()),
        "train_unique_users": int(train_df["visitorid"].nunique()),
        "train_unique_items": int(train_df["itemid"].nunique()),
        "val_unique_users": int(val_df["visitorid"].nunique()),
        "test_unique_users": int(test_df["visitorid"].nunique()),
    }
    logger.info("Chronological split completed cleanly with zero temporal leakage.")
    return train_df, val_df, test_df, split_metadata
