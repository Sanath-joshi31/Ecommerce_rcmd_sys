"""End-to-End Data Preparation Pipeline for Retailrocket Recommender System."""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_data import verify_dataset_files, load_events, load_category_tree
from src.data.preprocess import validate_raw_events, clean_events, split_chronological
from src.features.build_features import (
    aggregate_user_item_interactions,
    build_popularity_features,
    build_sparse_interaction_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3: DATA PREPARATION & PROCESSING PIPELINE")
    logger.info("=" * 70)

    # 1. Dataset Verification
    data_dir = Path("dataset")
    out_dir = Path("data/processed")
    metrics_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    file_report = verify_dataset_files(data_dir)
    logger.info("Verified raw dataset files:")
    for fname, meta in file_report.items():
        logger.info("  - %s: %.2f MB", fname, meta["size_mb"])

    # 2. Ingest Events
    logger.info("Ingesting raw events.csv...")
    raw_events = load_events(data_dir)
    raw_validation = validate_raw_events(raw_events)
    logger.info("Raw events validation status: %s", raw_validation["status"])

    # 3. Clean Events
    logger.info("Cleaning events dataset...")
    clean_df, cleaning_meta = clean_events(raw_events)
    clean_events_path = out_dir / "events_clean.parquet"
    clean_df.to_parquet(clean_events_path, index=False)
    logger.info("Saved cleaned events to: %s", clean_events_path)

    # 4. User-Product Interaction Aggregation
    logger.info("Aggregating user-product interaction confidence scores...")
    user_item_df = aggregate_user_item_interactions(clean_df)
    user_item_path = out_dir / "user_item_interactions.parquet"
    user_item_df.to_parquet(user_item_path, index=False)
    logger.info("Saved aggregated interactions to: %s", user_item_path)

    # 5. Chronological Splitting (70% Train, 15% Val, 15% Test)
    logger.info("Executing chronological train-val-test split...")
    train_df, val_df, test_df, split_meta = split_chronological(clean_df, 0.70, 0.15, 0.15)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info("Saved Train split (%d rows) to: %s", len(train_df), train_path)
    logger.info("Saved Val split (%d rows) to: %s", len(val_df), val_path)
    logger.info("Saved Test split (%d rows) to: %s", len(test_df), test_path)

    # 6. Popularity Features (Strictly on Train split)
    logger.info("Computing global popularity features on training split...")
    pop_df = build_popularity_features(train_df)
    pop_path = out_dir / "popularity_features.parquet"
    pop_df.to_parquet(pop_path, index=False)
    logger.info("Saved popularity features to: %s", pop_path)

    # 7. Training Interaction Matrix & Sparsity
    train_user_item = aggregate_user_item_interactions(train_df)
    sparse_mat, mat_meta = build_sparse_interaction_matrix(train_user_item)

    # Save mapping dictionaries as pickle for Phase 4
    import pickle
    mappings_path = Path("artifacts/mappings/train_mappings.pkl")
    with open(mappings_path, "wb") as f:
        pickle.dump(
            {
                "visitorid_to_idx": mat_meta["visitorid_to_idx"],
                "idx_to_visitorid": mat_meta["idx_to_visitorid"],
                "itemid_to_idx": mat_meta["itemid_to_idx"],
                "idx_to_itemid": mat_meta["idx_to_itemid"],
            },
            f,
        )
    logger.info("Saved index mappings pickle to: %s", mappings_path)

    matrix_summary = {
        "num_users": int(mat_meta["num_users"]),
        "num_items": int(mat_meta["num_items"]),
        "non_zeros": int(mat_meta["non_zeros"]),
        "sparsity_pct": float(mat_meta["sparsity_pct"]),
        "density_pct": float(mat_meta["density_pct"]),
    }

    # 8. Compile Quality & Preparation Report
    duration = time.time() - start_time
    full_report = {
        "execution_duration_sec": round(duration, 2),
        "raw_files": file_report,
        "raw_validation": raw_validation,
        "cleaning": cleaning_meta,
        "splitting": split_meta,
        "training_matrix": matrix_summary,
    }

    report_path = metrics_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Saved comprehensive Data Quality Report to: %s", report_path)

    logger.info("=" * 70)
    logger.info("PHASE 3 DATA PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", duration)
    logger.info("=" * 70)
    return full_report


if __name__ == "__main__":
    run_pipeline()
