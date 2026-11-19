"""Train and comparatively evaluate all recommendation models on validation split."""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np

# Project path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.popularity import PopularityRecommender
from src.models.collaborative_filtering import ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.recommendation.hybrid import HybridRecommender
from src.features.build_features import aggregate_user_item_interactions, build_sparse_interaction_matrix
from src.evaluation.metrics import evaluate_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4: MACHINE LEARNING MODEL DEVELOPMENT & EVALUATION")
    logger.info("=" * 70)

    data_dir = Path("data/processed")
    raw_dir = Path("dataset")
    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    mappings_dir = Path("artifacts/mappings")

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    logger.info("Loading training and validation parquet splits...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    logger.info("Loaded Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    # Catalog of all items in training set
    catalog_items = set(train_df["itemid"].unique())
    logger.info("Catalog size (unique training items): %d", len(catalog_items))

    # 2. Build & Train Model 1: Popularity Baseline
    logger.info("\n--- Training Model 1: Popularity Recommender ---")
    pop_model = PopularityRecommender()
    pop_model.load_from_parquet(data_dir / "popularity_features.parquet")

    # 3. Build & Train Model 2: Implicit ALS
    logger.info("\n--- Training Model 2: Implicit ALS Collaborative Filtering ---")
    train_user_item = aggregate_user_item_interactions(train_df)
    sparse_mat, mappings = build_sparse_interaction_matrix(train_user_item)

    als_model = ALSRecommender(factors=64, regularization=0.1, iterations=20, random_state=42)
    als_model.fit(sparse_mat, mappings)

    # Save ALS artifact
    als_path = models_dir / "als_model.pkl"
    als_model.save(als_path)

    # 4. Build & Train Model 3: Content-Based Recommender
    logger.info("\n--- Training Model 3: Attribute-Based Content Recommender ---")
    logger.info("Ingesting item properties sample for attribute profiles...")
    props_df = pd.read_csv(raw_dir / "item_properties_part1.csv", nrows=300000)
    content_model = ContentBasedRecommender(max_features=5000, min_df=2)
    content_model.fit(props_df, train_user_item)

    content_path = models_dir / "content_model.pkl"
    content_model.save(content_path)

    # 5. Build Model 4: Hybrid Recommender
    logger.info("\n--- Initializing Model 4: Hybrid Recommendation Engine ---")
    hybrid_model = HybridRecommender(
        popularity_model=pop_model,
        als_model=als_model,
        content_model=content_model,
        collaborative_weight=0.5,
        content_weight=0.3,
        popularity_weight=0.2,
    )

    # 6. Prepare Evaluation Set (Warm Validation Users)
    logger.info("\n--- Preparing Validation Evaluation Population ---")
    # Ground truth: items interacted with in validation split
    val_ground_truth: Dict[int, Set[int]] = (
        val_df.groupby("visitorid")["itemid"]
        .apply(lambda x: set(x))
        .to_dict()
    )

    # Filter to warm users present in training set
    known_users = set(mappings["visitorid_to_idx"].keys())
    warm_val_users = [u for u in val_ground_truth.keys() if u in known_users]
    logger.info("Total validation users: %d | Warm validation users: %d", len(val_ground_truth), len(warm_val_users))

    # Sample 2,000 warm users for rapid, statistically robust offline evaluation
    np.random.seed(42)
    sample_size = min(2000, len(warm_val_users))
    eval_user_sample = np.random.choice(warm_val_users, size=sample_size, replace=False).tolist()
    eval_ground_truth = {u: val_ground_truth[u] for u in eval_user_sample}
    logger.info("Sampled %d warm evaluation users for benchmark comparison.", sample_size)

    # 7. Generate Recommendations & Evaluate Across Models
    logger.info("\n--- Running Comparative Benchmark Evaluation ---")
    models = {
        "Popularity Baseline": pop_model,
        "Collaborative Filtering (ALS)": als_model,
        "Content-Based (TF-IDF)": content_model,
        "Hybrid Engine (ALS+Content+Pop)": hybrid_model,
    }

    comparison_results = {}
    k_eval_list = [5, 10, 20]

    for model_name, model in models.items():
        logger.info("Evaluating: %s...", model_name)
        recs_dict: Dict[int, List[int]] = {}
        t0 = time.time()

        for u in eval_user_sample:
            if model_name == "Hybrid Engine (ALS+Content+Pop)":
                recs, _ = model.recommend(visitor_id=u, top_n=20, filter_already_interacted=True)
            elif model_name == "Popularity Baseline":
                recs = model.recommend(visitor_id=u, top_n=20, filter_items=None)
            else:
                recs = model.recommend(visitor_id=u, top_n=20, filter_already_interacted=True)

            recs_dict[u] = [item_id for item_id, _ in recs]

        eval_duration = time.time() - t0
        metrics = evaluate_recommendations(recs_dict, eval_ground_truth, catalog_items, k_eval_list)
        metrics["inference_time_sec"] = round(eval_duration, 2)
        metrics["avg_latency_ms"] = round((eval_duration / sample_size) * 1000, 2)
        comparison_results[model_name] = metrics

    # 8. Log and Save Comparison Results
    comparison_path = metrics_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info("Saved model comparison results to: %s", comparison_path)

    # 9. Format Summary Comparison Table
    logger.info("\n" + "=" * 90)
    logger.info("OFFLINE VALIDATION BENCHMARK RESULTS (Top-10 Ranking Metrics)")
    logger.info("=" * 90)
    logger.info(
        f"{'Model Name':<35} | {'Recall@10':<10} | {'Prec@10':<10} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Coverage%':<10} | {'Latency':<8}"
    )
    logger.info("-" * 90)
    for mname, m in comparison_results.items():
        logger.info(
            f"{mname:<35} | {m.get('recall@10', 0):<10.5f} | {m.get('precision@10', 0):<10.5f} | "
            f"{m.get('ndcg@10', 0):<10.5f} | {m.get('map@10', 0):<10.5f} | {m.get('coverage@10_pct', 0):<10.2f} | "
            f"{m.get('avg_latency_ms', 0):<5.2f} ms"
        )
    logger.info("=" * 90)
    logger.info("Phase 4 Model Training and Evaluation completed successfully in %.2f seconds.", time.time() - start_time)


if __name__ == "__main__":
    main()
