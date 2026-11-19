"""MLflow Experiment Tracking and Model Management for Retailrocket Recommender."""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Retailrocket-Recommendations"
TRACKING_URI = "sqlite:///mlflow.db"


def run_tracking():
    logger.info("=" * 70)
    logger.info("PHASE 5: MLFLOW EXPERIMENT TRACKING & MODEL MANAGEMENT")
    logger.info("=" * 70)

    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        exp_id = client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=str(Path("artifacts/mlruns").resolve()),
        )
        logger.info("Created new MLflow experiment: %s (ID: %s)", EXPERIMENT_NAME, exp_id)
    else:
        exp_id = experiment.experiment_id
        logger.info("Using existing MLflow experiment: %s (ID: %s)", EXPERIMENT_NAME, exp_id)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load Evaluation Benchmarks
    metrics_path = Path("artifacts/metrics/model_comparison.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation results at {metrics_path}")

    with open(metrics_path, "r") as f:
        benchmarks = json.load(f)

    # 3. Model Run Definitions
    runs_to_track = [
        {
            "run_name": "Popularity-Baseline",
            "model_type": "popularity",
            "params": {
                "algorithm": "PopularityRecommender",
                "weights_view": 1.0,
                "weights_cart": 3.0,
                "weights_trans": 5.0,
                "dataset_version": "1.0.0",
                "training_cutoff": "2015-08-02T23:29:02Z",
                "feature_version": "v1_weighted_frequency",
            },
            "metrics": benchmarks.get("Popularity Baseline", {}),
            "artifacts": ["data/processed/popularity_features.parquet"],
        },
        {
            "run_name": "Collaborative-Filtering-ALS",
            "model_type": "als",
            "params": {
                "algorithm": "ImplicitALS",
                "factors": 64,
                "regularization": 0.1,
                "iterations": 20,
                "random_seed": 42,
                "dataset_version": "1.0.0",
                "training_cutoff": "2015-08-02T23:29:02Z",
                "feature_version": "v1_csr_implicit_matrix",
            },
            "metrics": benchmarks.get("Collaborative Filtering (ALS)", {}),
            "artifacts": [
                "artifacts/models/als_model.pkl",
                "artifacts/mappings/train_mappings.pkl",
            ],
        },
        {
            "run_name": "Content-Based-TFIDF",
            "model_type": "content_based",
            "params": {
                "algorithm": "ContentBasedTFIDF",
                "max_features": 5000,
                "min_df": 2,
                "similarity_metric": "sparse_cosine",
                "dataset_version": "1.0.0",
                "training_cutoff": "2015-08-02T23:29:02Z",
                "feature_version": "v1_anonymized_property_tokens",
            },
            "metrics": benchmarks.get("Content-Based (TF-IDF)", {}),
            "artifacts": [
                "artifacts/models/content_model.pkl",
                "artifacts/mappings/train_mappings.pkl",
            ],
        },
        {
            "run_name": "Hybrid-Engine-Balanced",
            "model_type": "hybrid",
            "params": {
                "algorithm": "HybridRankFusion",
                "collaborative_weight": 0.5,
                "content_weight": 0.3,
                "popularity_weight": 0.2,
                "normalization": "min_max",
                "filter_already_interacted": True,
                "dataset_version": "1.0.0",
                "training_cutoff": "2015-08-02T23:29:02Z",
                "feature_version": "v1_hybrid_blend",
            },
            "metrics": benchmarks.get("Hybrid Engine (ALS+Content+Pop)", {}),
            "artifacts": [
                "artifacts/models/als_model.pkl",
                "artifacts/models/content_model.pkl",
                "data/processed/popularity_features.parquet",
                "artifacts/mappings/train_mappings.pkl",
                "configs/config.yaml",
            ],
        },
    ]

    # 4. Log Each Run to MLflow
    logged_runs = {}
    for run_spec in runs_to_track:
        run_name = run_spec["run_name"]
        logger.info("Logging MLflow Run: %s...", run_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Log Parameters
            mlflow.log_params(run_spec["params"])

            # Log Metrics (Sanitize keys: MLflow forbids '@' in metric names)
            for m_key, m_val in run_spec["metrics"].items():
                if isinstance(m_val, (int, float)):
                    sanitized_key = m_key.replace("@", "_at_")
                    mlflow.log_metric(sanitized_key, m_val)

            # Log Artifacts
            for art_path in run_spec["artifacts"]:
                p = Path(art_path)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="model_package")
                else:
                    logger.warning("Artifact not found to log: %s", art_path)

            # Tag Run
            mlflow.set_tag("model_type", run_spec["model_type"])
            mlflow.set_tag("pipeline_stage", "offline_evaluation")
            mlflow.set_tag("project", "Retailrocket-Recommendations")

            logged_runs[run_name] = {
                "run_id": run.info.run_id,
                "artifact_uri": run.info.artifact_uri,
                "recall@10": run_spec["metrics"].get("recall@10", 0.0),
                "coverage@10_pct": run_spec["metrics"].get("coverage@10_pct", 0.0),
                "avg_latency_ms": run_spec["metrics"].get("avg_latency_ms", 0.0),
            }
            logger.info("Run '%s' logged with ID: %s", run_name, run.info.run_id)

    # 5. Save Experiment Lineage JSON
    lineage_path = Path("artifacts/metrics/mlflow_experiment_summary.json")
    with open(lineage_path, "w") as f:
        json.dump(logged_runs, f, indent=2)
    logger.info("Saved MLflow experiment summary to: %s", lineage_path)
    return logged_runs


if __name__ == "__main__":
    run_tracking()
