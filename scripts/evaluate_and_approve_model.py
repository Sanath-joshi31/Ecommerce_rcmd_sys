"""Automated Model Approval Gate and Registration for Production Serving."""

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "Retailrocket-Hybrid-Recommender"


def evaluate_and_approve():
    logger.info("=" * 70)
    logger.info("RUNNING AUTOMATED MODEL APPROVAL QUALITY GATE")
    logger.info("=" * 70)

    metrics_path = Path("artifacts/metrics/model_comparison.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics at {metrics_path}")

    with open(metrics_path, "r") as f:
        benchmarks = json.load(f)

    baseline_metrics = benchmarks.get("Popularity Baseline", {})
    candidate_metrics = benchmarks.get("Hybrid Engine (ALS+Content+Pop)", {})
    als_metrics = benchmarks.get("Collaborative Filtering (ALS)", {})

    baseline_recall = baseline_metrics.get("recall@10", 0.0)
    candidate_recall = candidate_metrics.get("recall@10", 0.0)
    candidate_latency = candidate_metrics.get("avg_latency_ms", 999.0)
    candidate_coverage = candidate_metrics.get("coverage@10_pct", 0.0)

    # 1. Quality Gate Rules
    relative_lift = ((candidate_recall - baseline_recall) / (baseline_recall + 1e-8)) * 100.0

    checks = [
        {
            "criterion": "Recall@10 Superiority Over Baseline",
            "required": ">= 30% relative lift",
            "observed": f"{relative_lift:.2f}% relative lift ({candidate_recall:.5f} vs {baseline_recall:.5f})",
            "passed": relative_lift >= 30.0,
        },
        {
            "criterion": "Inference Latency Budget",
            "required": "< 80.0 ms per user request",
            "observed": f"{candidate_latency:.2f} ms",
            "passed": candidate_latency < 80.0,
        },
        {
            "criterion": "Catalog Exploration & Diversity",
            "required": "> 0.1% catalog coverage",
            "observed": f"{candidate_coverage:.3f}% coverage",
            "passed": candidate_coverage > 0.1,
        },
        {
            "criterion": "Artifact Completeness (ALS Weights)",
            "required": "artifacts/models/als_model.pkl exists",
            "observed": "Exists" if Path("artifacts/models/als_model.pkl").exists() else "Missing",
            "passed": Path("artifacts/models/als_model.pkl").exists(),
        },
        {
            "criterion": "Artifact Completeness (Content Weights)",
            "required": "artifacts/models/content_model.pkl exists",
            "observed": "Exists" if Path("artifacts/models/content_model.pkl").exists() else "Missing",
            "passed": Path("artifacts/models/content_model.pkl").exists(),
        },
        {
            "criterion": "Artifact Completeness (Index Mappings)",
            "required": "artifacts/mappings/train_mappings.pkl exists",
            "observed": "Exists" if Path("artifacts/mappings/train_mappings.pkl").exists() else "Missing",
            "passed": Path("artifacts/mappings/train_mappings.pkl").exists(),
        },
        {
            "criterion": "Artifact Completeness (Popularity Baseline)",
            "required": "data/processed/popularity_features.parquet exists",
            "observed": "Exists" if Path("data/processed/popularity_features.parquet").exists() else "Missing",
            "passed": Path("data/processed/popularity_features.parquet").exists(),
        },
    ]

    all_passed = all(c["passed"] for c in checks)

    logger.info("Quality Gate Assessment Results:")
    for c in checks:
        status_str = "[PASS]" if c["passed"] else "[FAIL]"
        logger.info("  %s %s: Observed=%s (Req: %s)", status_str, c["criterion"], c["observed"], c["required"])

    if not all_passed:
        logger.error("MODEL APPROVAL GATE FAILED: Model does not meet production standards.")
        return False

    logger.info("\nALL QUALITY CHECKS PASSED: Candidate Model Approved for Production Promotion.")

    # 2. Package Approved Artifacts for Serving
    approved_dir = Path("artifacts/approved")
    approved_dir.mkdir(parents=True, exist_ok=True)

    artifacts_to_copy = [
        ("artifacts/models/als_model.pkl", approved_dir / "als_model.pkl"),
        ("artifacts/models/content_model.pkl", approved_dir / "content_model.pkl"),
        ("data/processed/popularity_features.parquet", approved_dir / "popularity_features.parquet"),
        ("artifacts/mappings/train_mappings.pkl", approved_dir / "train_mappings.pkl"),
    ]

    for src, dst in artifacts_to_copy:
        shutil.copy2(src, dst)
        logger.info("Copied approved artifact: %s -> %s", src, dst)

    # 3. Write Approved Model Metadata
    approval_metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": "1.0.0",
        "approval_timestamp": datetime.now(timezone.utc).isoformat(),
        "approval_status": "APPROVED",
        "selected_architecture": "Hybrid (ALS + Content-Based TF-IDF + Popularity Baseline)",
        "weights": {
            "collaborative_weight": 0.5,
            "content_weight": 0.3,
            "popularity_weight": 0.2,
        },
        "performance_metrics": {
            "recall@10": candidate_recall,
            "precision@10": candidate_metrics.get("precision@10", 0.0),
            "ndcg@10": candidate_metrics.get("ndcg@10", 0.0),
            "map@10": candidate_metrics.get("map@10", 0.0),
            "coverage@10_pct": candidate_coverage,
            "relative_lift_over_baseline_pct": round(relative_lift, 2),
            "avg_latency_ms": candidate_latency,
        },
        "quality_gate_checks": checks,
    }

    meta_path = approved_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(approval_metadata, f, indent=2)
    logger.info("Saved approved model metadata to: %s", meta_path)

    # 4. Register in MLflow Model Registry
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient(tracking_uri=TRACKING_URI)

        # Retrieve Hybrid run ID
        summary_path = Path("artifacts/metrics/mlflow_experiment_summary.json")
        if summary_path.exists():
            with open(summary_path, "r") as f:
                exp_summary = json.load(f)
            hybrid_run = exp_summary.get("Hybrid-Engine-Balanced", {})
            run_id = hybrid_run.get("run_id")

            if run_id:
                # Create or get registered model
                try:
                    client.create_registered_model(
                        name=REGISTERED_MODEL_NAME,
                        description="Production Retailrocket Hybrid Recommendation Engine blending ALS, Content, and Popularity.",
                    )
                    logger.info("Created registered model in MLflow: %s", REGISTERED_MODEL_NAME)
                except Exception:
                    logger.info("Registered model %s already exists.", REGISTERED_MODEL_NAME)

                # Create model version
                model_version = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=f"runs:/{run_id}/model_package",
                    run_id=run_id,
                    description=f"Approved hybrid model with Recall@10={candidate_recall:.5f} and latency={candidate_latency:.2f}ms",
                )
                logger.info("Created Model Version %s in MLflow Model Registry", model_version.version)

                # Set alias/tag
                client.set_model_version_tag(
                    name=REGISTERED_MODEL_NAME,
                    version=model_version.version,
                    key="deployment_status",
                    value="approved_champion",
                )
    except Exception as e:
        logger.warning("MLflow Model Registry call noted non-critical warning: %s", e)

    logger.info("=" * 70)
    logger.info("MODEL PROMOTION COMPLETE: Production package is sealed in artifacts/approved/")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    evaluate_and_approve()
