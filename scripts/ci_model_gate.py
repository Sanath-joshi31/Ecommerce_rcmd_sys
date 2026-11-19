"""Automated Model Quality Gate for CI/CD Deployment Pipeline.

This script executes in the GitHub Actions CI pipeline to ensure that
candidate models satisfy all production performance criteria, safety bounds,
and artifact completeness requirements before allowing Docker build and cloud deployment.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CI-Quality-Gate")


def evaluate_ci_gate(metadata_path: str = "artifacts/approved/model_metadata.json") -> bool:
    logger.info("=" * 70)
    logger.info("EXECUTING AUTOMATED CI/CD MODEL QUALITY GATE")
    logger.info("=" * 70)

    meta_file = Path(metadata_path)
    if not meta_file.exists():
        logger.error("FATAL: Approved model metadata file '%s' not found!", metadata_path)
        return False

    with open(meta_file, "r") as f:
        meta: Dict[str, Any] = json.load(f)

    metrics = meta.get("performance_metrics", {})
    lift_pct = metrics.get("relative_lift_over_baseline_pct", 0.0)
    avg_latency = metrics.get("avg_latency_ms", 999.0)
    coverage_pct = metrics.get("coverage@10_pct", 0.0)
    recall_10 = metrics.get("recall@10", 0.0)
    ndcg_10 = metrics.get("ndcg@10", 0.0)
    status = meta.get("approval_status", "UNKNOWN")

    pop_artifact = Path("artifacts/approved/popularity_features.parquet")

    checks: List[Dict[str, Any]] = [
        {
            "id": "CHK-01",
            "name": "Model Promotion Status",
            "required": "APPROVED",
            "observed": status,
            "passed": status == "APPROVED",
        },
        {
            "id": "CHK-02",
            "name": "Recall@10 Superiority Lift",
            "required": ">= 30.0% relative lift",
            "observed": f"{lift_pct:.2f}% lift",
            "passed": lift_pct >= 30.0,
        },
        {
            "id": "CHK-03",
            "name": "Inference Latency SLA",
            "required": "< 80.0 ms budget",
            "observed": f"{avg_latency:.2f} ms",
            "passed": avg_latency < 80.0,
        },
        {
            "id": "CHK-04",
            "name": "Catalog Diversity & Exploration",
            "required": "> 0.1% catalog coverage",
            "observed": f"{coverage_pct:.3f}% coverage",
            "passed": coverage_pct > 0.1,
        },
        {
            "id": "CHK-05",
            "name": "Ranking Metric Sanity Bounds",
            "required": "0.0 < Recall, NDCG <= 1.0",
            "observed": f"Recall={recall_10:.4f}, NDCG={ndcg_10:.4f}",
            "passed": (0.0 < recall_10 <= 1.0) and (0.0 < ndcg_10 <= 1.0),
        },
        {
            "id": "CHK-06",
            "name": "Popularity Artifact Completeness",
            "required": "popularity_features.parquet exists",
            "observed": "Present" if pop_artifact.exists() else "Missing",
            "passed": pop_artifact.exists(),
        },
    ]

    all_passed = all(c["passed"] for c in checks)

    logger.info("Quality Assessment Summary:")
    for c in checks:
        icon = "[PASS]" if c["passed"] else "[FAIL]"
        logger.info("  %s %s: Observed=%s | Required=%s", icon, c["name"], c["observed"], c["required"])

    # If running inside GitHub Actions, emit Markdown step summary table
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as sf:
                sf.write("## 🎯 Automated Model Quality Gate Report\n\n")
                sf.write(f"**Model Name:** `{meta.get('model_name')}` | **Version:** `{meta.get('model_version')}`\n\n")
                sf.write(f"**Gate Outcome:** {'✅ **PASSED** - Safe for Deployment' if all_passed else '❌ **FAILED** - Build Blocked'}\n\n")
                sf.write("| Gate ID | Quality Criterion | Observed Value | Production Threshold | Status |\n")
                sf.write("| :---: | :--- | :--- | :--- | :---: |\n")
                for c in checks:
                    status_badge = "✅ PASS" if c["passed"] else "❌ FAIL"
                    sf.write(f"| {c['id']} | {c['name']} | `{c['observed']}` | `{c['required']}` | {status_badge} |\n")
                sf.write("\n---\n")
        except Exception as e:
            logger.warning("Could not write GITHUB_STEP_SUMMARY: %s", e)

    if not all_passed:
        logger.error("CI GATE REJECTED: Model violates one or more production SLA criteria.")
        return False

    logger.info("=" * 70)
    logger.info("CI QUALITY GATE PASSED: Proceeding to Container Build & Deployment.")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = evaluate_ci_gate()
    sys.exit(0 if success else 1)
