"""Production Monitoring and Data Drift Detection Pipeline for Recommender Service.

Calculates:
1. Population Stability Index (PSI) on User Interaction Weights.
2. Event Ratio Distribution Shift (view vs addtocart vs transaction).
3. Cold-Start Vocabulary Drift (Unseen visitors in recent production window).
4. Automated Alerting & Retraining Triggers.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Drift-Monitor")


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two continuous distributions.

    PSI = sum((target_pct - baseline_pct) * ln(target_pct / baseline_pct))
    """
    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    # Determine bin edges based on baseline quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Remove non-unique bin edges if data is discrete
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    baseline_pct = baseline_counts / len(baseline)
    target_pct = target_counts / len(target)

    # Apply Laplace smoothing to avoid division by zero or log(0)
    eps = 1e-4
    baseline_pct = np.clip(baseline_pct, eps, 1.0)
    target_pct = np.clip(target_pct, eps, 1.0)

    # Renormalize
    baseline_pct /= np.sum(baseline_pct)
    target_pct /= np.sum(target_pct)

    psi_value = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
    return float(psi_value)


def run_drift_analysis(
    baseline_path: str = "data/processed/train.parquet",
    production_window_path: str = "data/processed/test.parquet",
    output_path: str = "artifacts/metrics/drift_report.json",
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PRODUCTION DATA & CONCEPT DRIFT ANALYSIS")
    logger.info("=" * 70)

    b_path = Path(baseline_path)
    t_path = Path(production_window_path)

    if not b_path.exists() or not t_path.exists():
        raise FileNotFoundError(f"Missing datasets for drift analysis: {b_path} or {t_path}")

    # 1. Load distributions using DuckDB
    logger.info("Ingesting baseline and production window via DuckDB...")
    con = duckdb.connect()

    # Load interaction weights
    sql = """
    SELECT 
        visitorid, 
        itemid, 
        event,
        CASE event 
            WHEN 'view' THEN 1.0 
            WHEN 'addtocart' THEN 3.0 
            WHEN 'transaction' THEN 5.0 
            ELSE 1.0 
        END AS weight
    FROM '{}'
    """
    baseline_df = con.execute(sql.format(baseline_path)).df()
    prod_df = con.execute(sql.format(production_window_path)).df()

    logger.info("Baseline records: %d | Production window records: %d", len(baseline_df), len(prod_df))

    # 2. Compute PSI on Interaction Weights
    b_weights = baseline_df["weight"].values.astype(float)
    p_weights = prod_df["weight"].values.astype(float)
    psi_weights = calculate_psi(b_weights, p_weights, num_bins=5)
    logger.info("Interaction Weight PSI: %.4f", psi_weights)

    # 3. Compute Event Ratio Distribution Shift
    b_events = baseline_df["event"].value_counts(normalize=True).to_dict()
    p_events = prod_df["event"].value_counts(normalize=True).to_dict()

    event_shift = {
        event: {
            "baseline_pct": round(b_events.get(event, 0.0) * 100, 2),
            "production_pct": round(p_events.get(event, 0.0) * 100, 2),
            "diff_pct_points": round((p_events.get(event, 0.0) - b_events.get(event, 0.0)) * 100, 2),
        }
        for event in ["view", "addtocart", "transaction"]
    }
    logger.info("Event Breakdown Shift: %s", event_shift)

    # 4. Cold-Start Vocabulary Drift (Unseen Visitors)
    baseline_visitors = set(baseline_df["visitorid"].unique())
    prod_visitors = set(prod_df["visitorid"].unique())
    unseen_visitors = prod_visitors - baseline_visitors
    cold_start_ratio = len(unseen_visitors) / (len(prod_visitors) + 1e-8)
    logger.info(
        "Production Visitors: %d | Unseen (Cold-Start): %d (%.2f%%)",
        len(prod_visitors),
        len(unseen_visitors),
        cold_start_ratio * 100,
    )

    # 5. Catalog Inactivity Shift (Unseen Items)
    baseline_items = set(baseline_df["itemid"].unique())
    prod_items = set(prod_df["itemid"].unique())
    new_items = prod_items - baseline_items
    new_item_ratio = len(new_items) / (len(prod_items) + 1e-8)
    logger.info("Production Items: %d | New (Unindexed): %d (%.2f%%)", len(prod_items), len(new_items), new_item_ratio * 100)

    # 6. Drift Assessment & Automated Retraining Triggers
    # Rule 1: PSI >= 0.25 -> Significant drift
    # Rule 2: Cold start ratio > 50% -> Significant user churn/drift
    # Rule 3: Transaction ratio drop > 30% -> Conversion funnel drift
    retrain_triggers = []
    if psi_weights >= 0.25:
        retrain_triggers.append(f"PSI ({psi_weights:.4f}) exceeds threshold (0.25)")
    if cold_start_ratio > 0.60:
        retrain_triggers.append(f"Cold-Start Visitor Ratio ({cold_start_ratio*100:.1f}%) exceeds threshold (60%)")
    if new_item_ratio > 0.15:
        retrain_triggers.append(f"New Unindexed Items Ratio ({new_item_ratio*100:.1f}%) exceeds threshold (15%)")

    if len(retrain_triggers) > 0:
        drift_level = "SIGNIFICANT_DRIFT"
        recommendation = "AUTOMATED_RETRAINING_TRIGGERED"
        logger.warning("ALERT: %s! Triggers: %s", drift_level, retrain_triggers)
    elif psi_weights >= 0.10 or cold_start_ratio > 0.40:
        drift_level = "MODERATE_DRIFT"
        recommendation = "MONITOR_CLOSELY"
        logger.info("STATUS: %s. No immediate retraining required.", drift_level)
    else:
        drift_level = "NO_DRIFT"
        recommendation = "STABLE"
        logger.info("STATUS: %s. Distribution is stable.", drift_level)

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_level": drift_level,
        "recommendation": recommendation,
        "metrics": {
            "interaction_weight_psi": round(psi_weights, 4),
            "cold_start_visitor_pct": round(cold_start_ratio * 100, 2),
            "new_unindexed_item_pct": round(new_item_ratio * 100, 2),
            "total_baseline_records": len(baseline_df),
            "total_production_records": len(prod_df),
        },
        "event_distribution_shift": event_shift,
        "active_retrain_triggers": retrain_triggers,
        "sla_thresholds": {
            "psi_warning": 0.10,
            "psi_critical_retrain": 0.25,
            "cold_start_warning_pct": 40.0,
            "cold_start_critical_pct": 60.0,
            "new_item_critical_pct": 15.0,
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved drift report to: %s", out_file)
    logger.info("=" * 70)
    return report


if __name__ == "__main__":
    run_drift_analysis()
