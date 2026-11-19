"""Data loading utilities for Retailrocket dataset using DuckDB and Pandas."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FILES = [
    "events.csv",
    "category_tree.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
]


def verify_dataset_files(data_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Verify that all expected dataset files exist and profile their size.

    Args:
        data_dir: Path to directory containing raw CSV files.

    Returns:
        Dict mapping filename to file metadata (exists, size_mb, path).

    Raises:
        FileNotFoundError: If any expected file is missing.
    """
    path = Path(data_dir)
    report = {}
    missing = []

    for fname in EXPECTED_FILES:
        fpath = path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            report[fname] = {
                "exists": True,
                "size_mb": round(size_mb, 2),
                "path": str(fpath.resolve()),
            }
            logger.info("Found %s (%.2f MB)", fname, size_mb)
        else:
            report[fname] = {"exists": False, "size_mb": 0.0, "path": str(fpath.resolve())}
            missing.append(fname)
            logger.error("Missing expected dataset file: %s", fname)

    if missing:
        raise FileNotFoundError(f"Missing required dataset files in {data_dir}: {missing}")

    return report


def load_events(data_dir: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load events.csv with DuckDB if available, otherwise Pandas.

    Args:
        data_dir: Directory containing events.csv.
        nrows: Optional row limit for sampling.

    Returns:
        Pandas DataFrame containing raw events.
    """
    fpath = Path(data_dir) / "events.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Events file not found at: {fpath}")

    logger.info("Loading events from %s (nrows=%s)...", fpath, nrows)
    try:
        import duckdb
        query = f"SELECT * FROM read_csv_auto('{fpath.as_posix()}')"
        if nrows:
            query += f" LIMIT {nrows}"
        df = duckdb.query(query).to_df()
        logger.info("Loaded %d event records via DuckDB", len(df))
    except (ImportError, Exception) as e:
        logger.warning("DuckDB load failed or unavailable (%s). Falling back to Pandas.", e)
        df = pd.read_csv(fpath, nrows=nrows)
        logger.info("Loaded %d event records via Pandas", len(df))

    return df


def load_category_tree(data_dir: str | Path) -> pd.DataFrame:
    """Load category_tree.csv.

    Args:
        data_dir: Directory containing category_tree.csv.

    Returns:
        DataFrame containing category_id and parent_id mappings.
    """
    fpath = Path(data_dir) / "category_tree.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Category tree file not found at: {fpath}")

    logger.info("Loading category tree from %s...", fpath)
    df = pd.read_csv(fpath)
    logger.info("Loaded %d category relations", len(df))
    return df


def load_item_properties(
    data_dir: str | Path, nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load and concatenate item_properties_part1 and part2.

    Args:
        data_dir: Directory containing item_properties files.
        nrows: Optional limit per part.

    Returns:
        Concatenated DataFrame of item properties.
    """
    p1 = Path(data_dir) / "item_properties_part1.csv"
    p2 = Path(data_dir) / "item_properties_part2.csv"

    if not p1.exists() or not p2.exists():
        raise FileNotFoundError("Missing one or both item_properties files.")

    logger.info("Loading item properties parts 1 and 2...")
    df1 = pd.read_csv(p1, nrows=nrows)
    df2 = pd.read_csv(p2, nrows=nrows)
    combined = pd.concat([df1, df2], ignore_index=True)
    logger.info("Loaded %d combined item property records", len(combined))
    return combined
