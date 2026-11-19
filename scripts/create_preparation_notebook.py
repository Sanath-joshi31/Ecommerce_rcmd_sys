"""Generate notebooks/02_data_preparation.ipynb."""

import json
from pathlib import Path


def make_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 02 - Data Preparation, Cleaning & Chronological Splitting\n",
                    "## E-Commerce Product Recommendation System (Retailrocket Dataset)\n",
                    "\n",
                    "**Objective:**\n",
                    "Transform the raw event logs into reproducible, high-quality analytical datasets ready for machine learning:\n",
                    "1. Validate essential entity IDs and filter invalid records.\n",
                    "2. Standardize Unix millisecond timestamps into UTC datetimes.\n",
                    "3. Investigate duplicate events (accidental double-clicks vs. legitimate repeated views).\n",
                    "4. Assign implicit behavioral confidence weights (`view=1.0`, `addtocart=3.0`, `transaction=5.0`).\n",
                    "5. Aggregate interactions at the `(visitorid, itemid)` level.\n",
                    "6. Execute strict chronological train/val/test splits (70% / 15% / 15%) guaranteeing zero temporal leakage.\n",
                    "7. Export optimized Parquet files into `data/processed/`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Add project root to sys.path\n",
                    "ROOT_DIR = Path('..').resolve()\n",
                    "if str(ROOT_DIR) not in sys.path:\n",
                    "    sys.path.insert(0, str(ROOT_DIR))\n",
                    "\n",
                    "from src.data.load_data import load_events\n",
                    "from src.data.preprocess import clean_events, split_chronological\n",
                    "from src.features.build_features import aggregate_user_item_interactions, build_popularity_features\n",
                    "\n",
                    "DATA_DIR = ROOT_DIR / 'dataset'\n",
                    "OUT_DIR = ROOT_DIR / 'data' / 'processed'\n",
                    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "print(f'Ready to process from: {DATA_DIR} to {OUT_DIR}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Ingest Raw Events"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "raw_events = load_events(DATA_DIR)\n",
                    "print(f'Ingested Raw Events: {len(raw_events):,}')\n",
                    "display(raw_events.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Execute Data Cleaning & Standardization\n",
                    "We apply our reproducible cleaning rules via `clean_events()`:\n",
                    "- Check essential non-nullity (`visitorid`, `itemid`, `timestamp`, `event`).\n",
                    "- Deduplicate identical rapid clicks at the exact same millisecond.\n",
                    "- Cast timestamps to UTC datetimes and extract temporal features."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "clean_df, cleaning_meta = clean_events(raw_events)\n",
                    "print('\\n--- CLEANING SUMMARY ---')\n",
                    "for k, v in cleaning_meta.items():\n",
                    "    print(f'{k}: {v}')\n",
                    "\n",
                    "# Save cleaned events\n",
                    "clean_path = OUT_DIR / 'events_clean.parquet'\n",
                    "clean_df.to_parquet(clean_path, index=False)\n",
                    "print(f'Saved cleaned events to: {clean_path}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Aggregate User-Product Interactions & Implicit Feedback Scoring\n",
                    "We calculate behavioral weights:\n",
                    "- View: $w = 1.0$\n",
                    "- Add-to-cart: $w = 3.0$\n",
                    "- Transaction: $w = 5.0$\n",
                    "\n",
                    "Total weight per `(visitorid, itemid)` pair represents our implicit interaction confidence."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "user_item_df = aggregate_user_item_interactions(clean_df)\n",
                    "print(f'Aggregated User-Item Interactions: {len(user_item_df):,}')\n",
                    "display(user_item_df.head(10))\n",
                    "\n",
                    "user_item_path = OUT_DIR / 'user_item_interactions.parquet'\n",
                    "user_item_df.to_parquet(user_item_path, index=False)\n",
                    "print(f'Saved user-item interactions to: {user_item_path}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Chronological Splitting (70% Train, 15% Validation, 15% Test)\n",
                    "To guarantee zero future information leakage, we enforce a strict temporal boundary split:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_df, val_df, test_df, split_meta = split_chronological(clean_df, 0.70, 0.15, 0.15)\n",
                    "print('\\n--- CHRONOLOGICAL SPLIT SUMMARY ---')\n",
                    "for k, v in split_meta.items():\n",
                    "    print(f'{k}: {v}')\n",
                    "\n",
                    "# Verify temporal boundaries\n",
                    "assert train_df['timestamp'].max() < val_df['timestamp'].min(), 'Leakage detected!'\n",
                    "assert val_df['timestamp'].max() < test_df['timestamp'].min(), 'Leakage detected!'\n",
                    "print('\\n[PASS] Zero data leakage assertion verified successfully.')\n",
                    "\n",
                    "# Export splits to Parquet\n",
                    "train_df.to_parquet(OUT_DIR / 'train.parquet', index=False)\n",
                    "val_df.to_parquet(OUT_DIR / 'val.parquet', index=False)\n",
                    "test_df.to_parquet(OUT_DIR / 'test.parquet', index=False)\n",
                    "print(f'Splits successfully written to {OUT_DIR}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Popularity Features (Strictly on Train Split)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pop_df = build_popularity_features(train_df)\n",
                    "print('Top 10 Most Popular Products in Training Period:')\n",
                    "display(pop_df.head(10))\n",
                    "pop_df.to_parquet(OUT_DIR / 'popularity_features.parquet', index=False)\n",
                    "print(f'Saved popularity features to: {OUT_DIR / \"popularity_features.parquet\"}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    p = Path("notebooks/02_data_preparation.ipynb")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {p}")


if __name__ == "__main__":
    make_notebook()
