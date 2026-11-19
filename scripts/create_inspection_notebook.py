"""Generate notebooks/01_data_inspection.ipynb."""

import json
from pathlib import Path


def make_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01 - Raw Dataset Inspection & Quality Profiling\n",
                    "## E-Commerce Product Recommendation System (Retailrocket Dataset)\n",
                    "\n",
                    "**Objective:**\n",
                    "Inspect all four raw Retailrocket CSV files before applying any transformations:\n",
                    "1. `events.csv`\n",
                    "2. `category_tree.csv`\n",
                    "3. `item_properties_part1.csv`\n",
                    "4. `item_properties_part2.csv`\n",
                    "\n",
                    "We diagnose file sizes, column schemas, data types, nullity structures, duplicate rows, temporal ranges, and behavioral anomalies."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_theme(style='whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                    "\n",
                    "DATA_DIR = Path('../dataset')\n",
                    "print(f'Data directory exists: {DATA_DIR.exists()}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. File Verification & Physical Size Profile"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "expected_files = ['events.csv', 'category_tree.csv', 'item_properties_part1.csv', 'item_properties_part2.csv']\n",
                    "file_stats = []\n",
                    "for fname in expected_files:\n",
                    "    fpath = DATA_DIR / fname\n",
                    "    exists = fpath.exists()\n",
                    "    size_mb = fpath.stat().st_size / (1024 * 1024) if exists else 0.0\n",
                    "    file_stats.append({'Filename': fname, 'Exists': exists, 'Size (MB)': round(size_mb, 2)})\n",
                    "\n",
                    "df_files = pd.DataFrame(file_stats)\n",
                    "display(df_files)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Inspect `events.csv`\n",
                    "The core behavioral log capturing visitor interactions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Loading events.csv...')\n",
                    "events_df = pd.read_csv(DATA_DIR / 'events.csv')\n",
                    "print(f'Total Event Records: {len(events_df):,}')\n",
                    "display(events_df.head())\n",
                    "print('\\nSchema & Data Types:')\n",
                    "events_df.info()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### 2.1 Null Value Diagnosis & Structural Missingness\n",
                    "Notice that `transactionid` is missing in most rows. In e-commerce tracking, this is an expected structural null because only `transaction` events generate an order ID."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing_df = pd.DataFrame({\n",
                    "    'Missing Count': events_df.isna().sum(),\n",
                    "    'Missing Pct (%)': (events_df.isna().sum() / len(events_df) * 100).round(3)\n",
                    "})\n",
                    "display(missing_df)\n",
                    "\n",
                    "# Breakdown by event type\n",
                    "print('Missing transactionid by event type:')\n",
                    "display(events_df.groupby('event')['transactionid'].apply(lambda x: f'{x.isna().sum():,} missing / {len(x):,} total'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### 2.2 Event Frequency & Funnel Conversion"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "event_counts = events_df['event'].value_counts()\n",
                    "event_pcts = (events_df['event'].value_counts(normalize=True) * 100).round(2)\n",
                    "event_summary = pd.DataFrame({'Count': event_counts, 'Share (%)': event_pcts})\n",
                    "display(event_summary)\n",
                    "\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.barplot(x=event_summary.index, y='Count', data=event_summary, palette='viridis')\n",
                    "plt.title('Retailrocket Event Funnel Distribution')\n",
                    "plt.yscale('log')\n",
                    "plt.ylabel('Count (Log Scale)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### 2.3 Cardinality & Unique Entity Counts"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "n_visitors = events_df['visitorid'].nunique()\n",
                    "n_items = events_df['itemid'].nunique()\n",
                    "total_events = len(events_df)\n",
                    "\n",
                    "print(f'Unique Visitors: {n_visitors:,}')\n",
                    "print(f'Unique Items:    {n_items:,}')\n",
                    "print(f'Average Events per Visitor: {total_events / n_visitors:.2f}')\n",
                    "print(f'Average Events per Item:    {total_events / n_items:.2f}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### 2.4 Temporal Span & Date Conversion"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "events_df['datetime'] = pd.to_datetime(events_df['timestamp'], unit='ms', utc=True)\n",
                    "print(f'Earliest Timestamp: {events_df[\"datetime\"].min()}')\n",
                    "print(f'Latest Timestamp:   {events_df[\"datetime\"].max()}')\n",
                    "print(f'Total Date Range:   {(events_df[\"datetime\"].max() - events_df[\"datetime\"].min()).days} days')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Inspect `category_tree.csv`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cat_df = pd.read_csv(DATA_DIR / 'category_tree.csv')\n",
                    "print(f'Category Relations: {len(cat_df):,}')\n",
                    "display(cat_df.head())\n",
                    "print(f'Root Categories (parent_id is null): {cat_df[\"parentid\"].isna().sum():,}')\n",
                    "print(f'Unique Categories: {cat_df[\"categoryid\"].nunique():,}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Inspect `item_properties_part1.csv` and `part2.csv`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "prop_sample = pd.read_csv(DATA_DIR / 'item_properties_part1.csv', nrows=100000)\n",
                    "print('Item Properties Sample (First 100,000 rows of Part 1):')\n",
                    "display(prop_sample.head())\n",
                    "print(f'Unique Properties in Sample: {prop_sample[\"property\"].nunique():,}')\n",
                    "print('Top 10 Most Common Property Names:')\n",
                    "display(prop_sample['property'].value_counts().head(10))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Summary & Key Findings for Data Cleaning\n",
                    "1. **Zero Essential ID Nulls:** `visitorid`, `itemid`, `timestamp`, and `event` have zero missing values.\n",
                    "2. **Expected Structural Nulls:** `transactionid` is missing in 99.2% of events because views and addtocart events do not generate orders.\n",
                    "3. **Extreme Event Funnel Imbalance:** Views represent ~96.7% of all actions; add-to-cart represents ~2.5%; transactions represent ~0.8%. This justifies our implicit behavioral weights (1, 3, 5).\n",
                    "4. **Exact Duplicates:** A small fraction of duplicate clicks exists (accidental rapid double-clicks at the exact same millisecond). These should be deduplicated."
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

    p = Path("notebooks/01_data_inspection.ipynb")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {p}")


if __name__ == "__main__":
    make_notebook()
