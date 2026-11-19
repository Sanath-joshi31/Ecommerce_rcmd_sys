# Phase 3: Data Preparation, Quality Audit & Interactive EDA Documentation

## 1. Executive Summary & Objective
In e-commerce recommendation systems, data quality and temporal integrity determine the ceiling of model performance. Real-world behavioral event streams are noisy, sparse, and temporally non-stationary. Training a model on corrupted IDs, uncalibrated behavioral weights, or future data leaks produces deceptive offline metrics that completely collapse in production.

This document details the complete **Phase 3 implementation**, transforming the raw 987 MB Retailrocket dataset into certified, reproducible, and leakage-free analytical datasets for Phase 4 machine learning modeling.

---

## 2. Raw Dataset Profiling & Verification

We verified and ingested all four core Retailrocket CSV files from `dataset/`:

| Dataset File | File Size (MB) | Total Records | Key Columns | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`events.csv`** | 89.87 MB | 2,756,101 | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` | Core behavioral interaction logs (views, carts, transactions). |
| **`category_tree.csv`** | 0.01 MB | 1,669 | `categoryid`, `parentid` | Hierarchical taxonomy mapping products into categories. |
| **`item_properties_part1.csv`** | 461.88 MB | 10,999,999 | `timestamp`, `itemid`, `property`, `value` | Temporal snapshot 1 of anonymized product attributes. |
| **`item_properties_part2.csv`** | 389.99 MB | 9,275,903 | `timestamp`, `itemid`, `property`, `value` | Temporal snapshot 2 of anonymized product attributes. |

### 2.1 Inherent Funnel Imbalance
Analysis of raw event frequencies reveals the characteristic e-commerce conversion funnel:
- **`view`:** 2,664,312 records ($96.67\%$)
- **`addtocart`:** 69,332 records ($2.52\%$)
- **`transaction`:** 22,457 records ($0.81\%$)

This extreme imbalance confirms why standard explicit rating algorithms (like 1–5 star matrix factorization) fail, and strongly justifies our implicit feedback confidence weighting.

---

## 3. Data Cleaning Decisions & Audit Trail

We enforce strict data engineering rules via `src/data/preprocess.py`:

```
Raw Events Ingested (2,756,101)
  │
  ├── 1. Essential Key Validation (timestamp, visitorid, itemid, event)
  │      └── Dropped: 0 records (100% complete)
  │
  ├── 2. Event Type Validation ({view, addtocart, transaction})
  │      └── Dropped: 0 records (100% valid)
  │
  ├── 3. Exact Duplicate Click Deduplication
  │      └── Removed: 460 records (Accidental client double-clicks at identical millisecond)
  │
  ├── 4. Legitimate Repeated Interactions
  │      └── RETAINED: Repeated views across distinct sessions/timestamps (True user intent)
  │
  └── 5. UTC Datetime Conversion & Sorting
         └── Result: 2,755,641 Clean Records (99.9833% Retention Rate)
```

### Critical Domain Cleaning Nuances:
1. **Handling Missing `transactionid`:**
   - In `events.csv`, 2,733,644 rows have missing `transactionid`. 
   - **Audit finding:** Exactly $100\%$ of these nulls belong to `view` and `addtocart` actions. Exactly $0\%$ of `transaction` events are missing a `transactionid`. This is an expected structural property of web event tracking, not data corruption.
2. **Preserving Repeated Views:**
   - A visitor viewing an expensive laptop 4 times across two days is expressing strong purchase intent. Automatically deduplicating distinct interaction timestamps would destroy critical behavioral signal. We only remove exact duplicate rows occurring at the identical millisecond timestamp.
3. **Preserving Low-Activity Visitors:**
   - Single-interaction visitors represent real cold-start shoppers. They are preserved in the cleaned interaction store so that our cold-start popularity fallback can be rigorously evaluated.

---

## 4. Feature Engineering & Implicit Confidence Scoring

### 4.1 Behavioral Confidence Weighting
We formulate the interaction confidence score $c_{u,i}$ according to the calibrated purchase intent:
$$\text{Weight}(e) = \begin{cases} 1.0 & \text{if } e = \text{view} \\ 3.0 & \text{if } e = \text{addtocart} \\ 5.0 & \text{if } e = \text{transaction} \end{cases}$$

### 4.2 User-Item Interaction Aggregation
We aggregate interactions at the `(visitorid, itemid)` grain:
$$\text{Confidence}(u, i) = \sum_{e \in E_{u,i}} \text{Weight}(e)$$
- **Total Unique User-Product Pairs:** 2,145,179
- Each aggregated record stores: `total_weight`, `event_count`, `view_count`, `cart_count`, `purchase_count`, and `last_timestamp`.

### 4.3 Training Matrix Sparsity
From the training split (earliest 70% of data):
- **Unique Visitors ($|U_{\text{train}}|$):** 978,906
- **Unique Products ($|I_{\text{train}}|$):** 200,974
- **Observed Interactions:** 1,496,538
- **Theoretical Matrix Size:** $978,906 \times 200,974 = 1.967 \times 10^{11} \text{ cells}$
- **Matrix Sparsity:** **$99.9992\%$**
- **Matrix Density:** **$0.000761\%$**

By utilizing Compressed Sparse Row (`scipy.sparse.csr_matrix`) structures, we compress this matrix from ~786 Gigabytes down to **< 35 Megabytes** of memory.

---

## 5. Chronological Train, Validation, and Test Splitting

To eliminate future information leakage, we strictly partition interactions temporally using timestamp quantiles:

```
[================= TRAIN (70%) =================] [==== VAL (15%) ====] [==== TEST (15%) ====]
May 03, 2015 ──────────────────> Aug 02, 2015 ──> Aug 25, 2015 ──────> Sep 18, 2015
(1,928,948 events)                                (413,346 events)       (413,347 events)
```

### Verified Temporal Boundaries:
- **Train Split:** `2015-05-03 03:00:04 UTC` to `2015-08-02 23:29:02 UTC`
- **Validation Split:** `2015-08-02 23:29:03 UTC` to `2015-08-25 22:30:59 UTC`
- **Test Split:** `2015-08-25 22:31:01 UTC` to `2015-09-18 02:59:47 UTC`

$$\max(\text{Train Timestamp}) < \min(\text{Val Timestamp}) < \min(\text{Test Timestamp})$$
**Result:** Guaranteed **zero temporal lookahead leakage**. Popularity baselines and ALS latent factors are trained strictly on data prior to August 2, 2015.

---

## 6. Interactive EDA Dashboard Architecture (`03_interactive_eda.ipynb`)

We built an interactive, production-grade Exploratory Data Analysis dashboard in [notebooks/03_interactive_eda.ipynb](file:///d:/projects%20&%20datasets/Ecommerce_rcmd_sys/notebooks/03_interactive_eda.ipynb) featuring Plotly and ipywidgets:

```
Interactive EDA Dashboard Structure:
├── SECTION A: Dataset Overview
│   ├── Viz 1: KPI Indicator Cards (Total Events, Unique Visitors, Items, Transactions)
│   ├── Viz 2: Missingness Heatmap (Red/Blue nullity co-occurrence across columns)
│   └── Viz 3: Before-vs-After Cleaning Audit (Original vs Retained vs Dropped)
├── SECTION B: Customer Behavior Analysis
│   ├── Viz 4: Funnel Event Distribution (Log scale bar chart)
│   ├── Viz 5: Interactive Daily Activity Timeline (Unified hover & date zoom)
│   ├── Viz 6: Activity per Visitor Histogram (Interactive Log/Linear scale toggle)
│   └── Viz 7: Top-N Interacted Products (Interactive Event dropdown & Top-N slider)
├── SECTION C: Recommendation System Readiness
│   ├── Viz 8: Sampled User-Product Matrix Heatmap (Top 30 Users x Top 30 Items)
│   ├── Viz 9: Matrix Sparsity vs Density Donut Chart & Metrics Table
│   └── Viz 10: Category Hierarchy & Subcategory Distribution
└── SECTION D: Cleaned Data Quality Report
    └── PASS / WARNING / FAIL automated certification table
```

---

## 7. Phase 3 Deliverables Catalog

All processed artifacts are persisted in `data/processed/` and `artifacts/`:

| Artifact Name | Path | Format | Rows / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Cleaned Events** | `data/processed/events_clean.parquet` | Parquet | 2,755,641 rows | Cleaned, typed, and timestamped behavioral logs. |
| **User-Item Interactions** | `data/processed/user_item_interactions.parquet` | Parquet | 2,145,179 rows | Aggregated $(u, i)$ confidence weights and counts. |
| **Train Split** | `data/processed/train.parquet` | Parquet | 1,928,948 rows | Earliest 70% of chronological interactions. |
| **Validation Split** | `data/processed/val.parquet` | Parquet | 413,346 rows | Subsequent 15% for hyperparameter tuning. |
| **Test Split** | `data/processed/test.parquet` | Parquet | 413,347 rows | Final 15% held-out evaluation benchmark. |
| **Popularity Features** | `data/processed/popularity_features.parquet` | Parquet | 200,974 items | Precomputed global popularity scores and ranks. |
| **Index Mappings** | `artifacts/mappings/train_mappings.pkl` | Pickle | 978,906 users, 200,974 items | Synchronized bidirectional ID-to-index mappings. |
| **Data Quality Report** | `artifacts/metrics/data_quality_report.json` | JSON | 72 lines | Machine-readable execution and audit metrics. |
| **Inspection Notebook** | `notebooks/01_data_inspection.ipynb` | Jupyter | 5 sections | Profiling raw CSV files and nullity. |
| **Preparation Notebook** | `notebooks/02_data_preparation.ipynb` | Jupyter | 5 sections | Cleaning, weighting, and splitting execution. |
| **Interactive Dashboard** | `notebooks/03_interactive_eda.ipynb` | Jupyter | 4 sections (10 viz) | Full Plotly & ipywidgets interactive dashboard. |

---

## 8. Automated Certification Gate

| Validation Check | Observed Result | Benchmark Standard | Status |
| :--- | :--- | :--- | :--- |
| **Essential Key Non-Nullity** | $0$ missing visitor IDs, item IDs, timestamps | $0$ nulls allowed | **PASS** |
| **Event Type Permissibility** | $100\%$ within `{view, addtocart, transaction}` | $100\%$ valid | **PASS** |
| **Duplicate Click Deduplication** | $460$ exact millisecond duplicates removed | Deduplicated | **PASS** |
| **Repeated Views Preserved** | Multi-view sessions retained | Must preserve intent | **PASS** |
| **Transaction ID Missingness** | $0$ nulls on transactions; $100\%$ on non-purchases | Structural nullity | **PASS** |
| **Chronological Boundary Isolation** | $\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$ | Zero future leakage | **PASS** |
| **Sparse Matrix Compressibility** | Density: $0.000761\%$, Sparsity: $99.9992\%$ | $> 99.5\%$ sparsity | **PASS** |
| **Execution Performance** | End-to-end pipeline executes in $13.54\text{ seconds}$ | $< 60\text{ seconds}$ | **PASS** |

**Conclusion:** The prepared dataset is certified healthy, leakage-free, and fully verified for Phase 4 Machine Learning Model Development.
