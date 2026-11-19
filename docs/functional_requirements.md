# Functional Requirements Specification (FRS)

## 1. Overview
This document specifies the functional requirements for the **E-Commerce Product Recommendation System**. Functional requirements define the specific behaviors, computations, data flows, and capabilities that the system must deliver across data preparation, machine learning modeling, artifact management, and REST API serving.

---

## 2. Functional Requirements Breakdown

### Module 1: Data Ingestion & Preparation (DP)
- **FR-DP-01: Raw Data Ingestion & Integrity Verification**
  - The system shall ingest the four core Retailrocket files: `events.csv`, `category_tree.csv`, `item_properties_part1.csv`, and `item_properties_part2.csv`.
  - The system shall verify row counts, column schemas, data types, and check for missing critical identifiers (`visitorid`, `itemid`, `timestamp`, `event`).
- **FR-DP-02: Timestamp Standardization**
  - The system shall convert Unix epoch timestamps (in milliseconds) into standard UTC `datetime` objects and extract temporal attributes (date, hour, day-of-week).
- **FR-DP-03: Event Signal Validation & Weighting**
  - The system shall validate that the `event` column contains only permissible values (`view`, `addtocart`, `transaction`).
  - The system shall compute an implicit interaction confidence score using the documented weighting schema:
    $$\text{Interaction Weight} = 1.0 \times \mathbb{I}_{\text{view}} + 3.0 \times \mathbb{I}_{\text{addtocart}} + 5.0 \times \mathbb{I}_{\text{transaction}}$$
- **FR-DP-04: User-Product Interaction Aggregation**
  - The system shall aggregate interactions at the `(visitorid, itemid)` level, producing:
    - Total interaction count.
    - Weighted interaction score sum.
    - Last interaction timestamp.
- **FR-DP-05: Chronological Train-Validation-Test Splitting**
  - The system shall enforce a temporal boundary split (e.g., 70% Train, 15% Validation, 15% Test) based on timestamps.
  - No future interaction data shall be present in the training set (zero temporal leakage).

---

### Module 2: Model Engines (ME)

- **FR-ME-01: Popularity-Based Baseline Engine (Global & Cold-Start)**
  - The system shall aggregate item popularity exclusively using the training set.
  - The popularity score shall be computed as:
    $$\text{Popularity Score}(i) = \sum_{u \in U_{\text{train}}} \text{Interaction Weight}(u, i)$$
  - The engine shall produce top-$N$ ranked items globally to serve as a baseline and as the primary cold-start fallback.

- **FR-ME-02: Collaborative Filtering via Implicit ALS Engine**
  - The system shall build a sparse user-item interaction matrix $R \in \mathbb{R}^{|U| \times |I|}$ using `scipy.sparse.csr_matrix`.
  - The system shall train an Alternating Least Squares (ALS) model with latent factors $k$ (default 64), regularization $\lambda$ (default 0.1), and iterations (default 20).
  - The system shall maintain bidirectional index mappings: `visitorid_to_idx`, `idx_to_visitorid`, `itemid_to_idx`, and `idx_to_itemid`.
  - The ALS engine shall predict top-$N$ recommended items for known visitors.

- **FR-ME-03: Attribute-Based Content Similarity Engine**
  - The system shall aggregate anonymized item properties into a consolidated tokenized profile for each product.
  - The system shall generate sparse TF-IDF vectors for products and compute cosine similarity profiles.
  - For a given visitor, the engine shall recommend items structurally similar to the visitor's previously interacted items.

- **FR-ME-04: Hybrid Recommendation Engine**
  - The system shall implement a candidate generation and weighted rank-fusion pipeline combining:
    1. Collaborative Filtering scores ($S_{\text{cf}}$)
    2. Content-Based scores ($S_{\text{content}}$)
    3. Global Popularity scores ($S_{\text{pop}}$)
  - Scores shall be min-max normalized to $[0, 1]$ before linear blending:
    $$\text{Score}_{\text{hybrid}}(u, i) = \alpha \cdot \hat{S}_{\text{cf}}(u, i) + \beta \cdot \hat{S}_{\text{content}}(u, i) + \gamma \cdot \hat{S}_{\text{pop}}(i)$$
    where $\alpha + \beta + \gamma = 1.0$ (e.g., initial baseline: $\alpha=0.5, \beta=0.3, \gamma=0.2$).

- **FR-ME-05: Filtering & Deduplication**
  - The recommendation pipeline shall support an optional parameter `filter_already_interacted` (default: `True`).
  - When enabled, products that the user has already viewed or purchased during training shall be excluded from the recommended top-$N$ list.
  - The system shall guarantee that all top-$N$ recommendations contain unique product IDs (zero duplicates).

- **FR-ME-06: Graceful Cold-Start Degradation**
  - When a requested `visitor_id` is unknown (not present in the training index), the engine shall gracefully degrade to the Popularity-Based engine without raising an unhandled exception or 500 error.
  - The response payload shall explicitly set the recommendation source attribute (`"source": "popularity_fallback"`).

---

### Module 3: Evaluation & Experiment Tracking (ET)

- **FR-ET-01: Offline Top-$K$ Metric Computation**
  - The system shall compute industry-standard ranking metrics on validation and test splits:
    - **Precision@K:** Fraction of recommended top-$K$ items that were actually interacted with.
    - **Recall@K:** Fraction of held-out relevant items captured in top-$K$.
    - **NDCG@K:** Normalized Discounted Cumulative Gain accounting for rank position.
    - **MAP@K:** Mean Average Precision across all evaluated visitors.
    - **Catalog Coverage:** Percentage of the unique item catalog appearing at least once in top-$K$ recommendations across test users.
    - **Diversity:** Average pairwise distance between recommended items.
- **FR-ET-02: MLflow Experiment Logging**
  - Every training run shall log to an MLflow experiment named `Retailrocket-Recommendations`.
  - Logged entities shall include:
    - **Parameters:** Algorithm name, latent factors, regularization, iterations, TF-IDF max features, hybrid weights ($\alpha, \beta, \gamma$).
    - **Metrics:** Precision@K, Recall@K, NDCG@K, MAP@K, Coverage@K, Training Duration (seconds).
    - **Artifacts:** Serialized model weights (`model.npz` or `.pkl`), index mapping dictionaries (`mappings.pkl`), popularity score vectors (`popularity.pkl`), and evaluation summary (`evaluation_results.json`).
- **FR-ET-03: Model Registry & Quality Gate**
  - Only models that exceed the baseline Popularity engine's Recall@10 on the validation set shall be marked as eligible for promotion.

---

### Module 4: REST API Serving (API)

- **FR-API-01: Root Information Endpoint**
  - `GET /`
  - Returns service metadata: API name, version, status, and link to interactive documentation (`/docs`).
- **FR-API-02: Health & Readiness Probe**
  - `GET /health`
  - Returns system health status, model artifact loading status, uptime, and memory consumption.
  - Returns HTTP 200 if model artifacts are loaded and ready; returns HTTP 503 if models are uninitialized.
- **FR-API-03: Global Popularity Endpoint**
  - `GET /popular?top_n={N}`
  - Returns the top-$N$ globally popular products.
  - Default `top_n = 10`, clamped between 1 and 100.
- **FR-API-04: Personalized Recommendation Endpoint**
  - `GET /recommend/{visitor_id}?top_n={N}&filter_seen={true/false}`
  - Returns top-$N$ ranked product recommendations for the given `visitor_id`.
  - If the visitor is known, returns personalized hybrid recommendations (`"source": "hybrid"` or `"collaborative"`).
  - If the visitor is unknown, returns popular items with `"source": "popularity_fallback"`.
- **FR-API-05: Input Validation & Error Handling**
  - If `visitor_id` is not a valid integer or `top_n < 1`, return HTTP 422 (Unprocessable Entity) with clear error details.
  - All responses must conform to strict Pydantic schemas.

---

### Module 5: Automated Testing & Packaging (CI/CD)

- **FR-CI-01: Automated Test Suite**
  - Pytest suite covering:
    - Data preprocessing logic and weight calculations.
    - Metric calculation functions against known analytical vectors.
    - Model inference logic (known user, unknown user, out-of-range top-$N$).
    - API endpoint integration tests via FastAPI `TestClient`.
- **FR-CI-02: Container Packaging**
  - Standalone Dockerfile building a lightweight container image (e.g., Python 3.11 slim).
  - Model artifacts bundled or pulled on startup; container exposes port 8000.
