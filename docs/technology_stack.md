# Technology Stack Selection and Justification Specification

## 1. Executive Overview
Building an enterprise-ready machine learning system requires selecting tools that optimize for **computational efficiency, engineering ergonomics, operational reliability, and deployment simplicity**. In e-commerce recommendation systems, the primary engineering bottlenecks are **sparse matrix memory consumption, inference latency under concurrent web traffic, and reproducible experiment lineage**.

This document outlines the selected technology stack for the **E-Commerce Product Recommendation System**, comparing each choice against industry alternatives and justifying its adoption within our 8-phase ML-SDLC framework.

---

## 2. Technology Selection Matrix

| Project Activity | Selected Technology | Version (Target) | Purpose in Project | Why It Is Required (Core Justification) | Evaluated Alternatives & Reason Rejected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Language** | **Python** | `3.11 / 3.12` | Core programming language across ML pipelines, API serving, and test automation. | Rich ecosystem for numerical computing (NumPy, SciPy), native C-extension bindings, standard in modern MLOps. | *Java/Scala:* High verbosity, slower prototyping in scientific ML.<br>*Rust/C++:* Excessive engineering overhead for pipeline orchestration. |
| **In-Memory Analytical Querying** | **DuckDB** | `^1.1.0` | High-performance columnar SQL queries over raw 1GB CSV logs without database servers. | Executes vectorized OLAP queries directly on multi-hundred megabyte CSVs in milliseconds; zero external daemon required; seamless zero-copy Arrow/Pandas interop. | *SQLite:* Row-based, poor analytical aggregation performance on millions of rows.<br>*PySpark:* Massive JVM overhead, unnecessary complexity for single-node sub-10GB datasets.<br>*PostgreSQL:* Requires local server administration and port binding. |
| **Data Manipulation & Preprocessing** | **Pandas** | `^2.2.0` | In-memory tabular transformations, grouping, schema casting, and temporal slicing. | De facto standard for tabular data processing; native datetime parsing; robust index mapping capabilities. | *Polars:* Excellent performance, but Pandas has wider compatibility with downstream legacy ML tools and visualization bindings. |
| **Sparse Numerical Computation** | **NumPy & SciPy** | `numpy>=1.26`, `scipy>=1.12` | Dense array operations, linear algebra, and Compressed Sparse Row (`csr_matrix`) generation. | $99.9\%$ of user-item pairs are zero. `scipy.sparse.csr_matrix` compresses memory from 1.3 TB to < 50 MB, enabling fast sparse matrix-vector dot products. | *Pure Python dicts:* $O(N)$ lookup overhead, high memory pointers, incompatible with BLAS/LAPACK. |
| **Collaborative Filtering Engine** | **Implicit (ALS)** | `^0.7.2` | Alternating Least Squares (ALS) matrix factorization tailored for implicit feedback datasets. | Implements Hu, Koren & Volinsky (2008) implicit feedback algorithm with Cython/OpenMP multi-threading; orders of magnitude faster than naive SGD; handles unobserved signals as implicit negatives. | *Surprise Library:* Designed strictly for explicit ratings (1–5 stars); cannot properly optimize confidence-weighted implicit feedback.<br>*TensorFlow/PyTorch (NCF):* High GPU dependency, massive deployment image sizes (2-4GB), severe cold-start latency overhead. |
| **Content-Based Similarity** | **Scikit-learn** | `^1.4.0` | TF-IDF token vectorization, cosine similarity, train-test splitting utilities. | Optimized C-implemented TF-IDF vectorizer; efficient sparse cosine kernels; standard, bug-free implementations of numerical preprocessing pipelines. | *Gensim / FastText:* Overkill for anonymized, tokenized categorical properties without syntactic semantics. |
| **Interactive EDA & Dashboarding** | **Plotly + ipywidgets** | `plotly^5.20`, `ipywidgets^8.1` | Rich interactive charting (hover, zoom, dynamic sliders, event dropdowns) in Jupyter. | Plotly provides declarative, publication-quality web-ready graphs; ipywidgets provides reactive dropdowns for slicing millions of events without re-running entire notebooks. | *Matplotlib / Seaborn alone:* Static images only, no dynamic drill-down or hover tooltips for dense time series. |
| **Data Missingness Diagnosis** | **Missingno** | `^0.5.2` | Visualizing nullity patterns across high-cardinality multi-table datasets. | Instantly surfaces systematic vs random missingness across event types (e.g. `transactionid` only populated on `transaction` events). | *Manual df.isnull().sum():* Does not illustrate structural co-occurrence of missing fields. |
| **Experiment Tracking & Registry** | **MLflow** | `^2.11.0` | Tracking model hyperparameters, metrics (NDCG, Recall), run artifacts, and model versioning. | Open-source, vendor-agnostic, zero-cost self-hosted tracking; stores metrics, parameters, and serialized pickle/npz artifacts; provides visual web UI. | *Weights & Biases (W&B):* Proprietary SaaS, requires external API keys and cloud egress.<br>*Neptune.ai:* Commercial SaaS requiring proprietary client authentication. |
| **Production Serving API** | **FastAPI + Uvicorn** | `fastapi^0.110`, `uvicorn^0.29` | High-performance asynchronous REST API serving `/health`, `/popular`, and `/recommend`. | Asynchronous ASGI framework; native Pydantic v2 data validation; auto-generated OpenAPI/Swagger interactive UI (`/docs`); sub-millisecond route overhead. | *Flask:* Synchronous WSGI, manual request parsing, lack of native typing/Pydantic validation.<br>*Django REST Framework:* Heavyweight ORM overhead, excessive boilerplate for microservice inference. |
| **Data Validation & Schemas** | **Pydantic** | `^2.6.0` | Strict request and response payload typing and validation. | Rust-backed deserialization performance; enforces typed contracts (e.g., `visitor_id: int`, `top_n: int = 10`); auto-generates JSON schema. | *Marshmallow:* Slower Python-only validation, less ergonomic FastAPI integration. |
| **Automated Testing Suite** | **Pytest + HTTPX** | `pytest^8.1`, `httpx^0.27` | Unit tests for data/features, integration tests for recommendation models, API test client. | Fast test execution; fixture architecture; HTTPX enables seamless asynchronous integration testing of FastAPI endpoints without spinning up live sockets. | *Unittest:* Legacy xUnit syntax, verbose assertions, poor parametrization support. |
| **Containerization & Packaging** | **Docker** | `v25+ (OCI)` | Packaging the FastAPI service, model weights, and runtime into an immutable Linux container. | Eliminates environment discrepancies; guarantees exact library versions and system binaries; encapsulates memory and CPU limits. | *Bare Metal / Virtualenv:* High risk of host OS drift, missing C++ runtime libraries on target servers. |
| **CI/CD Automation** | **GitHub Actions** | `YAML Workflows` | Continuous integration pipeline executing linting, unit tests, artifact checks, and Docker builds. | Native integration with GitHub repository; free runner minutes for open source; declarative YAML workflows with matrix testing and secret management. | *Jenkins:* High administrative overhead, requires self-hosted master/worker servers.<br>*GitLab CI:* Requires hosting or migrating code repository to GitLab. |
| **Cloud Deployment Platform** | **Render / Railway** | `Cloud PaaS` | Hosting the live containerized recommendation API with automated HTTPS and health monitoring. | Free or low-cost hobby tiers suitable for portfolio demonstrations; native Docker deployment via GitHub webhook; zero Kubernetes cluster management overhead. | *AWS EKS / GCP GKE:* Enterprise-grade but expensive ($70+/month base cluster cost) and over-engineered for a single inference microservice. |

---

## 3. Technology Version Compatibility Plan

To avoid binary conflicts and dependency incompatibilities (especially on Windows development environments and Linux Docker containers), the following verified package dependency matrix is established:

```ini
# Core Scientific & Data Processing
python>=3.11,<3.13
numpy>=1.26.0,<2.4.0
scipy>=1.12.0,<1.17.0
pandas>=2.2.0,<2.4.0
duckdb>=1.1.0

# Visualization & EDA
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0
ipywidgets>=8.1.0
missingno>=0.5.2

# Machine Learning
scikit-learn>=1.4.0
implicit>=0.7.2

# MLOps & Experiment Tracking
mlflow>=2.11.0

# Serving & API
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0

# Testing & Quality
pytest>=8.1.0
httpx>=0.27.0
pytest-cov>=5.0.0
```

---

## 4. Operational Trade-Off Analysis

### 4.1 In-Memory Inference vs. Database-Backed Inference
* **Trade-off:** We choose to store precomputed popularity vectors and latent ALS factor arrays in **container memory** rather than querying an external database (e.g., Redis or PostgreSQL) during inference.
* **Why:** In-memory array dot products take $< 5\text{ ms}$, whereas database roundtrips across network boundaries add $15\text{--}40\text{ ms}$ latency and introduce external infrastructure dependencies that complicate portfolio deployment.
* **Risk Mitigation:** Since our sparse item factor representations and popularity arrays total $< 150\text{ MB}$, the memory footprint remains safely below our $1.5\text{ GB}$ container ceiling.

### 4.2 Offline Batch Model Training vs. Online Real-Time Updating
* **Trade-off:** The ALS collaborative filtering model is trained **offline in batch**, rather than updated continuously on every HTTP request.
* **Why:** Online retraining during serving introduces race conditions, severe memory spikes, and risks destabilizing the model with anomalous user behavior (feedback poisoning). Batch retraining governed by an MLflow quality gate ensures that only mathematically superior models enter production.
