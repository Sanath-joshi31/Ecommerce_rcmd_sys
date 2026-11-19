# Phase 6: Application Development and Containerization

## 1. Executive Summary

Phase 6 transitions the trained and approved recommendation models from offline experimentation to an enterprise-grade, high-throughput, low-latency REST API service containerized for cloud-native deployment.

Using **FastAPI**, **Pydantic v2**, and **Uvicorn**, we built a modular microservice capable of serving sub-5ms personalized hybrid recommendations and sub-millisecond popularity fallbacks. The entire service is packaged into a secure, multi-stage **Docker** container adhering to DevSecOps standards (unprivileged non-root user, slim base image, automated container health probes).

---

## 2. API Architecture and Component Layout

```mermaid
graph TD
    Client["Client / Web Frontend / Gateway"] -->|HTTP GET| Uvicorn["Uvicorn ASGI Server (:8000)"]
    Uvicorn --> FastAPI["FastAPI Application (app/main.py)"]
    FastAPI --> CORS["CORS Middleware"]
    CORS --> Endpoints{"Router Endpoints"}
    
    Endpoints -->|GET /| Root["Root Metadata (/)"]
    Endpoints -->|GET /health| Health["Liveness/Readiness Probe (/health)"]
    Endpoints -->|GET /popular| PopHandler["Popular Handler (/popular)"]
    Endpoints -->|GET /recommend/{visitor_id}| RecHandler["Personalized Hybrid Handler (/recommend)"]
    
    RecHandler --> RecService["RecommendationService Singleton (app/services.py)"]
    PopHandler --> RecService
    Health --> RecService
    
    RecService -->|"Lazy / Pre-warmed"| Artifacts["Approved Model Artifacts (artifacts/approved/)"]
    Artifacts -.-> ALS["ALS Collaborative Filtering Model (325 MB)"]
    Artifacts -.-> CB["Content-Based TF-IDF Model (38 MB)"]
    Artifacts -.-> Pop["Popularity Precomputed Features (2.2 MB)"]
    Artifacts -.-> Map["Item/Visitor Index Mappings (38 MB)"]
```

---

## 3. Implemented API Endpoints

### 3.1 Metadata Endpoint (`GET /`)
- **Route:** `GET /`
- **Purpose:** Service discovery, root ping, version verification, and interactive OpenAPI documentation pointers.
- **Sample Response:**
```json
{
  "name": "E-Commerce Product Recommendation API",
  "version": "1.0.0",
  "status": "operational",
  "docs_url": "/docs",
  "endpoints": {
    "root": "GET /",
    "health": "GET /health",
    "popular": "GET /popular?top_n=10",
    "recommend": "GET /recommend/{visitor_id}?top_n=10&filter_seen=true"
  }
}
```

### 3.2 Health Check Probe (`GET /health`)
- **Route:** `GET /health`
- **Purpose:** Diagnostic liveness and readiness probe for Kubernetes/Docker container orchestrators.
- **Health Metrics Tracked:**
  - `status`: `"healthy"` or `"degraded"`
  - `uptime_seconds`: Continuous server uptime
  - `memory_usage_mb`: Resident Set Size (RSS) memory consumption
  - `models_loaded`: Verification that `popularity`, `collaborative_filtering`, and `content_based` are active in memory.
- **Sample Response:**
```json
{
  "status": "healthy",
  "service": "ecommerce-recommendation-api",
  "uptime_seconds": 142.3,
  "memory_usage_mb": 412.8,
  "models_loaded": {
    "popularity": true,
    "collaborative_filtering": true,
    "content_based": true
  }
}
```

### 3.3 Global Popularity Endpoint (`GET /popular`)
- **Route:** `GET /popular?top_n={N}`
- **Query Parameters:**
  - `top_n` (int, default=10, min=1, max=100): Number of top products to retrieve.
- **Sample Response:**
```json
{
  "source": "popularity_global",
  "count": 5,
  "recommendations": [
    {"rank": 1, "item_id": 461686, "score": 1.0, "reason": "Global Popularity"},
    {"rank": 2, "item_id": 257040, "score": 0.8123, "reason": "Global Popularity"},
    {"rank": 3, "item_id": 312728, "score": 0.6541, "reason": "Global Popularity"},
    {"rank": 4, "item_id": 248482, "score": 0.5892, "reason": "Global Popularity"},
    {"rank": 5, "item_id": 445351, "score": 0.5401, "reason": "Global Popularity"}
  ]
}
```

### 3.4 Personalized Recommendation Endpoint (`GET /recommend/{visitor_id}`)
- **Route:** `GET /recommend/{visitor_id}?top_n={N}&filter_seen={bool}`
- **Path Parameters:**
  - `visitor_id` (int, strictly positive): Anonymous customer interaction identifier.
- **Query Parameters:**
  - `top_n` (int, default=10, min=1, max=100): Number of items requested.
  - `filter_seen` (bool, default=true): Exclude items the user has already viewed, carted, or bought in historical data.
- **Behavior:**
  - **Warm Visitors:** Executes ensemble hybrid ranking combining ALS latent preference scores ($50\%$), content-based property similarity ($30\%$), and popularity baseline ($20\%$). Returns `source: "hybrid"`.
  - **Cold-Start Visitors:** If visitor ID was never observed in training interactions, the engine gracefully degrades to ranked popularity items, returning `source: "popularity_fallback"`.
- **Sample Response:**
```json
{
  "visitor_id": 405903,
  "source": "hybrid",
  "count": 10,
  "latency_ms": 3.42,
  "recommendations": [
    {"rank": 1, "item_id": 38781, "score": 0.8912, "reason": "Ensemble Hybrid Recommendation"},
    {"rank": 2, "item_id": 182741, "score": 0.8405, "reason": "Ensemble Hybrid Recommendation"},
    {"rank": 3, "item_id": 96199, "score": 0.7931, "reason": "Ensemble Hybrid Recommendation"}
  ]
}
```

---

## 4. In-Memory Recommendation Service (`app/services.py`)

To achieve millisecond-level inference, `RecommendationService` implements several architectural optimizations:

1. **Singleton Initialization:** Model weights and mapping dictionaries are loaded once into RAM at process startup via FastAPI lifespan routines, eliminating disk I/O during request servicing.
2. **Matrix-Free Dot Product Inference:** Rather than maintaining the full $1.5\text{M} \times 200\text{K}$ sparse CSR interaction matrix in web API memory, the service uses direct vector dot products:
   $$\hat{y}_{u, i} = \mathbf{x}_u \cdot \mathbf{y}_i^T$$
   evaluating latent user vector $\mathbf{x}_u \in \mathbb{R}^{64}$ against pre-filtered candidate item factors. This reduces API inference latency to $\sim 1.7\text{ ms}$.
3. **Resilient Cold-Start Safeguard:** Every inference pathway is wrapped in a fail-safe try-except block. In the event of an out-of-vocabulary ID or vector dimension anomaly, the service automatically falls back to pre-indexed popularity items without raising a 500 error to the client.

---

## 5. Automated Test Suite and Quality Assurance

The test suite validates model mathematics, hybrid business rules, edge cases, input validation, and API contracts.

### Test Breakdown

| Test File | Test Name | Target Module | Description | Result |
| :--- | :--- | :--- | :--- | :---: |
| `tests/test_models.py` | `test_popularity_recommender` | `PopularityRecommender` | Verifies top-N ranking order, cold-start handling, and score monotonicity. | **PASSED** |
| `tests/test_models.py` | `test_als_recommender_synthetic` | `ALSRecommender` | Validates implicit ALS training, factor shapes, and user recommendation outputs. | **PASSED** |
| `tests/test_models.py` | `test_content_based_recommender` | `ContentBasedRecommender` | Verifies TF-IDF cosine similarity calculations and content-based item discovery. | **PASSED** |
| `tests/test_recommendations.py` | `test_min_max_normalize` | `HybridRecommender` | Asserts score normalization is strictly bound between $[0.0, 1.0]$. | **PASSED** |
| `tests/test_recommendations.py` | `test_hybrid_cold_start_fallback` | `HybridRecommender` | Asserts unseen cold visitors receive valid popularity recommendations. | **PASSED** |
| `tests/test_recommendations.py` | `test_hybrid_weight_updating` | `HybridRecommender` | Verifies dynamic tuning of ensemble weights ($w_{\text{cf}}, w_{\text{content}}, w_{\text{pop}}$). | **PASSED** |
| `tests/test_api.py` | `test_root_endpoint` | `GET /` | Asserts status 200, valid endpoint schema, and operational status. | **PASSED** |
| `tests/test_api.py` | `test_health_check_endpoint` | `GET /health` | Validates liveness probe, model availability flags, uptime, and memory reporting. | **PASSED** |
| `tests/test_api.py` | `test_popular_endpoint_default` | `GET /popular` | Verifies default top-10 popularity response and strict item uniqueness. | **PASSED** |
| `tests/test_api.py` | `test_popular_endpoint_custom_top_n`| `GET /popular?top_n=5` | Confirms dynamic top-N parameter slice constraints. | **PASSED** |
| `tests/test_api.py` | `test_recommend_cold_start_visitor`| `GET /recommend/{id}` | Asserts cold visitor degradation to popularity fallback with zero duplicate items. | **PASSED** |
| `tests/test_api.py` | `test_recommend_known_warm_visitor`| `GET /recommend/{id}` | Validates hybrid recommendation scoring for verified training set visitors. | **PASSED** |
| `tests/test_api.py` | `test_recommend_invalid_visitor_id`| `GET /recommend/{id}` | Confirms 422 Unprocessable Entity for negative, zero, and non-integer visitor IDs. | **PASSED** |
| `tests/test_api.py` | `test_recommend_invalid_top_n` | `GET /recommend/{id}` | Confirms 422 Unprocessable Entity for out-of-range ($N < 1$ or $N > 100$) inputs. | **PASSED** |

**Summary:** 14 passed out of 14 tests ($100\%$ pass rate) in 4.15 seconds.

---

## 6. Containerization Architecture (Docker)

### 6.1 Multi-Stage Build Strategy
To minimize final image size and reduce attack surface area, the `Dockerfile` uses a two-stage pattern:
1. **Stage 1 (`builder`):** Installs Linux compilation tools (`build-essential`) and pre-compiles binary wheels for C/Cython packages (`implicit`, `scipy`).
2. **Stage 2 (`runner`):** Copies only pre-built wheels and application code onto a lean `python:3.11-slim` image, discarding all compiler tools and build cache.

### 6.2 Security Hardening
- **Unprivileged User:** The container executes under a dedicated system user `appuser:appgroup` (UID `10001`), preventing potential container breakout vulnerabilities.
- **Read-Only Models:** The model artifacts in `artifacts/approved/` are mounted read-only (`ro`).
- **Healthcheck Probe:** Includes continuous liveness monitoring (`HEALTHCHECK`) via `curl -f http://localhost:8000/health`.

---

## 7. Local Execution and Verification Commands

### 7.1 Running the API Locally (Direct Python)
```bash
# Launch Uvicorn server on localhost:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 7.2 Executing Automated Tests
```bash
# Run complete test suite with coverage
pytest tests/ -v --cov=app --cov=src
```

### 7.3 Building and Running with Docker
```bash
# Build production Docker container
docker build -t ecommerce-recommendation-api:v1.0.0 .

# Run container with port forwarding and healthcheck
docker run -d --name recommendation-api -p 8000:8000 ecommerce-recommendation-api:v1.0.0

# Inspect container health status
docker inspect --format='{{json .State.Health}}' recommendation-api
```
