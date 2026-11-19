# Phase 7: Continuous Integration and Continuous Deployment (CI/CD)

## 1. Executive Summary

Phase 7 implements a robust, automated **MLOps CI/CD Pipeline** using **GitHub Actions**, containerizing every release upon code commit or pull request. 

Unlike traditional web applications where CI/CD only runs unit tests and builds images, our MLOps pipeline enforces a dedicated **Machine Learning Quality Gate**: validating model lift ($+213.8\%$ over baseline), catalog coverage ($2.25\%$), and serving latency SLAs ($< 80\text{ ms}$) before authorizing container builds and deployment.

---

## 2. CI/CD Architecture and Pipeline Workflow

```mermaid
flowchart TD
    Commit["Git Push / PR to main"] --> Stage1["Stage 1: Code Quality & Linting"]
    
    subgraph "Stage 1: Static Analysis"
        Stage1 --> Flake8["Flake8 (PEP8, Syntax, Unused Imports)"]
        Stage1 --> Black["Black & isort (Formatting)"]
    end

    Flake8 --> Stage2["Stage 2: Automated Testing Matrix"]
    
    subgraph "Stage 2: Pytest Matrix"
        Stage2 --> Py311["Python 3.11 Environment"]
        Stage2 --> Py312["Python 3.12 Environment"]
        Py311 --> UnitTests["Model & Hybrid Unit Tests"]
        Py311 --> ApiTests["FastAPI HTTP Integration Tests"]
        Py311 --> CovReport["Coverage Enforcement (> 85% core)"]
    end

    UnitTests --> Stage3["Stage 3: Automated ML Quality Gate"]
    
    subgraph "Stage 3: ML Quality Gate"
        Stage3 --> QualityScript["scripts/ci_model_gate.py"]
        QualityScript --> CheckRecall["Recall@10 Lift >= 30%"]
        QualityScript --> CheckLatency["Serving Latency < 80ms"]
        QualityScript --> CheckCoverage["Catalog Coverage > 0.1%"]
        QualityScript --> CheckArtifacts["Serving Artifact Integrity"]
    end

    QualityScript --> Stage4["Stage 4: Container Build & Smoke Test"]
    
    subgraph "Stage 4: Docker Smoke Testing"
        Stage4 --> Buildx["Buildx Multi-Stage Docker Build"]
        Stage4 --> SpinContainer["Launch Container on CI Runner"]
        Stage4 --> ProbeHealth["Probe GET /health"]
        Stage4 --> ProbeInference["Probe GET /recommend/999999"]
    end

    ProbeInference --> Stage5{"Stage 5: Main Branch Push?"}
    Stage5 -->|Yes| GHCR["Push to GitHub Container Registry (ghcr.io)"]
    GHCR --> CloudDeploy["Trigger Cloud Deploy Webhook (Render / Railway)"]
    Stage5 -->|No (PR)| PRComment["Report Passed Check to PR"]
```

---

## 3. Pipeline Stages Detailed Breakdown

### Stage 1: Code Quality & Static Analysis
- **Workflow Job:** `code-quality`
- **Runner:** `ubuntu-latest` (Python 3.11)
- **Tooling:** `flake8`, `black`, `isort`
- **Configuration:** Bound by `.flake8` with `max-line-length = 120`.
- **Purpose:** Catches syntax errors, undefined variables, and unused imports before running heavy test matrices.

### Stage 2: Automated Unit & Integration Testing Matrix
- **Workflow Job:** `test-and-evaluate`
- **Matrix:** Python 3.11 and Python 3.12 runners.
- **Coverage Tooling:** `pytest-cov`, reporting line-by-line coverage in XML format.
- **Test Scenarios:**
  - `tests/test_models.py`: Mathematically validates Popularity, ALS factor multiplication, and Content TF-IDF cosine similarity.
  - `tests/test_recommendations.py`: Tests min-max score normalization, dynamic weight updates, and cold-start fallback.
  - `tests/test_api.py`: Tests `/`, `/health`, `/popular`, and `/recommend/{id}` endpoints, verifying schemas, duplicate elimination, and 422 input validation.

### Stage 3: Automated Model Quality Gate
- **Workflow Job:** `model-quality-gate`
- **Script:** `scripts/ci_model_gate.py`
- **Evaluation Criteria:**
  1. **Recall Lift:** $\Delta \text{Recall@10} \ge 30\%$ relative to Popularity baseline (Champion achieved $+213.8\%$).
  2. **Latency SLA:** Average recommendation inference latency $< 80.0\text{ ms}$ (Champion achieved $4.95\text{ ms}$).
  3. **Exploration & Diversity:** Catalog coverage $> 0.1\%$ (Champion achieved $2.255\%$).
  4. **Artifact Completeness:** Verification that model metadata, popularity parquet, and mappings exist and are uncorrupted.
- **Reporting:** Emits a real-time markdown summary table directly to the GitHub Actions job dashboard (`$GITHUB_STEP_SUMMARY`).

### Stage 4: Docker Build & Live Smoke Test
- **Workflow Job:** `docker-smoke-test`
- **Builder:** `docker/setup-buildx-action@v3` with GitHub Actions Layer Caching (`type=gha`).
- **Live Smoke Testing Routine:**
  1. Builds image `ecommerce-recommendation-api:ci-test`.
  2. Spawns container in detached mode on CI runner (`docker run -d -p 8000:8000 ...`).
  3. Probes `GET /health` with retry loop to confirm ASGI startup.
  4. Dispatches live inference query `GET /recommend/999999?top_n=5` and asserts HTTP 200 return code.
  5. Inspects container logs and safely cleans up container resources.

### Stage 5: Container Registry Push & Cloud Deployment
- **Workflow Job:** `publish-and-deploy`
- **Trigger:** Only executed on `push` to `main` (branch merges).
- **Registry:** GitHub Container Registry (`ghcr.io`).
- **Image Tagging:** Automatically tags image with both `:latest` and the short git commit SHA (`:${{ github.sha }}`).
- **Continuous Deployment Webhook:** Calls cloud provider deploy webhook (`Render` / `Railway`) using repository secret `DEPLOY_HOOK_URL`.

---

## 4. Local Reproduction & Verification Commands

Developers can execute each CI/CD stage locally before pushing code to Git:

### Step 1: Run Static Linting
```bash
flake8 app src tests
```

### Step 2: Execute Pytest with Coverage
```bash
pytest tests/ -v --cov=app --cov=src/models --cov=src/recommendation --cov-report=term-missing
```

### Step 3: Run Model Quality Gate
```bash
python scripts/ci_model_gate.py
```

### Step 4: Run Container Smoke Test
```bash
# Build container
docker build -t ecommerce-recommendation-api:local-test .

# Run container
docker run -d --name local-api -p 8000:8000 ecommerce-recommendation-api:local-test

# Probe endpoints
curl -f http://localhost:8000/health
curl -f http://localhost:8000/recommend/12345?top_n=5

# Cleanup
docker stop local-api && docker rm local-api
```
