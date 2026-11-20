# Multi-Stage Production Dockerfile for E-Commerce Recommendation API
# Base: Python 3.11 Slim (Debian Bookworm)

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies required for compiling C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to wheels cache
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements-serving.txt


# ==========================================
# Final Runtime Stage
# ==========================================
FROM python:3.11-slim AS runner

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install runtime utilities (curl for container healthcheck probe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

# Copy application code, sources, configuration, and approved model artifacts
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup artifacts/approved/ ./artifacts/approved/

# Switch to unprivileged non-root user
USER appuser

# Expose internal microservice port
EXPOSE 8000

# Container liveness/readiness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production WSGI/ASGI entrypoint via Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
