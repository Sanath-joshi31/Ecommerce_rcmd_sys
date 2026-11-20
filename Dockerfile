# Production Dockerfile for E-Commerce Recommendation API
# Base: Python 3.11 Slim (Debian Bookworm)

FROM python:3.11-slim

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install runtime utilities (curl for container healthcheck probe, libgomp1 for OpenMP in implicit/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies directly from PyPI binary wheels
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serving.txt

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000 \
    PATH="/home/appuser/.local/bin:/usr/local/bin:$PATH"

# Copy application code, sources, configuration, and approved model artifacts
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup artifacts/approved/ ./artifacts/approved/

RUN chown -R appuser:appgroup /app

# Switch to unprivileged non-root user
USER appuser

# Expose internal microservice port
EXPOSE 8000

# Container liveness/readiness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI entrypoint via python -m uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
