"""Automated integration tests for FastAPI Recommendation Endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient executing the application lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verify that the root endpoint returns 200 OK and valid metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "E-Commerce Product Recommendation API"
    assert data["status"] == "operational"
    assert "endpoints" in data
    assert "recommend" in data["endpoints"]


def test_health_check_endpoint(client):
    """Verify that /health reports service readiness and model availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["service"] == "ecommerce-recommendation-api"
    assert data["uptime_seconds"] >= 0.0
    assert data["memory_usage_mb"] > 0.0
    assert "popularity" in data["models_loaded"]


def test_popular_endpoint_default(client):
    """Verify /popular returns default top 10 ranked products."""
    response = client.get("/popular")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert data["source"] == "popularity_global"
    assert len(data["recommendations"]) == 10

    # Verify rank order and uniqueness
    ranks = [item["rank"] for item in data["recommendations"]]
    assert ranks == list(range(1, 11))

    item_ids = [item["item_id"] for item in data["recommendations"]]
    assert len(item_ids) == len(set(item_ids)), "Duplicate item IDs returned!"


def test_popular_endpoint_custom_top_n(client):
    """Verify /popular respects custom top_n parameter."""
    response = client.get("/popular?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5


def test_recommend_cold_start_visitor(client):
    """Verify that an unseen visitor ID gracefully degrades to popularity fallback."""
    unseen_visitor_id = 9999999999
    response = client.get(f"/recommend/{unseen_visitor_id}?top_n=10")
    assert response.status_code == 200
    data = response.json()
    assert data["visitor_id"] == unseen_visitor_id
    assert data["count"] == 10
    assert data["source"] == "popularity_fallback"
    assert len(data["recommendations"]) == 10

    # Ensure zero duplicates
    item_ids = [item["item_id"] for item in data["recommendations"]]
    assert len(item_ids) == len(set(item_ids))


def test_recommend_known_warm_visitor(client):
    """Verify recommendation generation for a known warm visitor from training set."""
    service = app.state.service
    if service.als_model and service.als_model.visitorid_to_idx:
        sample_visitor = int(list(service.als_model.visitorid_to_idx.keys())[10])
        response = client.get(f"/recommend/{sample_visitor}?top_n=10&filter_seen=true")
        assert response.status_code == 200
        data = response.json()
        assert data["visitor_id"] == sample_visitor
        assert data["count"] <= 10
        assert data["source"] in ("hybrid", "popularity_fallback")
        assert len(data["recommendations"]) > 0


def test_recommend_invalid_visitor_id(client):
    """Verify that invalid non-positive visitor IDs return 422 Unprocessable Entity."""
    # Negative ID
    res_neg = client.get("/recommend/-1")
    assert res_neg.status_code == 422

    # Zero ID
    res_zero = client.get("/recommend/0")
    assert res_zero.status_code == 422

    # String ID
    res_str = client.get("/recommend/invalid_string_id")
    assert res_str.status_code == 422


def test_recommend_invalid_top_n(client):
    """Verify that out-of-range top_n values return 422 Unprocessable Entity."""
    # Zero top_n
    res_zero = client.get("/recommend/12345?top_n=0")
    assert res_zero.status_code == 422

    # Exceeding maximum (top_n > 100)
    res_high = client.get("/recommend/12345?top_n=101")
    assert res_high.status_code == 422

    # Negative top_n
    res_neg = client.get("/recommend/12345?top_n=-5")
    assert res_neg.status_code == 422
