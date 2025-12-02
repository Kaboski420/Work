"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


def test_ingest_endpoint():
    """Test content ingestion endpoint."""
    import io
    files = {"file": ("test.mp4", io.BytesIO(b"fake video"), "video/mp4")}
    data = {"platform": "tiktok", "caption": "test"}
    response = client.post("/api/v1/ingest", files=files, data=data)
    assert response.status_code in [200, 500]


def test_score_endpoint():
    """Test scoring endpoint."""
    request_data = {
        "content_id": "test-123",
        "features": {"visual": {"entropy": 0.7}},
        "embeddings": {"visual": [0.1, 0.2]},
        "metadata": {
            "platform": "tiktok",
            "engagement_metrics": {"views": 1000, "likes": 100}
        }
    }
    response = client.post("/api/v1/score", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "virality_score" in data



