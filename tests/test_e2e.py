"""End-to-end tests for Virality Engine."""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from src.api.main import app
from src.db.connection import init_db, get_db_session
from src.db.models import ContentItem, ViralityScore, FeedbackLoop
from src.config import settings

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_database():
    """Initialize database for testing."""
    init_db()
    yield


@pytest.fixture
def dummy_content_data():
    """Generate dummy content data for testing."""
    return {
        "platform": "tiktok",
        "caption": "Check out this amazing video! #viral #trending",
        "description": "A test video for virality prediction",
        "hashtags": ["viral", "trending", "fyp"],
        "engagement_metrics": {
            "views": 10000.0,
            "likes": 500.0,
            "shares": 50.0,
            "comments": 100.0,
            "saves": 200.0,
            "reach": 10000.0,
            "momentum_score": 0.7
        },
        "comments": [
            {"text": "This is amazing!", "timestamp": datetime.utcnow().isoformat()},
            {"text": "Love this content", "timestamp": datetime.utcnow().isoformat()},
            {"text": "So good!", "timestamp": datetime.utcnow().isoformat()}
        ]
    }


@pytest.fixture
def dummy_file():
    """Create a dummy file for upload."""
    import io
    return ("test_video.mp4", io.BytesIO(b"fake video content"), "video/mp4")


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_ingestion_endpoint(self, dummy_file, dummy_content_data):
        """Test content ingestion."""
        files = {
            "file": dummy_file
        }
        data = {
            "platform": dummy_content_data["platform"],
            "caption": dummy_content_data["caption"],
            "description": dummy_content_data["description"],
            "hashtags": ",".join(dummy_content_data["hashtags"])
        }
        
        response = client.post("/api/v1/ingest", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "content_id" in result
        assert "embeddings" in result
        assert "features" in result
        assert result["platform"] == dummy_content_data["platform"]
        
        return result
    
    def test_scoring_endpoint(self, dummy_content_data):
        """Test virality scoring."""
        request_data = {
            "content_id": str(uuid.uuid4()),
            "features": {
                "visual": {
                    "entropy": 0.7,
                    "variance": {"frame_variance": 0.5}
                },
                "audio": {
                    "bpm": 120.0,
                    "loudness": {"rms": 0.6, "peak": 0.8}
                },
                "text": {
                    "trend_proximity": {"trend_score": 0.8},
                    "hook_efficiency": {"hook_score": 0.7}
                }
            },
            "embeddings": {
                "visual": [0.1] * 512,
                "audio": [0.2] * 128,
                "text": [0.3] * 384,
                "contextual": [0.4] * 256
            },
            "metadata": dummy_content_data
        }
        
        response = client.post("/api/v1/score", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "virality_score" in result
        assert "virality_probability" in result
        assert "viral_classification" in result
        assert "confidence_interval" in result
        assert "confidence_level" in result
        assert "virality_dynamics" in result
        assert "model_lineage" in result
        assert "attribution_insights" in result
        assert "recommendations" in result
        assert 0.0 <= result["virality_score"] <= 100.0
        assert 0.0 <= result["virality_probability"] <= 1.0
        assert result["viral_classification"] in ["yes", "no"]
        assert result["virality_dynamics"] in ["deep", "broad", "hybrid"]
        assert 0.0 <= result["confidence_level"] <= 1.0
        assert len(result["attribution_insights"]) >= 5
        
        return result
    
    def test_analyze_endpoint(self, dummy_file, dummy_content_data):
        """Test analyze endpoint (ingest + score)."""
        files = {
            "file": dummy_file
        }
        data = {
            "platform": dummy_content_data["platform"],
            "caption": dummy_content_data["caption"],
            "description": dummy_content_data["description"],
            "hashtags": ",".join(dummy_content_data["hashtags"])
        }
        
        response = client.post("/api/v1/analyze", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "content_id" in result
        assert "embeddings" in result
        assert "features" in result
        assert "scoring" in result or "virality_score" in result
        
        return result
    
    def test_feedback_endpoint(self):
        """Test feedback collection."""
        feedback_data = {
            "content_id": str(uuid.uuid4()),
            "predicted_probability": 0.75,
            "actual_performance": 0.82
        }
        
        response = client.post("/api/v1/feedback", json=feedback_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
        assert "feedback" in result
        with get_db_session() as session:
            feedback = session.query(FeedbackLoop).filter(
                FeedbackLoop.content_id == feedback_data["content_id"]
            ).first()
            assert feedback is not None
            assert feedback.predicted_probability == feedback_data["predicted_probability"]
            assert feedback.actual_performance == feedback_data["actual_performance"]
    
    def test_complete_pipeline(self, dummy_file, dummy_content_data):
        """Test complete pipeline: ingest -> score -> feedback."""
        files = {"file": dummy_file}
        data = {
            "platform": dummy_content_data["platform"],
            "caption": dummy_content_data["caption"],
            "hashtags": ",".join(dummy_content_data["hashtags"])
        }
        
        ingest_response = client.post("/api/v1/ingest", files=files, data=data)
        assert ingest_response.status_code == 200
        ingest_result = ingest_response.json()
        content_id = ingest_result["content_id"]
        score_request = {
            "content_id": content_id,
            "features": ingest_result["features"],
            "embeddings": ingest_result["embeddings"],
            "metadata": {**ingest_result["metadata"], **dummy_content_data}
        }
        
        score_response = client.post("/api/v1/score", json=score_request)
        assert score_response.status_code == 200
        score_result = score_response.json()
        assert "virality_score" in score_result
        assert score_result["virality_score"] >= 0.0
        feedback_request = {
            "content_id": content_id,
            "predicted_probability": score_result["virality_probability"],
            "actual_performance": 0.85
        }
        
        feedback_response = client.post("/api/v1/feedback", json=feedback_request)
        assert feedback_response.status_code == 200
        assert ingest_result["content_id"] == content_id
        assert score_result["content_id"] == content_id
        
        return {
            "ingest": ingest_result,
            "score": score_result,
            "feedback": feedback_response.json()
        }
    
    def test_algorithm1_output_format(self, dummy_content_data):
        """Test that Algorithm 1 output format is correct."""
        request_data = {
            "content_id": str(uuid.uuid4()),
            "features": {
                "visual": {"entropy": 0.7},
                "text": {
                    "trend_proximity": {"trend_score": 0.8},
                    "hook_efficiency": {"hook_score": 0.7}
                }
            },
            "embeddings": {
                "visual": [0.1] * 512,
                "audio": [0.2] * 128,
                "text": [0.3] * 384,
                "contextual": [0.4] * 256
            },
            "metadata": dummy_content_data
        }
        
        response = client.post("/api/v1/score", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "virality_score" in result
        assert "virality_probability" in result
        assert "viral_classification" in result
        assert "confidence_interval" in result
        assert "model_lineage" in result
        assert "virality_dynamics" in result
        
        ci = result["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] <= result["virality_probability"] <= ci["upper"]
        
        lineage = result["model_lineage"]
        assert "model_version" in lineage
        assert "reproducibility_hash" in lineage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

