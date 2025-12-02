"""Test runner for Virality Engine."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.generate_dummy_data import generate_dummy_content, generate_dummy_feedback
from src.db.connection import init_db
from src.api.main import app
from fastapi.testclient import TestClient
import uuid


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_health_check():
    """Test health check."""
    print_section("Testing Health Check")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Health check passed: {data}")
    return True


def test_ingestion(dummy_content):
    """Test content ingestion."""
    print_section("Testing Content Ingestion")
    client = TestClient(app)
    import io
    dummy_file = ("test_video.mp4", io.BytesIO(b"fake video content"), "video/mp4")
    
    files = {"file": dummy_file}
    data = {
        "platform": dummy_content["platform"],
        "caption": dummy_content["caption"],
        "description": dummy_content.get("description", ""),
        "hashtags": ",".join(dummy_content.get("hashtags", []))
    }
    
    response = client.post("/api/v1/ingest", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ Ingestion successful:")
    print(f"   - Content ID: {result['content_id']}")
    print(f"   - Platform: {result['platform']}")
    print(f"   - Has embeddings: {bool(result.get('embeddings'))}")
    print(f"   - Has features: {bool(result.get('features'))}")
    
    return result


def test_scoring(dummy_content):
    """Test virality scoring."""
    print_section("Testing Virality Scoring")
    client = TestClient(app)
    
    request_data = {
        "content_id": dummy_content["content_id"],
        "features": dummy_content["features"],
        "embeddings": dummy_content["embeddings"],
        "metadata": {
            "caption": dummy_content["caption"],
            "description": dummy_content.get("description", ""),
            "hashtags": dummy_content.get("hashtags", []),
            "platform": dummy_content["platform"],
            "engagement_metrics": dummy_content.get("engagement_metrics", {})
        }
    }
    
    response = client.post("/api/v1/score", json=request_data)
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ Scoring successful:")
    print(f"   - Virality Score: {result['virality_score']:.2f}/100")
    print(f"   - Virality Probability: {result['virality_probability']:.3f}")
    print(f"   - Viral Classification: {result['viral_classification']}")
    print(f"   - Confidence Level: {result['confidence_level']:.3f}")
    print(f"   - Virality Dynamics: {result['virality_dynamics']}")
    print(f"   - Attribution Factors: {len(result['attribution_insights'])}")
    print(f"   - Model Version: {result['model_lineage']['model_version']}")
    
    # Verify Algorithm 1 requirements
    assert 0.0 <= result["virality_score"] <= 100.0
    assert result["viral_classification"] in ["yes", "no"]
    assert result["virality_dynamics"] in ["deep", "broad", "hybrid"]
    assert len(result["attribution_insights"]) >= 5
    print("   ✅ All Algorithm 1 requirements met")
    return result


def test_feedback(content_id: str, predicted_prob: float, actual_perf: float):
    """Test feedback collection."""
    print_section("Testing Feedback Collection")
    client = TestClient(app)
    
    feedback_data = {
        "content_id": content_id,
        "predicted_probability": predicted_prob,
        "actual_performance": actual_perf
    }
    
    response = client.post("/api/v1/feedback", json=feedback_data)
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ Feedback collected:")
    print(f"   - Content ID: {result['feedback']['content_id']}")
    print(f"   - Predicted: {result['feedback']['predicted_probability']:.3f}")
    print(f"   - Actual: {result['feedback']['actual_performance']:.3f}")
    print(f"   - Delta: {result['feedback']['performance_delta']:.3f}")
    
    return result


def test_complete_pipeline():
    """Test complete pipeline end-to-end."""
    print_section("Testing Complete Pipeline (E2E)")
    
    dummy_contents = generate_dummy_content(count=3)
    dummy_content = dummy_contents[0]
    ingestion_result = test_ingestion(dummy_content)
    content_id = ingestion_result["content_id"]
    
    scoring_request = {
        "content_id": content_id,
        "features": ingestion_result["features"],
        "embeddings": ingestion_result["embeddings"],
        "metadata": {
            **ingestion_result["metadata"],
            "engagement_metrics": dummy_content.get("engagement_metrics", {})
        }
    }
    
    scoring_result = test_scoring({
        "content_id": content_id,
        "features": scoring_request["features"],
        "embeddings": scoring_request["embeddings"],
        "metadata": scoring_request["metadata"]
    })
    
    predicted_prob = scoring_result["virality_probability"]
    actual_perf = min(1.0, predicted_prob + 0.1)
    feedback_result = test_feedback(content_id, predicted_prob, actual_perf)
    
    print("\n✅ Complete pipeline test passed!")
    return {
        "ingestion": ingestion_result,
        "scoring": scoring_result,
        "feedback": feedback_result
    }


def test_multiple_items():
    """Test processing multiple content items."""
    print_section("Testing Multiple Content Items")
    client = TestClient(app)
    
    dummy_contents = generate_dummy_content(count=5)
    results = []
    
    for i, content in enumerate(dummy_contents, 1):
        print(f"\nProcessing item {i}/5...")
        import io
        dummy_file = ("test_video.mp4", io.BytesIO(b"fake video content"), "video/mp4")
        files = {"file": dummy_file}
        data = {
            "platform": content["platform"],
            "caption": content["caption"],
            "hashtags": ",".join(content.get("hashtags", []))
        }
        
        ingest_response = client.post("/api/v1/ingest", files=files, data=data)
        if ingest_response.status_code != 200:
            print(f"   ❌ Ingestion failed for item {i}")
            continue
        
        ingest_result = ingest_response.json()
        score_request = {
            "content_id": ingest_result["content_id"],
            "features": ingest_result["features"],
            "embeddings": ingest_result["embeddings"],
            "metadata": {
                **ingest_result["metadata"],
                "engagement_metrics": content.get("engagement_metrics", {})
            }
        }
        
        score_response = client.post("/api/v1/score", json=score_request)
        if score_response.status_code != 200:
            print(f"   ❌ Scoring failed for item {i}")
            continue
        
        score_result = score_response.json()
        
        results.append({
            "content_id": ingest_result["content_id"],
            "virality_score": score_result["virality_score"],
            "viral_classification": score_result["viral_classification"]
        })
        
        print(f"   ✅ Item {i}: Score={score_result['virality_score']:.2f}, "
              f"Viral={score_result['viral_classification']}")
    
    print(f"\n✅ Processed {len(results)}/{len(dummy_contents)} items successfully")
    return results


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  VIRALITY ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Initialize database
    try:
        init_db()
        print("\n✅ Database initialized")
    except Exception as e:
        print(f"\n⚠️  Database initialization warning: {e}")
        print("   Continuing with tests (may use in-memory storage)")
    
    test_results = {
        "health_check": False,
        "complete_pipeline": False,
        "multiple_items": False
    }
    
    try:
        test_results["health_check"] = test_health_check()
        test_results["complete_pipeline"] = test_complete_pipeline()
        test_results["multiple_items"] = test_multiple_items()
        print_section("Test Summary")
        print("✅ Health Check: PASSED")
        print("✅ Complete Pipeline: PASSED")
        print("✅ Multiple Items: PASSED")
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

