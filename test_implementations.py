#!/usr/bin/env python3
"""
Quick test script to verify implementations.
Tests key functionality without requiring full infrastructure.
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_media_service():
    """Test Media Intelligence Service."""
    print("\n" + "="*70)
    print("TEST: Media Intelligence Service")
    print("="*70)
    
    try:
        from src.services.media.service import MediaIntelligenceService
        
        service = MediaIntelligenceService()
        print("✅ MediaIntelligenceService initialized")
        
        # Test with dummy frames
        dummy_frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        
        # Test visual analysis
        result = await service.analyze_visual_content(dummy_frames, {})
        print(f"✅ Visual analysis: {list(result.keys())}")
        
        # Test audio analysis
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        audio_result = await service.analyze_audio_content(dummy_audio, 16000, {})
        print(f"✅ Audio analysis: {list(audio_result.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_text_service():
    """Test Text Understanding Service."""
    print("\n" + "="*70)
    print("TEST: Text Understanding Service")
    print("="*70)
    
    try:
        from src.services.text.service import TextUnderstandingService
        
        service = TextUnderstandingService()
        print("✅ TextUnderstandingService initialized")
        
        # Test text analysis
        result = await service.analyze_text_content(
            caption="Amazing viral video! #viral #trending",
            description="Check this out!",
            hashtags=["viral", "trending", "fyp"],
            metadata={"comments": [{"text": "Love it!"}]}
        )
        
        print(f"✅ Text analysis: {list(result.keys())}")
        print(f"   Trend score: {result.get('trend_proximity', {}).get('trend_score', 0):.3f}")
        print(f"   Hook score: {result.get('hook_efficiency', {}).get('hook_score', 0):.3f}")
        print(f"   Comment quality: {result.get('comment_quality', {}).get('quality_score', 0):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ingestion_integration():
    """Test Ingestion Service integration."""
    print("\n" + "="*70)
    print("TEST: Ingestion Service Integration")
    print("="*70)
    
    try:
        from src.services.ingestion.service import IngestionService
        
        service = IngestionService()
        print("✅ IngestionService initialized")
        print(f"   MediaIntelligenceService: {'✅' if hasattr(service, 'media_service') else '❌'}")
        
        # Test feature extraction (without actual content)
        features = await service._extract_features(
            content_type="text",
            content_data=b"dummy",
            metadata={
                "caption": "Test caption",
                "hashtags": ["test"]
            },
            embeddings={}
        )
        
        print(f"✅ Feature extraction: {list(features.keys())}")
        print(f"   Text features: {list(features.get('text', {}).keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_monitoring_service():
    """Test Monitoring Service."""
    print("\n" + "="*70)
    print("TEST: Monitoring Service")
    print("="*70)
    
    try:
        from src.services.monitoring.service import MonitoringService
        
        service = MonitoringService()
        print("✅ MonitoringService initialized")
        
        # Test retraining eligibility
        eligibility = await service.check_retraining_eligibility()
        print(f"✅ Retraining eligibility check: {eligibility.get('eligible', False)}")
        print(f"   Reason: {eligibility.get('reason', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("="*70)
    print("IMPLEMENTATION VERIFICATION TESTS")
    print("="*70)
    
    results = []
    
    results.append(await test_media_service())
    results.append(await test_text_service())
    results.append(await test_ingestion_integration())
    results.append(await test_monitoring_service())
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(results)}")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    print("="*70)
    
    if all(results):
        print("\n✅ ALL IMPLEMENTATIONS VERIFIED!")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED - Check errors above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

