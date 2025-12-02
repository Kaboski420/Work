#!/bin/bash
# Quick Test Script for Virality Engine

echo "🚀 Virality Engine - Quick Test"
echo "================================"
echo ""

# Step 1: Check if Docker is running
echo "1. Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi
echo "✅ Docker is running"
echo ""

# Step 2: Start services
echo "2. Starting services..."
docker-compose -f docker-compose.test.yml up -d
echo "✅ Services starting..."
sleep 5
echo ""

# Step 3: Wait for API
echo "3. Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    echo "   Attempt $i/30: Waiting..."
    sleep 2
done
echo ""

# Step 4: Test health endpoint
echo "4. Testing health endpoint..."
HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo "✅ Health check passed"
    echo "   Response: $HEALTH"
else
    echo "❌ Health check failed"
    echo "   Response: $HEALTH"
fi
echo ""

# Step 5: Test scoring endpoint
echo "5. Testing scoring endpoint..."
SCORE_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "test-123",
    "features": {"visual": {"entropy": 0.7}, "audio": {"bpm": 120.0}, "text": {"trend_score": 0.6}},
    "embeddings": {"visual": [0.1, 0.2], "audio": [0.2, 0.3], "text": [0.3, 0.4]},
    "metadata": {"platform": "tiktok", "engagement_metrics": {"views": 1000, "likes": 100, "shares": 50, "comments": 25}}
  }')

if echo "$SCORE_RESPONSE" | grep -q "virality_score"; then
    echo "✅ Scoring endpoint works"
    echo "   Score: $(echo "$SCORE_RESPONSE" | grep -o '"virality_score":[0-9.]*' | cut -d: -f2)"
else
    echo "❌ Scoring endpoint failed"
    echo "   Response: $SCORE_RESPONSE"
fi
echo ""

# Step 6: Run test suite
echo "6. Running test suite..."
if command -v python3 &> /dev/null; then
    python3 test_sprint25_complete.py 2>&1 | tail -10
else
    echo "⚠️  Python3 not found, skipping test suite"
fi
echo ""

echo "================================"
echo "✅ Quick test complete!"
echo ""
echo "Next steps:"
echo "- View API docs: http://localhost:8000/docs"
echo "- Run full tests: python test_sprint25_complete.py"
echo "- Performance test: python scripts/performance_test.py"
