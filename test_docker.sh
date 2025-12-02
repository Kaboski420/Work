#!/bin/bash
# Simple test script to run in Docker environment

echo "=========================================="
echo "Testing Virality Engine in Docker"
echo "=========================================="

# Wait for API to be ready
echo "Waiting for API to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
API_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        API_READY=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES: API not ready yet, waiting..."
    sleep 2
done

if [ "$API_READY" = false ]; then
    echo "ERROR: API did not become ready after $MAX_RETRIES attempts"
    exit 1
fi

echo "API is ready!"

# Test health endpoint
echo ""
echo "1. Testing Health Check..."
curl -s http://localhost:8000/health | python3 -m json.tool || echo "Health check failed"

# Test with dummy data
echo ""
echo "2. Testing Content Ingestion..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@/dev/null" \
  -F "platform=tiktok" \
  -F "caption=Test video #viral" \
  -F "hashtags=viral,trending" 2>/dev/null)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo "$BODY" | python3 -m json.tool
else
    echo "Ingestion test failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi

echo ""
echo "3. Testing Virality Scoring..."
# Create test JSON with proper arrays (using Python to generate arrays)
python3 << 'PYEOF' > /tmp/test_score.json
import json

test_data = {
    "content_id": "test-123",
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
    "metadata": {
        "platform": "tiktok",
        "engagement_metrics": {
            "views": 10000,
            "likes": 500,
            "shares": 50,
            "comments": 100,
            "saves": 200
        }
    }
}
print(json.dumps(test_data))
PYEOF

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_score.json 2>/dev/null)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo "$BODY" | python3 -m json.tool
else
    echo "Scoring test failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi

echo ""
echo "4. Testing Feedback Collection..."
cat > /tmp/test_feedback.json << 'EOF'
{
  "content_id": "test-123",
  "predicted_probability": 0.75,
  "actual_performance": 0.82
}
EOF

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_feedback.json 2>/dev/null)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo "$BODY" | python3 -m json.tool
else
    echo "Feedback test failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi

echo ""
echo "=========================================="
echo "Tests Complete"
echo "=========================================="

