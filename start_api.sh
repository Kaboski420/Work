#!/bin/bash
# Start the Virality Engine API

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Virality Engine API"
echo "=========================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import fastapi; import uvicorn; import multipart" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip3 install -q fastapi uvicorn pydantic-settings python-multipart
fi

echo "✓ Dependencies OK"
echo ""
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the server
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

