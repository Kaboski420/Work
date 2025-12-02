#!/bin/bash
# Setup script for running Virality Engine on the server

set -e

echo "=========================================="
echo "VIRALITY ENGINE SERVER SETUP"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/api/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📦 Installing/updating dependencies..."
pip3 install -q -r requirements.txt

echo ""
echo "📁 Creating necessary directories..."
mkdir -p models
mkdir -p data/clickhouse_imports
mkdir -p audit_logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model on full ClickHouse data:"
echo "   PYTHONPATH=. python3 scripts/train_on_full_clickhouse.py --limit 10000"
echo ""
echo "2. Run scoring on full dataset:"
echo "   PYTHONPATH=. python3 scripts/score_full_clickhouse.py --limit 10000"
echo ""
echo "3. Start API server:"
echo "   ./start_api.sh"
echo ""

