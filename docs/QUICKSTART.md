# Quick Start Guide

## Prerequisites

- Python 3.10+
- Docker and Docker Compose

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Services

#### Option A: Full Stack
```bash
docker-compose up -d
```

#### Option B: Minimal Test Stack (Recommended)
```bash
docker-compose -f docker-compose.test.yml up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- MinIO (ports 9000, 9002)
- API (port 8000)

### 3. Start API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Ingest Content

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@sample_video.mp4" \
  -F "platform=tiktok" \
  -F "caption=Check out this content!" \
  -F "hashtags=viral,trending"
```

### Score Content

```bash
curl -X POST "http://localhost:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "content-123",
    "features": {
      "visual": {"entropy": 0.7},
      "text": {"trend_proximity": {"trend_score": 0.8}}
    },
    "embeddings": {
      "visual": [0.1, 0.2, ...],
      "audio": [0.2, 0.3, ...],
      "text": [0.3, 0.4, ...]
    },
    "metadata": {
      "platform": "tiktok",
      "engagement_metrics": {
        "views": 10000,
        "likes": 500,
        "shares": 50,
        "comments": 100
      }
    }
  }'
```

## Testing

### Run Tests

```bash
python test_sprint25_complete.py
```

### Docker Testing

```bash
docker-compose -f docker-compose.test.yml up -d
./test_docker.sh
```

## Troubleshooting

### API Not Starting

```bash
# Install dependencies
pip install -r requirements.txt

# Check if app imports correctly
python -c "from src.api.main import app; print('OK')"
```

### Port Conflicts

Change port in uvicorn command:
```bash
uvicorn src.api.main:app --port 8001
```

### Database Connection

Check PostgreSQL is running:
```bash
docker ps
```
