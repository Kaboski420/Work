# Virality Prediction Engine

Predicts viral potential of multimedia content across social platforms (TikTok, Instagram, YouTube Shorts, Reels).

## Overview

The system analyzes video, audio, and text content to generate virality scores using machine learning and rule-based algorithms.

## Core Services

1. **Ingestion Service** - Processes content and extracts features
2. **Media Service** - Analyzes video and audio
3. **Text Service** - Processes captions, descriptions, and comments
4. **Temporal Service** - Predicts engagement patterns over time
5. **Scoring Service** - Calculates virality scores
6. **Monitoring Service** - Tracks system health and model performance

## Technology Stack

- **API**: FastAPI, gRPC
- **Databases**: PostgreSQL, ClickHouse, Redis
- **Storage**: MinIO
- **ML**: PyTorch, HuggingFace, scikit-learn
- **Monitoring**: Prometheus, Grafana
- **Orchestration**: Kubernetes, Docker Compose

## Quick Start

### Prerequisites
- Python 3.10+
- Docker and Docker Compose

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start services:
```bash
docker-compose -f docker-compose.test.yml up -d
```

3. Test the API:
```bash
curl http://localhost:8000/health
```

4. Run tests:
```bash
python test_sprint25_complete.py
```

For detailed setup instructions, see `docs/QUICKSTART.md`.

## Documentation

- **Architecture**: `docs/ARCHITECTURE.md`
- **Quick Start**: `docs/QUICKSTART.md`
- **Acceptance Criteria**: `docs/acceptance_criteria.md`
- **Testing Guide**: See `QUICK_TEST.sh` for automated testing

