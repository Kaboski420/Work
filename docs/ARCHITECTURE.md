# System Architecture

## Overview

The Virality Prediction Engine analyzes multimedia content to predict viral potential. It uses a microservice architecture with six core services.

## Core Services

### 1. Ingestion Service
- **Location**: `src/services/ingestion/`
- **Purpose**: Processes uploaded content and extracts features
- **Output**: Features and embeddings stored in object storage

### 2. Media Service
- **Location**: `src/services/media/`
- **Purpose**: Analyzes video and audio content
- **Output**: Visual and audio analytics

### 3. Text Service
- **Location**: `src/services/text/`
- **Purpose**: Analyzes captions, descriptions, and comments
- **Output**: Text analytics and comment quality scores

### 4. Temporal Service
- **Location**: `src/services/temporal/`
- **Purpose**: Predicts engagement patterns over time
- **Output**: Momentum scores and engagement forecasts

### 5. Scoring Service
- **Location**: `src/services/scoring/`
- **Purpose**: Calculates virality scores using rule-based and ML models
- **Output**: Virality scores with confidence intervals

### 6. Monitoring Service
- **Location**: `src/services/monitoring/`
- **Purpose**: Tracks system health and model performance
- **Output**: Metrics, alerts, and drift detection

## Data Flow

```
Content Upload → Ingestion → Feature Extraction → Scoring → Results
```

## Infrastructure

### Databases
- **PostgreSQL**: Stores content metadata and scores
- **ClickHouse**: Stores time-series engagement data
- **Redis**: Caches scoring results for fast retrieval
- **MinIO**: Stores feature vectors and embeddings

### Messaging
- **Kafka**: Handles asynchronous communication between services

### Deployment
- **Docker Compose**: Local development and testing
- **Kubernetes**: Production deployment

## API Endpoints

### REST API
- `GET /health` - Health check
- `POST /api/v1/ingest` - Upload and process content
- `POST /api/v1/score` - Calculate virality score
- `POST /api/v1/analyze` - Complete analysis (ingest + score)
- `POST /api/v1/feedback` - Submit feedback for model improvement

### Response Format
```json
{
  "virality_score": 0-100,
  "virality_probability": 0-1,
  "viral_classification": "yes" | "no",
  "confidence_interval": {"lower": 0-1, "upper": 0-1},
  "model_lineage": {"hash": "...", "version": "..."},
  "virality_dynamics": "deep" | "broad" | "hybrid"
}
```

## Security

- **Authentication**: Keycloak with role-based access control
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Audit Logs**: Immutable logs for compliance

## Performance Targets

- **Latency**: < 2 seconds at 95th percentile
- **Throughput**: ≥ 100 scoring operations per second
- **Storage**: ≤ 50 KB per minute of processed content
