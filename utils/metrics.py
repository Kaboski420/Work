"""Prometheus metrics for the Virality Engine."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Metrics will be disabled.")
    
    # Create dummy classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, value=1):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass
    
    CONTENT_TYPE_LATEST = "text/plain"
    
    def generate_latest(registry):
        return b"# Prometheus metrics not available\n"
    
    class CollectorRegistry:
        pass

# Create a custom registry
if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()
else:
    registry = None

# Request metrics (only create if prometheus_client is available)
if PROMETHEUS_AVAILABLE:
    http_requests_total = Counter(
        'http_requests_total',
        'Total number of HTTP requests',
        ['method', 'endpoint', 'status'],
        registry=registry
    )
else:
    http_requests_total = Counter()

if PROMETHEUS_AVAILABLE:
    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint'],
        registry=registry
    )
    
    # Content processing metrics
    content_ingested_total = Counter(
        'content_ingested_total',
        'Total number of content items ingested',
        ['platform', 'content_type'],
        registry=registry
    )
    
    content_scored_total = Counter(
        'content_scored_total',
        'Total number of content items scored',
        ['platform'],
        registry=registry
    )
    
    # Virality prediction metrics
    virality_probability = Histogram(
        'virality_probability',
        'Distribution of virality probability scores',
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        registry=registry
    )
    
    prediction_confidence = Histogram(
        'prediction_confidence',
        'Distribution of prediction confidence scores',
        buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        registry=registry
    )
    
    prediction_latency_seconds = Histogram(
        'prediction_latency_seconds',
        'Time taken to generate predictions',
        ['operation'],  # ingest, score, analyze
        registry=registry
    )
    
    # Model metrics
    model_version = Gauge(
        'model_version_info',
        'Model version information',
        ['version', 'hash'],
        registry=registry
    )
    
    # System metrics
    active_connections = Gauge(
        'active_connections',
        'Number of active connections',
        registry=registry
    )
    
    cache_hits_total = Counter(
        'cache_hits_total',
        'Total number of cache hits',
        ['cache_type'],
        registry=registry
    )
    
    cache_misses_total = Counter(
        'cache_misses_total',
        'Total number of cache misses',
        ['cache_type'],
        registry=registry
    )
    
    # Error metrics
    errors_total = Counter(
        'errors_total',
        'Total number of errors',
        ['error_type', 'endpoint'],
        registry=registry
    )
    
    # Feature extraction metrics
    embedding_extraction_duration_seconds = Histogram(
        'embedding_extraction_duration_seconds',
        'Time taken to extract embeddings',
        ['embedding_type'],  # visual, audio, text, contextual
        registry=registry
    )
else:
    # Create dummy metrics when prometheus_client is not available
    http_request_duration_seconds = Histogram()
    content_ingested_total = Counter()
    content_scored_total = Counter()
    virality_probability = Histogram()
    prediction_confidence = Histogram()
    prediction_latency_seconds = Histogram()
    model_version = Gauge()
    active_connections = Gauge()
    cache_hits_total = Counter()
    cache_misses_total = Counter()
    errors_total = Counter()
    embedding_extraction_duration_seconds = Histogram()


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    if PROMETHEUS_AVAILABLE and registry:
        return generate_latest(registry)
    else:
        return b"# Prometheus metrics not available\n# Install prometheus_client to enable metrics\n"


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST

