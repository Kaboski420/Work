"""OpenTelemetry instrumentation and observability utilities."""

import logging

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, but make it optional
try:
    from opentelemetry import trace
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    
    OPENTELEMETRY_AVAILABLE = True
    
    # Initialize tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Initialize metrics
    metric_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Observability features will be disabled.")


def setup_observability(app):
    """
    Set up OpenTelemetry instrumentation for FastAPI app.
    
    Must meet criteria: at least 90% of microservices instrumented.
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not installed. Skipping observability setup.")
        return
    
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument HTTP requests
        RequestsInstrumentor().instrument()
        
        logger.info("OpenTelemetry instrumentation configured")
    except Exception as e:
        logger.error(f"Failed to set up observability: {str(e)}")


def get_tracer(name: str):
    """Get a tracer for a specific component."""
    if not OPENTELEMETRY_AVAILABLE:
        return None
    return trace.get_tracer(name)

