"""Configuration management for the Virality Engine."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    service_name: str = "virality-engine"
    service_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "virality_engine"
    postgres_user: str = "virality_user"
    postgres_password: str = "changeme"
    
    clickhouse_host: str = "clickhouse.bragmant.noooo.art"
    clickhouse_port: int = 443
    clickhouse_db: str = "crawler"
    clickhouse_user: str = "dev2"
    clickhouse_password: str = "730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102"  # SHA256 hash
    clickhouse_secure: bool = True  # Set to True for SSL/HTTPS connections
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "virality-features"
    minio_secure: bool = False
    
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_ingestion_topic: str = "content-ingestion"
    kafka_scoring_topic: str = "scoring-requests"
    kafka_results_topic: str = "scoring-results"
    
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "virality-prediction"
    
    feast_repo_path: str = "./feature_store"
    
    model_cache_dir: str = "./models"
    device: str = "cuda"
    batch_size: int = 32
    
    max_inference_latency_ms: int = 2000
    target_throughput_ops_per_sec: int = 100
    
    keycloak_url: str = "http://localhost:8080"
    keycloak_realm: str = "virality-engine"
    jwt_secret_key: Optional[str] = None
    encryption_key: Optional[str] = None
    
    prometheus_port: int = 9090
    otel_service_name: str = "virality-engine"
    otel_exporter_otlp_endpoint: Optional[str] = None
    
    retraining_interval_days: int = 14
    drift_detection_threshold: float = 0.1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()



