"""SQLAlchemy models for the Virality Engine."""

from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class ContentItem(Base):
    """Model for content items."""
    __tablename__ = "content_items"
    
    content_id = Column(String(36), primary_key=True)
    platform = Column(String(50), nullable=False)
    content_type = Column(String(50), nullable=False)
    creator_id = Column(String(100))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    # Column name in DB remains "metadata" for backwards compatibility; attribute
    # name avoids clashing with SQLAlchemy's Base.metadata attribute.
    content_metadata = Column("metadata", JSON)
    
    __table_args__ = (
        Index('idx_platform_created', 'platform', 'created_at'),
        Index('idx_creator_id', 'creator_id'),
    )


class ViralityScore(Base):
    """Model for virality scores."""
    __tablename__ = "virality_scores"
    
    id = Column(String(36), primary_key=True)
    content_id = Column(String(36), nullable=False, index=True)
    virality_probability = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)
    attribution_insights = Column(JSON)
    model_version = Column(String(50))
    model_hash = Column(String(64))
    recommendations = Column(JSON)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_content_created', 'content_id', 'created_at'),
        Index('idx_probability', 'virality_probability'),
    )


class EngagementMetrics(Base):
    """Model for engagement metrics time-series data."""
    __tablename__ = "engagement_metrics"
    
    id = Column(String(36), primary_key=True)
    content_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    views = Column(Float, default=0.0)
    likes = Column(Float, default=0.0)
    shares = Column(Float, default=0.0)
    comments = Column(Float, default=0.0)
    additional_metrics = Column(JSON)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_content_timestamp', 'content_id', 'timestamp'),
    )


class FeedbackLoop(Base):
    """Model for feedback loop data for retraining."""
    __tablename__ = "feedback_loop"
    
    id = Column(String(36), primary_key=True)
    content_id = Column(String(36), nullable=False, index=True)
    predicted_probability = Column(Float, nullable=False)
    actual_performance = Column(Float)  # Actual virality metric
    performance_delta = Column(Float)  # Difference between predicted and actual
    feedback_timestamp = Column(DateTime, default=func.now(), nullable=False)
    used_for_training = Column(String(1), default='N')  # Y/N flag
    training_run_id = Column(String(100))
    
    __table_args__ = (
        Index('idx_feedback_timestamp', 'feedback_timestamp'),
        Index('idx_training_flag', 'used_for_training'),
    )



