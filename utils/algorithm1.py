"""Algorithm 1 (Automatic Virality Detection) utilities."""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("numpy not available. Some features will be limited.")


# Algorithm 1 Thresholds (configurable)
THRESHOLDS = {
    "engagement_rate": {
        "high": 0.10,  # 10% engagement rate
        "medium": 0.05,  # 5% engagement rate
        "low": 0.02  # 2% engagement rate
    },
    "momentum_score": {
        "high": 0.8,  # High momentum
        "medium": 0.5,  # Medium momentum
        "low": 0.2  # Low momentum
    },
    "saves_likes_ratio": {
        "high": 0.15,  # 15% saves/likes ratio
        "medium": 0.08,  # 8% saves/likes ratio
        "low": 0.03  # 3% saves/likes ratio
    },
    "comment_quality": {
        "high": 0.75,  # High quality comments
        "medium": 0.50,  # Medium quality
        "low": 0.25  # Low quality
    },
    "viral_classification": 0.6  # Threshold for viral yes/no (0-1 probability)
}


def calculate_engagement_rate(
    views: float,
    likes: float,
    shares: float,
    comments: float
) -> float:
    """Calculate engagement rate: (Likes + Shares + Comments) / Views"""
    if views <= 0:
        return 0.0
    
    total_engagements = likes + shares + comments
    engagement_rate = total_engagements / views
    
    return float(min(engagement_rate, 1.0))


def calculate_saves_likes_ratio(
    saves: float,
    likes: float
) -> float:
    """Calculate saves/likes ratio."""
    if likes <= 0:
        return 0.0
    
    ratio = saves / likes
    return float(min(ratio, 1.0))


def rule_based_scoring(
    engagement_rate: float,
    momentum_score: float,
    saves_likes_ratio: float,
    comment_quality_score: float,
    visual_entropy: float = 0.0,
    text_trend_score: float = 0.0,
    hook_efficiency: float = 0.0
) -> Tuple[float, bool]:
    """Rule-based scoring with thresholds (Algorithm 1, Part A)."""
    score = 0.0
    points = 0.0
    max_points = 0.0
    
    max_points += 30
    if engagement_rate >= THRESHOLDS["engagement_rate"]["high"]:
        points += 30
    elif engagement_rate >= THRESHOLDS["engagement_rate"]["medium"]:
        points += 20
    elif engagement_rate >= THRESHOLDS["engagement_rate"]["low"]:
        points += 10
    else:
        points += 5
    
    max_points += 25
    if momentum_score >= THRESHOLDS["momentum_score"]["high"]:
        points += 25
    elif momentum_score >= THRESHOLDS["momentum_score"]["medium"]:
        points += 15
    elif momentum_score >= THRESHOLDS["momentum_score"]["low"]:
        points += 8
    else:
        points += 3
    
    max_points += 20
    if saves_likes_ratio >= THRESHOLDS["saves_likes_ratio"]["high"]:
        points += 20
    elif saves_likes_ratio >= THRESHOLDS["saves_likes_ratio"]["medium"]:
        points += 12
    elif saves_likes_ratio >= THRESHOLDS["saves_likes_ratio"]["low"]:
        points += 6
    else:
        points += 2
    
    max_points += 15
    if comment_quality_score >= THRESHOLDS["comment_quality"]["high"]:
        points += 15
    elif comment_quality_score >= THRESHOLDS["comment_quality"]["medium"]:
        points += 9
    elif comment_quality_score >= THRESHOLDS["comment_quality"]["low"]:
        points += 4
    else:
        points += 1
    
    max_points += 10
    if visual_entropy >= 0.7:
        points += 10
    elif visual_entropy >= 0.4:
        points += 6
    else:
        points += 2
    
    max_points += 10
    if text_trend_score >= 0.7:
        points += 10
    elif text_trend_score >= 0.4:
        points += 6
    else:
        points += 2
    
    max_points += 10
    if hook_efficiency >= 0.7:
        points += 10
    elif hook_efficiency >= 0.4:
        points += 6
    else:
        points += 2
    
    if max_points > 0:
        score = (points / max_points) * 100
    else:
        score = 0.0
    
    is_viral = score >= (THRESHOLDS["viral_classification"] * 100)
    
    return float(min(max(score, 0.0), 100.0)), bool(is_viral)


def classify_virality_dynamics(
    engagement_rate: float,
    reach_metrics: Dict[str, float],
    momentum_score: float,
    depth_threshold: float = 0.08,
    breadth_threshold: float = 50000.0
) -> str:
    """Classify virality dynamics: Deep, Broad, or Hybrid."""
    views = reach_metrics.get("views", 0.0)
    
    high_engagement = engagement_rate >= depth_threshold
    high_reach = views >= breadth_threshold
    high_momentum = momentum_score >= THRESHOLDS["momentum_score"]["high"]
    
    if high_engagement and high_reach:
        return "hybrid"
    elif high_engagement and not high_reach:
        return "deep"
    elif high_reach and not high_engagement:
        return "broad"
    elif high_momentum:
        # If momentum is high, likely hybrid or deep
        if engagement_rate >= depth_threshold * 0.7:
            return "hybrid"
        else:
            return "deep"
    else:
        # Default based on engagement rate
        if engagement_rate >= depth_threshold * 0.5:
            return "deep"
        else:
            return "broad"


def calculate_confidence_interval(
    probability: float,
    confidence_level: float,
    method: str = "standard"
) -> Dict[str, float]:
    """Calculate confidence interval for virality probability."""
    if method == "standard":
        margin = (1 - confidence_level) / 2
        lower = max(0.0, probability - margin)
        upper = min(1.0, probability + margin)
        
        if confidence_level > 0.8:
            interval_width = 0.1
        elif confidence_level > 0.6:
            interval_width = 0.15
        else:
            interval_width = 0.2
        
        lower = max(0.0, probability - interval_width / 2)
        upper = min(1.0, probability + interval_width / 2)
        
        return {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": float(confidence_level)
        }
    else:
        return {
            "lower": max(0.0, probability - 0.1),
            "upper": min(1.0, probability + 0.1),
            "confidence_level": float(confidence_level)
        }


def check_training_data_availability(
    model_info: Dict[str, Any],
    min_samples: int = 100
) -> bool:
    """Check if sufficient training data is available for ML model."""
    if not model_info:
        return False
    
    model_hash = model_info.get("hash", "default")
    if model_hash == "default":
        return False
    
    model_version = model_info.get("version", "0.0.0")
    try:
        major, minor, patch = map(int, model_version.split(".")[:3])
        if major > 0 or (major == 0 and minor > 1):
            return True
    except:
        pass
    
    training_timestamp = model_info.get("training_timestamp")
    if training_timestamp and training_timestamp != datetime.utcnow().isoformat():
        return True
    
    return False

