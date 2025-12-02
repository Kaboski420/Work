"""
Temporal Behavior Modeling Service

Predicts time-based audience resonance using open-source forecasting.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.config import settings
from src.utils.timeseries import TimeSeriesService

logger = logging.getLogger(__name__)

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Temporal modeling will be limited.")


class TemporalModelingService:
    """Service for temporal behavior modeling and forecasting."""
    
    def __init__(self):
        self.service_id = str(uuid.uuid4())
        
        # Initialize ClickHouse service
        self.timeseries = TimeSeriesService(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_db,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure
        )
        
        logger.info(f"Initialized TemporalModelingService: {self.service_id}")
    
    async def predict_engagement_trajectory(
        self,
        content_id: str,
        initial_metrics: Dict[str, float],
        historical_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict engagement trajectory over time.
        
        Args:
            content_id: Content identifier
            initial_metrics: Initial engagement metrics (views, likes, shares, etc.)
            historical_patterns: Historical engagement patterns for similar content
        
        Returns:
            Dictionary containing:
            - velocity: First-30-minute reaction velocity
            - plateau_detection: Saturation plateau detection
            - decay_curve: Decay curve analysis
            - rebound_probability: Probability of engagement rebound
            - expansion_index: Organic audience acceleration index
        """
        logger.info(f"Predicting engagement trajectory for {content_id}")
        
        # Calculate velocity metrics
        velocity = await self._calculate_velocity(initial_metrics)
        
        # Detect plateau
        plateau = await self._detect_plateau(initial_metrics, historical_patterns)
        
        # Predict decay curve
        decay_curve = await self._predict_decay_curve(initial_metrics, historical_patterns)
        
        # Calculate rebound probability
        rebound_prob = await self._calculate_rebound_probability(
            initial_metrics, historical_patterns
        )
        
        # Calculate expansion index
        expansion_index = await self._calculate_expansion_index(
            initial_metrics, historical_patterns
        )
        
        result = {
            "content_id": content_id,
            "timestamp": datetime.utcnow().isoformat(),
            "velocity": velocity,
            "plateau_detection": plateau,
            "decay_curve": decay_curve,
            "rebound_probability": rebound_prob,
            "expansion_index": expansion_index
        }
        
        logger.info(f"Generated trajectory prediction for {content_id}")
        return result
    
    async def _calculate_velocity(
        self,
        initial_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate first-30-minute reaction velocity and 60-minute momentum score.
        
        Returns:
            Dictionary with velocity metrics and momentum score
        """
        content_id = initial_metrics.get("content_id", "")
        
        # Calculate 30-minute velocity
        velocity = self.timeseries.get_velocity(
            content_id=content_id,
            minutes=30
        )
        
        # Calculate 60-minute momentum score (Algorithm 1 requirement)
        momentum_velocity = self.timeseries.get_velocity(
            content_id=content_id,
            minutes=60
        )
        
        # Momentum Score: normalized growth rate over 60 minutes
        momentum_score = 0.0
        if momentum_velocity.get("views_per_minute", 0) > 0:
            # Normalize momentum (higher = better)
            views_per_min = momentum_velocity["views_per_minute"]
            likes_per_min = momentum_velocity["likes_per_minute"]
            
            # Combine views and likes momentum
            momentum_score = (
                min(views_per_min / 100.0, 1.0) * 0.6 +  # Views momentum (normalized)
                min(likes_per_min / 10.0, 1.0) * 0.4     # Likes momentum (normalized)
            )
        
        velocity["momentum_score"] = float(momentum_score)
        velocity["momentum_60min"] = momentum_velocity
        
        # Calculate overall velocity score (weighted average)
        velocity["overall_velocity_score"] = (
            velocity["views_per_minute"] * 0.4 +
            velocity["likes_per_minute"] * 0.3 +
            velocity["shares_per_minute"] * 0.2 +
            velocity["comments_per_minute"] * 0.1
        )
        
        return velocity
    
    async def _detect_plateau(
        self,
        initial_metrics: Dict[str, float],
        historical_patterns: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Detect saturation plateau in engagement.
        
        Uses historical patterns and velocity to estimate plateau.
        
        Returns:
            Dictionary with plateau detection results
        """
        try:
            # Get historical data if available
            if historical_patterns:
                # Convert to DataFrame
                df = pd.DataFrame(historical_patterns)
                if 'timestamp' in df.columns and 'views' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Calculate growth rate
                    if len(df) > 1:
                        # Calculate rolling mean to smooth
                        df['views_rolling'] = df['views'].rolling(window=min(5, len(df)), center=True).mean()
                        
                        # Calculate rate of change
                        df['growth_rate'] = df['views_rolling'].pct_change()
                        
                        # Detect plateau (growth rate < threshold)
                        threshold = 0.01  # 1% growth rate threshold
                        recent_growth = df['growth_rate'].tail(3).mean()
                        
                        if recent_growth < threshold and recent_growth > -threshold:
                            # Plateau detected
                            plateau_level = df['views'].max()
                            estimated_time = datetime.utcnow() + timedelta(hours=12)  # Estimate from pattern
                            
                            return {
                                "detected": True,
                                "estimated_time_to_plateau": estimated_time.isoformat(),
                                "plateau_level": float(plateau_level),
                                "confidence": min(abs(recent_growth) / threshold, 1.0)
                            }
            
            # Use velocity to estimate plateau
            velocity = initial_metrics.get("views_per_minute", 0.0)
            current_views = initial_metrics.get("views", 0.0)
            
            # Estimate plateau based on velocity decay
            if velocity > 0:
                # Assume exponential decay
                # Plateau reached when velocity < 1% of initial
                decay_rate = 0.95  # 5% decay per hour
                hours_to_plateau = -np.log(0.01) / np.log(1 / decay_rate)  # ~90 hours
                estimated_time = datetime.utcnow() + timedelta(hours=hours_to_plateau)
                
                # Estimate plateau level (integrate velocity decay)
                plateau_level = current_views + velocity * 60 * (1 / (1 - decay_rate))
                
                return {
                    "detected": False,
                    "estimated_time_to_plateau": estimated_time.isoformat(),
                    "plateau_level": float(plateau_level),
                    "confidence": 0.5  # Medium confidence from velocity
                }
            
            # No data available
            return {
                "detected": False,
                "estimated_time_to_plateau": None,
                "plateau_level": None,
                "confidence": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error detecting plateau: {e}", exc_info=True)
            return {
                "detected": False,
                "estimated_time_to_plateau": None,
                "plateau_level": None,
                "confidence": 0.0
            }
    
    async def _predict_decay_curve(
        self,
        initial_metrics: Dict[str, float],
        historical_patterns: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Predict decay curve for engagement using Prophet.
        
        Returns:
            Dictionary with decay curve parameters
        """
        try:
            current_views = initial_metrics.get("views", 0.0)
            
            # Prepare data for Prophet
            if historical_patterns and len(historical_patterns) > 2:
                # Use historical data if available
                df = pd.DataFrame(historical_patterns)
                if 'timestamp' in df.columns and 'views' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Prepare Prophet format
                    prophet_df = pd.DataFrame({
                        'ds': df['timestamp'],
                        'y': df['views']
                    })
                    
                    if PROPHET_AVAILABLE:
                        # Fit Prophet model
                        model = Prophet(
                            yearly_seasonality=False,
                            weekly_seasonality=True,
                            daily_seasonality=True,
                            changepoint_prior_scale=0.05  # Lower for smoother decay
                        )
                        model.fit(prophet_df)
                        
                        # Forecast future
                        future = model.make_future_dataframe(periods=7*24, freq='H')  # 7 days hourly
                        forecast = model.predict(future)
                        
                        # Calculate decay rate (from recent trend)
                        recent_forecast = forecast.tail(48)  # Last 48 hours
                        decay_rate = (recent_forecast['yhat'].iloc[-1] - recent_forecast['yhat'].iloc[0]) / recent_forecast['yhat'].iloc[0]
                        
                        # Half-life (time for 50% decay)
                        if decay_rate < 0:
                            half_life_hours = -np.log(0.5) / abs(decay_rate) * 24
                        else:
                            half_life_hours = None
                        
                        # Projections
                        projected_24h = float(forecast[forecast['ds'] <= datetime.utcnow() + timedelta(hours=24)]['yhat'].iloc[-1])
                        projected_7d = float(forecast[forecast['ds'] <= datetime.utcnow() + timedelta(days=7)]['yhat'].iloc[-1])
                        
                        return {
                            "decay_rate": float(decay_rate),
                            "half_life_hours": float(half_life_hours) if half_life_hours else None,
                            "projected_engagement_24h": projected_24h,
                            "projected_engagement_7d": projected_7d
                        }
            
            # Fallback: Use exponential decay model
            velocity = initial_metrics.get("views_per_minute", 0.0)
            if velocity > 0 and current_views > 0:
                # Estimate exponential decay: views(t) = views(0) * e^(-decay_rate * t)
                # Decay rate based on velocity pattern
                decay_rate = 0.05  # 5% per hour
                
                # Half-life
                half_life_hours = np.log(2) / decay_rate
                
                # Projections
                projected_24h = current_views * np.exp(-decay_rate * 24) + velocity * 60 * 24
                projected_7d = current_views * np.exp(-decay_rate * 7 * 24) + velocity * 60 * 7 * 24
                
                return {
                    "decay_rate": decay_rate,
                    "half_life_hours": float(half_life_hours),
                    "projected_engagement_24h": float(projected_24h),
                    "projected_engagement_7d": float(projected_7d)
                }
            
            # No data available
            return {
                "decay_rate": 0.0,
                "half_life_hours": None,
                "projected_engagement_24h": None,
                "projected_engagement_7d": None
            }
            
        except Exception as e:
            logger.error(f"Error predicting decay curve: {e}", exc_info=True)
            return {
                "decay_rate": 0.0,
                "half_life_hours": None,
                "projected_engagement_24h": None,
                "projected_engagement_7d": None
            }
    
    async def _calculate_rebound_probability(
        self,
        initial_metrics: Dict[str, float],
        historical_patterns: Optional[List[Dict[str, Any]]]
    ) -> float:
        """
        Calculate probability of engagement rebound.
        
        Uses historical patterns to detect rebound patterns.
        
        Returns:
            Probability value between 0 and 1
        """
        try:
            # Analyze historical patterns for rebound indicators
            if historical_patterns and len(historical_patterns) > 3:
                df = pd.DataFrame(historical_patterns)
                if 'timestamp' in df.columns and 'views' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Calculate velocity changes
                    df['views_diff'] = df['views'].diff()
                    df['velocity_change'] = df['views_diff'].diff()
                    
                    # Detect rebound pattern (negative velocity followed by positive)
                    negative_velocity = df['views_diff'] < 0
                    positive_acceleration = df['velocity_change'] > 0
                    
                    # Count rebound patterns
                    rebound_count = 0
                    for i in range(1, len(df)):
                        if negative_velocity.iloc[i-1] and positive_acceleration.iloc[i]:
                            rebound_count += 1
                    
                    # Calculate probability
                    if len(df) > 3:
                        rebound_probability = min(rebound_count / (len(df) - 2), 0.8)  # Cap at 0.8
                        return float(rebound_probability)
            
            # Use velocity metrics to estimate rebound
            velocity = initial_metrics.get("views_per_minute", 0.0)
            likes = initial_metrics.get("likes", 0.0)
            shares = initial_metrics.get("shares", 0.0)
            views = initial_metrics.get("views", 0.0)
            
            # Higher engagement ratio suggests potential for rebound
            if views > 0:
                engagement_ratio = (likes + shares * 2) / views  # Shares weighted more
                
                # Velocity decline suggests potential rebound if engagement is high
                if velocity > 0:
                    # Lower velocity with high engagement = higher rebound probability
                    rebound_probability = min(engagement_ratio * 2, 0.6)
                    return float(rebound_probability)
            
            # Default: low probability
            return 0.2
            
        except Exception as e:
            logger.error(f"Error calculating rebound probability: {e}", exc_info=True)
            return 0.2
    
    async def _calculate_expansion_index(
        self,
        initial_metrics: Dict[str, float],
        historical_patterns: Optional[List[Dict[str, Any]]]
    ) -> float:
        """
        Calculate organic audience acceleration index.
        
        Measures how fast engagement is growing organically.
        
        Returns:
            Expansion index value (0-1, higher = faster expansion)
        """
        try:
            views = initial_metrics.get("views", 0.0)
            likes = initial_metrics.get("likes", 0.0)
            shares = initial_metrics.get("shares", 0.0)
            comments = initial_metrics.get("comments", 0.0)
            views_per_minute = initial_metrics.get("views_per_minute", 0.0)
            
            # Calculate engagement velocity components
            engagement_velocity = views_per_minute
            
            # Calculate share rate (viral coefficient)
            if views > 0:
                share_rate = shares / views
            else:
                share_rate = 0.0
            
            # Calculate comment rate (discussion/engagement)
            if views > 0:
                comment_rate = comments / views
            else:
                comment_rate = 0.0
            
            # Calculate like rate (general appeal)
            if views > 0:
                like_rate = likes / views
            else:
                like_rate = 0.0
            
            # Expansion index combines:
            # 1. High velocity (fast growth)
            # 2. High share rate (viral spread)
            # 3. High engagement rates (active audience)
            
            # Normalize velocity (assuming max 1000 views/min)
            normalized_velocity = min(engagement_velocity / 1000.0, 1.0)
            
            # Weighted combination
            expansion_index = (
                normalized_velocity * 0.4 +      # Velocity weight
                min(share_rate * 10, 1.0) * 0.3 +  # Share rate weight (cap at 0.1 share rate)
                min(comment_rate * 20, 1.0) * 0.2 +  # Comment rate weight (cap at 0.05 comment rate)
                min(like_rate * 5, 1.0) * 0.1     # Like rate weight (cap at 0.2 like rate)
            )
            
            # Use historical patterns to refine
            if historical_patterns and len(historical_patterns) > 2:
                df = pd.DataFrame(historical_patterns)
                if 'timestamp' in df.columns and 'views' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Calculate acceleration (rate of velocity change)
                    if len(df) > 1:
                        df['views_diff'] = df['views'].diff()
                        df['acceleration'] = df['views_diff'].diff()
                        
                        # Recent acceleration
                        recent_acceleration = df['acceleration'].tail(3).mean()
                        
                        # Boost index if accelerating
                        if recent_acceleration > 0:
                            expansion_index = min(expansion_index * 1.2, 1.0)
            
            return float(min(max(expansion_index, 0.0), 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating expansion index: {e}", exc_info=True)
            return 0.0

