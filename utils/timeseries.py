"""ClickHouse integration for time-series engagement metrics."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Try to import clickhouse
try:
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    logger.warning("clickhouse_connect not available. Time-series storage will be disabled.")


class TimeSeriesService:
    """ClickHouse service for storing and querying engagement metrics."""
    
    def __init__(self, host: str = "localhost", port: int = 8123,
                 database: str = "metrics", username: str = "default",
                 password: str = "changeme", secure: bool = False):
        """
        Initialize ClickHouse service.
        
        Args:
            host: ClickHouse host
            port: ClickHouse HTTP port
            database: Database name
            username: Username
            password: Password
            secure: Use HTTPS/SSL connection (default: False)
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.secure = secure
        self.client = None
        
        if CLICKHOUSE_AVAILABLE:
            try:
                self.client = clickhouse_connect.get_client(
                    host=host,
                    port=port,
                    database=database,
                    username=username,
                    password=password,
                    secure=secure
                )
                self._ensure_table()
                protocol = "https" if secure else "http"
                logger.info(f"ClickHouse connected: {protocol}://{host}:{port}/{database}")
            except Exception as e:
                logger.warning(f"ClickHouse not available: {e}. Time-series storage disabled.")
                self.client = None
        else:
            logger.warning("ClickHouse library not available. Time-series storage disabled.")
    
    def _ensure_table(self):
        """Ensure engagement_metrics table exists."""
        if not self.client:
            return
        
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS engagement_metrics (
                id String,
                content_id String,
                timestamp DateTime,
                views Float64,
                likes Float64,
                shares Float64,
                comments Float64,
                additional_metrics String,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (content_id, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            """
            self.client.command(create_table_query)
            logger.info("Engagement metrics table ensured")
        except Exception as e:
            logger.error(f"Error ensuring table: {e}")
    
    def store_metrics(
        self,
        content_id: str,
        timestamp: datetime,
        views: float = 0.0,
        likes: float = 0.0,
        shares: float = 0.0,
        comments: float = 0.0,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store engagement metrics.
        
        Args:
            content_id: Content identifier
            timestamp: Metric timestamp
            views: View count
            likes: Like count
            shares: Share count
            comments: Comment count
            additional_metrics: Additional metrics as dict
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            import uuid
            import json
            
            metric_id = str(uuid.uuid4())
            additional_str = json.dumps(additional_metrics) if additional_metrics else ""
            
            insert_query = """
            INSERT INTO engagement_metrics 
            (id, content_id, timestamp, views, likes, shares, comments, additional_metrics)
            VALUES
            """
            
            data = [[
                metric_id,
                content_id,
                timestamp,
                views,
                likes,
                shares,
                comments,
                additional_str
            ]]
            
            self.client.insert("engagement_metrics", data)
            logger.debug(f"Stored metrics for {content_id} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            return False
    
    def get_metrics(
        self,
        content_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get engagement metrics for content.
        
        Args:
            content_id: Content identifier
            start_time: Start time (optional)
            end_time: End time (optional)
            
        Returns:
            List of metric records
        """
        if not self.client:
            return []
        
        try:
            query = f"""
            SELECT timestamp, views, likes, shares, comments, additional_metrics
            FROM engagement_metrics
            WHERE content_id = '{content_id}'
            """
            
            if start_time:
                query += f" AND timestamp >= '{start_time.isoformat()}'"
            if end_time:
                query += f" AND timestamp <= '{end_time.isoformat()}'"
            
            query += " ORDER BY timestamp"
            
            result = self.client.query(query)
            
            metrics = []
            for row in result.result_rows:
                metrics.append({
                    "timestamp": row[0],
                    "views": row[1],
                    "likes": row[2],
                    "shares": row[3],
                    "comments": row[4],
                    "additional_metrics": json.loads(row[5]) if row[5] else {}
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    def get_velocity(
        self,
        content_id: str,
        minutes: int = 30
    ) -> Dict[str, float]:
        """
        Calculate engagement velocity (rate of change).
        
        Args:
            content_id: Content identifier
            minutes: Time window in minutes
            
        Returns:
            Dictionary with velocity metrics
        """
        if not self.client:
            return {
                "views_per_minute": 0.0,
                "likes_per_minute": 0.0,
                "shares_per_minute": 0.0,
                "comments_per_minute": 0.0
            }
        
        try:
            import json
            
            end_time = datetime.utcnow()
            start_time = datetime.utcnow() - timedelta(minutes=minutes)
            
            query = f"""
            SELECT 
                sum(views) / {minutes} as views_per_min,
                sum(likes) / {minutes} as likes_per_min,
                sum(shares) / {minutes} as shares_per_min,
                sum(comments) / {minutes} as comments_per_min
            FROM engagement_metrics
            WHERE content_id = '{content_id}'
            AND timestamp >= '{start_time.isoformat()}'
            AND timestamp <= '{end_time.isoformat()}'
            """
            
            result = self.client.query(query)
            if result.result_rows:
                row = result.result_rows[0]
                return {
                    "views_per_minute": float(row[0] or 0.0),
                    "likes_per_minute": float(row[1] or 0.0),
                    "shares_per_minute": float(row[2] or 0.0),
                    "comments_per_minute": float(row[3] or 0.0)
                }
            
            return {
                "views_per_minute": 0.0,
                "likes_per_minute": 0.0,
                "shares_per_minute": 0.0,
                "comments_per_minute": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating velocity: {e}")
            return {
                "views_per_minute": 0.0,
                "likes_per_minute": 0.0,
                "shares_per_minute": 0.0,
                "comments_per_minute": 0.0
            }

