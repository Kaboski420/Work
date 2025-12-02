"""Redis caching utilities."""

import logging
import json
from typing import Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)

# Try to import metrics
try:
    from src.utils.metrics import cache_hits_total, cache_misses_total
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Try to import redis
try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not available. Caching will be disabled.")


class CacheService:
    """Redis cache service for scoring results and feature vectors."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 password: Optional[str] = None, db: int = 0,
                 default_ttl: int = 3600):
        """
        Initialize cache service.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.default_ttl = default_ttl
        self.client = None
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.client.ping()
                logger.info(f"Redis cache connected: {host}:{port}")
            except (ConnectionError, TimeoutError, Exception) as e:
                logger.warning(f"Redis not available: {e}. Caching disabled.")
                self.client = None
        else:
            logger.warning("Redis library not available. Caching disabled.")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                if METRICS_AVAILABLE:
                    cache_hits_total.labels(cache_type="redis").inc()
                return json.loads(value)
            else:
                if METRICS_AVAILABLE:
                    cache_misses_total.labels(cache_type="redis").inc()
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            if METRICS_AVAILABLE:
                cache_misses_total.labels(cache_type="redis").inc()
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if not self.client:
            return False
        
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "score:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0



