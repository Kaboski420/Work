"""MinIO storage utilities for feature vectors and embeddings."""

import logging
import json
import io
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import minio
try:
    from minio import Minio
    from minio.error import S3Error
    from minio.commonconfig import REPLACE
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("minio/pandas/pyarrow not available. Storage will be disabled.")


class StorageService:
    """MinIO storage service for feature vectors and embeddings."""
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, secure: bool = False):
        """
        Initialize storage service.
        
        Args:
            endpoint: MinIO endpoint (e.g., "localhost:9000")
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket: Bucket name
            secure: Use HTTPS
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self.client = None
        
        if MINIO_AVAILABLE:
            try:
                self.client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure
                )
                # Ensure bucket exists
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
                logger.info(f"MinIO storage connected: {endpoint}/{bucket}")
            except Exception as e:
                logger.warning(f"MinIO not available: {e}. Storage disabled.")
                self.client = None
        else:
            logger.warning("MinIO library not available. Storage disabled.")
    
    def store_features(
        self,
        content_id: str,
        features: Dict[str, Any],
        embeddings: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Store features and embeddings as Parquet in MinIO.
        
        Args:
            content_id: Content identifier
            features: Extracted features
            embeddings: Embeddings
            metadata: Metadata
            
        Returns:
            Object path or None
        """
        if not self.client or not MINIO_AVAILABLE:
            return None
        
        try:
            # Combine all data
            data = {
                "content_id": content_id,
                "timestamp": datetime.utcnow().isoformat(),
                **features,
                **embeddings,
                "metadata": metadata
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Convert to Parquet
            buffer = io.BytesIO()
            table = pa.Table.from_pandas(df)
            pq.write_table(table, buffer)
            buffer.seek(0)
            
            # Store in MinIO
            object_name = f"features/{content_id}.parquet"
            self.client.put_object(
                self.bucket,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/parquet"
            )
            
            logger.info(f"Stored features for {content_id} at {object_name}")
            return object_name
            
        except Exception as e:
            logger.error(f"Error storing features for {content_id}: {e}")
            return None
    
    def load_features(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Load features from MinIO.
        
        Args:
            content_id: Content identifier
            
        Returns:
            Features dictionary or None
        """
        if not self.client or not MINIO_AVAILABLE:
            return None
        
        try:
            object_name = f"features/{content_id}.parquet"
            
            # Get object
            response = self.client.get_object(self.bucket, object_name)
            buffer = io.BytesIO(response.read())
            
            # Read Parquet
            table = pq.read_table(buffer)
            df = table.to_pandas()
            
            # Convert to dict
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return None
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"Features not found for {content_id}")
            else:
                logger.error(f"Error loading features for {content_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading features for {content_id}: {e}")
            return None
    
    def store_embedding(self, content_id: str, embedding_type: str, 
                       embedding: List[float]) -> Optional[str]:
        """
        Store single embedding vector.
        
        Args:
            content_id: Content identifier
            embedding_type: Type (visual, audio, text, contextual)
            embedding: Embedding vector
            
        Returns:
            Object path or None
        """
        if not self.client:
            return None
        
        try:
            # Store as JSON
            data = {
                "content_id": content_id,
                "embedding_type": embedding_type,
                "embedding": embedding,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            object_name = f"embeddings/{content_id}/{embedding_type}.json"
            buffer = io.BytesIO(json.dumps(data).encode())
            
            self.client.put_object(
                self.bucket,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/json"
            )
            
            return object_name
            
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return None
    
    def delete_features(self, content_id: str) -> bool:
        """
        Delete features for content.
        
        Args:
            content_id: Content identifier
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            object_name = f"features/{content_id}.parquet"
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"Deleted features for {content_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting features for {content_id}: {e}")
            return False




