#!/usr/bin/env python3
"""
Train Virality Predictor on full ClickHouse dataset.

This script:
1. Connects to ClickHouse
2. Queries the full ig_post table (or specified table)
3. Extracts features for all posts
4. Creates labels based on engagement metrics
5. Trains the ML model
6. Saves the model for use by ScoringService
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import clickhouse_connect
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.models.virality_predictor import ViralityPredictor
from src.services.text.service import TextUnderstandingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ClickHouse connection config
CLICKHOUSE_CONFIG = {
    'host': 'clickhouse.bragmant.noooo.art',
    'port': 443,
    'username': 'dev2',
    'password': '730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
    'database': 'crawler',
    'secure': True
}

def connect_to_clickhouse():
    """Connect to ClickHouse database."""
    try:
        client = clickhouse_connect.create_client(**CLICKHOUSE_CONFIG)
        logger.info(f"✅ Connected to ClickHouse: {CLICKHOUSE_CONFIG['host']}")
        return client
    except Exception as e:
        logger.error(f"❌ Error connecting to ClickHouse: {e}")
        raise

def get_full_dataset(client, table_name='ig_post', limit=None):
    """
    Get full dataset from ClickHouse.
    
    Args:
        client: ClickHouse client
        table_name: Table to query
        limit: Optional limit (None = all rows)
    
    Returns:
        pandas DataFrame
    """
    logger.info(f"📊 Querying table: {table_name}")
    
    if limit:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        logger.info(f"   Limiting to {limit} rows")
    else:
        # Get total count first
        count_query = f"SELECT count() FROM {table_name}"
        total_count = client.query(count_query).result_rows[0][0]
        logger.info(f"   Total rows in table: {total_count:,}")
        query = f"SELECT * FROM {table_name}"
    
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    
    logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df

def create_labels(df, top_percentile=20):
    """
    Create binary viral labels based on engagement metrics.
    
    Args:
        df: DataFrame with engagement columns
        top_percentile: Top X% to label as viral (default: 20%)
    
    Returns:
        numpy array of labels (0 = non-viral, 1 = viral)
    """
    logger.info(f"🏷️  Creating labels (top {top_percentile}% = viral)")
    
    # Calculate engagement score
    # Use like_count as primary metric, with comment_count as secondary
    df['engagement_score'] = (
        df.get('like_count', 0).fillna(0) * 1.0 +
        df.get('comment_count', 0).fillna(0) * 2.0 +  # Comments weighted more
        df.get('video_view_count', 0).fillna(0) * 0.1  # Views weighted less
    )
    
    # Label top percentile as viral
    threshold = df['engagement_score'].quantile(1 - (top_percentile / 100))
    labels = (df['engagement_score'] >= threshold).astype(int)
    
    viral_count = labels.sum()
    non_viral_count = len(labels) - viral_count
    
    logger.info(f"   Viral (1): {viral_count:,} posts")
    logger.info(f"   Non-viral (0): {non_viral_count:,} posts")
    logger.info(f"   Threshold: {threshold:.2f}")
    
    return labels.values

def extract_features_for_training(df, text_service=None):
    """
    Extract features for all posts in the dataset.
    
    Args:
        df: DataFrame with post data
        text_service: Optional TextUnderstandingService instance
    
    Returns:
        numpy array of feature vectors
    """
    logger.info("🔧 Extracting features for training...")
    
    if text_service is None:
        text_service = TextUnderstandingService()
    
    predictor = ViralityPredictor()
    feature_vectors = []
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            logger.info(f"   Processed {idx + 1:,} / {total:,} posts ({100*(idx+1)/total:.1f}%)")
        
        # Extract text features
        caption = str(row.get('caption', '') or '')
        hashtags_str = str(row.get('caption_hashtags', '') or '')
        hashtags = [h.strip('#') for h in hashtags_str.split(',') if h.strip()] if hashtags_str else []
        
        # Build features dict (minimal for CSV data)
        features = {
            "visual": {},
            "audio": {},
            "text": {
                "trend_proximity": {"trend_score": 0.0},
                "hook_efficiency": {"hook_score": 0.0},
            },
            "temporal": {},
        }
        
        # Try to get text features from TextUnderstandingService (synchronous fallback)
        try:
            # Use simple heuristics for now (async would require asyncio.run)
            hashtag_count = len(hashtags)
            caption_len = len(caption)
            features["text"]["trend_proximity"]["trend_score"] = min(hashtag_count / 10.0, 1.0)
            features["text"]["hook_efficiency"]["hook_score"] = min(caption_len / 200.0, 1.0) if caption else 0.0
        except Exception as e:
            # Fallback: simple heuristic
            features["text"]["trend_proximity"]["trend_score"] = min(len(hashtags) / 10.0, 1.0)
            features["text"]["hook_efficiency"]["hook_score"] = min(len(caption) / 200.0, 1.0) if caption else 0.0
        
        # Build metadata
        metadata = {
            "caption": caption,
            "description": "",
            "hashtags": hashtags,
            "platform": "instagram",
            "like_count": float(row.get('like_count', 0) or 0),
            "comment_count": float(row.get('comment_count', 0) or 0),
        }
        
        # Extract feature vector
        embeddings = {}  # Empty embeddings for CSV data
        try:
            fv = predictor._extract_features(embeddings, features, metadata)
            if isinstance(fv, list):
                fv = np.array(fv, dtype=np.float32)
            feature_vectors.append(fv)
        except Exception as e:
            logger.warning(f"   Error extracting features for row {idx}: {e}")
            # Use zero vector as fallback
            feature_vectors.append(np.zeros(35, dtype=np.float32))
    
    logger.info(f"✅ Extracted features for {len(feature_vectors):,} posts")
    return np.array(feature_vectors, dtype=np.float32)


def train_model(X, y, model_path="./models/virality_model.pkl"):
    """
    Train the virality prediction model.
    
    Args:
        X: Feature matrix
        y: Labels
        model_path: Path to save the model
    """
    logger.info("🧠 Training model...")
    logger.info(f"   Feature matrix shape: {X.shape}")
    logger.info(f"   Label distribution: {np.bincount(y)} (classes 0,1)")
    
    # Create model directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize and train predictor
    predictor = ViralityPredictor(model_path=model_path)
    predictor.train(X, y)
    predictor.save()
    
    logger.info(f"✅ Model trained and saved to {model_path}")
    logger.info(f"   Model file size: {os.path.getsize(model_path) / 1024:.1f} KB")

def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train virality model on full ClickHouse data')
    parser.add_argument('--table', default='ig_post', help='ClickHouse table name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows (None = all)')
    parser.add_argument('--top-percentile', type=int, default=20, help='Top X%% to label as viral (default: 20)')
    parser.add_argument('--model-path', default='./models/virality_model.pkl', help='Path to save model')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING VIRALITY PREDICTOR ON FULL CLICKHOUSE DATA")
    print("=" * 80)
    print(f"Table: {args.table}")
    print(f"Limit: {args.limit or 'ALL ROWS'}")
    print(f"Top {args.top_percentile}% = viral")
    print("=" * 80)
    print()
    
    try:
        # Connect to ClickHouse
        client = connect_to_clickhouse()
        
        # Get full dataset
        df = get_full_dataset(client, table_name=args.table, limit=args.limit)
        
        # Create labels
        y = create_labels(df, top_percentile=args.top_percentile)
        
        # Extract features
        X = extract_features_for_training(df)
        
        # Train model
        train_model(X, y, model_path=args.model_path)
        
        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        print(f"Model saved to: {args.model_path}")
        print(f"On next API / scoring service start, this model will be loaded automatically.")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

