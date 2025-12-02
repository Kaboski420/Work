#!/usr/bin/env python3
"""
Score all posts from ClickHouse using the trained model.

This script:
1. Connects to ClickHouse
2. Queries all posts (or specified limit)
3. Scores each post using ScoringService
4. Saves results to CSV
"""

import sys
import os
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import clickhouse_connect
import pandas as pd
from datetime import datetime
import logging

from src.services.scoring.service import ScoringService
from src.services.ingestion.service import IngestionService

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

def get_posts(client, table_name='ig_post', limit=None):
    """Get posts from ClickHouse."""
    logger.info(f"📊 Querying table: {table_name}")
    
    if limit:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        logger.info(f"   Limiting to {limit} rows")
    else:
        count_query = f"SELECT count() FROM {table_name}"
        total_count = client.query(count_query).result_rows[0][0]
        logger.info(f"   Total rows in table: {total_count:,}")
        query = f"SELECT * FROM {table_name}"
    
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    
    logger.info(f"✅ Loaded {len(df):,} rows")
    return df

async def score_post(row, scoring_service, ingestion_service):
    """Score a single post."""
    try:
        content_id = str(row.get('post_id', row.get('id', f"post_{row.name}")))
        
        # Prepare metadata
        metadata = {
            "platform": "instagram",
            "caption": str(row.get('caption', '') or ''),
            "description": "",
            "hashtags": [],
            "engagement_metrics": {
                "views": float(row.get('video_view_count', 0) or 0),
                "likes": float(row.get('like_count', 0) or 0),
                "shares": 0.0,  # Not available in CSV
                "comments": float(row.get('comment_count', 0) or 0),
                "saves": 0.0,  # Not available in CSV
            }
        }
        
        # Extract simple features (minimal for CSV data)
        features = {
            "visual": {},
            "audio": {},
            "text": {
                "trend_proximity": {"trend_score": 0.0},
                "hook_efficiency": {"hook_score": 0.0},
            },
            "temporal": {},
        }
        
        embeddings = {}  # Empty for CSV data
        
        # Score the content
        result = await scoring_service.score_content(
            content_id=content_id,
            features=features,
            embeddings=embeddings,
            metadata=metadata
        )
        
        return {
            "content_id": content_id,
            "post_id": row.get('post_id'),
            "owner_ig_username": row.get('owner_ig_username'),
            "caption": metadata["caption"][:200],  # Truncate
            "like_count": metadata["engagement_metrics"]["likes"],
            "comment_count": metadata["engagement_metrics"]["comments"],
            "video_view_count": metadata["engagement_metrics"]["views"],
            "virality_score": result.get("virality_score", 0.0),
            "virality_probability": result.get("virality_probability", 0.0),
            "viral_classification": result.get("viral_classification", "no"),
            "confidence_lower": result.get("confidence_interval", {}).get("lower", 0.0),
            "confidence_upper": result.get("confidence_interval", {}).get("upper", 0.0),
            "confidence_level": result.get("confidence_level", 0.0),
            "virality_dynamics": result.get("virality_dynamics", "broad"),
            "engagement_rate": result.get("engagement_rate", 0.0),
            "momentum_score": result.get("momentum_score", 0.0),
            "comment_quality_score": result.get("comment_quality_score", 0.5),
            "prediction_source": result.get("prediction_source", "rule_based"),
        }
    except Exception as e:
        logger.warning(f"Error scoring post {row.get('post_id', 'unknown')}: {e}")
        return None

async def score_all_posts(df, scoring_service, ingestion_service, batch_size=10):
    """Score all posts in batches."""
    logger.info(f"🎯 Scoring {len(df):,} posts...")
    
    results = []
    total = len(df)
    
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            score_post(row, scoring_service, ingestion_service)
            for _, row in batch.iterrows()
        ])
        
        # Filter out None results
        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)
        
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= total:
            logger.info(f"   Scored {min(i + batch_size, total):,} / {total:,} posts ({100*min(i+batch_size, total)/total:.1f}%)")
    
    logger.info(f"✅ Scored {len(results):,} posts")
    return results

def main():
    """Main scoring pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Score all posts from ClickHouse')
    parser.add_argument('--table', default='ig_post', help='ClickHouse table name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows (None = all)')
    parser.add_argument('--output', default=None, help='Output CSV path (auto-generated if not specified)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for async processing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SCORING POSTS FROM CLICKHOUSE")
    print("=" * 80)
    print(f"Table: {args.table}")
    print(f"Limit: {args.limit or 'ALL ROWS'}")
    print("=" * 80)
    print()
    
    try:
        # Connect to ClickHouse
        client = connect_to_clickhouse()
        
        # Get posts
        df = get_posts(client, table_name=args.table, limit=args.limit)
        
        # Initialize services
        logger.info("🔧 Initializing services...")
        scoring_service = ScoringService()
        ingestion_service = IngestionService()
        
        # Score all posts
        results = asyncio.run(score_all_posts(df, scoring_service, ingestion_service, batch_size=args.batch_size))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            limit_str = f"_{args.limit}rows" if args.limit else "_all"
            output_path = f"data/clickhouse_imports/{args.table}{limit_str}_{timestamp}_scored.csv"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        print()
        print("=" * 80)
        print("✅ SCORING COMPLETE")
        print("=" * 80)
        print(f"Output saved to: {output_path}")
        print()
        print("Summary:")
        print(results_df[['virality_score', 'virality_probability', 'viral_classification', 'prediction_source']].describe())
        print()
        print(f"Prediction sources:")
        print(results_df['prediction_source'].value_counts())
        print()
        print(f"Viral classifications:")
        print(results_df['viral_classification'].value_counts())
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Scoring failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

