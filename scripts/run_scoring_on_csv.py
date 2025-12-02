#!/usr/bin/env python3
"""
Run Virality Engine scoring on exported ClickHouse data.

This script:
- Loads the latest CSV from `data/clickhouse_imports/` (e.g. `ig_post_100rows_*.csv`)
- Maps each Instagram post row into a minimal `metadata` / `features` structure
- Calls `ScoringService.score_content` for each row
- Prints a compact summary and writes results to a new CSV next to the input
"""

import asyncio
import glob
import uuid
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.services.scoring.service import ScoringService


DATA_DIR = Path("data/clickhouse_imports")


def _find_latest_csv() -> Path:
    """Find the latest ig_post_100rows CSV in the data folder."""
    pattern = str(DATA_DIR / "ig_post_100rows_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV files matching {pattern}")
    return Path(matches[-1])


def _row_to_request(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a single DataFrame row into (content_id, features, embeddings, metadata)
    suitable for `ScoringService.score_content`.
    """
    # Use post_id if available, otherwise generate a UUID
    content_id = str(row.get("post_id") or uuid.uuid4())

    caption = str(row.get("caption") or "") or None

    # `caption_hashtags` is likely a list-like string; keep it simple for now
    raw_hashtags = row.get("caption_hashtags")
    hashtags: List[str] = []
    if isinstance(raw_hashtags, str) and raw_hashtags.strip():
        # Very lightweight parsing: split on commas or spaces and strip '#'
        parts = [p.strip() for p in raw_hashtags.replace(",", " ").split()]
        hashtags = [p.lstrip("#") for p in parts if p]

    # Engagement metrics derived from ig_post schema
    views = float(row.get("video_view_count") or 0.0)
    likes = float(row.get("like_count") or 0.0)
    comments = float(row.get("comment_count") or 0.0)

    metadata: Dict[str, Any] = {
        "platform": "instagram",
        "caption": caption,
        "description": None,
        "hashtags": hashtags,
        "additional_metadata": {
            "owner_ig_username": row.get("owner_ig_username"),
            "shortcode": row.get("shortcode"),
            "permalink": row.get("permalink"),
            "is_video": bool(row.get("is_video")),
            "post_type": row.get("post_type"),
        },
        "engagement_metrics": {
            "views": views,
            "likes": likes,
            "shares": 0.0,
            "comments": comments,
            "saves": 0.0,
        },
    }

    # We don't have precomputed multimodal features/embeddings here,
    # so we pass empty structures and let Algorithm 1 fall back to
    # engagement- and text-based heuristics.
    features: Dict[str, Any] = {
        "visual": {},
        "audio": {},
        "text": {},
        "temporal": {},
    }
    embeddings: Dict[str, Any] = {}

    return {
        "content_id": content_id,
        "features": features,
        "embeddings": embeddings,
        "metadata": metadata,
    }


async def score_dataframe(df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    """Run scoring on each row of the DataFrame and return a new DataFrame with results."""
    service = ScoringService()

    if limit is not None:
        df = df.head(limit)

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        req = _row_to_request(row)
        result = await service.score_content(
            content_id=req["content_id"],
            features=req["features"],
            embeddings=req["embeddings"],
            metadata=req["metadata"],
        )

        results.append(
            {
                "content_id": result["content_id"],
                "owner_ig_username": row.get("owner_ig_username"),
                "post_id": row.get("post_id"),
                "shortcode": row.get("shortcode"),
                "permalink": row.get("permalink"),
                "caption": row.get("caption"),
                "like_count": row.get("like_count"),
                "comment_count": row.get("comment_count"),
                "video_view_count": row.get("video_view_count"),
                "virality_score": result["virality_score"],
                "virality_probability": result["virality_probability"],
                "viral_classification": result["viral_classification"],
                "confidence_lower": result["confidence_interval"]["lower"],
                "confidence_upper": result["confidence_interval"]["upper"],
                "confidence_level": result["confidence_level"],
                "virality_dynamics": result["virality_dynamics"],
                "engagement_rate": result.get("engagement_rate"),
                "momentum_score": result.get("momentum_score"),
                "comment_quality_score": result.get("comment_quality_score"),
                "prediction_source": result.get("prediction_source"),
            }
        )

    return pd.DataFrame(results)


async def main():
    csv_path = _find_latest_csv()
    print(f"📄 Using input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    # Score all rows; adjust limit here if you want a subset
    scored_df = await score_dataframe(df, limit=None)

    # Derive output path
    output_path = csv_path.with_name(csv_path.stem + "_scored.csv")
    scored_df.to_csv(output_path, index=False)

    print(f"\n✅ Scoring completed for {len(scored_df)} posts")
    print(f"   Output saved to: {output_path}")

    # Simple summary
    print("\nSummary:")
    print(scored_df[["virality_score", "virality_probability", "viral_classification"]].describe(include="all"))


if __name__ == "__main__":
    asyncio.run(main())


