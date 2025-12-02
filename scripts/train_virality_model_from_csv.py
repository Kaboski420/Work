#!/usr/bin/env python3
"""
Train the ViralityPredictor ML model from exported ClickHouse data.

Workflow:
1. Load the latest `ig_post_100rows_*.csv` from `data/clickhouse_imports/`
2. Build simple features + metadata from each row
3. Create a binary "viral" label (1 = viral, 0 = non-viral) using a heuristic
4. Use ViralityPredictor._extract_features(...) to create feature vectors
5. Train the ML model (if scikit-learn is available) and save it to ./models/virality_model.pkl

After successful training, ScoringService will automatically load the saved
model on startup via ViralityPredictor.load().
"""

import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from src.models.virality_predictor import ViralityPredictor, NUMPY_AVAILABLE, SKLEARN_AVAILABLE  # type: ignore


DATA_DIR = Path("data/clickhouse_imports")


def find_latest_csv() -> Path:
    """Find the latest ig_post_100rows CSV file."""
    pattern = str(DATA_DIR / "ig_post_100rows_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV files found matching pattern: {pattern}")
    return Path(matches[-1])


def parse_hashtags(raw: Any) -> List[str]:
    """Parse hashtags from the `caption_hashtags` field."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    # Caption hashtags are often stored like "['tag1', 'tag2']" or "tag1, tag2"
    text = raw.strip()
    # Remove brackets/quotes if present
    for ch in ["[", "]", "'", "\""]:
        text = text.replace(ch, "")
    parts = [p.strip() for p in text.replace(",", " ").split()]
    return [p.lstrip("#") for p in parts if p]


def simple_trend_score(caption: str, hashtags: List[str]) -> float:
    """
    Very simple proxy for trend proximity:
    - Count occurrences of known trending keywords/hashtags in caption + hashtags.
    - Normalize to [0, 1].
    """
    if not caption and not hashtags:
        return 0.0

    trending_keywords = [
        "viral", "trending", "fyp", "foryou", "foryoupage", "explore",
        "reels", "tiktok", "mustsee", "watchthis", "amazing", "incredible",
        "wow", "crazy", "insane", "funny", "meme"
    ]

    text = (caption or "").lower()
    ht_lower = [h.lower() for h in hashtags]

    matches = 0
    for kw in trending_keywords:
        if kw in text:
            matches += 1
        for h in ht_lower:
            if kw in h:
                matches += 1

    max_possible = len(trending_keywords) + len(ht_lower) if ht_lower else len(trending_keywords)
    if max_possible <= 0:
        return 0.0
    return float(min(matches / max_possible, 1.0))


def simple_hook_score(caption: str) -> float:
    """
    Simple hook efficiency proxy based on:
    - presence of hooky words in first few words
    - punctuation and length
    """
    if not caption:
        return 0.0

    text = caption.strip()
    words = text.split()
    first_words = " ".join(words[:3]).lower()

    hook_keywords = [
        "you", "your", "this", "these", "watch", "look", "secret",
        "crazy", "amazing", "unbelievable", "wow"
    ]
    hook_hits = sum(1 for kw in hook_keywords if kw in first_words)
    hook_loc_score = min(hook_hits / 3.0, 1.0)

    has_question = "?" in text[:80]
    has_excl = "!" in text[:80]
    punct_score = 0.5 if (has_question or has_excl) else 0.0

    length = len(text)
    if 20 <= length <= 200:
        length_score = 1.0
    elif length < 20:
        length_score = length / 20.0
    else:
        length_score = max(0.0, 1.0 - (length - 200) / 300.0)

    hook_score = (
        hook_loc_score * 0.4 +
        punct_score * 0.3 +
        length_score * 0.3
    )
    return float(min(max(hook_score, 0.0), 1.0))


def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for training using ViralityPredictor._extract_features.
    Label definition (very simple baseline):
      - y = 1 if like_count is in the top 20% of the sample, else 0
    """
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is not available; cannot build feature matrix.")

    like_counts = df["like_count"].fillna(0).astype(float).values
    threshold = np.percentile(like_counts, 80)  # top 20% as 'viral'
    labels = (like_counts >= threshold).astype(int)

    vp = ViralityPredictor()

    features_list: List[np.ndarray] = []

    for _, row in df.iterrows():
        caption = str(row.get("caption") or "")
        hashtags = parse_hashtags(row.get("caption_hashtags"))

        embeddings: Dict[str, Any] = {}

        text_trend = simple_trend_score(caption, hashtags)
        hook = simple_hook_score(caption)

        features: Dict[str, Any] = {
            "visual": {},
            "audio": {},
            "text": {
                "trend_proximity": {"trend_score": text_trend},
                "hook_efficiency": {"hook_score": hook},
            },
            "temporal": {},
        }

        metadata: Dict[str, Any] = {
            "caption": caption,
            "description": "",
            "hashtags": hashtags,
            "creator_id": str(row.get("owner_ig_user_id") or ""),
            "platform": "instagram",
            "additional_metadata": {
                "owner_ig_username": row.get("owner_ig_username"),
                "post_id": row.get("post_id"),
                "shortcode": row.get("shortcode"),
                "permalink": row.get("permalink"),
                "is_video": bool(row.get("is_video")),
                "post_type": row.get("post_type"),
            },
        }

        fv = vp._extract_features(embeddings=embeddings, features=features, metadata=metadata)

        if isinstance(fv, list):
            fv_arr = np.array(fv, dtype=np.float32)
        else:
            fv_arr = fv.astype(np.float32)

        features_list.append(fv_arr)

    X = np.vstack(features_list)
    y = labels.astype(np.int32)
    return X, y


def main() -> int:
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn is not available in this environment.")
        print("   Training cannot run. Please install scikit-learn (and its dependencies) in your environment,")
        print("   then re-run this script. The project requirements.txt already includes scikit-learn.")
        return 1

    try:
        csv_path = find_latest_csv()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    print("==============================================")
    print(" TRAINING VIRALITY PREDICTOR FROM CSV DATA")
    print("==============================================")
    print(f"📄 Using input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    X, y = build_dataset(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)} (classes 0,1)")

    vp = ViralityPredictor(model_path="./models/virality_model.pkl")

    print("\n🧠 Training model...")
    vp.train(X, y)

    print("💾 Saving model to ./models/virality_model.pkl ...")
    vp.save()

    print("✅ Training complete.")
    print("   On next API / scoring service start, this model will be loaded automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


