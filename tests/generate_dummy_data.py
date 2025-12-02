"""Generate dummy data for testing."""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np


def generate_dummy_profile(
    profile_id: Optional[str] = None,
    account_type: str = "creator"
) -> Dict[str, Any]:
    """Generate dummy profile data based on Daten Punkte specification."""
    if profile_id is None:
        profile_id = f"profile_{random.randint(100000, 999999)}"
    
    username = f"user_{random.randint(1000, 9999)}"
    is_viral_creator = random.random() > 0.7
    
    if is_viral_creator:
        follower_count = random.randint(50000, 5000000)
        following_count = random.randint(100, 2000)
        post_count = random.randint(50, 500)
    else:
        follower_count = random.randint(100, 10000)
        following_count = random.randint(50, 500)
        post_count = random.randint(10, 100)
    
    account_types = ["creator", "business", "personal"]
    account_type = random.choice(account_types) if account_type == "creator" else account_type
    
    profile = {
        "profile_id": profile_id,
        "username": username,
        "profile_name": f"{username.replace('_', ' ').title()}",
        "profile_picture_url": f"https://example.com/profiles/{profile_id}/pic.jpg",
        "bio_text": f"🎨 Creative content creator | ✨ {random.choice(['Artist', 'Influencer', 'Creator', 'Entrepreneur'])} | 📍 {random.choice(['NYC', 'LA', 'London', 'Berlin'])}",
        "pronouns": random.choice(["he/him", "she/her", "they/them", None]),
        "external_links": [
            {"type": "website", "url": f"https://{username}.com"},
            {"type": "linktree", "url": f"https://linktr.ee/{username}"}
        ] if random.random() > 0.3 else [],
        "verification_status": is_viral_creator and random.random() > 0.5,
        "account_type": account_type,
        "category": random.choice(["Artist", "Influencer", "RetailCompany", "Fitness", "Photographer", None]),
        "contact_data": {
            "email": f"contact@{username}.com" if random.random() > 0.7 else None,
            "phone": f"+1{random.randint(2000000000, 9999999999)}" if random.random() > 0.8 else None
        },
        "is_private": random.random() > 0.85,
        "follower_count": follower_count,
        "following_count": following_count,
        "post_count": post_count,
        "last_post_time": (datetime.utcnow() - timedelta(hours=random.randint(1, 72))).isoformat(),
        "shop_url": f"https://shop.{username}.com" if account_type == "business" and random.random() > 0.5 else None,
        "language": random.choice(["en", "de", "es", "fr", "it"]),
        "registration_date": (datetime.utcnow() - timedelta(days=random.randint(365, 3650))).isoformat(),
        "created_at": (datetime.utcnow() - timedelta(days=random.randint(365, 3650))).isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    if account_type == "business":
        profile.update({
            "business_email": profile["contact_data"]["email"],
            "business_phone": profile["contact_data"]["phone"],
            "business_address": {
                "street": f"{random.randint(1, 999)} Main St",
                "city": random.choice(["New York", "Los Angeles", "London", "Berlin"]),
                "country": random.choice(["USA", "UK", "Germany"])
            },
            "shop_platform": random.choice(["Shopify", "WooCommerce", "Etsy", None]),
            "company_name": f"{username.replace('_', ' ').title()} Inc."
        })
    
    if account_type == "creator":
        profile.update({
            "creator_category": random.choice(["Photographer", "Fitness", "Artist", "Influencer", "Musician"]),
            "has_reels": random.random() > 0.3,
            "has_igtv": random.random() > 0.5,
            "external_portfolio_links": [
                {"platform": "YouTube", "url": f"https://youtube.com/@{username}"},
                {"platform": "TikTok", "url": f"https://tiktok.com/@{username}"}
            ] if random.random() > 0.5 else []
        })
    
    return profile


def generate_dummy_post(
    profile_id: str,
    platform: str = "instagram",
    post_type: str = "video"
) -> Dict[str, Any]:
    """Generate dummy post data."""
    post_id = f"{platform}_{random.randint(1000000000000000000, 9999999999999999999)}"
    shortcode = post_id[-11:] if platform == "instagram" else post_id
    
    captions = [
        "Check out this amazing video! #viral #trending 🌟",
        "You won't believe what happened! #shocking #viral 😱",
        "This is so good! #fyp #trending 🔥",
        "Amazing content right here! #viral #fyp ✨",
        "This blew my mind! #trending #viral 💥",
        "Best video ever! #fyp #viral 🎬",
        "Incredible! #trending #amazing ⚡",
        "This is fire! #viral #hot 🔥",
        "So good! #fyp #trending 🌈",
        "Must watch! #viral #fyp 👀"
    ]
    
    hashtags_pool = [
        ["viral", "trending", "fyp"],
        ["viral", "fyp", "amazing"],
        ["trending", "hot", "fire"],
        ["viral", "shocking", "wow"],
        ["fyp", "trending", "best"],
        ["viral", "amazing", "incredible"],
        ["trending", "fyp", "mustwatch"],
        ["viral", "fire", "hot"],
        ["fyp", "viral", "trending"],
        ["viral", "best", "amazing"]
    ]
    
    is_viral = random.random() > 0.5
    publication_date = datetime.utcnow() - timedelta(hours=random.randint(1, 720))
    
    if is_viral:
        views = random.uniform(50000, 1000000)
        likes = views * random.uniform(0.05, 0.15)
        shares = views * random.uniform(0.01, 0.03)
        comments_count = views * random.uniform(0.005, 0.02)
        saves = likes * random.uniform(0.1, 0.2)
    else:
        views = random.uniform(100, 10000)
        likes = views * random.uniform(0.01, 0.05)
        shares = views * random.uniform(0.001, 0.01)
        comments_count = views * random.uniform(0.001, 0.005)
        saves = likes * random.uniform(0.05, 0.1)
    
    caption = random.choice(captions)
    hashtags = random.choice(hashtags_pool)
    caption_hashtags = [tag for tag in caption.split() if tag.startswith('#')]
    caption_mentions = [tag for tag in caption.split() if tag.startswith('@')]
    all_hashtags = list(set(hashtags + [h.replace('#', '') for h in caption_hashtags]))
    
    post = {
        "post_id": post_id,
        "shortcode": shortcode,
        "profile_id": profile_id,
        "platform": platform,
        "publication_date": publication_date.isoformat(),
        "post_type": post_type,
        "caption": caption,
        "caption_entities": {
            "hashtags": all_hashtags,
            "mentions": [m.replace('@', '') for m in caption_mentions],
            "urls": []
        },
        "media_urls": {
            "original": [f"https://cdn.example.com/{post_id}/original.mp4"],
            "cdn": [f"https://cdn.example.com/{post_id}/cdn.mp4"],
            "thumbnail": f"https://cdn.example.com/{post_id}/thumb.jpg"
        },
        "video_duration": random.uniform(15, 300) if post_type == "video" else None,
        "video_resolution": random.choice(["1920x1080", "1080x1920", "1080x1080"]),
        "video_frame_rate": random.choice([24, 30, 60]),
        "video_bitrate": random.uniform(2000, 8000),
        "image_dimensions": "1080x1080" if post_type == "image" else None,
        "like_count": int(likes),
        "comment_count": int(comments_count),
        "share_count": int(shares),
        "save_count": int(saves),
        "tagged_accounts": [
            {
                "handle": f"friend_{random.randint(1, 10)}",
                "user_id": f"user_{random.randint(100, 999)}",
                "bbox": [0.2, 0.3, 0.4, 0.5] if random.random() > 0.5 else None
            }
        ] if random.random() > 0.7 else [],
        "audio_id": f"audio_{random.randint(100000, 999999)}" if post_type == "reel" else None,
        "audio_track": random.choice(["Trending Song", "Original Audio", "Popular Track"]) if post_type == "reel" else None,
        "audio_is_original": random.random() > 0.7 if post_type == "reel" else None,
        "effect_name": random.choice(["Vintage", "Vibrant", "Warm", None]),
        "is_pinned": random.random() > 0.9,
        "created_at": publication_date.isoformat(),
        "updated_at": publication_date.isoformat()
    }
    
    return post


def generate_dummy_comment(
    post_id: str,
    profile_id: str,
    parent_comment_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate dummy comment data."""
    comment_id = f"comment_{random.randint(1000000000000000000, 9999999999999999999)}"
    commenter_profile_id = f"profile_{random.randint(100000, 999999)}"
    
    comments = [
        "This is amazing! 🔥",
        "Love this content! ❤️",
        "So good! 👏",
        "Incredible! 😍",
        "Best video ever! 🎬",
        "Wow! This is fire! 🔥",
        "Amazing work! ✨",
        "Great content! 👍",
        "This is so cool! 😎",
        "Love it! 💕"
    ]
    
    comment_text = random.choice(comments)
    timestamp = datetime.utcnow() - timedelta(minutes=random.randint(1, 1440))
    
    comment = {
        "comment_id": comment_id,
        "post_id": post_id,
        "comment_text": comment_text,
        "timestamp": timestamp.isoformat(),
        "commentator_username": f"commenter_{random.randint(1000, 9999)}",
        "commentator_profile_name": f"Commenter {random.randint(1, 100)}",
        "commentator_profile_id": commenter_profile_id,
        "commentator_profile_picture_url": f"https://cdn.example.com/profiles/{commenter_profile_id}/pic.jpg",
        "commentator_is_verified": random.random() > 0.9,
        "commentator_is_private": random.random() > 0.7,
        "commentator_account_type": random.choice(["Private", "Creator", "Brand"]),
        "like_count": random.randint(0, 100),
        "parent_comment_id": parent_comment_id,
        "is_reply": parent_comment_id is not None,
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat()
    }
    
    return comment


def generate_dummy_content(
    count: int = 10,
    platforms: List[str] = ["tiktok", "instagram"]
) -> List[Dict[str, Any]]:
    """Generate dummy content items."""
    content_items = []
    profiles = {}
    
    for i in range(count):
        platform = random.choice(platforms)
        post_type = random.choice(["video", "image", "carousel", "reel"])
        profile_id = f"profile_{random.randint(1, 20)}"
        if profile_id not in profiles:
            profiles[profile_id] = generate_dummy_profile(profile_id=profile_id)
        
        post = generate_dummy_post(
            profile_id=profile_id,
            platform=platform,
            post_type=post_type
        )
        
        num_comments = random.randint(3, 10)
        comments = []
        for j in range(num_comments):
            comment = generate_dummy_comment(
                post_id=post["post_id"],
                profile_id=profile_id
            )
            comments.append(comment)
        
        num_likers = min(post["like_count"], random.randint(10, 50))
        likers = []
        for j in range(num_likers):
            liker = {
                "username": f"liker_{random.randint(1000, 9999)}",
                "profile_name": f"Liker {random.randint(1, 100)}",
                "profile_id": f"profile_{random.randint(100000, 999999)}",
                "profile_picture_url": f"https://cdn.example.com/profiles/{random.randint(100000, 999999)}/pic.jpg",
                "verification_status": random.random() > 0.9,
                "is_private": random.random() > 0.7
            }
            likers.append(liker)
        
        views = post.get("like_count", 0) / random.uniform(0.05, 0.15) if post.get("like_count", 0) > 0 else random.uniform(100, 10000)
        is_viral = views > 50000
        
        content_item = {
            "content_id": post["post_id"],
            "platform": platform,
            "content_type": post_type,
            "creator_id": profile_id,
            "created_at": post["publication_date"],
            "updated_at": post.get("updated_at", post["publication_date"]),
            "profile": profiles[profile_id],
            "post": post,
            "comments": comments,
            "likers": likers[:10],
            "engagement_metrics": {
                "views": float(views),
                "likes": float(post["like_count"]),
                "shares": float(post["share_count"]),
                "comments": float(post["comment_count"]),
                "saves": float(post["save_count"]),
                "reach": float(views * random.uniform(0.8, 1.2)),
                "momentum_score": random.uniform(0.3, 0.9) if is_viral else random.uniform(0.1, 0.4),
                "views_per_minute": random.uniform(10, 1000) if is_viral else random.uniform(0.1, 10),
                "likes_per_minute": random.uniform(1, 100) if is_viral else random.uniform(0.01, 1),
                "shares_per_minute": random.uniform(0.1, 10) if is_viral else random.uniform(0.001, 0.1),
                "comments_per_minute": random.uniform(0.1, 10) if is_viral else random.uniform(0.001, 0.1)
            },
            "features": {
                "visual": {
                    "entropy": random.uniform(0.3, 0.9),
                    "variance": {"frame_variance": random.uniform(0.2, 0.8)},
                    "motion_index": random.uniform(0.2, 0.8),
                    "color_gamut": {"saturation_score": random.uniform(0.3, 0.9)},
                    "cut_density": random.uniform(0.1, 0.7)
                },
                "audio": {
                    "bpm": random.uniform(80, 160),
                    "loudness": {
                        "rms": random.uniform(0.3, 0.9),
                        "peak": random.uniform(0.5, 1.0)
                    }
                },
                "text": {
                    "trend_proximity": {"trend_score": random.uniform(0.2, 0.9)},
                    "hook_efficiency": {"hook_score": random.uniform(0.3, 0.8)},
                    "comment_quality": {
                        "quality_score": random.uniform(0.4, 0.9),
                        "sentiment_score": random.uniform(0.5, 0.9),
                        "comment_count": len(comments),
                        "sentiment_distribution": {
                            "positive": random.uniform(0.4, 0.8),
                            "negative": random.uniform(0.05, 0.2),
                            "neutral": random.uniform(0.1, 0.4)
                        }
                    }
                }
            },
            "embeddings": {
                "visual": np.random.rand(512).astype(np.float32).tolist(),
                "audio": np.random.rand(128).astype(np.float32).tolist(),
                "text": np.random.rand(384).astype(np.float32).tolist(),
                "contextual": np.random.rand(256).astype(np.float32).tolist()
            },
            "caption": post["caption"],
            "description": None,
            "hashtags": post["caption_entities"]["hashtags"]
        }
        
        content_items.append(content_item)
    
    return content_items


def generate_dummy_feedback(
    content_items: List[Dict[str, Any]],
    predicted_probabilities: List[float]
) -> List[Dict[str, Any]]:
    """Generate dummy feedback data."""
    feedback_records = []
    
    for content_item, predicted_prob in zip(content_items, predicted_probabilities):
        noise = random.uniform(-0.2, 0.2)
        actual_performance = max(0.0, min(1.0, predicted_prob + noise))
        
        feedback = {
            "content_id": content_item["content_id"],
            "predicted_probability": predicted_prob,
            "actual_performance": actual_performance,
            "performance_delta": actual_performance - predicted_prob,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        feedback_records.append(feedback)
    
    return feedback_records


def save_dummy_data(content_items: List[Dict[str, Any]], filename: str = "dummy_data.json"):
    """Save dummy data to JSON file."""
    with open(filename, "w") as f:
        json.dump(content_items, f, indent=2)
    print(f"Saved {len(content_items)} content items to {filename}")


def main():
    """Generate and save dummy data."""
    print("Generating dummy data...")
    
    content_items = generate_dummy_content(count=20)
    profiles = {}
    for item in content_items:
        if "profile" in item and item["profile"]["profile_id"] not in profiles:
            profiles[item["profile"]["profile_id"]] = item["profile"]
    
    predicted_probabilities = [
        random.uniform(0.3, 0.9) for _ in content_items
    ]
    
    feedback_records = generate_dummy_feedback(content_items, predicted_probabilities)
    save_dummy_data(content_items, "tests/dummy_content.json")
    profiles_list = list(profiles.values())
    with open("tests/dummy_profiles.json", "w") as f:
        json.dump(profiles_list, f, indent=2)
    print(f"Saved {len(profiles_list)} profiles to tests/dummy_profiles.json")
    
    with open("tests/dummy_feedback.json", "w") as f:
        json.dump(feedback_records, f, indent=2)
    print(f"Saved {len(feedback_records)} feedback records to tests/dummy_feedback.json")
    
    print("\nDummy data generation complete!")
    print(f"- Content items: {len(content_items)}")
    print(f"- Profiles: {len(profiles_list)}")
    print(f"- Feedback records: {len(feedback_records)}")
    print(f"- Platforms: {set(item['platform'] for item in content_items)}")
    print(f"- Post types: {set(item['content_type'] for item in content_items)}")
    print("\nData structure includes:")
    print("  ✓ Profile data (comprehensive)")
    print("  ✓ Post data (with all metadata)")
    print("  ✓ Comments (with commentator info)")
    print("  ✓ Engagement metrics (with momentum)")
    print("  ✓ Features and embeddings")


if __name__ == "__main__":
    main()

