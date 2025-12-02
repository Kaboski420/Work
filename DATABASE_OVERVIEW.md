# ClickHouse Crawler Database Overview

## ✅ Database Status: ACCESSIBLE

**Database**: `crawler`  
**Tables Found**: 12 tables  
**Total Data**: ~200+ MB

---

## 📊 Available Tables

### Instagram Data (`ig_*` tables)

#### 1. `ig_post` - Instagram Posts
- **Rows**: 16,569 posts
- **Size**: 87.52 MB
- **Key Features**:
  - ✅ Has `raw_json` field (full API response)
  - Engagement metrics: `like_count`, `comment_count`, `shares_visible`
  - Post metadata: `caption`, `hashtags`, `mentions`, `post_type`
  - Media info: `video_view_count`, `display_url`, `is_video`, `is_carousel`
  - Timestamps: `publication_timestamp`, `taken_at_timestamp`
  - Owner info: `owner_ig_username`, `owner_ig_user_id`

**Perfect for virality prediction!** Contains:
- Like counts
- Comment counts  
- Video view counts
- Caption analysis (hashtags, mentions)
- Post types (image, video, carousel, reel)

#### 2. `ig_comment` - Instagram Comments
- **Rows**: 94,956 comments
- **Size**: 45.00 MB
- **Key Features**:
  - ✅ Has `raw_json` field
  - Comment engagement: `comment_like_count`, `child_comment_count`
  - User info: `username`, `user_is_verified`
  - Text content: `text`
  - Relationships: `parent_comment_id` (thread structure)

**Useful for**:
- Engagement analysis
- Comment quality metrics
- User interaction patterns

#### 3. `ig_users` - Instagram User Profiles
- **Rows**: 2,702 users
- **Size**: 101.17 MB
- **Key Features**:
  - ✅ Has `raw_json` field
  - Follower metrics: `follower_count`, `following_count`, `media_count`
  - Profile info: `biography`, `is_verified`, `is_private`, `full_name`
  - Account type: `is_business`, `account_type`, `category`

**Perfect for**:
- Influencer analysis
- Account type classification
- Follower growth tracking

#### 4. `ig_users_discvered` - Discovered Users
- **Rows**: 45,726 users
- **Size**: 3.41 MB
- Tracks users discovered in posts

#### 5. `ig_stories` - Instagram Stories
- Story data with media URLs, timestamps, viewer lists

#### 6. `ig_stories_highlights` - Story Highlights
- Highlight sets with cover images and story IDs

---

### TikTok Data (`tt_*` tables)

#### 7. `tt_post` - TikTok Posts
- Similar structure to `ig_post`
- Engagement metrics: `likes_visible`, `comments_visible`, `shares_visible`
- ✅ Has `raw_post_json` field
- Media info, audio, effects, tags

#### 8. `tt_comment` - TikTok Comments
- Comment data with engagement counts
- ✅ Has `raw_comment_json` field

#### 9. `tt_users` - TikTok User Profiles
- User profiles with follower counts and metadata

#### 10. `tt_stories` - TikTok Stories
- Story content and metadata

---

### Supporting Tables

#### 11. `proxies` - Proxy Management
- **Rows**: 10 proxies
- Proxy configuration for scraping
- Health scores and usage statistics

#### 12. `users` - User Tracking
- **Rows**: 1 user
- Tracks crawl status and platform availability

---

## 🎯 Key Features for Virality Prediction

### Engagement Metrics Available:
- ✅ **Likes**: `like_count`, `comment_like_count`
- ✅ **Comments**: `comment_count`, `child_comment_count`
- ✅ **Shares**: `shares_visible`
- ✅ **Saves**: `saves_visible`
- ✅ **Views**: `video_view_count`

### Content Features:
- ✅ **Text**: Captions, comments, biographies
- ✅ **Hashtags**: `caption_hashtags`
- ✅ **Mentions**: `caption_mentions`
- ✅ **Media Type**: Image, video, carousel, reel
- ✅ **Post Type**: Enum values

### User Features:
- ✅ **Follower Count**: `follower_count`
- ✅ **Following Count**: `following_count`
- ✅ **Media Count**: `media_count`
- ✅ **Verified Status**: `is_verified`
- ✅ **Account Type**: Business, personal, etc.

### Raw Data:
- ✅ **Raw JSON**: Full API responses in `raw_json` field
- Perfect for extracting additional features not in structured columns

---

## 📝 Example Queries

### Get Top Posts by Engagement
```sql
SELECT 
    post_id,
    owner_ig_username,
    caption,
    like_count,
    comment_count,
    publication_timestamp
FROM ig_post
ORDER BY like_count DESC
LIMIT 10;
```

### Get Comments for a Post
```sql
SELECT 
    comment_id,
    username,
    text,
    comment_like_count,
    comment_created_at
FROM ig_comment
WHERE owner_post_id = '3755871336960025910'
ORDER BY comment_like_count DESC;
```

### Get User Stats
```sql
SELECT 
    username,
    follower_count,
    following_count,
    media_count,
    is_verified,
    is_business
FROM ig_users
ORDER BY follower_count DESC
LIMIT 10;
```

### Extract from Raw JSON
```sql
SELECT 
    post_id,
    raw_json
FROM ig_post
WHERE raw_json != ''
LIMIT 5;
```

---

## 🚀 Next Steps for Virality Prediction

1. **Feature Engineering**:
   - Engagement rate (likes/followers)
   - Comment-to-like ratio
   - Time-based features (hour, day of week)
   - Hashtag count and diversity
   - Caption length and sentiment

2. **Data Extraction**:
   - Use `raw_json` to extract additional features
   - Combine post and user data for richer features
   - Track engagement over time

3. **Model Training**:
   - Use engagement metrics as target variables
   - Create train/test splits based on timestamp
   - Feature engineering from structured + raw JSON data

---

## 📦 Quick Access

Use the provided tools:
- `query_table.py` - Query any table by name
- `explore_clickhouse.py` - Full database exploration
- `click_house.py` - Simple connection test

Example:
```bash
python query_table.py ig_post --limit 5
```

