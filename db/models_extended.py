"""Extended SQLAlchemy models based on Daten Punkte specification.

Comprehensive data models for Instagram/TikTok profile, post, comment, story data.
"""

from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer, Boolean, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Profile(Base):
    """Profile/Account model based on Daten Punkte specification."""
    __tablename__ = "profiles"
    
    # Core Profile Fields
    profile_id = Column(String(100), primary_key=True)  # Internal Profile ID (numeric or string)
    username = Column(String(100), nullable=False, index=True)  # Username (Handle)
    profile_name = Column(String(200))  # Profile Name (Display Name)
    profile_picture_url = Column(Text)  # Profile picture URLs (all resolutions)
    bio_text = Column(Text)  # Bio-text (incl. emojis)
    pronouns = Column(String(50))  # Pronouns (if visible)
    external_links = Column(JSON)  # External links (website, linktree, shop, others)
    verification_status = Column(Boolean, default=False)  # Verification status
    account_type = Column(String(50))  # Account Type Flags (Creator/Business/Personal)
    category = Column(String(100))  # Category/industry (e.g., "Artist", "RetailCompany")
    contact_data = Column(JSON)  # Contact buttons & data (e-mail, telephone, address)
    is_private = Column(Boolean, default=False)  # Public/Private Account Status
    follower_count = Column(Integer)  # Follower number (as shown)
    following_count = Column(Integer)  # Following number
    post_count = Column(Integer)  # Number of posts (posts/reels)
    last_post_time = Column(DateTime)  # Time of last post
    shop_url = Column(String(500))  # Shop presence/Shop URL (if available)
    
    # Brand/Business Account Fields
    business_email = Column(String(200))
    business_phone = Column(String(50))
    business_address = Column(JSON)
    facebook_page_id = Column(String(100))  # Connected Facebook/Meta Page ID
    shop_catalog_id = Column(String(100))
    company_name = Column(String(200))
    shop_platform = Column(String(50))  # Shopify, WooCommerce, etc.
    
    # Creator Account Fields
    creator_category = Column(String(100))  # Creator Category/Niche Tag
    total_likes_sum = Column(Integer)
    has_reels = Column(Boolean, default=False)
    has_igtv = Column(Boolean, default=False)
    external_portfolio_links = Column(JSON)  # YouTube, TikTok, Linktree
    
    # Additional Fields
    language = Column(String(10))
    registration_date = Column(DateTime)  # Account-Age
    archived_posts_count = Column(Integer)
    raw_html_snapshot = Column(Text)  # Raw HTML of the profile page
    raw_json_data = Column(JSON)  # Raw JSON/GraphQL answer(s)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    posts = relationship("Post", back_populates="profile", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_username', 'username'),
        Index('idx_account_type', 'account_type'),
        Index('idx_verification', 'verification_status'),
        Index('idx_follower_count', 'follower_count'),
    )


class Post(Base):
    """Post/Reel/Carousel model based on Daten Punkte specification."""
    __tablename__ = "posts"
    
    # Core Post Fields
    post_id = Column(String(100), primary_key=True)  # Post-ID (Shortcode + numeric)
    shortcode = Column(String(50), nullable=False, index=True)  # Shortcode
    permalink = Column(String(500), nullable=False)  # Permalink/URL
    profile_id = Column(String(100), ForeignKey('profiles.profile_id'), nullable=False, index=True)
    platform = Column(String(50), nullable=False)  # tiktok, instagram
    publication_date = Column(DateTime, nullable=False, index=True)  # Publication date & time
    post_type = Column(String(50), nullable=False)  # image, video, carousel, reel
    
    # Content Fields
    caption = Column(Text)  # Caption (complete including hashtags, @mentions, emojis)
    caption_entities = Column(JSON)  # Extracted hashtags, mentions, URLs
    media_urls = Column(JSON)  # Media URLs (images/video; original/CDN links; cover/thumbnail)
    
    # Video Metadata
    video_duration = Column(Float)  # Duration in seconds
    video_resolution = Column(String(50))  # e.g., "1920x1080"
    video_frame_rate = Column(Float)
    video_bitrate = Column(Float)
    video_download_url = Column(String(500))  # Video download URL (mp4)
    
    # Image Metadata
    image_dimensions = Column(String(50))  # e.g., "1080x1080"
    image_file_size = Column(Integer)  # Bytes
    image_mime_type = Column(String(50))
    image_exif_data = Column(JSON)
    
    # Location
    geotag_name = Column(String(200))  # Geotag/place name
    geotag_place_id = Column(String(100))
    geotag_address = Column(JSON)  # Address/coordinates
    
    # Tags and Mentions
    tagged_accounts = Column(JSON)  # Tagged accounts in image/video (handle, ID, bbox)
    collaboration_info = Column(JSON)  # Collaboration/Co-Author Info for Collab Posts
    branded_content_label = Column(String(200))  # "Paid Partnership with..." Partner-Handle
    branded_content_partner_id = Column(String(100))
    
    # Shopping/Product Tags
    shopping_tags = Column(JSON)  # ProductID, ProductName, DestinationURL
    
    # Audio/Sound (for Reels)
    audio_id = Column(String(100))
    audio_track = Column(String(200))  # Track name
    audio_is_original = Column(Boolean)
    audio_is_remix = Column(Boolean)
    audio_url = Column(String(500))
    audio_original_creator = Column(String(100))
    
    # Effects/Filters
    effect_name = Column(String(200))
    effect_id = Column(String(100))
    filter_name = Column(String(200))
    filter_id = Column(String(100))
    ar_effect = Column(JSON)  # AR Effect data
    
    # Engagement Metrics
    like_count = Column(Integer)  # Visible like number (if not hidden)
    comment_count = Column(Integer)  # Visible number of comments
    share_count = Column(Integer)  # Visible shares/saves (if derivable)
    save_count = Column(Integer)
    
    # Status Flags
    is_pinned = Column(Boolean, default=False)  # Pinned status
    
    # Additional Data
    interactive_elements = Column(JSON)  # Polls, quizzes, questions
    raw_post_json = Column(JSON)  # Raw Post JSON/embedded data
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    profile = relationship("Profile", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    likers = relationship("PostLiker", back_populates="post", cascade="all, delete-orphan")
    engagement_metrics = relationship("EngagementMetrics", back_populates="post", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_profile_date', 'profile_id', 'publication_date'),
        Index('idx_post_type', 'post_type'),
        Index('idx_publication_date', 'publication_date'),
    )


class PostLiker(Base):
    """Model for post likers (extracted from like overlay/popup)."""
    __tablename__ = "post_likers"
    
    id = Column(String(36), primary_key=True)
    post_id = Column(String(100), ForeignKey('posts.post_id'), nullable=False, index=True)
    username = Column(String(100))
    profile_name = Column(String(200))
    profile_id = Column(String(100))
    profile_picture_url = Column(String(500))
    verification_status = Column(Boolean, default=False)
    is_private = Column(Boolean)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    post = relationship("Post", back_populates="likers")
    
    __table_args__ = (
        Index('idx_post_liker', 'post_id', 'profile_id'),
    )


class Comment(Base):
    """Comment model based on Daten Punkte specification."""
    __tablename__ = "comments"
    
    comment_id = Column(String(100), primary_key=True)  # Comment ID
    post_id = Column(String(100), ForeignKey('posts.post_id'), nullable=False, index=True)
    comment_text = Column(Text)  # Comment text (incl. emojis, hashtags, @mentions, URLs)
    timestamp = Column(DateTime, nullable=False, index=True)  # Timestamp of the comment
    
    # Commentator Info
    commentator_username = Column(String(100))
    commentator_profile_name = Column(String(200))
    commentator_profile_id = Column(String(100))
    commentator_profile_picture_url = Column(String(500))
    commentator_is_verified = Column(Boolean, default=False)
    commentator_is_private = Column(Boolean)
    commentator_account_type = Column(String(50))  # Brand, Creator, Private
    
    # Engagement
    like_count = Column(Integer)  # Likes on comment (if visible)
    
    # Thread/Reply Info
    parent_comment_id = Column(String(100), ForeignKey('comments.comment_id'))  # Parent/Thread ID
    is_reply = Column(Boolean, default=False)
    thread_chain = Column(JSON)  # Full thread chain (recursive)
    
    # Additional Data
    raw_comment_json = Column(JSON)  # Raw comment JSON
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    parent = relationship("Comment", remote_side=[comment_id], backref="replies")
    
    __table_args__ = (
        Index('idx_post_timestamp', 'post_id', 'timestamp'),
        Index('idx_parent_comment', 'parent_comment_id'),
    )


class Story(Base):
    """Story model based on Daten Punkte specification."""
    __tablename__ = "stories"
    
    story_id = Column(String(100), primary_key=True)  # Story ID
    profile_id = Column(String(100), ForeignKey('profiles.profile_id'), nullable=False, index=True)
    publication_time = Column(DateTime, nullable=False, index=True)  # Time of publication
    media_url = Column(String(500))  # Media URL (image/video; CDN link)
    media_type = Column(String(50))  # image, video
    duration = Column(Float)  # Duration for video (seconds)
    
    # Link Sticker
    link_sticker_url = Column(String(500))  # Link Sticker Destination URL
    
    # Tags and Mentions
    tagged_accounts = Column(JSON)  # Tagged accounts (@mentions in the story)
    
    # Music/Sound
    music_title = Column(String(200))
    music_artist = Column(String(200))
    music_audio_id = Column(String(100))
    
    # Effects/Filters
    effect_name = Column(String(200))
    effect_id = Column(String(100))
    filter_name = Column(String(200))
    filter_id = Column(String(100))
    
    # Additional Data
    viewer_list = Column(JSON)  # Story viewer list (if accessible)
    interactive_sticker_answers = Column(JSON)  # Answers to surveys/questions
    raw_story_json = Column(JSON)  # Raw Story JSON
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_profile_publication', 'profile_id', 'publication_time'),
    )


class StoryHighlight(Base):
    """Story Highlight model."""
    __tablename__ = "story_highlights"
    
    highlight_set_id = Column(String(100), primary_key=True)
    profile_id = Column(String(100), ForeignKey('profiles.profile_id'), nullable=False, index=True)
    highlight_title = Column(String(200))
    highlight_cover_image_url = Column(String(500))
    order_position = Column(Integer)
    included_story_ids = Column(JSON)  # List of story IDs included in highlight
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_profile_highlight', 'profile_id', 'order_position'),
    )


class ExternalLink(Base):
    """External/Deep Link model."""
    __tablename__ = "external_links"
    
    id = Column(String(36), primary_key=True)
    profile_id = Column(String(100), ForeignKey('profiles.profile_id'), index=True)
    post_id = Column(String(100), ForeignKey('posts.post_id'), index=True)
    link_type = Column(String(50))  # bio_link, shop_link, story_link, post_link
    destination_url = Column(String(1000))  # Destination URL (after redirects)
    page_title = Column(String(500))
    meta_description = Column(Text)
    contact_emails = Column(JSON)  # Extracted emails
    phone_numbers = Column(JSON)  # Extracted phone numbers
    domain_type = Column(String(50))  # Shop, SocialMedia, Blog, Affiliate, Portfolio
    shop_platform = Column(String(50))  # Shopify, WooCommerce, etc.
    raw_html = Column(Text)  # Raw HTML of the landing page
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_profile_links', 'profile_id'),
        Index('idx_post_links', 'post_id'),
    )


# Keep existing models for backward compatibility
from src.db.models import ContentItem, ViralityScore, EngagementMetrics, FeedbackLoop

__all__ = [
    'Base',
    'Profile',
    'Post',
    'PostLiker',
    'Comment',
    'Story',
    'StoryHighlight',
    'ExternalLink',
    'ContentItem',
    'ViralityScore',
    'EngagementMetrics',
    'FeedbackLoop',
]

