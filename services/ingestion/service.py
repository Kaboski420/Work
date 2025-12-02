"""
Data Ingestion & Feature Extraction Service

Handles all input modalities (video, audio, text, engagement telemetry).
Orchestrated via Apache Airflow.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import numpy as np

from src.config import settings
from src.models.embeddings import (
    VisualEmbeddingModel,
    AudioEmbeddingModel,
    TextEmbeddingModel,
    MultimodalFusion
)
from src.services.media.service import MediaIntelligenceService
from src.utils.storage import StorageService
from src.utils.messaging import KafkaMessagingService

logger = logging.getLogger(__name__)

# Try to import video/audio processing libraries
try:
    import cv2
    import io
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Video frame extraction will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio extraction will be limited.")


class IngestionService:
    """Service for ingesting and extracting features from multimedia content."""
    
    def __init__(self):
        self.service_id = str(uuid.uuid4())
        # Initialize embedding models
        self.visual_model = VisualEmbeddingModel()
        self.audio_model = AudioEmbeddingModel()
        self.text_model = TextEmbeddingModel()
        self.fusion_model = MultimodalFusion()
        
        # Initialize text understanding service for comment quality analysis (TECH-309)
        from src.services.text.service import TextUnderstandingService
        self.text_service = TextUnderstandingService()
        
        # Initialize media intelligence service
        self.media_service = MediaIntelligenceService()
        
        # Initialize storage service
        self.storage = StorageService(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure
        )
        
        # Initialize Kafka messaging
        self.messaging = KafkaMessagingService(
            bootstrap_servers=settings.kafka_bootstrap_servers
        )
        
        logger.info(f"Initialized IngestionService: {self.service_id}")
    
    async def ingest_content(
        self,
        content_type: str,
        content_data: bytes,
        metadata: Dict[str, Any],
        platform: str
    ) -> Dict[str, Any]:
        """
        Ingest content and extract features.
        
        Args:
            content_type: Type of content (video, audio, text, etc.)
            content_data: Raw content bytes
            metadata: Content metadata (hashtags, captions, etc.)
            platform: Source platform (tiktok, instagram, etc.)
        
        Returns:
            Dictionary containing content_id, features, and embeddings
        """
        content_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        logger.info(f"Ingesting content {content_id} of type {content_type} from {platform}")
        
        # Generate multimodal embeddings
        embeddings = await self._extract_embeddings(
            content_type=content_type,
            content_data=content_data,
            metadata=metadata
        )
        
        # Extract features
        features = await self._extract_features(
            content_type=content_type,
            content_data=content_data,
            metadata=metadata,
            embeddings=embeddings
        )
        
        result = {
            "content_id": content_id,
            "platform": platform,
            "content_type": content_type,
            "timestamp": timestamp.isoformat(),
            "embeddings": embeddings,
            "features": features,
            "metadata": metadata
        }
        
        # Store features in MinIO
        storage_path = self.storage.store_features(
            content_id=content_id,
            features=features,
            embeddings=embeddings,
            metadata=metadata
        )
        if storage_path:
            result["storage_path"] = storage_path
        
        # Publish to Kafka ingestion topic
        self.messaging.produce(
            topic=settings.kafka_ingestion_topic,
            value=result,
            key=content_id
        )
        
        logger.info(f"Successfully ingested content {content_id}")
        return result
    
    async def _extract_embeddings(
        self,
        content_type: str,
        content_data: bytes,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract multimodal embeddings from content.
        
        Returns:
            Dictionary with visual, audio, text, and contextual embeddings
        """
        embeddings = {
            "visual": None,
            "audio": None,
            "text": None,
            "contextual": None
        }
        
        # TODO: Implement actual embedding extraction
        # - Visual: OpenCLIP/LAION CLIP
        # - Audio: YAMNet, VGGish, Hubert
        # - Text: Sentence-BERT, InstructorXL, E5
        # - Contextual: Multimodal fusion
        
        if content_type in ["video", "image"]:
            embeddings["visual"] = await self._extract_visual_embeddings(content_data)
        
        if content_type in ["video", "audio"]:
            embeddings["audio"] = await self._extract_audio_embeddings(content_data)
        
        if metadata.get("caption") or metadata.get("description"):
            embeddings["text"] = await self._extract_text_embeddings(metadata)
        
        embeddings["contextual"] = await self._fuse_embeddings(embeddings)
        
        return embeddings
    
    async def _extract_visual_embeddings(self, content_data: bytes) -> Optional[list]:
        """Extract visual embeddings using OpenCLIP/LAION CLIP."""
        try:
            embedding = self.visual_model.extract(content_data)
            if embedding is not None:
                return embedding.tolist()
            return None
        except Exception as e:
            logger.error(f"Error extracting visual embeddings: {e}")
            return None
    
    async def _extract_audio_embeddings(self, content_data: bytes) -> Optional[list]:
        """Extract audio embeddings using YAMNet, VGGish, or Hubert."""
        try:
            embedding = self.audio_model.extract(content_data)
            if embedding is not None:
                return embedding.tolist()
            return None
        except Exception as e:
            logger.error(f"Error extracting audio embeddings: {e}")
            return None
    
    async def _extract_text_embeddings(self, metadata: Dict[str, Any]) -> Optional[list]:
        """Extract text embeddings using Sentence-BERT, InstructorXL, or E5."""
        try:
            # Combine caption, description, and hashtags
            text_parts = []
            if metadata.get("caption"):
                text_parts.append(metadata["caption"])
            if metadata.get("description"):
                text_parts.append(metadata["description"])
            if metadata.get("hashtags"):
                text_parts.append(" ".join(metadata["hashtags"]))
            
            if not text_parts:
                return None
            
            text = " ".join(text_parts)
            embedding = self.text_model.extract(text)
            if embedding is not None:
                # Handle both list and numpy array
                if isinstance(embedding, list):
                    return embedding
                elif hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                else:
                    return list(embedding) if embedding else None
            return None
        except Exception as e:
            logger.error(f"Error extracting text embeddings: {e}")
            return None
    
    async def _fuse_embeddings(self, embeddings: Dict[str, Any]) -> Optional[list]:
        """Fuse multimodal embeddings using transformer-based fusion."""
        try:
            visual = embeddings.get("visual")
            audio = embeddings.get("audio")
            text = embeddings.get("text")
            
            fused = self.fusion_model.fuse(visual=visual, audio=audio, text=text)
            return fused
        except Exception as e:
            logger.error(f"Error fusing embeddings: {e}")
            return None
    
    async def _extract_features(
        self,
        content_type: str,
        content_data: bytes,
        metadata: Dict[str, Any],
        embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract features from content.
        
        Returns:
            Dictionary of extracted features
        """
        features = {
            "visual": {},
            "audio": {},
            "text": {},
            "temporal": {}
        }
        
        # Text analysis including comment quality (TECH-309: GPT/BERT Integration)
        try:
            text_analysis = await self.text_service.analyze_text_content(
                caption=metadata.get("caption"),
                description=metadata.get("description"),
                hashtags=metadata.get("hashtags", []),
                metadata=metadata
            )
            
            # Populate text features with analysis results
            features["text"] = {
                "trend_proximity": text_analysis.get("trend_proximity", {}),
                "hook_efficiency": text_analysis.get("hook_efficiency", {}),
                "comment_quality": text_analysis.get("comment_quality", {}),
                "emotion": text_analysis.get("emotion", {}),
                "intent": text_analysis.get("intent", {}),
                "virality_triggers": text_analysis.get("virality_triggers", []),
                "brand_safety": text_analysis.get("brand_safety", {})
            }
            
            logger.debug(f"Extracted text features including comment quality for content")
        except Exception as e:
            logger.warning(f"Error in text analysis: {e}. Using empty text features.")
            # Fallback: ensure comment_quality exists
            features["text"]["comment_quality"] = {"quality_score": 0.5}
        
        # Extract visual features using MediaIntelligenceService
        if content_type in ["video", "image"]:
            try:
                video_frames = await self._extract_video_frames(content_data, content_type)
                if video_frames:
                    visual_analysis = await self.media_service.analyze_visual_content(
                        video_frames=video_frames,
                        metadata=metadata
                    )
                    features["visual"] = {
                        "variance": visual_analysis.get("variance", {}),
                        "entropy": visual_analysis.get("entropy", 0.0),
                        "motion_index": visual_analysis.get("motion_index", 0.0),
                        "color_gamut": visual_analysis.get("color_gamut", {}),
                        "brand_detection": visual_analysis.get("brand_detection", []),
                        "cut_density": visual_analysis.get("cut_density", 0.0),
                        "hook_timing": visual_analysis.get("hook_timing", {}),
                        "narrative_rhythm": visual_analysis.get("narrative_rhythm", {})
                    }
                    logger.debug("Extracted visual features using MediaIntelligenceService")
            except Exception as e:
                logger.warning(f"Error extracting visual features: {e}")
        
        # Extract audio features using MediaIntelligenceService
        if content_type in ["video", "audio"]:
            try:
                audio_data, sample_rate = await self._extract_audio_data(content_data, content_type)
                if audio_data is not None and len(audio_data) > 0:
                    audio_analysis = await self.media_service.analyze_audio_content(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        metadata=metadata
                    )
                    features["audio"] = {
                        "bpm": audio_analysis.get("bpm", 0.0),
                        "loudness": audio_analysis.get("loudness", {}),
                        "speech_music_segmentation": audio_analysis.get("speech_music_segmentation", {}),
                        "trending_sound_similarity": audio_analysis.get("trending_sound_similarity", []),
                        "harmonic_fingerprint": audio_analysis.get("harmonic_fingerprint", {})
                    }
                    logger.debug("Extracted audio features using MediaIntelligenceService")
            except Exception as e:
                logger.warning(f"Error extracting audio features: {e}")
        
        return features
    
    async def _extract_video_frames(
        self,
        content_data: bytes,
        content_type: str
    ) -> List:
        """Extract video frames from content."""
        frames = []
        
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available. Cannot extract video frames.")
            return frames
        
        try:
            if content_type == "image":
                # Single frame for images
                nparr = np.frombuffer(content_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            elif content_type == "video":
                # Extract frames from video
                nparr = np.frombuffer(content_data, np.uint8)
                # Create temporary file-like object
                video_stream = io.BytesIO(content_data)
                
                # Use cv2.VideoCapture with BytesIO (requires writing to temp file or using different approach)
                # For now, sample frames at regular intervals
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(content_data)
                    tmp_path = tmp_file.name
                
                try:
                    cap = cv2.VideoCapture(tmp_path)
                    frame_count = 0
                    max_frames = 30  # Limit to 30 frames for performance
                    
                    while cap.isOpened() and len(frames) < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Sample every nth frame
                        if frame_count % 10 == 0:  # Every 10th frame
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        
                        frame_count += 1
                    
                    cap.release()
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error extracting video frames: {e}")
        
        return frames
    
    async def _extract_audio_data(
        self,
        content_data: bytes,
        content_type: str
    ) -> tuple:
        """Extract audio data from content."""
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available. Cannot extract audio data.")
            return None, 16000
        
        try:
            if content_type == "audio":
                # Direct audio file
                audio_stream = io.BytesIO(content_data)
                audio_data, sample_rate = librosa.load(audio_stream, sr=None, duration=30.0)  # Limit to 30 seconds
                return audio_data, sample_rate
            elif content_type == "video":
                # Extract audio from video
                import tempfile
                import os
                import subprocess
                
                # Write video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(content_data)
                    tmp_video_path = tmp_video.name
                
                # Extract audio using ffmpeg (if available) or librosa
                try:
                    # Try librosa first (simpler)
                    audio_stream = io.BytesIO(content_data)
                    # librosa can sometimes handle video files, but may fail
                    try:
                        audio_data, sample_rate = librosa.load(audio_stream, sr=None, duration=30.0)
                        return audio_data, sample_rate
                    except Exception:
                        # Fallback: use ffmpeg to extract audio
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                            tmp_audio_path = tmp_audio.name
                        
                        try:
                            # Use ffmpeg to extract audio
                            subprocess.run(
                                ['ffmpeg', '-i', tmp_video_path, '-vn', '-acodec', 'pcm_s16le', 
                                 '-ar', '44100', '-ac', '1', '-y', tmp_audio_path],
                                check=True,
                                capture_output=True
                            )
                            
                            # Load extracted audio
                            audio_data, sample_rate = librosa.load(tmp_audio_path, sr=None, duration=30.0)
                            return audio_data, sample_rate
                        except Exception as e:
                            logger.warning(f"Could not extract audio from video: {e}")
                            return None, 16000
                        finally:
                            if os.path.exists(tmp_audio_path):
                                os.unlink(tmp_audio_path)
                finally:
                    if os.path.exists(tmp_video_path):
                        os.unlink(tmp_video_path)
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return None, 16000
        
        return None, 16000


