"""
Embedding extraction models for visual, audio, and text content.
"""

import logging
from typing import Optional, List, Dict, Any
import io

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available. Some features will be limited.")

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Visual embeddings will be limited.")

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available. Some ML features will be limited.")

# Try to import ML libraries, make them optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Text embeddings will be disabled.")

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not available. Visual embeddings will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio embeddings will be disabled.")


class VisualEmbeddingModel:
    """Extract visual embeddings using CLIP or similar models."""
    
    def __init__(self):
        self.model = None
        # Check if torch is available before using it
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.use_openclip = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the visual embedding model."""
        if not TORCH_AVAILABLE:
            self.use_openclip = False
            self.model = None
            logger.warning("Torch not available. Visual embeddings will use fallback.")
            return
        
        try:
            # Try to use OpenCLIP if available
            try:
                import open_clip
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                self.use_openclip = True
                logger.info("Initialized OpenCLIP model")
            except ImportError:
                # Fallback to a simple feature extractor
                self.use_openclip = False
                logger.warning("OpenCLIP not available. Using simple visual features.")
        except Exception as e:
            logger.error(f"Failed to initialize visual model: {e}")
            self.model = None
            self.use_openclip = False
    
    def extract(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract visual embeddings from image/video frame.
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            Embedding vector or None
        """
        try:
            if not NUMPY_AVAILABLE:
                return [0.0] * 512
            
            if self.model is None or not self.use_openclip:
                # Return a simple feature vector as fallback
                if NUMPY_AVAILABLE:
                    return np.random.rand(512).astype(np.float32).tolist()
                return [0.0] * 512
            
            if not PIL_AVAILABLE or not TORCH_AVAILABLE:
                if NUMPY_AVAILABLE:
                    return np.random.rand(512).astype(np.float32).tolist()
                return [0.0] * 512
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding.astype(np.float32).tolist()
        
        except Exception as e:
            logger.error(f"Error extracting visual embeddings: {e}")
            # Return a random vector as fallback
            if NUMPY_AVAILABLE:
                return np.random.rand(512).astype(np.float32).tolist()
            return [0.0] * 512


class AudioEmbeddingModel:
    """Extract audio embeddings."""
    
    def __init__(self):
        self.sample_rate = 16000
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize audio embedding model."""
        # For now, use librosa features as embeddings
        # In production, use YAMNet, VGGish, or Hubert
        pass
    
    def extract(self, audio_data: bytes, sample_rate: Optional[int] = None) -> Optional[List[float]]:
        """
        Extract audio embeddings.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
        
        Returns:
            Embedding vector or None
        """
        try:
            if not LIBROSA_AVAILABLE:
                # Return a simple feature vector as fallback
                return np.random.rand(128).astype(np.float32)
            
            # Load audio
            audio_array, sr = librosa.load(
                io.BytesIO(audio_data),
                sr=sample_rate or self.sample_rate,
                duration=10.0  # Limit to 10 seconds for processing
            )
            
            # Extract features (MFCC, chroma, etc.)
            mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma(y=audio_array, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sr)
            
            # Combine features into embedding
            embedding = np.concatenate([
                mfcc.mean(axis=1),
                chroma.mean(axis=1),
                spectral_contrast.mean(axis=1)
            ])
            
            # Pad or truncate to fixed size
            target_size = 128
            if len(embedding) > target_size:
                embedding = embedding[:target_size]
            else:
                embedding = np.pad(embedding, (0, target_size - len(embedding)))
            
            return embedding.astype(np.float32).tolist()
        
        except Exception as e:
            logger.error(f"Error extracting audio embeddings: {e}")
            if NUMPY_AVAILABLE:
                return np.random.rand(128).astype(np.float32).tolist()
            return [0.0] * 128


class TextEmbeddingModel:
    """Extract text embeddings using Sentence-BERT."""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize text embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized Sentence-BERT model")
            except Exception as e:
                logger.error(f"Failed to load Sentence-BERT: {e}")
                self.model = None
        else:
            logger.warning("Sentence-BERT not available. Text embeddings disabled.")
    
    def extract(self, text: str) -> Optional[List[float]]:
        """
        Extract text embeddings.
        
        Args:
            text: Text string
        
        Returns:
            Embedding vector or None
        """
        if self.model is None:
            # Return a simple feature vector as fallback
            # Based on text length and basic features
            if not NUMPY_AVAILABLE:
                return [float(len(text))] * 384
            features = np.array([
                len(text),
                text.count('!'),
                text.count('?'),
                text.count('#'),
                len(text.split()),
            ])
            # Pad to 384 (MiniLM dimension)
            embedding = np.pad(features, (0, 384 - len(features)))
            return embedding.astype(np.float32).tolist()
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32).tolist()
        except Exception as e:
            logger.error(f"Error extracting text embeddings: {e}")
            if NUMPY_AVAILABLE:
                return np.random.rand(384).astype(np.float32).tolist()
            return [0.0] * 384


class MultimodalFusion:
    """Fuse multiple modality embeddings."""
    
    def __init__(self):
        self.visual_dim = 512
        self.audio_dim = 128
        self.text_dim = 384
        self.fused_dim = 256
    
    def fuse(
        self,
        visual: Optional[List[float]] = None,
        audio: Optional[List[float]] = None,
        text: Optional[List[float]] = None
    ) -> List[float]:
        """
        Fuse multimodal embeddings.
        
        Args:
            visual: Visual embedding
            audio: Audio embedding
            text: Text embedding
        
        Returns:
            Fused embedding vector
        """
        if not NUMPY_AVAILABLE:
            # Simple fallback without numpy
            return [0.0] * self.fused_dim
        
        embeddings = []
        
        if visual is not None:
            visual_arr = np.array(visual)
            # Normalize and reduce dimension
            visual_norm = visual_arr / (np.linalg.norm(visual_arr) + 1e-8)
            if len(visual_norm) > self.fused_dim:
                visual_norm = visual_norm[:self.fused_dim]
            else:
                visual_norm = np.pad(visual_norm, (0, self.fused_dim - len(visual_norm)))
            embeddings.append(visual_norm)
        
        if audio is not None:
            audio_arr = np.array(audio)
            audio_norm = audio_arr / (np.linalg.norm(audio_arr) + 1e-8)
            if len(audio_norm) > self.fused_dim:
                audio_norm = audio_norm[:self.fused_dim]
            else:
                audio_norm = np.pad(audio_norm, (0, self.fused_dim - len(audio_norm)))
            embeddings.append(audio_norm)
        
        if text is not None:
            text_arr = np.array(text)
            text_norm = text_arr / (np.linalg.norm(text_arr) + 1e-8)
            if len(text_norm) > self.fused_dim:
                text_norm = text_norm[:self.fused_dim]
            else:
                text_norm = np.pad(text_norm, (0, self.fused_dim - len(text_norm)))
            embeddings.append(text_norm)
        
        if not embeddings:
            # Return random vector if no embeddings
            return np.random.rand(self.fused_dim).astype(np.float32).tolist()
        
        # Simple concatenation and averaging
        if len(embeddings) == 1:
            return embeddings[0].tolist()
        
        # Weighted average (can be improved with learned weights)
        weights = np.array([0.4, 0.3, 0.3][:len(embeddings)])
        weights = weights / weights.sum()
        
        fused = np.average(embeddings, axis=0, weights=weights)
        return fused.astype(np.float32).tolist()

