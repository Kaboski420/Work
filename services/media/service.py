"""
Media Intelligence Pipeline

High-throughput multimodal signal analysis using only open-source components.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Visual analysis will be limited.")

# Try to import librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio analysis will be limited.")

# Try to import ultralytics (YOLO)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available. Brand detection will be disabled.")


class MediaIntelligenceService:
    """Service for visual and audio analysis."""
    
    def __init__(self):
        self.service_id = str(uuid.uuid4())
        self.yolo_model = None
        
        # Initialize YOLO model for brand detection if available
        if YOLO_AVAILABLE:
            try:
                # Try to load a pre-trained YOLO model (YOLOv8n is lightweight)
                self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
                logger.info("YOLO model initialized for brand detection")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                self.yolo_model = None
        
        logger.info(f"Initialized MediaIntelligenceService: {self.service_id}")
    
    async def analyze_visual_content(
        self,
        video_frames: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze visual content from video frames.
        
        Args:
            video_frames: List of video frames as numpy arrays
            metadata: Video metadata
        
        Returns:
            Dictionary containing:
            - variance: Frame variance metrics
            - entropy: Visual entropy
            - motion_index: Motion indexing
            - color_gamut: Color gamut + saturation scoring
            - brand_detection: Brand/logo detection results
            - cut_density: Cut density metrics
            - hook_timing: Hook timing analysis
            - narrative_rhythm: Narrative rhythm detection
        """
        logger.info(f"Analyzing visual content: {len(video_frames)} frames")
        
        # Calculate visual metrics
        variance = await self._calculate_variance(video_frames)
        entropy = await self._calculate_entropy(video_frames)
        motion_index = await self._calculate_motion_index(video_frames)
        color_gamut = await self._analyze_color_gamut(video_frames)
        brand_detection = await self._detect_brands(video_frames)
        cut_density = await self._calculate_cut_density(video_frames, metadata)
        hook_timing = await self._analyze_hook_timing(video_frames, metadata)
        narrative_rhythm = await self._detect_narrative_rhythm(video_frames, metadata)
        
        result = {
            "variance": variance,
            "entropy": entropy,
            "motion_index": motion_index,
            "color_gamut": color_gamut,
            "brand_detection": brand_detection,
            "cut_density": cut_density,
            "hook_timing": hook_timing,
            "narrative_rhythm": narrative_rhythm
        }
        
        logger.info("Visual analysis complete")
        return result
    
    async def analyze_audio_content(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze audio content.
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Audio sample rate
            metadata: Audio metadata
        
        Returns:
            Dictionary containing:
            - bpm: BPM calculation
            - loudness: Loudness metrics
            - speech_music_segmentation: Speech/music segmentation
            - trending_sound_similarity: Similarity to trending sounds
            - harmonic_fingerprint: Harmonic fingerprint
        """
        logger.info(f"Analyzing audio content: {len(audio_data)} samples at {sample_rate}Hz")
        
        bpm = await self._calculate_bpm(audio_data, sample_rate)
        loudness = await self._calculate_loudness(audio_data)
        segmentation = await self._segment_speech_music(audio_data, sample_rate)
        trending_similarity = await self._compare_trending_sounds(audio_data, sample_rate)
        harmonic_fingerprint = await self._extract_harmonic_fingerprint(audio_data, sample_rate)
        
        result = {
            "bpm": bpm,
            "loudness": loudness,
            "speech_music_segmentation": segmentation,
            "trending_sound_similarity": trending_similarity,
            "harmonic_fingerprint": harmonic_fingerprint
        }
        
        logger.info("Audio analysis complete")
        return result
    
    # Visual analysis methods
    async def _calculate_variance(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Calculate frame variance metrics."""
        if not frames or len(frames) == 0:
            return {"frame_variance": 0.0, "temporal_variance": 0.0}
        
        try:
            # Calculate spatial variance for each frame
            frame_variances = []
            for frame in frames:
                if CV2_AVAILABLE and len(frame.shape) == 3:
                    # Convert to grayscale if needed
                    if frame.shape[2] == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = frame
                    frame_var = float(np.var(gray))
                else:
                    frame_var = float(np.var(frame))
                frame_variances.append(frame_var)
            
            # Average frame variance
            avg_frame_variance = np.mean(frame_variances) if frame_variances else 0.0
            
            # Temporal variance (variance of frame variances over time)
            temporal_variance = float(np.var(frame_variances)) if len(frame_variances) > 1 else 0.0
            
            return {
                "frame_variance": float(avg_frame_variance),
                "temporal_variance": temporal_variance
            }
        except Exception as e:
            logger.error(f"Error calculating variance: {e}")
            return {"frame_variance": 0.0, "temporal_variance": 0.0}
    
    async def _calculate_entropy(self, frames: List[np.ndarray]) -> float:
        """Calculate visual entropy."""
        if not frames or len(frames) == 0:
            return 0.0
        
        try:
            entropies = []
            for frame in frames:
                if CV2_AVAILABLE and len(frame.shape) == 3:
                    # Convert to grayscale
                    if frame.shape[2] == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = frame
                else:
                    gray = frame if len(frame.shape) == 2 else frame.flatten().reshape(int(np.sqrt(len(frame))), -1)
                
                # Calculate histogram
                hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
                hist = hist[hist > 0]  # Remove zeros
                
                # Normalize histogram
                hist = hist / hist.sum()
                
                # Calculate entropy: -sum(p * log2(p))
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)
            
            # Average entropy across frames
            avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            # Normalize to 0-1 range (max entropy for 256 bins is log2(256) = 8)
            normalized_entropy = avg_entropy / 8.0
            
            return float(min(max(normalized_entropy, 0.0), 1.0))
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0
    
    async def _calculate_motion_index(self, frames: List[np.ndarray]) -> float:
        """Calculate motion indexing using optical flow."""
        if not frames or len(frames) < 2:
            return 0.0
        
        if not CV2_AVAILABLE:
            # Fallback: simple frame difference
            try:
                motion_scores = []
                for i in range(1, len(frames)):
                    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                return float(np.mean(motion_scores) / 255.0) if motion_scores else 0.0
            except Exception as e:
                logger.error(f"Error calculating motion (fallback): {e}")
                return 0.0
        
        try:
            # Use optical flow for better motion detection
            motion_magnitudes = []
            
            for i in range(1, min(len(frames), 10)):  # Limit to first 10 frames for performance
                prev_frame = frames[i-1]
                curr_frame = frames[i]
                
                # Convert to grayscale
                if len(prev_frame.shape) == 3:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
                else:
                    prev_gray = prev_frame
                    curr_gray = curr_frame
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate magnitude of flow vectors
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_magnitude = np.mean(magnitude)
                motion_magnitudes.append(avg_magnitude)
            
            # Normalize motion index (typical range: 0-50 pixels)
            motion_index = float(np.mean(motion_magnitudes) / 50.0) if motion_magnitudes else 0.0
            return float(min(max(motion_index, 0.0), 1.0))
        except Exception as e:
            logger.error(f"Error calculating motion index: {e}")
            return 0.0
    
    async def _analyze_color_gamut(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze color gamut and saturation."""
        if not frames or len(frames) == 0:
            return {"saturation_score": 0.0, "color_diversity": 0.0}
        
        try:
            saturation_scores = []
            color_diversities = []
            
            for frame in frames[:10]:  # Sample first 10 frames
                if CV2_AVAILABLE and len(frame.shape) == 3:
                    # Convert to HSV
                    if frame.shape[2] == 3:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    else:
                        hsv = frame
                    
                    # Extract saturation channel
                    saturation = hsv[..., 1].astype(float) / 255.0
                    avg_saturation = float(np.mean(saturation))
                    saturation_scores.append(avg_saturation)
                    
                    # Color diversity: standard deviation of hue
                    hue = hsv[..., 0].astype(float)
                    hue_std = float(np.std(hue))
                    # Normalize (max std for 360 degrees is ~104)
                    color_diversity = hue_std / 104.0
                    color_diversities.append(color_diversity)
                else:
                    # Fallback for grayscale
                    saturation_scores.append(0.0)
                    color_diversities.append(0.0)
            
            return {
                "saturation_score": float(np.mean(saturation_scores)) if saturation_scores else 0.0,
                "color_diversity": float(np.mean(color_diversities)) if color_diversities else 0.0
            }
        except Exception as e:
            logger.error(f"Error analyzing color gamut: {e}")
            return {"saturation_score": 0.0, "color_diversity": 0.0}
    
    async def _detect_brands(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect brands/logos using YOLO."""
        if not self.yolo_model or not frames:
            return []
        
        try:
            all_detections = []
            
            # Process first few frames (for performance)
            for i, frame in enumerate(frames[:5]):
                try:
                    # Run YOLO detection
                    results = self.yolo_model(frame, verbose=False)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Filter for relevant classes (person, car, etc. - adjust based on brand detection needs)
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Only include high-confidence detections
                            if confidence > 0.5:
                                all_detections.append({
                                    "frame": i,
                                    "class_id": class_id,
                                    "confidence": confidence,
                                    "bbox": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                                })
                except Exception as e:
                    logger.warning(f"Error detecting brands in frame {i}: {e}")
                    continue
            
            return all_detections
        except Exception as e:
            logger.error(f"Error in brand detection: {e}")
            return []
    
    async def _calculate_cut_density(self, frames: List[np.ndarray], metadata: Dict[str, Any]) -> float:
        """Calculate cut density (scene changes per second)."""
        if not frames or len(frames) < 2:
            return 0.0
        
        try:
            cuts = 0
            threshold = 0.3  # Threshold for scene change detection
            
            for i in range(1, len(frames)):
                try:
                    # Calculate histogram difference
                    if CV2_AVAILABLE:
                        if len(frames[i-1].shape) == 3:
                            hist1 = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                            hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        else:
                            hist1 = cv2.calcHist([frames[i-1]], [0], None, [256], [0, 256])
                            hist2 = cv2.calcHist([frames[i]], [0], None, [256], [0, 256])
                        
                        # Compare histograms
                        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                        if correlation < (1.0 - threshold):
                            cuts += 1
                    else:
                        # Fallback: simple difference
                        diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                        if diff > threshold * 255.0:
                            cuts += 1
                except Exception as e:
                    logger.warning(f"Error comparing frames {i-1} and {i}: {e}")
                    continue
            
            # Calculate cut density (cuts per second)
            # Assume metadata has duration, or estimate from frame count
            duration = metadata.get("duration", len(frames) / 30.0)  # Default 30 fps
            cut_density = cuts / max(duration, 1.0)
            
            # Normalize (typical range: 0-10 cuts/second)
            normalized_density = min(cut_density / 10.0, 1.0)
            
            return float(normalized_density)
        except Exception as e:
            logger.error(f"Error calculating cut density: {e}")
            return 0.0
    
    async def _analyze_hook_timing(self, frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hook timing within first few seconds."""
        if not frames:
            return {"hook_detected": False, "hook_timestamp": None, "hook_score": 0.0}
        
        try:
            # Analyze first 3 seconds (assuming 30 fps = 90 frames)
            hook_frames = frames[:min(90, len(frames))]
            
            if len(hook_frames) < 2:
                return {"hook_detected": False, "hook_timestamp": None, "hook_score": 0.0}
            
            # Detect rapid changes in first few frames (hook indicators)
            hook_scores = []
            
            for i in range(1, min(len(hook_frames), 30)):  # First second
                try:
                    # Calculate frame difference
                    diff = np.mean(np.abs(hook_frames[i].astype(float) - hook_frames[i-1].astype(float)))
                    normalized_diff = diff / 255.0
                    hook_scores.append(normalized_diff)
                except Exception:
                    continue
            
            if hook_scores:
                avg_hook_score = float(np.mean(hook_scores))
                max_hook_score = float(np.max(hook_scores))
                
                # Hook detected if there's significant activity in first second
                hook_detected = avg_hook_score > 0.1 or max_hook_score > 0.3
                hook_timestamp = 0.0 if hook_detected else None
                
                return {
                    "hook_detected": hook_detected,
                    "hook_timestamp": hook_timestamp,
                    "hook_score": float(avg_hook_score)
                }
            
            return {"hook_detected": False, "hook_timestamp": None, "hook_score": 0.0}
        except Exception as e:
            logger.error(f"Error analyzing hook timing: {e}")
            return {"hook_detected": False, "hook_timestamp": None, "hook_score": 0.0}
    
    async def _detect_narrative_rhythm(self, frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Detect narrative rhythm patterns."""
        if not frames or len(frames) < 10:
            return {"rhythm_score": 0.0, "pattern": "unknown"}
        
        try:
            # Analyze cut frequency pattern
            cut_intervals = []
            prev_cut_frame = 0
            
            for i in range(1, len(frames)):
                try:
                    # Simple scene change detection
                    diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                    if diff > 0.2 * 255.0:  # Scene change threshold
                        interval = i - prev_cut_frame
                        if interval > 0:
                            cut_intervals.append(interval)
                        prev_cut_frame = i
                except Exception:
                    continue
            
            if len(cut_intervals) < 2:
                return {"rhythm_score": 0.0, "pattern": "static"}
            
            # Calculate rhythm consistency (lower std = more consistent rhythm)
            intervals_std = float(np.std(cut_intervals))
            intervals_mean = float(np.mean(cut_intervals))
            
            # Coefficient of variation (normalized consistency)
            cv = intervals_std / max(intervals_mean, 1.0)
            rhythm_score = 1.0 - min(cv, 1.0)  # Higher score = more consistent
            
            # Classify pattern
            if intervals_mean < 10:
                pattern = "fast"
            elif intervals_mean < 30:
                pattern = "medium"
            else:
                pattern = "slow"
            
            if cv < 0.3:
                pattern += "_consistent"
            else:
                pattern += "_varied"
            
            return {
                "rhythm_score": float(rhythm_score),
                "pattern": pattern,
                "avg_cut_interval": float(intervals_mean)
            }
        except Exception as e:
            logger.error(f"Error detecting narrative rhythm: {e}")
            return {"rhythm_score": 0.0, "pattern": "unknown"}
    
    # Audio analysis methods
    async def _calculate_bpm(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate BPM using librosa."""
        if not LIBROSA_AVAILABLE or len(audio_data) == 0:
            return 0.0
        
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Limit length for performance (first 30 seconds)
            max_samples = sample_rate * 30
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # Calculate tempo using librosa
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) and len(tempo) > 0 else float(tempo)
            
            # Normalize to reasonable range (60-180 BPM typical for music)
            normalized_bpm = (bpm - 60) / 120.0 if bpm > 0 else 0.0
            return float(min(max(normalized_bpm, 0.0), 1.0))
        except Exception as e:
            logger.error(f"Error calculating BPM: {e}")
            return 0.0
    
    async def _calculate_loudness(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate loudness metrics."""
        if len(audio_data) == 0:
            return {"rms": 0.0, "peak": 0.0, "lufs": 0.0}
        
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # RMS (Root Mean Square) - average power
            rms = float(np.sqrt(np.mean(audio_data**2)))
            
            # Peak level
            peak = float(np.max(np.abs(audio_data)))
            
            # LUFS approximation (simplified - full LUFS requires ITU-R BS.1770)
            # Using RMS as approximation
            if LIBROSA_AVAILABLE:
                try:
                    # Use librosa's loudness estimation if available
                    lufs = librosa.feature.rms(y=audio_data)[0].mean()
                    # Convert to approximate LUFS scale (-60 to 0 dB)
                    lufs_normalized = (lufs + 60) / 60.0
                    lufs = float(min(max(lufs_normalized, 0.0), 1.0))
                except Exception:
                    # Fallback: use RMS as proxy
                    lufs = float(min(max(rms * 2.0, 0.0), 1.0))
            else:
                # Fallback: use RMS as proxy
                lufs = float(min(max(rms * 2.0, 0.0), 1.0))
            
            return {
                "rms": float(rms),
                "peak": float(peak),
                "lufs": lufs
            }
        except Exception as e:
            logger.error(f"Error calculating loudness: {e}")
            return {"rms": 0.0, "peak": 0.0, "lufs": 0.0}
    
    async def _segment_speech_music(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Segment speech vs music."""
        if not LIBROSA_AVAILABLE or len(audio_data) == 0:
            return {"speech_ratio": 0.0, "music_ratio": 0.0, "segments": []}
        
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Limit length for performance
            max_samples = sample_rate * 30
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # Extract features for classification
            # Music typically has more harmonic content, speech has more spectral centroid variation
            
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # Simple heuristic: music has more stable spectral centroid, speech has more variation
            centroid_variance = float(np.var(spectral_centroids))
            zcr_mean = float(np.mean(zero_crossing_rate))
            
            # Segment into windows
            window_size = sample_rate * 2  # 2 second windows
            num_windows = len(audio_data) // window_size
            segments = []
            music_frames = 0
            speech_frames = 0
            
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                window = audio_data[start:end]
                
                if len(window) > 0:
                    # Calculate features for this window
                    window_centroid = librosa.feature.spectral_centroid(y=window, sr=sample_rate)[0]
                    window_zcr = librosa.feature.zero_crossing_rate(window)[0]
                    
                    window_centroid_var = float(np.var(window_centroid))
                    window_zcr_mean = float(np.mean(window_zcr))
                    
                    # Heuristic classification
                    # Music: lower centroid variance, moderate ZCR
                    # Speech: higher centroid variance, higher ZCR
                    if window_centroid_var < centroid_variance * 0.7 and window_zcr_mean < 0.15:
                        classification = "music"
                        music_frames += 1
                    else:
                        classification = "speech"
                        speech_frames += 1
                    
                    segments.append({
                        "start": float(start / sample_rate),
                        "end": float(end / sample_rate),
                        "classification": classification
                    })
            
            total_frames = music_frames + speech_frames
            music_ratio = music_frames / max(total_frames, 1)
            speech_ratio = speech_frames / max(total_frames, 1)
            
            return {
                "speech_ratio": float(speech_ratio),
                "music_ratio": float(music_ratio),
                "segments": segments[:20]  # Limit to first 20 segments
            }
        except Exception as e:
            logger.error(f"Error segmenting speech/music: {e}")
            return {"speech_ratio": 0.0, "music_ratio": 0.0, "segments": []}
    
    async def _compare_trending_sounds(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Compare with trending sounds using vector DB (Qdrant)."""
        # Note: This requires Qdrant setup and trending sound database
        # For now, return empty list - can be implemented when Qdrant is configured
        try:
            # TODO: Implement Qdrant integration when available
            # 1. Extract audio fingerprint/embedding
            # 2. Query Qdrant for similar sounds
            # 3. Return similarity scores
            
            logger.debug("Trending sound comparison not yet implemented (requires Qdrant setup)")
            return []
        except Exception as e:
            logger.error(f"Error comparing trending sounds: {e}")
            return []
    
    async def _extract_harmonic_fingerprint(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Extract harmonic fingerprint."""
        if not LIBROSA_AVAILABLE or len(audio_data) == 0:
            return {"fingerprint": [], "similarity_scores": []}
        
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Limit length for performance
            max_samples = sample_rate * 30
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # Extract harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
            
            # Calculate chroma features (harmonic fingerprint)
            chroma = librosa.feature.chroma(y=y_harmonic, sr=sample_rate)
            
            # Average chroma across time to get fingerprint
            fingerprint = np.mean(chroma, axis=1).tolist()
            
            # Calculate spectral features for similarity
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1).tolist()
            
            return {
                "fingerprint": fingerprint,  # 12-dimensional chroma vector
                "mfcc_features": mfcc_mean,  # 13-dimensional MFCC vector
                "similarity_scores": []  # To be populated when comparing with other sounds
            }
        except Exception as e:
            logger.error(f"Error extracting harmonic fingerprint: {e}")
            return {"fingerprint": [], "similarity_scores": []}


