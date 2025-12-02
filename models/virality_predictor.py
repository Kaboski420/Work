"""
Virality Prediction Model

Predicts the viral potential of content based on multimodal features.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # Set to None so type hints work
    logger.warning("numpy not available. Some features will be limited.")

# For type hints when numpy is not available
if TYPE_CHECKING:
    try:
        import numpy as np
    except ImportError:
        pass

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Using simple prediction model.")


class ViralityPredictor:
    """
    Virality prediction model.
    
    Uses ensemble methods to predict viral potential with calibrated probabilities.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model_path = model_path or "./models/virality_model.pkl"
        self.feature_names = []
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the prediction model."""
        if SKLEARN_AVAILABLE:
            # Use Gradient Boosting for better performance
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            # Calibrate for better probability estimates (meets Brier Score requirement)
            self.model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv=3
            )
            logger.info("Initialized virality prediction model")
        else:
            self.model = None
            logger.warning("scikit-learn not available. Using simple heuristic model.")
    
    def _extract_features(
        self,
        embeddings: Dict[str, Any],
        features: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Any:  # Returns np.ndarray if numpy available, otherwise list
        """
        Extract feature vector from embeddings, features, and metadata.
        
        Args:
            embeddings: Multimodal embeddings
            features: Extracted features
            metadata: Content metadata
        
        Returns:
            Feature vector
        """
        feature_list = []
        
        # Embedding statistics
        if embeddings.get("visual") is not None and NUMPY_AVAILABLE:
            visual = np.array(embeddings["visual"])
            feature_list.extend([
                float(visual.mean()),
                float(visual.std()),
                float(np.linalg.norm(visual)),
                float(visual.max()),
                float(visual.min())
            ])
        else:
            feature_list.extend([0.0] * 5)
        
        if embeddings.get("audio") is not None and NUMPY_AVAILABLE:
            audio = np.array(embeddings["audio"])
            feature_list.extend([
                float(audio.mean()),
                float(audio.std()),
                float(np.linalg.norm(audio)),
                float(audio.max()),
                float(audio.min())
            ])
        else:
            feature_list.extend([0.0] * 5)
        
        if embeddings.get("text") is not None and NUMPY_AVAILABLE:
            text = np.array(embeddings["text"])
            feature_list.extend([
                float(text.mean()),
                float(text.std()),
                float(np.linalg.norm(text)),
                float(text.max()),
                float(text.min())
            ])
        else:
            feature_list.extend([0.0] * 5)
        
        if embeddings.get("contextual") is not None and NUMPY_AVAILABLE:
            contextual = np.array(embeddings["contextual"])
            feature_list.extend([
                float(contextual.mean()),
                float(contextual.std()),
                float(np.linalg.norm(contextual))
            ])
        else:
            feature_list.extend([0.0] * 3)
        
        # Metadata features
        caption = metadata.get("caption", "")
        description = metadata.get("description", "")
        hashtags = metadata.get("hashtags", [])
        
        feature_list.extend([
            len(caption),
            len(description),
            len(hashtags),
            caption.count("!"),
            caption.count("?"),
            len(caption.split()) if caption else 0,
            len(description.split()) if description else 0,
        ])
        
        # Visual features if available
        visual_features = features.get("visual", {})
        feature_list.extend([
            visual_features.get("variance", {}).get("frame_variance", 0.0),
            visual_features.get("entropy", 0.0),
            visual_features.get("motion_index", 0.0),
            visual_features.get("color_gamut", {}).get("saturation_score", 0.0),
            visual_features.get("cut_density", 0.0),
        ])
        
        # Audio features if available
        audio_features = features.get("audio", {})
        feature_list.extend([
            audio_features.get("bpm", 0.0),
            audio_features.get("loudness", {}).get("rms", 0.0),
            audio_features.get("loudness", {}).get("peak", 0.0),
        ])
        
        # Text features if available
        text_features = features.get("text", {})
        feature_list.extend([
            text_features.get("trend_proximity", {}).get("trend_score", 0.0),
            text_features.get("hook_efficiency", {}).get("hook_score", 0.0),
        ])
        
        if NUMPY_AVAILABLE:
            return np.array(feature_list, dtype=np.float32)
        return feature_list
    
    def predict(
        self,
        embeddings: Dict[str, Any],
        features: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict virality probability.
        
        Args:
            embeddings: Multimodal embeddings
            features: Extracted features
            metadata: Content metadata
        
        Returns:
            Dictionary with probability and confidence
        """
        try:
            # Extract features
            feature_vector = self._extract_features(embeddings, features, metadata)
            
            if self.model is None or not SKLEARN_AVAILABLE:
                # Simple heuristic model as fallback
                # Based on text length, hashtags, and basic features
                score = 0.5  # Base score
                
                # Boost for hashtags
                hashtags = metadata.get("hashtags", [])
                score += min(len(hashtags) * 0.05, 0.2)
                
                # Boost for caption length (optimal range)
                caption_len = len(metadata.get("caption", ""))
                if 20 <= caption_len <= 200:
                    score += 0.1
                
                # Boost for exclamation marks
                caption = metadata.get("caption", "")
                score += min(caption.count("!") * 0.02, 0.1)
                
                # Normalize to [0, 1]
                probability = min(max(score, 0.0), 1.0)
                confidence = 0.7  # Medium confidence for heuristic
                
                return {
                    "probability": float(probability),
                    "confidence": float(confidence)
                }
            
            # Reshape for sklearn
            if NUMPY_AVAILABLE:
                feature_vector = feature_vector.reshape(1, -1)
            elif isinstance(feature_vector, list):
                feature_vector = [feature_vector]
            
            # Scale features
            if hasattr(self, 'scaler_fitted') and self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            else:
                # Fit scaler on first prediction (in production, fit on training data)
                self.scaler.fit(feature_vector)
                self.scaler_fitted = True
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                try:
                    # Check if model is fitted
                    if hasattr(self.model, 'calibrated_classifiers_'):
                        probabilities = self.model.predict_proba(feature_vector)[0]
                        # Get probability of positive class (virality)
                        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                    else:
                        # Model not fitted, use fallback
                        raise ValueError("Model not fitted")
                except (ValueError, AttributeError):
                    # Fallback if model not trained
                    probability = 0.5
            else:
                # Fallback if model not trained
                probability = 0.5
            
            # Calculate confidence based on prediction certainty
            confidence = abs(probability - 0.5) * 2  # Higher confidence when closer to 0 or 1
            
            return {
                "probability": float(probability),
                "confidence": float(min(confidence, 1.0))
            }
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            # Return default values on error
            return {
                "probability": 0.5,
                "confidence": 0.5
            }
    
    def train(self, X: Any, y: Any):
        """
        Train the model on data.
        
        Args:
            X: Feature matrix
            y: Labels (0 = not viral, 1 = viral)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train model: scikit-learn not available")
            return
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.scaler_fitted = True
            
            logger.info(f"Model trained on {len(X)} samples")
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
    
    def save(self, path: Optional[str] = None):
        """Save the model to disk."""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'scaler_fitted': getattr(self, 'scaler_fitted', False)
                }, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, path: Optional[str] = None):
        """Load the model from disk."""
        path = path or self.model_path
        
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                self.scaler = data.get('scaler')
                self.scaler_fitted = data.get('scaler_fitted', False)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

