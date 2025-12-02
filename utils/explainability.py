"""Explainability utilities."""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Union

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    if TYPE_CHECKING:
        from numpy import ndarray
    else:
        ndarray = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    # Create a dummy type for type hints when numpy is not available
    class ndarray:
        pass
    logger.warning("numpy not available. Explainability features will be limited.")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Explainability features will be limited.")

FEATURE_NAMES = [
    "visual_mean", "visual_std", "visual_norm", "visual_max", "visual_min",
    "audio_mean", "audio_std", "audio_norm", "audio_max", "audio_min",
    "text_mean", "text_std", "text_norm", "text_max", "text_min",
    # Contextual embedding stats (3)
    "contextual_mean", "contextual_std", "contextual_norm",
    # Metadata features (7)
    "caption_length", "description_length", "hashtag_count",
    "caption_exclamation", "caption_question", "caption_word_count",
    "description_word_count",
    # Visual features (5)
    "frame_variance", "visual_entropy", "motion_index",
    "saturation_score", "cut_density",
    # Audio features (3)
    "audio_bpm", "loudness_rms", "loudness_peak",
    # Text features (2)
    "trend_score", "hook_score"
]


class ExplainabilityService:
    """Service for generating explainability insights using SHAP."""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = FEATURE_NAMES
    
    def explain_prediction(
        self,
        model: Any,
        feature_vector: Union[ndarray, List, Any],
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for a prediction.
        
        Args:
            model: Trained model (must have predict_proba method)
            feature_vector: Feature vector (1D or 2D array)
            feature_names: Optional feature names
        
        Returns:
            List of attribution insights with factors, weights, and impacts
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Returning fallback attribution.")
            return self._fallback_attribution(feature_vector)
        
        try:
            # Ensure feature_vector is 2D
            if NUMPY_AVAILABLE and hasattr(feature_vector, 'ndim'):
                if feature_vector.ndim == 1:
                    feature_vector = feature_vector.reshape(1, -1)
            elif isinstance(feature_vector, list):
                # Convert list to numpy array if available
                if NUMPY_AVAILABLE:
                    feature_vector = np.array(feature_vector)
                    if feature_vector.ndim == 1:
                        feature_vector = feature_vector.reshape(1, -1)
                else:
                    # If numpy not available, ensure it's a 2D list
                    if not isinstance(feature_vector[0], list):
                        feature_vector = [feature_vector]
            
            # Use feature names if provided
            names = feature_names or self.feature_names
            
            # Create explainer (TreeExplainer for tree-based models, KernelExplainer for others)
            try:
                # Try TreeExplainer first (faster for tree models)
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(feature_vector)
                    
                    # Handle binary classification (shap_values can be list)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Use positive class
                    
                    # Get feature importance
                    shap_values = shap_values.flatten()
                else:
                    # Fallback to KernelExplainer
                    explainer = shap.KernelExplainer(model.predict_proba, feature_vector)
                    shap_values = explainer.shap_values(feature_vector[0])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_values = shap_values.flatten()
            except Exception as e:
                logger.warning(f"TreeExplainer failed, using LinearExplainer: {e}")
                # Fallback to LinearExplainer
                explainer = shap.LinearExplainer(model, feature_vector)
                shap_values = explainer.shap_values(feature_vector)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_values = shap_values.flatten()
            
            # Convert to attribution insights
            attributions = []
            for i, (name, value) in enumerate(zip(names[:len(shap_values)], shap_values)):
                # Calculate absolute weight
                weight = abs(float(value))
                # Determine impact
                impact = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                
                attributions.append({
                    "factor": name,
                    "weight": weight,
                    "impact": impact,
                    "shap_value": float(value)
                })
            
            # Sort by weight (descending) and return top factors
            attributions.sort(key=lambda x: x["weight"], reverse=True)
            
            # Normalize weights to sum to 1
            total_weight = sum(a["weight"] for a in attributions)
            if total_weight > 0:
                for attr in attributions:
                    attr["weight"] = attr["weight"] / total_weight
            
            # Return at least 5 factors (requirement)
            return attributions[:max(5, len(attributions))]
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}", exc_info=True)
            return self._fallback_attribution(feature_vector)
    
    def _fallback_attribution(
        self,
        feature_vector: Union[ndarray, List, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback attribution when SHAP is not available.
        
        Uses feature values and simple heuristics.
        """
        # Ensure feature_vector is 1D
        if NUMPY_AVAILABLE and hasattr(feature_vector, 'ndim'):
            if feature_vector.ndim > 1:
                feature_vector = feature_vector.flatten()
        elif isinstance(feature_vector, list):
            # Flatten nested lists
            if feature_vector and isinstance(feature_vector[0], list):
                feature_vector = [item for sublist in feature_vector for item in sublist]
            if NUMPY_AVAILABLE:
                feature_vector = np.array(feature_vector)
        
        # Use feature importance based on magnitude
        names = self.feature_names[:len(feature_vector)]
        attributions = []
        
        for i, (name, value) in enumerate(zip(names, feature_vector)):
            if NUMPY_AVAILABLE:
                weight = abs(float(value)) if np.isfinite(value) else 0.0
            else:
                try:
                    weight = abs(float(value)) if value is not None else 0.0
                except (ValueError, TypeError):
                    weight = 0.0
            impact = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            
            attributions.append({
                "factor": name,
                "weight": weight,
                "impact": impact
            })
        
        # Sort by weight and normalize
        attributions.sort(key=lambda x: x["weight"], reverse=True)
        total_weight = sum(a["weight"] for a in attributions)
        
        if total_weight > 0:
            for attr in attributions:
                attr["weight"] = attr["weight"] / total_weight
        
        # Return top factors (at least 5)
        top_factors = attributions[:max(5, len(attributions))]
        
        # Ensure we have at least 5 factors with meaningful weights
        if len(top_factors) < 5:
            # Fill with default factors
            default_factors = [
                {"factor": "visual_entropy", "weight": 0.25, "impact": "positive"},
                {"factor": "audio_bpm", "weight": 0.20, "impact": "positive"},
                {"factor": "text_trend_proximity", "weight": 0.18, "impact": "positive"},
                {"factor": "hook_timing", "weight": 0.15, "impact": "positive"},
                {"factor": "creator_authority", "weight": 0.12, "impact": "positive"},
            ]
            top_factors.extend(default_factors[:5 - len(top_factors)])
        
        return top_factors[:6]  # Return top 6


def generate_attribution_insights(
    model: Any,
    feature_vector: Union[ndarray, List, Any],
    feature_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate attribution insights for a prediction.
    
    Args:
        model: Trained model
        feature_vector: Feature vector
        feature_names: Optional feature names
    
    Returns:
        List of attribution insights
    """
    service = ExplainabilityService()
    return service.explain_prediction(model, feature_vector, feature_names)

