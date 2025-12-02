"""MLflow model registry integration."""

import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Model registry features will be disabled.")


class MLflowRegistry:
    """MLflow model registry service for model versioning and loading."""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLflow registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = None
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                
                # Get or create experiment
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(experiment_name)
                
                # Initialize MLflow client
                self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                
                logger.info(f"MLflow registry initialized: {tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}. Model registry disabled.")
                self.client = None
        else:
            logger.warning("MLflow not available. Model registry disabled.")
    
    def load_latest_model(
        self,
        model_name: str = "virality-predictor",
        stage: str = "Production"
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest model from MLflow registry.
        
        Args:
            model_name: Model name in registry
            stage: Model stage (Production, Staging, None)
        
        Returns:
            Dictionary with model info and loaded model, or None
        """
        if not MLFLOW_AVAILABLE or not self.client:
            logger.warning("MLflow not available. Returning default model info.")
            return self._get_default_model_info()
        
        try:
            # Try to get latest version from registry
            try:
                if stage:
                    latest_version = self.client.get_latest_versions(
                        model_name,
                        stages=[stage]
                    )
                else:
                    # Get all versions and sort
                    versions = self.client.search_model_versions(
                        f"name='{model_name}'"
                    )
                    if versions:
                        latest_version = [max(versions, key=lambda v: int(v.version))]
                    else:
                        latest_version = []
                
                if latest_version:
                    model_version = latest_version[0]
                    version = model_version.version
                    run_id = model_version.run_id
                    
                    # Get run info
                    run = self.client.get_run(run_id)
                    
                    # Load model
                    model_uri = f"models:/{model_name}/{stage}" if stage else f"models:/{model_name}/{version}"
                    
                    try:
                        model = mlflow.sklearn.load_model(model_uri)
                    except Exception as e:
                        logger.warning(f"Could not load model from {model_uri}: {e}")
                        # Fallback to local model
                        return self._get_default_model_info()
                    
                    # Get model hash from run tags or compute from artifact
                    model_hash = run.data.tags.get("model_hash", self._compute_model_hash(model))
                    
                    # Get training timestamp
                    training_timestamp = run.info.start_time
                    if training_timestamp:
                        training_time = datetime.fromtimestamp(
                            training_timestamp / 1000
                        ).isoformat()
                    else:
                        training_time = datetime.utcnow().isoformat()
                    
                    logger.info(f"Loaded model {model_name} version {version} from stage {stage}")
                    
                    return {
                        "model": model,
                        "version": version,
                        "hash": model_hash,
                        "training_timestamp": training_time,
                        "stage": stage,
                        "run_id": run_id
                    }
                else:
                    logger.warning(f"No model found in registry for {model_name} stage {stage}")
                    return self._get_default_model_info()
                    
            except Exception as e:
                logger.warning(f"Error loading from registry: {e}. Using default model.")
                return self._get_default_model_info()
                
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}", exc_info=True)
            return self._get_default_model_info()
    
    def register_model(
        self,
        model: Any,
        model_name: str = "virality-predictor",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Register a model in MLflow registry.
        
        Args:
            model: Model to register
            model_name: Model name
            run_id: Optional run ID (creates new run if None)
            tags: Optional tags to add
        
        Returns:
            Model version or None
        """
        if not MLFLOW_AVAILABLE or not self.client:
            logger.warning("MLflow not available. Cannot register model.")
            return None
        
        try:
            # Compute model hash
            model_hash = self._compute_model_hash(model)
            
            # Start or use existing run
            if run_id:
                mlflow.start_run(run_id=run_id)
            else:
                mlflow.start_run()
            
            try:
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
                
                # Add tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Add model hash
                mlflow.set_tag("model_hash", model_hash)
                mlflow.set_tag("model_name", model_name)
                
                # Get model version
                latest_versions = self.client.get_latest_versions(model_name, stages=[])
                if latest_versions:
                    version = latest_versions[0].version
                    logger.info(f"Registered model {model_name} version {version}")
                    return version
                else:
                    logger.warning("Model registered but version not found")
                    return None
                    
            finally:
                mlflow.end_run()
                
        except Exception as e:
            logger.error(f"Error registering model: {e}", exc_info=True)
            return None
    
    def _compute_model_hash(self, model: Any) -> str:
        """
        Compute hash of model for reproducibility.
        
        Args:
            model: Model object
        
        Returns:
            SHA256 hash string
        """
        try:
            import pickle
            import hashlib
            
            # Serialize model
            model_bytes = pickle.dumps(model)
            # Compute hash
            hash_obj = hashlib.sha256(model_bytes)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute model hash: {e}")
            return "unknown"
    
    def _get_default_model_info(self) -> Dict[str, Any]:
        """Get default model info when MLflow is not available."""
        return {
            "model": None,  # Will use predictor's default model
            "version": "0.1.0",
            "hash": "default",
            "training_timestamp": datetime.utcnow().isoformat(),
            "stage": "Development"
        }


def get_model_registry() -> MLflowRegistry:
    """Get or create MLflow registry instance."""
    from src.config import settings
    return MLflowRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name
    )

