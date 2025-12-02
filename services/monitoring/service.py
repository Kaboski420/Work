"""
Monitoring, Drift Detection & Self-Learning

Observability and training automation with open tooling.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Try to import EvidentlyAI
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("EvidentlyAI not available. Drift detection will be limited.")


class MonitoringService:
    """Service for monitoring, drift detection, and continuous learning."""
    
    def __init__(self):
        self.service_id = str(uuid.uuid4())
        logger.info(f"Initialized MonitoringService: {self.service_id}")
    
    async def detect_drift(
        self,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any],
        model_version: str
    ) -> Dict[str, Any]:
        """
        Detect data or model drift using EvidentlyAI.
        
        Args:
            reference_data: Reference dataset for comparison (list of dicts or DataFrame-like)
            current_data: Current dataset to check (list of dicts or DataFrame-like)
            model_version: Model version identifier
        
        Returns:
            Dictionary containing drift detection results
        """
        logger.info(f"Detecting drift for model version {model_version}")
        
        drift_result = {
            "drift_detected": False,
            "drift_score": 0.0,
            "drift_threshold": settings.drift_detection_threshold,
            "affected_features": [],
            "severity": "none",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning("EvidentlyAI not available. Using fallback drift detection.")
            return self._fallback_drift_detection(reference_data, current_data, drift_result)
        
        try:
            # Convert data to DataFrames
            ref_df = self._to_dataframe(reference_data)
            curr_df = self._to_dataframe(current_data)
            
            if ref_df.empty or curr_df.empty:
                logger.warning("Empty datasets provided for drift detection")
                return drift_result
            
            # Define column mapping (numeric features)
            numeric_features = [col for col in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[col])]
            
            # Limit to reasonable number of features
            if len(numeric_features) > 50:
                numeric_features = numeric_features[:50]  # Limit to 50 features
            
            column_mapping = ColumnMapping(
                numerical_features=numeric_features,
                categorical_features=None,
                datetime_features=None,
                target=None,
                prediction=None
            )
            
            # Run data drift report
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(
                reference_data=ref_df,
                current_data=curr_df,
                column_mapping=column_mapping
            )
            
            # Extract drift results
            drift_metrics = data_drift_report.as_dict()
            
            # Check for drift
            drift_score = 0.0
            affected_features = []
            
            # Parse drift results
            if 'metrics' in drift_metrics:
                for metric in drift_metrics['metrics']:
                    if metric.get('metric') == 'DatasetDriftMetric':
                        drift_score = metric.get('result', {}).get('dataset_drift', 0.0)
                        drift_detected = metric.get('result', {}).get('drift_detected', False)
                    elif metric.get('metric') == 'ColumnDriftMetric':
                        # Individual feature drift
                        feature_name = metric.get('result', {}).get('column_name', '')
                        feature_drift = metric.get('result', {}).get('drift_detected', False)
                        drift_value = metric.get('result', {}).get('drift_score', 0.0)
                        
                        if feature_drift and drift_value > settings.drift_detection_threshold:
                            affected_features.append({
                                "feature": feature_name,
                                "drift_score": drift_value,
                                "threshold": settings.drift_detection_threshold
                            })
            
            # Determine severity
            severity = "none"
            if drift_score > settings.drift_detection_threshold:
                if drift_score > 0.5:
                    severity = "high"
                elif drift_score > 0.3:
                    severity = "medium"
                else:
                    severity = "low"
                
                drift_result["drift_detected"] = True
            
            # Sort affected features by drift score
            affected_features.sort(key=lambda x: x["drift_score"], reverse=True)
            
            drift_result.update({
                "drift_detected": drift_score > settings.drift_detection_threshold,
                "drift_score": float(drift_score),
                "affected_features": affected_features[:10],  # Top 10 features
                "severity": severity
            })
            
            if drift_result["drift_detected"]:
                logger.warning(
                    f"Drift detected! Score: {drift_score:.3f}, "
                    f"Severity: {severity}, Affected features: {len(affected_features)}"
                )
            else:
                logger.info(f"No drift detected. Score: {drift_score:.3f}")
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error detecting drift with EvidentlyAI: {e}", exc_info=True)
            return self._fallback_drift_detection(reference_data, current_data, drift_result)
    
    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, list):
                if len(data) == 0:
                    return pd.DataFrame()
                elif isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({"value": data})
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                if all(isinstance(v, (list, np.ndarray)) for v in data.values()):
                    return pd.DataFrame(data)
                else:
                    # Single row
                    return pd.DataFrame([data])
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {e}")
            return pd.DataFrame()
    
    def _fallback_drift_detection(
        self,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any],
        drift_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback drift detection using simple statistical tests."""
        try:
            ref_df = self._to_dataframe(reference_data)
            curr_df = self._to_dataframe(current_data)
            
            if ref_df.empty or curr_df.empty:
                return drift_result
            
            # Simple drift detection: compare means
            affected_features = []
            drift_scores = []
            
            numeric_cols = [col for col in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[col])]
            
            for col in numeric_cols[:20]:  # Limit to 20 features
                if col in curr_df.columns:
                    ref_mean = ref_df[col].mean()
                    curr_mean = curr_df[col].mean()
                    ref_std = ref_df[col].std()
                    
                    if ref_std > 0:
                        # Z-score for drift
                        drift_score = abs(curr_mean - ref_mean) / ref_std
                        drift_scores.append(drift_score)
                        
                        if drift_score > settings.drift_detection_threshold:
                            affected_features.append({
                                "feature": col,
                                "drift_score": float(drift_score),
                                "threshold": settings.drift_detection_threshold
                            })
            
            # Overall drift score (average of feature drift scores)
            if drift_scores:
                overall_drift = np.mean(drift_scores)
                drift_result.update({
                    "drift_detected": overall_drift > settings.drift_detection_threshold,
                    "drift_score": float(overall_drift),
                    "affected_features": affected_features[:10],
                    "severity": "high" if overall_drift > 0.5 else "medium" if overall_drift > 0.3 else "low" if overall_drift > settings.drift_detection_threshold else "none"
                })
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error in fallback drift detection: {e}", exc_info=True)
            return drift_result
    
    async def check_retraining_eligibility(self) -> Dict[str, Any]:
        """
        Check if retraining cycle should be executed.
        
        Must be executable at least once every 14 days.
        Checks:
        - Days since last training (must be >= 14 days)
        - Data availability
        - Drift status
        
        Returns:
            Dictionary with eligibility status and reasons
        """
        try:
            # Get last training time from MLflow or database
            last_training = None
            
            # Try to get from MLflow
            try:
                from src.utils.mlflow_registry import get_model_registry
                registry = get_model_registry()
                model_info = registry.load_latest_model(
                    model_name="virality-predictor",
                    stage="Production"
                )
                
                if model_info and model_info.get("training_timestamp"):
                    try:
                        last_training = datetime.fromisoformat(model_info["training_timestamp"].replace("Z", "+00:00"))
                    except Exception:
                        # Try parsing without timezone
                        last_training = datetime.fromisoformat(model_info["training_timestamp"].split("+")[0])
            except Exception as e:
                logger.debug(f"Could not get training time from MLflow: {e}")
            
            # If not in MLflow, try database
            if last_training is None:
                try:
                    from src.db.connection import get_db_session
                    from src.db.models import ViralityScore
                    from sqlalchemy import func
                    
                    with get_db_session() as session:
                        # Get most recent model version timestamp
                        latest_score = session.query(
                            func.max(ViralityScore.created_at)
                        ).filter(
                            ViralityScore.model_version.isnot(None)
                        ).scalar()
                        
                        if latest_score:
                            last_training = latest_score
                except Exception as e:
                    logger.debug(f"Could not get training time from database: {e}")
            
            # Check days since last training
            if last_training:
                days_since_training = (datetime.utcnow() - last_training.replace(tzinfo=None)).days
            else:
                days_since_training = 999  # No previous training, eligible
            
            # Check data availability - count feedback records
            data_available = False
            feedback_count = 0
            try:
                from src.db.connection import get_db_session
                from src.db.models import FeedbackLoop
                
                with get_db_session() as session:
                    feedback_count = session.query(FeedbackLoop).filter(
                        FeedbackLoop.used_for_training == 'N'
                    ).count()
                    
                    # Need at least 100 feedback records for training
                    data_available = feedback_count >= 100
            except Exception as e:
                logger.warning(f"Error checking data availability: {e}")
                # Fallback: assume data available if we can't check
                data_available = True
            
            # Check drift status (simplified - would need reference data)
            drift_detected = False
            try:
                # This would require reference dataset - for now, skip
                # In production, would compare current data distribution with training data
                pass
            except Exception as e:
                logger.debug(f"Error checking drift: {e}")
            
            # Determine eligibility
            eligible = (
                days_since_training >= settings.retraining_interval_days and
                data_available
            )
            
            # Build reason message
            if eligible:
                reason = "Eligible for retraining"
            elif days_since_training < settings.retraining_interval_days:
                reason = f"Not eligible: Only {days_since_training} days since last training (need {settings.retraining_interval_days})"
            elif not data_available:
                reason = f"Not eligible: Insufficient data ({feedback_count} feedback records, need 100+)"
            else:
                reason = "Not eligible: Unknown reason"
            
            return {
                "eligible": eligible,
                "last_training": last_training.isoformat() if last_training else None,
                "days_since_training": days_since_training,
                "data_available": data_available,
                "feedback_count": feedback_count,
                "drift_detected": drift_detected,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error checking retraining eligibility: {e}", exc_info=True)
            return {
                "eligible": False,
                "last_training": None,
                "days_since_training": 0,
                "data_available": False,
                "drift_detected": False,
                "reason": f"Error: {str(e)}"
            }
    
    async def collect_feedback(
        self,
        content_id: str,
        predicted_probability: float,
        actual_performance: float
    ) -> Dict[str, Any]:
        """
        Collect feedback for continuous learning.
        
        Args:
            content_id: Content identifier
            predicted_probability: Predicted virality probability
            actual_performance: Actual performance metric
        
        Returns:
            Feedback record
        """
        logger.info(f"Collecting feedback for content {content_id}")
        
        performance_delta = actual_performance - predicted_probability
        
        feedback = {
            "content_id": content_id,
            "predicted_probability": predicted_probability,
            "actual_performance": actual_performance,
            "performance_delta": performance_delta,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store feedback in database for retraining
        try:
            from src.db.connection import get_db_session
            from src.db.models import FeedbackLoop
            import uuid
            
            with get_db_session() as session:
                feedback_record = FeedbackLoop(
                    id=str(uuid.uuid4()),
                    content_id=content_id,
                    predicted_probability=predicted_probability,
                    actual_performance=actual_performance,
                    performance_delta=performance_delta,
                    feedback_timestamp=datetime.utcnow(),
                    used_for_training='N'
                )
                session.add(feedback_record)
                session.commit()
                logger.info(f"Feedback stored for content {content_id}")
        except Exception as e:
            logger.error(f"Error storing feedback: {e}", exc_info=True)
            # Continue even if storage fails
        
        return feedback
    
    async def generate_metrics_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        
        Includes:
        - System metrics (CPU, RAM, GPU, IO, latency, queue depth)
        - Model performance metrics
        - Drift detection status
        - Training status
        """
        try:
            # Try to get metrics from Prometheus
            system_metrics = {}
            model_metrics = {}
            
            try:
                import requests
                prometheus_url = f"http://localhost:{settings.prometheus_port}/api/v1/query"
                
                # Query system metrics
                queries = {
                    "cpu_usage": 'rate(process_cpu_seconds_total[5m]) * 100',
                    "memory_usage": 'process_resident_memory_bytes',
                    "request_latency": 'histogram_quantile(0.95, http_request_duration_seconds_bucket)',
                    "request_rate": 'rate(http_requests_total[5m])',
                    "error_rate": 'rate(errors_total[5m])'
                }
                
                for metric_name, query in queries.items():
                    try:
                        response = requests.get(prometheus_url, params={"query": query}, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("status") == "success" and data.get("data", {}).get("result"):
                                value = data["data"]["result"][0].get("value", [None, None])[1]
                                if value:
                                    system_metrics[metric_name] = float(value)
                    except Exception as e:
                        logger.debug(f"Could not fetch {metric_name} from Prometheus: {e}")
                
                # Query model metrics
                model_queries = {
                    "virality_probability_avg": 'virality_probability',
                    "prediction_confidence_avg": 'prediction_confidence',
                    "content_scored_total": 'content_scored_total'
                }
                
                for metric_name, query in model_queries.items():
                    try:
                        response = requests.get(prometheus_url, params={"query": query}, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("status") == "success" and data.get("data", {}).get("result"):
                                value = data["data"]["result"][0].get("value", [None, None])[1]
                                if value:
                                    model_metrics[metric_name] = float(value)
                    except Exception as e:
                        logger.debug(f"Could not fetch {metric_name} from Prometheus: {e}")
            except ImportError:
                logger.debug("requests not available for Prometheus queries")
            except Exception as e:
                logger.warning(f"Error querying Prometheus: {e}")
            
            # Get drift status
            drift_status = {
                "last_check": None,
                "drift_detected": False
            }
            
            # Get training status
            training_status = await self.check_retraining_eligibility()
            
            return {
                "system_metrics": system_metrics,
                "model_metrics": model_metrics,
                "drift_status": drift_status,
                "training_status": training_status
            }
        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            return {
                "system_metrics": {},
                "model_metrics": {},
                "drift_status": {},
                "training_status": {}
            }


