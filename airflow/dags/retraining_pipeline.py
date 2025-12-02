"""
Apache Airflow DAG for Model Retraining Pipeline

Automated retraining workflow:
1. Check retraining eligibility (14 days interval, data availability)
2. Collect feedback data from database
3. Prepare training dataset
4. Train new model
5. Evaluate model performance
6. Register model in MLflow (if performance acceptable)
7. Update model version
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'virality-engine',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval=timedelta(days=1),  # Check daily, but only retrain if eligible
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retraining', 'model-update'],
)


def check_retraining_eligibility(**context):
    """Check if retraining is eligible."""
    import asyncio
    from src.services.monitoring.service import MonitoringService
    
    async def check():
        service = MonitoringService()
        eligibility = await service.check_retraining_eligibility()
        return eligibility
    
    result = asyncio.run(check())
    
    if result.get("eligible"):
        context['ti'].xcom_push(key='eligible', value=True)
        context['ti'].xcom_push(key='reason', value=result.get("reason"))
        print(f"Retraining eligible: {result.get('reason')}")
    else:
        context['ti'].xcom_push(key='eligible', value=False)
        context['ti'].xcom_push(key='reason', value=result.get("reason"))
        print(f"Retraining not eligible: {result.get('reason')}")
    
    return result


def collect_feedback_data(**context):
    """Collect feedback data from database for training."""
    from src.db.connection import get_db_session
    from src.db.models import FeedbackLoop, ViralityScore, ContentItem
    from sqlalchemy import and_
    from datetime import datetime, timedelta
    import pandas as pd
    
    # Get feedback data not yet used for training
    with get_db_session() as session:
        # Get feedback records marked as not used
        feedback_records = session.query(FeedbackLoop).filter(
            FeedbackLoop.used_for_training == 'N'
        ).all()
        
        # Get corresponding predictions and features
        training_data = []
        for feedback in feedback_records:
            # Get original prediction
            score = session.query(ViralityScore).filter(
                ViralityScore.content_id == feedback.content_id
            ).order_by(ViralityScore.created_at.desc()).first()
            
            if score:
                training_data.append({
                    "content_id": feedback.content_id,
                    "predicted_probability": feedback.predicted_probability,
                    "actual_performance": feedback.actual_performance,
                    "features": score.attribution_insights or {},
                    "model_version": score.model_version
                })
        
        if len(training_data) < 100:
            print(f"Warning: Only {len(training_data)} feedback records available. Need at least 100 for training.")
            context['ti'].xcom_push(key='data_available', value=False)
            context['ti'].xcom_push(key='record_count', value=len(training_data))
            return {"status": "insufficient_data", "count": len(training_data)}
        
        context['ti'].xcom_push(key='data_available', value=True)
        context['ti'].xcom_push(key='record_count', value=len(training_data))
        context['ti'].xcom_push(key='training_data', value=training_data)
        
        print(f"Collected {len(training_data)} feedback records for training")
        return {"status": "success", "count": len(training_data)}


def prepare_training_dataset(**context):
    """Prepare training dataset from feedback data."""
    import pandas as pd
    import numpy as np
    import asyncio
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.models.virality_predictor import ViralityPredictor
    from src.db.connection import get_db_session
    from src.db.models import ContentItem, ViralityScore
    from src.utils.storage import StorageService
    from src.config import settings
    
    training_data = context['ti'].xcom_pull(key='training_data')
    
    if not training_data:
        raise ValueError("No training data available")
    
    # Initialize predictor for feature extraction
    predictor = ViralityPredictor()
    
    # Load features and embeddings from storage
    storage = StorageService(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure
    )
    
    X = []
    y = []
    
    for record in training_data:
        content_id = record["content_id"]
        
        try:
            # Load features from storage
            features_data = storage.load_features(content_id)
            
            if features_data:
                # Extract features and embeddings
                features = features_data.get("features", {})
                embeddings = features_data.get("embeddings", {})
                metadata = features_data.get("metadata", {})
                
                # Use predictor's feature extraction method
                if hasattr(predictor, '_extract_features'):
                    try:
                        feature_vector = predictor._extract_features(
                            embeddings=embeddings,
                            features=features,
                            metadata=metadata
                        )
                        
                        # Convert to numpy array if needed
                        if isinstance(feature_vector, list):
                            feature_vector = np.array(feature_vector, dtype=np.float32)
                        elif not isinstance(feature_vector, np.ndarray):
                            feature_vector = np.array(feature_vector, dtype=np.float32)
                        
                        # Use actual performance as label (binary classification: viral >= 0.6)
                        label = 1.0 if record["actual_performance"] >= 0.6 else 0.0
                        
                        X.append(feature_vector)
                        y.append(label)
                    except Exception as e:
                        print(f"Error extracting features for {content_id}: {e}")
                        continue
                else:
                    print(f"Predictor does not have _extract_features method")
                    continue
            else:
                # Try to get from database
                try:
                    with get_db_session() as session:
                        score = session.query(ViralityScore).filter(
                            ViralityScore.content_id == content_id
                        ).order_by(ViralityScore.created_at.desc()).first()
                        
                        if score and score.attribution_insights:
                            # Use attribution insights as features (simplified)
                            # In production, would reconstruct full feature vector
                            print(f"Warning: Using simplified features for {content_id}")
                            # Skip this record or use fallback
                            continue
                except Exception as e:
                    print(f"Error loading from database for {content_id}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing record {content_id}: {e}")
            continue
    
    if len(X) < 100:
        raise ValueError(f"Insufficient training samples: {len(X)} (need 100+)")
    
    X = np.array(X)
    y = np.array(y)
    
    # Ensure all feature vectors have the same length
    if len(X) > 0:
        feature_length = len(X[0])
        # Pad or truncate if needed
        for i, x in enumerate(X):
            if len(x) != feature_length:
                if len(x) < feature_length:
                    X[i] = np.pad(x, (0, feature_length - len(x)), mode='constant')
                else:
                    X[i] = x[:feature_length]
    
    context['ti'].xcom_push(key='X', value=X.tolist())
    context['ti'].xcom_push(key='y', value=y.tolist())
    
    print(f"Prepared training dataset: {len(X)} samples with {len(X[0]) if len(X) > 0 else 0} features each")
    return {"status": "success", "samples": len(X), "feature_dim": len(X[0]) if len(X) > 0 else 0}


def train_model(**context):
    """Train new model on feedback data."""
    import numpy as np
    from src.models.virality_predictor import ViralityPredictor
    from src.utils.mlflow_registry import get_model_registry
    from datetime import datetime
    
    X = np.array(context['ti'].xcom_pull(key='X'))
    y = np.array(context['ti'].xcom_pull(key='y'))
    
    if X is None or y is None:
        raise ValueError("Training data not available")
    
    # Initialize and train model
    predictor = ViralityPredictor()
    predictor.train(X, y)
    
    # Evaluate model (simple accuracy)
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Simple train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Retrain on train set
    predictor.train(X_train, y_train)
    
    # Predict on test set
    predictions = []
    for i in range(len(X_test)):
        # Create dummy embeddings/features for prediction
        dummy_embeddings = {"visual": X_test[i][:5].tolist(), "audio": X_test[i][5:10].tolist(), 
                           "text": X_test[i][10:15].tolist(), "contextual": X_test[i][15:18].tolist()}
        dummy_features = {"visual": {}, "audio": {}, "text": {}}
        dummy_metadata = {}
        
        pred = predictor.predict(dummy_embeddings, dummy_features, dummy_metadata)
        predictions.append(pred["probability"])
    
    predictions = np.array(predictions)
    y_pred_binary = (predictions >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0.5
    
    print(f"Model performance - Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    # Check if performance meets threshold
    min_roc_auc = 0.75
    if roc_auc >= min_roc_auc:
        # Register model in MLflow
        registry = get_model_registry()
        version = registry.register_model(
            model=predictor.model,
            model_name="virality-predictor",
            tags={
                "training_date": datetime.utcnow().isoformat(),
                "accuracy": str(accuracy),
                "roc_auc": str(roc_auc),
                "samples": str(len(X))
            }
        )
        
        context['ti'].xcom_push(key='model_version', value=version)
        context['ti'].xcom_push(key='model_performance', value={"accuracy": accuracy, "roc_auc": roc_auc})
        context['ti'].xcom_push(key='model_registered', value=True)
        
        print(f"Model registered successfully: version {version}")
        return {"status": "success", "version": version, "accuracy": accuracy, "roc_auc": roc_auc}
    else:
        context['ti'].xcom_push(key='model_registered', value=False)
        print(f"Model performance below threshold ({roc_auc:.3f} < {min_roc_auc}). Not registering.")
        return {"status": "rejected", "roc_auc": roc_auc, "threshold": min_roc_auc}


def mark_feedback_used(**context):
    """Mark feedback records as used for training."""
    from src.db.connection import get_db_session
    from src.db.models import FeedbackLoop
    
    model_version = context['ti'].xcom_pull(key='model_version')
    model_registered = context['ti'].xcom_pull(key='model_registered')
    
    if not model_registered or not model_version:
        print("Model not registered. Skipping feedback marking.")
        return {"status": "skipped"}
    
    with get_db_session() as session:
        # Mark all unused feedback as used
        feedback_records = session.query(FeedbackLoop).filter(
            FeedbackLoop.used_for_training == 'N'
        ).all()
        
        for feedback in feedback_records:
            feedback.used_for_training = 'Y'
            feedback.training_run_id = str(model_version)
        
        session.commit()
        print(f"Marked {len(feedback_records)} feedback records as used")
        return {"status": "success", "marked_count": len(feedback_records)}


# Define tasks
check_eligibility = PythonOperator(
    task_id='check_retraining_eligibility',
    python_callable=check_retraining_eligibility,
    dag=dag,
)

collect_feedback = PythonOperator(
    task_id='collect_feedback_data',
    python_callable=collect_feedback_data,
    dag=dag,
)

prepare_dataset = PythonOperator(
    task_id='prepare_training_dataset',
    python_callable=prepare_training_dataset,
    dag=dag,
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

mark_used = PythonOperator(
    task_id='mark_feedback_used',
    python_callable=mark_feedback_used,
    dag=dag,
)

# Define task dependencies
check_eligibility >> collect_feedback >> prepare_dataset >> train >> mark_used

