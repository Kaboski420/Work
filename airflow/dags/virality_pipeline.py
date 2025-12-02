"""
Apache Airflow DAG for Virality Engine Pipeline

Orchestrates the end-to-end workflow:
1. Content ingestion
2. Feature extraction
3. Multimodal analysis
4. Temporal modeling
5. Virality scoring
6. Result storage
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
    'virality_engine_pipeline',
    default_args=default_args,
    description='End-to-end virality prediction pipeline',
    schedule_interval=timedelta(minutes=1),  # Adjust as needed
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['virality', 'ml', 'prediction'],
)


def ingest_content_task(**context):
    """Task to ingest content and extract features."""
    import asyncio
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.services.ingestion.service import IngestionService
    
    async def ingest():
        service = IngestionService()
        
        # Get content from context or trigger
        # In production, this would come from a queue or trigger
        content_data = context.get("dag_run").conf.get("content_data", b"")
        content_type = context.get("dag_run").conf.get("content_type", "video")
        platform = context.get("dag_run").conf.get("platform", "tiktok")
        metadata = context.get("dag_run").conf.get("metadata", {})
        
        if not content_data:
            print("No content data provided in DAG run config")
            return {"content_id": None, "status": "skipped"}
        
        result = await service.ingest_content(
            content_type=content_type,
            content_data=content_data,
            metadata=metadata,
            platform=platform
        )
        
        # Push result to XCom for next task
        context['ti'].xcom_push(key='content_id', value=result.get("content_id"))
        context['ti'].xcom_push(key='ingestion_result', value=result)
        
        return {"content_id": result.get("content_id"), "status": "ingested"}
    
    return asyncio.run(ingest())


def extract_features_task(**context):
    """Task to extract multimodal features."""
    # Features are already extracted during ingestion
    ingestion_result = context['ti'].xcom_pull(key='ingestion_result')
    
    if not ingestion_result:
        print("No ingestion result found")
        return {"status": "skipped"}
    
    features = ingestion_result.get("features", {})
    embeddings = ingestion_result.get("embeddings", {})
    
    context['ti'].xcom_push(key='features', value=features)
    context['ti'].xcom_push(key='embeddings', value=embeddings)
    
    print(f"Features extracted: {list(features.keys())}")
    return {"status": "features_extracted", "feature_count": len(features)}


def analyze_media_task(**context):
    """Task to analyze visual and audio content."""
    # Media analysis is already done in ingestion via MediaIntelligenceService
    features = context['ti'].xcom_pull(key='features')
    
    if not features:
        print("No features found")
        return {"status": "skipped"}
    
    visual_features = features.get("visual", {})
    audio_features = features.get("audio", {})
    
    print(f"Visual features: {list(visual_features.keys())}")
    print(f"Audio features: {list(audio_features.keys())}")
    
    return {
        "status": "media_analyzed",
        "visual_features": len(visual_features),
        "audio_features": len(audio_features)
    }


def model_temporal_task(**context):
    """Task to model temporal behavior."""
    import asyncio
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.services.temporal.service import TemporalModelingService
    
    async def model_temporal():
        service = TemporalModelingService()
        content_id = context['ti'].xcom_pull(key='content_id')
        
        if not content_id:
            print("No content_id found")
            return {"status": "skipped"}
        
        # Get initial metrics (would come from engagement data in production)
        initial_metrics = {
            "content_id": content_id,
            "views": 0.0,
            "likes": 0.0,
            "shares": 0.0,
            "comments": 0.0
        }
        
        trajectory = await service.predict_engagement_trajectory(
            content_id=content_id,
            initial_metrics=initial_metrics,
            historical_patterns=None
        )
        
        context['ti'].xcom_push(key='trajectory', value=trajectory)
        
        return {"status": "temporal_modeled", "momentum_score": trajectory.get("velocity", {}).get("momentum_score", 0.0)}
    
    return asyncio.run(model_temporal())


def score_virality_task(**context):
    """Task to generate virality score."""
    import asyncio
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.services.scoring.service import ScoringService
    
    async def score():
        service = ScoringService()
        content_id = context['ti'].xcom_pull(key='content_id')
        features = context['ti'].xcom_pull(key='features')
        embeddings = context['ti'].xcom_pull(key='embeddings')
        ingestion_result = context['ti'].xcom_pull(key='ingestion_result')
        
        if not content_id or not features or not embeddings:
            print("Missing required data for scoring")
            return {"status": "skipped"}
        
        metadata = ingestion_result.get("metadata", {}) if ingestion_result else {}
        
        result = await service.score_content(
            content_id=content_id,
            features=features,
            embeddings=embeddings,
            metadata=metadata
        )
        
        context['ti'].xcom_push(key='scoring_result', value=result)
        
        return {
            "status": "scored",
            "virality_probability": result.get("virality_probability", 0.0),
            "virality_score": result.get("virality_score", 0.0)
        }
    
    return asyncio.run(score())


def store_results_task(**context):
    """Task to store results in database."""
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.db.connection import get_db_session
    from src.db.models import ViralityScore, ContentItem
    import uuid
    from datetime import datetime
    
    scoring_result = context['ti'].xcom_pull(key='scoring_result')
    content_id = context['ti'].xcom_pull(key='content_id')
    ingestion_result = context['ti'].xcom_pull(key='ingestion_result')
    
    if not scoring_result or not content_id:
        print("No scoring result to store")
        return {"status": "skipped"}
    
    try:
        with get_db_session() as session:
            # Store or update content item
            content_item = session.query(ContentItem).filter(
                ContentItem.content_id == content_id
            ).first()
            
            if not content_item and ingestion_result:
                content_item = ContentItem(
                    content_id=content_id,
                    platform=ingestion_result.get("platform", "unknown"),
                    content_type=ingestion_result.get("content_type", "unknown"),
                    content_metadata=ingestion_result.get("metadata", {})
                )
                session.add(content_item)
            
            # Store virality score
            score_record = ViralityScore(
                id=str(uuid.uuid4()),
                content_id=content_id,
                virality_probability=scoring_result.get("virality_probability", 0.0),
                confidence_level=scoring_result.get("confidence_level", 0.0),
                attribution_insights=scoring_result.get("attribution_insights", []),
                model_version=scoring_result.get("model_lineage", {}).get("model_version", "0.1.0"),
                model_hash=scoring_result.get("model_lineage", {}).get("reproducibility_hash", "default"),
                recommendations=scoring_result.get("recommendations", {}),
                created_at=datetime.utcnow()
            )
            session.add(score_record)
            session.commit()
            
            print(f"Stored results for content {content_id}")
            return {"status": "stored", "content_id": content_id}
    except Exception as e:
        print(f"Error storing results: {e}")
        return {"status": "error", "error": str(e)}


# Define tasks
ingest = PythonOperator(
    task_id='ingest_content',
    python_callable=ingest_content_task,
    dag=dag,
)

extract_features = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features_task,
    dag=dag,
)

analyze_media = PythonOperator(
    task_id='analyze_media',
    python_callable=analyze_media_task,
    dag=dag,
)

model_temporal = PythonOperator(
    task_id='model_temporal',
    python_callable=model_temporal_task,
    dag=dag,
)

score_virality = PythonOperator(
    task_id='score_virality',
    python_callable=score_virality_task,
    dag=dag,
)

store_results = PythonOperator(
    task_id='store_results',
    python_callable=store_results_task,
    dag=dag,
)

# Define task dependencies
ingest >> extract_features >> analyze_media >> model_temporal >> score_virality >> store_results



