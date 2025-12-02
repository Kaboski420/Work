"""gRPC server for Virality Engine."""

import logging
import asyncio
from concurrent import futures
import grpc
from typing import Dict, Any

from src.config import settings
from src.services.ingestion.service import IngestionService
from src.services.scoring.service import ScoringService

logger = logging.getLogger(__name__)

# Try to import generated protobuf code
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))
    
    # Generate protobuf files if needed
    proto_dir = os.path.join(os.path.dirname(__file__), 'proto')
    proto_file = os.path.join(proto_dir, 'virality.proto')
    
    if os.path.exists(proto_file):
        # Try to import generated code
        try:
            import virality_pb2
            import virality_pb2_grpc
            GRPC_AVAILABLE = True
        except ImportError:
            logger.warning("gRPC protobuf files not generated. Run: python -m grpc_tools.protoc -I src/api/proto --python_out=src/api/proto --grpc_python_out=src/api/proto src/api/proto/virality.proto")
            GRPC_AVAILABLE = False
    else:
        GRPC_AVAILABLE = False
        logger.warning("Proto file not found")
except Exception as e:
    GRPC_AVAILABLE = False
    logger.warning(f"gRPC setup failed: {e}")


class ViralityServicer(virality_pb2_grpc.ViralityServiceServicer if GRPC_AVAILABLE else object):
    """gRPC service implementation."""
    
    def __init__(self):
        self.ingestion_service = IngestionService()
        self.scoring_service = ScoringService()
    
    def HealthCheck(self, request, context):
        """Health check endpoint."""
        return virality_pb2.HealthResponse(
            status="healthy",
            service=settings.service_name,
            version=settings.service_version
        )
    
    def ScoreContent(self, request, context):
        """Score content endpoint."""
        try:
            # Convert protobuf to dict
            features = dict(request.features)
            embeddings = dict(request.embeddings)
            metadata = dict(request.metadata)
            
            # Call scoring service (async, but gRPC is sync)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.scoring_service.score_content(
                    content_id=request.content_id,
                    features=features,
                    embeddings=embeddings,
                    metadata=metadata
                )
            )
            loop.close()
            
            # Convert to protobuf response
            attribution_insights = [
                virality_pb2.AttributionInsight(
                    factor=insight["factor"],
                    weight=insight["weight"],
                    impact=insight["impact"]
                )
                for insight in result.get("attribution_insights", [])
            ]
            
            return virality_pb2.ScoringResponse(
                content_id=result["content_id"],
                timestamp=result["timestamp"],
                virality_probability=result["virality_probability"],
                confidence_level=result["confidence_level"],
                attribution_insights=attribution_insights,
                model_lineage=virality_pb2.ModelLineage(
                    model_version=result["model_lineage"]["model_version"],
                    reproducibility_hash=result["model_lineage"]["reproducibility_hash"],
                    training_timestamp=result["model_lineage"]["training_timestamp"]
                ),
                recommendations=virality_pb2.Recommendations(
                    tactical=result["recommendations"].get("tactical", []),
                    strategic=result["recommendations"].get("strategic", [])
                )
            )
        except Exception as e:
            logger.error(f"Error in ScoreContent: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return virality_pb2.ScoringResponse()
    
    def IngestContent(self, request, context):
        """Ingest content endpoint."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.ingestion_service.ingest_content(
                    content_type=request.content_type,
                    content_data=bytes(request.content_data),
                    metadata={
                        "caption": request.caption,
                        "description": request.description,
                        "hashtags": list(request.hashtags),
                        "filename": "uploaded_file"
                    },
                    platform=request.platform
                )
            )
            loop.close()
            
            return virality_pb2.IngestResponse(
                content_id=result["content_id"],
                platform=result["platform"],
                content_type=result["content_type"],
                timestamp=result["timestamp"],
                embeddings={k: str(v) for k, v in result.get("embeddings", {}).items()},
                features={k: str(v) for k, v in result.get("features", {}).items()},
                metadata={k: str(v) for k, v in result.get("metadata", {}).items()},
                storage_path=result.get("storage_path", "")
            )
        except Exception as e:
            logger.error(f"Error in IngestContent: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return virality_pb2.IngestResponse()
    
    def AnalyzeContent(self, request, context):
        """Analyze content (ingest + score) endpoint."""
        try:
            # First ingest
            ingest_result = self.IngestContent(request, context)
            if context.code() != grpc.StatusCode.OK:
                return virality_pb2.AnalyzeResponse()
            
            # Then score
            scoring_request = virality_pb2.ScoringRequest(
                content_id=ingest_result.content_id,
                features=ingest_result.features,
                embeddings=ingest_result.embeddings,
                metadata=ingest_result.metadata
            )
            scoring_result = self.ScoreContent(scoring_request, context)
            
            return virality_pb2.AnalyzeResponse(
                content_id=ingest_result.content_id,
                timestamp=ingest_result.timestamp,
                embeddings=ingest_result.embeddings,
                features=ingest_result.features,
                metadata=ingest_result.metadata,
                scoring=scoring_result
            )
        except Exception as e:
            logger.error(f"Error in AnalyzeContent: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return virality_pb2.AnalyzeResponse()


def serve(port: int = 50051):
    """Start gRPC server."""
    if not GRPC_AVAILABLE:
        logger.error("gRPC not available. Cannot start server.")
        return
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    virality_pb2_grpc.add_ViralityServiceServicer_to_server(
        ViralityServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        server.stop(0)


if __name__ == '__main__':
    serve()




