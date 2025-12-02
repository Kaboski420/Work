"""Kafka messaging utilities for async communication."""

import logging
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import kafka
try:
    from kafka import KafkaProducer
    try:
        from kafka import KafkaConsumer
    except ImportError:
        # Some versions have different import path
        try:
            from kafka.consumer import KafkaConsumer
        except ImportError:
            KafkaConsumer = None
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaConsumer = None
    logger.warning("kafka-python not available. Messaging will be disabled.")


class KafkaMessagingService:
    """Kafka messaging service for async communication."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """
        Initialize Kafka messaging service.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumers = {}
        
        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3
                )
                logger.info(f"Kafka producer connected: {bootstrap_servers}")
            except Exception as e:
                logger.warning(f"Kafka not available: {e}. Messaging disabled.")
                self.producer = None
        else:
            logger.warning("Kafka library not available. Messaging disabled.")
    
    def produce(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Produce message to Kafka topic.
        
        Args:
            topic: Topic name
            value: Message value (dict)
            key: Message key (optional)
            
        Returns:
            True if successful
        """
        if not self.producer:
            return False
        
        try:
            future = self.producer.send(topic, value=value, key=key)
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Message sent to {topic} "
                f"[partition={record_metadata.partition}, offset={record_metadata.offset}]"
            )
            return True
        except KafkaError as e:
            logger.error(f"Error producing message to {topic}: {e}")
            return False
    
    def create_consumer(
        self,
        topic: str,
        group_id: str,
        auto_offset_reset: str = "earliest"
    ) -> Optional[Any]:  # type: ignore
        """
        Create Kafka consumer.
        
        Args:
            topic: Topic name
            group_id: Consumer group ID
            auto_offset_reset: Offset reset policy
            
        Returns:
            KafkaConsumer or None
        """
        if not KAFKA_AVAILABLE or KafkaConsumer is None:
            return None
        
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True
            )
            logger.info(f"Kafka consumer created for topic {topic}, group {group_id}")
            return consumer
        except Exception as e:
            logger.error(f"Error creating consumer: {e}")
            return None
    
    def consume_messages(
        self,
        consumer: Any,
        callback: Callable[[Dict[str, Any]], None],
        timeout_ms: int = 1000
    ):
        """
        Consume messages from Kafka.
        
        Args:
            consumer: KafkaConsumer instance
            callback: Callback function to process messages
            timeout_ms: Timeout in milliseconds
        """
        if not consumer:
            return
        
        try:
            for message in consumer:
                try:
                    callback(message.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
    
    def close(self):
        """Close producer and consumers."""
        if self.producer:
            self.producer.close()
        for consumer in self.consumers.values():
            if consumer:
                consumer.close()

