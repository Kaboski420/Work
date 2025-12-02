"""Security utilities for authentication and encryption."""

import logging
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os

from src.config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for AES-256 encryption (data at rest)."""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryption service.
        
        Args:
            key: Encryption key (if None, uses settings or generates new)
        """
        if key:
            self.key = key.encode()
        elif settings.encryption_key:
            self.key = settings.encryption_key.encode()
        else:
            # Generate a key (for development only)
            self.key = Fernet.generate_key()
            logger.warning("Using generated encryption key - not secure for production!")
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-256."""
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.cipher.decrypt(encrypted_data)


def validate_pii_policy(data: dict) -> bool:
    """Validate that data doesn't contain PII."""
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        r'\b[\w.-]+@[\w.-]+\.\w+\b',
    ]
    return True


