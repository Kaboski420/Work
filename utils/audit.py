"""
Audit Logging Service with Immutability.

Implements append-only, tamper-proof audit logging for compliance.
"""

import logging
import hashlib
import json
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import os

from src.config import settings
from src.utils.security import EncryptionService

logger = logging.getLogger(__name__)


class AuditLogService:
    """
    Immutable audit logging service.
    
    Features:
    - Append-only logs (no modification/deletion)
    - Cryptographic hashing for tamper detection
    - Encrypted storage option
    - Chain of custody tracking
    """
    
    def __init__(self, log_dir: Optional[str] = None, encrypt: bool = False):
        """
        Initialize audit log service.
        
        Args:
            log_dir: Directory for audit logs (default: ./audit_logs)
            encrypt: Whether to encrypt audit logs
        """
        self.log_dir = Path(log_dir or "./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.encrypt = encrypt
        self.encryption_service = EncryptionService() if encrypt else None
        
        # Chain hash for tamper detection
        self.chain_hash_file = self.log_dir / "chain_hash.txt"
        self.last_hash = self._load_last_hash()
    
    def _load_last_hash(self) -> Optional[str]:
        """Load last chain hash from file."""
        if self.chain_hash_file.exists():
            try:
                return self.chain_hash_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Error loading chain hash: {e}")
        return None
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _calculate_chain_hash(self, previous_hash: Optional[str], current_data: str) -> str:
        """
        Calculate chain hash for tamper detection.
        
        Chain hash = SHA256(previous_hash + current_data)
        This creates an immutable chain where any modification breaks the chain.
        """
        if previous_hash:
            combined = previous_hash + current_data
        else:
            combined = current_data
        return self._calculate_hash(combined)
    
    def log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        action: str = "",
        resource: str = "",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ) -> str:
        """
        Log an audit event (append-only).
        
        Args:
            event_type: Type of event (e.g., "authentication", "data_access", "data_modification")
            user_id: User identifier
            action: Action performed
            resource: Resource accessed/modified
            details: Additional details
            ip_address: Client IP address
            user_agent: User agent string
            success: Whether action was successful
        
        Returns:
            Log entry ID (hash)
        """
        timestamp = datetime.utcnow()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success
        }
        
        # Serialize to JSON
        log_data = json.dumps(log_entry, sort_keys=True, default=str)
        
        # Calculate entry hash
        entry_hash = self._calculate_hash(log_data)
        
        # Calculate chain hash (includes previous hash for immutability)
        chain_hash = self._calculate_chain_hash(self.last_hash, log_data)
        
        # Add hash to entry
        log_entry["entry_hash"] = entry_hash
        log_entry["chain_hash"] = chain_hash
        
        # Final log entry with hash
        final_log_data = json.dumps(log_entry, sort_keys=True, default=str)
        
        # Encrypt if enabled
        if self.encrypt and self.encryption_service:
            encrypted_data = self.encryption_service.encrypt(final_log_data.encode())
            log_line = base64.b64encode(encrypted_data).decode('utf-8')
        else:
            log_line = final_log_data
        
        # Append to log file (append-only)
        log_file = self.log_dir / f"audit_{timestamp.strftime('%Y-%m-%d')}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
        
        # Update chain hash
        self.last_hash = chain_hash
        self.chain_hash_file.write_text(chain_hash)
        
        logger.info(f"Audit log entry created: {entry_hash[:16]}...")
        
        return entry_hash
    
    def verify_integrity(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify integrity of audit logs for a given date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            Verification result with integrity status
        """
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        
        log_file = self.log_dir / f"audit_{date}.log"
        
        if not log_file.exists():
            return {
                "date": date,
                "verified": False,
                "error": "Log file not found",
                "entries_checked": 0
            }
        
        verified = True
        entries_checked = 0
        previous_hash = None
        errors = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Decrypt if needed
                    if self.encrypt and self.encryption_service:
                        try:
                            encrypted_data = base64.b64decode(line)
                            decrypted_data = self.encryption_service.decrypt(encrypted_data)
                            log_entry = json.loads(decrypted_data.decode('utf-8'))
                        except Exception as e:
                            errors.append(f"Line {line_num}: Decryption failed: {e}")
                            verified = False
                            continue
                    else:
                        log_entry = json.loads(line)
                    
                    # Verify entry hash
                    entry_hash = log_entry.pop("entry_hash")
                    chain_hash = log_entry.pop("chain_hash")
                    
                    log_data = json.dumps(log_entry, sort_keys=True, default=str)
                    calculated_entry_hash = self._calculate_hash(log_data)
                    
                    if calculated_entry_hash != entry_hash:
                        errors.append(f"Line {line_num}: Entry hash mismatch")
                        verified = False
                    
                    # Verify chain hash
                    calculated_chain_hash = self._calculate_chain_hash(previous_hash, log_data)
                    if calculated_chain_hash != chain_hash:
                        errors.append(f"Line {line_num}: Chain hash mismatch - possible tampering")
                        verified = False
                    
                    previous_hash = chain_hash
                    entries_checked += 1
        
        except Exception as e:
            return {
                "date": date,
                "verified": False,
                "error": str(e),
                "entries_checked": entries_checke
            }
        
        return {
            "date": date,
            "verified": verified,
            "entries_checked": entries_checked,
            "errors": errors if errors else None
        }
    
    def query_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            event_type: Filter by event type
            user_id: Filter by user ID
            limit: Maximum number of results
        
        Returns:
            List of log entries
        """
        results = []
        
        # Determine date range
        if start_date is None:
            start_date = datetime.utcnow().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = start_date
        
        # Iterate through date range
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            log_file = self.log_dir / f"audit_{date_str}.log"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(results) >= limit:
                                break
                            
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Decrypt if needed
                            if self.encrypt and self.encryption_service:
                                try:
                                    encrypted_data = base64.b64decode(line)
                                    decrypted_data = self.encryption_service.decrypt(encrypted_data)
                                    log_entry = json.loads(decrypted_data.decode('utf-8'))
                                except Exception:
                                    continue
                            else:
                                log_entry = json.loads(line)
                            
                            # Apply filters
                            if event_type and log_entry.get("event_type") != event_type:
                                continue
                            if user_id and log_entry.get("user_id") != user_id:
                                continue
                            
                            results.append(log_entry)
                
                except Exception as e:
                    logger.warning(f"Error reading log file {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return results


# Global audit log service instance
audit_log = AuditLogService(
    log_dir=os.getenv("AUDIT_LOG_DIR", "./audit_logs"),
    encrypt=os.getenv("AUDIT_LOG_ENCRYPT", "false").lower() == "true"
)

